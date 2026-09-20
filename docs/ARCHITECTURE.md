# Architecture（完整数据流，逐箭头标注实现）

> 本文所有箭头均从代码实际追踪得出，标注 `文件 :: 函数/类`。公式不改写，只引用。
> 阅读本文前建议先看 `CURRENT_CODE_MAP.md` 了解文件分类。

## A. 渲染数据流（SMPL-X → RGB / Depth / Mask）

```text
BEDLAM clothing NPZ + SMPL-X motion
        ↓
   [canonical motion NPZ]
   tools/generate/load_real_smplx_motion.py :: main()
     （poses[165] 切片为 global_orient/body_pose/jaw/eye/hand；betas；可选 transl；帧选择）
   或 tools/generate/generate_p0_toy_motion.py :: main()   （程序化 toy motion，直接产几何）
        ↓
   SMPL-X forward（pytorch smplx 模型）
   tools/generate/generate_real_motion_geometry.py :: main()
     - smplx.SMPLX(model_dir, gender, num_betas, use_pca=False, flat_hand_mean=True)
     - vertices [T,10475,3] float32；faces = model.faces
        ↓
   DenseKP 3D（同一几何源）
   tools/generate/generate_real_motion_geometry.py :: main()
     - dense = np.einsum('kv,tvc->tkc', R, vertices)   # R=assets/densekp_regressor_R.npy [138,10475]
   （toy 路径等价实现在 tools/generate/generate_p0_toy_motion.py）
        ↓
   geometry/ 目录制品（vertices.npy / faces.npy / densekp_3d.npy /
   motion_params.npz / geometry_manifest.json）
        ↓
   [可选] 衣物几何
   tools/generate/prepare_p0_dressed_geometry.py :: main()
     - clothing_vertices [T,13027,3]，clothing_faces [26328,3]，UV，clothing_manifest.json
        ↓
   Blender headless 渲染
   tools/render/render_p0_blender.py :: main()      （由 wrapper 以 blender -b --python 调起：
     tools/render/run_eye_background_fix.py / tools/render/run_temporal_case.py /
     tools/render/run_camera_stress.py）
     - me.vertices.foreach_set('co', vs[f])        # 每帧原样写入 SMPL-X 局部坐标
     - conv = Matrix(((1,0,0,0),(0,0,-1,0),(0,1,0,0),(0,0,0,1)))   # SMPL-X→Blender
       ob.matrix_world = Translation((0,0,-min_y)) @ conv          # Z 抬升落地面
     - 相机：PERSP lens=50mm sensor 36mm HORIZONTAL，loc (0,-3.2,1)，euler (pi/2,0,0)
     - compositor 输出 multilayer EXR：Depth pass + Alpha(Mask) pass
     - 同时解析计算 DenseKP 2D（见 B 段）并与 world_to_camera_view 交叉验证
     - --camera-placement（来自 solve_camera_stress_placement.py）可逐帧替换 cam.matrix_world
        ↓
   后处理 / 注记
   tools/render/finalize_p0_outputs.py :: main()
     - EXR→NPY depth；mask=(Alpha>=.5)*255
     - depth 语义自动判定（已知平面测试 → CAMERA_SPACE_Z / EUCLIDEAN_RAY_DISTANCE）
     - visibility 状态：0 VISIBLE / 1 SELF_OCCLUDED / 2 OUT_OF_FRAME / 3 BEHIND_CAMERA
       （3×3 邻域最小 |depth - point_depth|，容差 max(0.002, 2*median/fx)）
     - overlay / contact sheet / p0_output_report.json
        ↓
   RGB (rgb/*.png) + Depth (depth/*.npy) + Mask (mask/*.png)
   + annotations/densekp_{2d,camera_xyz,depth,valid,visible,visibility_state}.npy
   + manifest.jsonl（每帧相机/内参/物体矩阵，可独立复算投影）
```

### A.1 相机压力分支（近距/边缘/截断）

```text
configs/perspective_camera_stress.json
        ↓
tools/render/solve_camera_stress_placement.py :: solve()/main()
   - least_squares 求相机局部平移（内参与几何不变），输出 outputs/.../configs/<name>.json
        ↓
tools/render/run_camera_stress.py :: main()
   - 每配置调用 tools/render/run_eye_background_fix.py --camera-placement
        ↓  （→ render_p0_blender.py，placement 直接写 cam.matrix_world）
tools/render/finalize_p0_outputs.py（每个 _renders/<name> 目录，产出 densekp_visible.npy）
        ↓
tools/render/finalize_camera_stress.py :: main()
   - 汇总 128 样本注记 + 验收 gate + manifests/all_samples.jsonl
```

### A.2 Fisheye / ERP / 场景分支

```text
geometry/（同 A）
        ↓  SMPL-X→Blender world 在 NumPy 中做等价变换
tools/fisheye/run_fisheye_camera_p0.py :: main()
   - lift = -min(y)；world = column_stack((x, -z, y)) + lift      # = conv 矩阵的列变换
   - pose(target, origin, yaw_deg)                                # camera_to_world，CV 约定
        ↓
Blender ERP 源渲染（PANO EQUIRECTANGULAR 相机）
   tools/fisheye/render_equirectangular_source.py            （HDRI-only）
   tools/fisheye/render_equirectangular_scene_source.py      （+ .blend 场景，Phase 18）
        ↓ import tools/scene/blender_scene_bridge.py :: link_blend_scene/set_visibility/scene_bounds
        ↓
tools/fisheye/run_fisheye_camera_p0.py :: warp_case()
   - read_exr → unproject_pixels_omni_radtan → ERP remap mapx/mapy
   - cv2.remap rgb(线性)/depth(最近邻)/index(最近邻)；mask=(warped_index>.5)
   - project_points_omni_radtan(points, camera, k) → densekp fisheye 2D
   - 可见性：投影点 5×5 邻域 |warped_depth - point_range| < 4 cm
        ↓
   center/ + seam/cam_left + seam/cam_right（Phase 17）注记与报告
        ↓ 驱动脚本
   tools/fisheye/run_fisheye_seam_sweep.py        （yaw 扫描）
   tools/fisheye/run_human_scale_sweep.py         （距离扫描）
   tools/fisheye/run_fisheye_scene_p0.py          （Phase 18：场景资产 spec 校验
        ↓ tools/scene/scene_loader.py :: SceneAssetSpec.validate()
          缺资产 → SCENE_ASSET_MISSING（不伪造））
```

## B. DenseKP GT 数据流（SMPL-X → 训练标注）

```text
SMPL-X vertices [10475,3]（几何制品内 = SMPL-X 局部坐标，米）
        ↓  138 点 correspondence
assets/densekp_regressor_R.npy  [138,10475] float32 线性顶点回归算子
   densekp_3d[t] = R @ vertices[t]
   （tools/generate/generate_real_motion_geometry.py、generate_p0_toy_motion.py、
    generate_bedlam_densekp.py :: main() 中 R@V）
        ↓  world transform（逐阶段不同，见 COORDINATE_SYSTEMS.md）
   perspective 渲染：tools/render/render_p0_blender.py :: main()
     world = ob.matrix_world @ Vector(p)     # conv + Z 抬升逐帧写 manifest
   fisheye：tools/fisheye/run_fisheye_camera_p0.py :: main()
     world = column_stack((x, -z, y + lift))（+ 场景 yaw/平移，Phase 18）
        ↓  camera transform
   perspective：q = cam.matrix_world.inverted() @ world（Blender 相机局部系）
        ↓  2D projection
   perspective：tools/render/render_p0_blender.py:199
     u = fx*q.x/z + cx ; v = cy - fy*q.y/z ; z = -q.z     # 左上原点
     交叉验证：bpy_extras.world_to_camera_view → (ndc.x*w, (1-ndc.y)*h)
   fisheye：tools/fisheye/omni_radtan.py :: project_points_omni_radtan
        ↓  visibility
   perspective：tools/render/finalize_p0_outputs.py（depth 状态 0-3，容差 max(0.002,2*median/fx)）
   fisheye：tools/fisheye/run_fisheye_camera_p0.py :: warp_case（4cm，5×5 邻域）
        ↓  训练格式（bbox → crop）
tools/annotation/generate_bedlam_densekp.py
   - project_official()：BEDLAM 官方 (points+t)/z 再左乘 K，输出 [x,y,1.0]
   - official_bbox()：前 22 个 joints2d、图内过滤、1.2 扩张 → center/scale=max(w,h)/200*1.2
   - 写 NPZ：imgname / center / scale / proj_verts（[N,138,3]，xy=原图像素，第三通道=1.0）
   或 tools/render/finalize_camera_stress.py（非训练格式：densekp_3d/2d/depth/valid/in_frame/visible/visibility_state）
        ↓
densekp/core/datasets/dataset_keypoints.py::DatasetKeypoints.__getitem__
   bbox_size = expand_to_aspect_ratio(scale*200, BBOX_SHAPE).max()
        ↓
densekp/core/datasets/utils.py::get_example_projverts
   同一 trans 变换图像与 proj_verts[:,:2]；xy/256-0.5
        ↓
item['proj_verts_cropped'] → densekp/core/densekp_trainer.py::DenseKP.compute_loss
   kpts_loss = mean( sum(gt_xy - pred_xy)² )（with_var 时除以 2σ² + log σ²）→ ×60
```

## C. 训练数据流（与 B 尾部衔接）

```text
densekp/train_keypoints.py :: main()           # Hydra: configs/hydra/train.yaml
        ↓
densekp/core/configs/__init__.py :: dataset_config()   # 冻结空 yacs 壳
        ↓
densekp/core/datasets/datamodule_keypoints.py::DataModule
   train: cfg.DATASETS.DATASETS_AND_RATIOS.split('_') → 多个 DatasetKeypoints → ConcatDataset
   val  : cfg.DATASETS.VAL_DATASETS.split('_') → 每个 val 数据集独立 DataLoader
        ↓
DatasetKeypoints.__getitem__ → item{img, img_disp, proj_verts, proj_verts_cropped,
                                    box_center, box_size, img_size, _scale,
                                    _trans, imgname, dataset}
        ↓
densekp_trainer.DenseKP.forward_step
   x[:,:,:,32:-32]                     # [B,3,256,192]
   → backbones/vit.py::vit()           # ViTPose-H backbone（create_backbone）
   → heads/smpl_head_keypoints.py::KeypointsHead
       token 展平 + TransformerDecoder（components/pose_transformer.py）
       → nn.Linear(dim, 138*3) → [B,138,3]
        ↓
densekp_trainer.compute_loss
   gt = batch['proj_verts_cropped'][:,:,:2]
   with_var: loss = mean(1000·d²/(2σ²)) + mean(log σ²)；else mean(d²)；总 ×60
        ↓
training_step（手动优化：zero_grad → manual_backward → step，AdamW）
```

## D. 验证/回归链（现成的 regression 机制）

| 机制 | 文件 :: 函数 | 校验内容 |
|---|---|---|
| mesh↔数组一致 | render_p0_blender :: main（foreach_get 回读，diff==0 才继续） | Blender 未改几何 |
| 投影交叉验证 | render_p0_blender :: main（解析 vs world_to_camera_view，阈值 0.1px） | 解析投影 = Blender 投影 |
| 几何指纹 | 各脚本的 blake2b-128 fingerprint（vertices/clothing/densekp_3d） | 整理/纹理接入前后几何不变 |
| depth 语义 | render_p0_blender :: plane_test + finalize_p0_outputs（自动判定） | Depth pass = camera-space Z |
| silhouette | tools/validate/p0_pyrender_crosscheck.py | Blender mask vs pyrender IoU≈0.999 |
| 相机模型单测 | tools/fisheye/test_omni_radtan.py | Omni-Radtan roundtrip <3e-8 rad |
| 训练闭环 | tools/validate/validate_bedlam_densekp_tiny.py | 生成 NPZ 能被 DatasetKeypoints 正确消费 |
| 相机压力 gate | finalize_camera_stress :: acceptance | bbox 高度比例/截断/边缘放置达标 |

## E. 尚未存在（后续要新建）的环节

根据 `docs/TODO/TODO.md` 与多阶段规划，以下能力**当前代码中没有**，不要假设已实现：

- 多人渲染与 person-person occlusion（当前 mask 只有单人体 pass_index 1/2；场景 pass_index 0）；
- fisheye/ERP 域的 `DatasetKeypoints` 训练 NPZ 生成（fisheye 注记是 [T,138,2] npy，不是训练 NPZ）；
- 骑行 mocap 重定向到 SMPL-X（audit 工具已备，retarget 未做）；
- 截断/近距样本的端到端训练验证（相机压力集已产出注记，但未转成训练 NPZ）。
