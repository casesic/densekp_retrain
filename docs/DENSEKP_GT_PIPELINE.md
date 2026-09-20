# DenseKP GT Pipeline（138 点从哪来、到哪去）

> 回答核心问题："这 138 个点究竟是怎么产生的？"
> 全部结论从代码与既有审计文档（BEDLAM_OPERATOR_AUDIT.md、TRAINING_AUDIT.md、DATA_FORMAT.md）确认，无推测处均标注。

## 1. 138 点 correspondence 存在哪里？

**存储形式：线性顶点回归矩阵（vertex regressor），不是 vertex ID 列表，也不是 barycentric 面。

- 文件：`assets/densekp_regressor_R.npy`
- shape：`[138, 10475]`，dtype `float32`（generate_bedlam_densekp.py:101 强制校验）
- 语义：第 `i` 行是对 SMPL-X 10475 个顶点的稀疏线性组合权重：
  `point_i = R[i] @ V`，`V` 为 `[10475,3]` SMPL-X 顶点
- 稀疏性：非零元素总数 204（TRAINING_AUDIT.md #6），即每行 1–3 个顶点的加权和；
  行权重和 ≈1（inspect_surface_operator.py 输出 row_sum 范围）
- 点的身份：**仅由行索引 0..137 决定**；仓库内**没有**任何解剖语义名称表
  （TRAINING_AUDIT.md #6 明确 UNKNOWN）
- 第三方对照：`tools/annotation/inspect_surface_operator.py` 可导出每行的顶点 ID 与权重 JSON；
  `tools/annotation/generate_bedlam_densekp.py::print_operator_rows` 打印 0/1/10/50/100/137 行的精确 support

因此：**不是 vertex ID、不是 barycentric mesh lookup，是逐行固定权重的线性顶点回归**。
它与 SMPL-X 拓扑（10475 顶点）绑定。

## 2. 生成路径（代码级）

### 2.1 SMPL-X → DenseKP 3D（本仓库三条等价实现）

| 场景 | 文件 :: 位置 | 公式 |
|---|---|---|
| toy motion | tools/generate/generate_p0_toy_motion.py（main 内 R@V） | `densekp_3d[t] = R @ vertices[t]` |
| 真实 motion（主路径） | tools/generate/generate_real_motion_geometry.py:15 | `np.einsum('kv,tvc->tkc', r, vertices, optimize=True).astype(np.float32)` |
| BEDLAM tiny | tools/annotation/generate_bedlam_densekp.py:156 | `densekp_3d = np.asarray(R, dtype=np.float64) @ vertices`（float64 复算） |

输出：`geometry/densekp_3d.npy`，shape `[T,138,3]`，**SMPL-X local 坐标，米**。

### 2.2 DenseKP 3D → world（分域）

- Perspective 渲染域：`tools/render/render_p0_blender.py:198`
  `world = ob.matrix_world @ Vector(p)`（conv + Z 抬升，逐帧写 manifest）
- Fisheye 域：`tools/fisheye/run_fisheye_camera_p0.py:77`
  `world = column_stack((x, -z, y+lift))`（Phase 18 再加 yaw/平移，run_fisheye_scene_p0.py:47-51）
- BEDLAM tiny 域：不做 world 变换；BEDLAM 记录自带 `camera_translation`，投影按官方 `(points+t)/z`

### 2.3 3D → 2D（分域投影）

| 域 | 实现 | 公式（原样） |
|---|---|---|
| Perspective (P0) | tools/render/render_p0_blender.py:199 | `u=fx*qx/z+cx; v=cy-fy*qy/z; z=-q.z`，交叉验证 `world_to_camera_view` |
| Fisheye | tools/fisheye/omni_radtan.py :: project_points_omni_radtan | `m=sinθ/(cosθ+xi); r=f(m+k1m³+k2m⁵); (cx+r·cosφ, cy+r·sinφ)` |
| BEDLAM 官方 | tools/annotation/generate_bedlam_densekp.py :: project_official | `p_cam = points + t; p_h = K @ (p_cam / z)`，输出 `[x,y,1]` |

### 2.4 visibility（分域，语义不同，逐域验证）

| 域 | 实现 | 规则 |
|---|---|---|
| Perspective | tools/render/finalize_p0_outputs.py :: main | 状态 0-3（VISIBLE/SELF_OCCLUDED/OUT_OF_FRAME/BEHIND_CAMERA）；3×3 邻域最小 \|render_depth − point_depth\|；容差 `max(0.002, 2·median_depth/fx)` |
| Fisheye | tools/fisheye/run_fisheye_camera_p0.py :: warp_case | 投影像素 5×5 邻域 `min|warped_depth − point_range|` < 4 cm；`point_range=‖camera_points‖`（欧氏） |
| 相机压力 | finalize_p0_outputs + finalize_camera_stress.py :: main | 状态数组与 in_frame/valid 分离保存 |

当前训练 loader **不消费**任何 visibility（TRAINING_AUDIT.md #7：真实标注是否编码可见性为 UNKNOWN）。

### 2.5 bbox → 训练 NPZ

`tools/annotation/generate_bedlam_densekp.py::official_bbox`：前 22 个 joints2d、图内过滤、≥8 个有效、1.2 扩张，
`center=[(xmin+xmax)/2,(ymin+ymax)/2]`，`scale=max(w,h)/200*1.2`。
输出 `datasets/densekp_bedlam_tiny/annotations/bedlam_densekp_tiny.npz`：
`imgname[N] / center[N,2] / scale[N] / proj_verts[N,138,3] float32`。

### 2.6 → DatasetKeypoints → loss

见 `TRAINING_DATA_FORMAT.md` 与 `ARCHITECTURE.md` §C。损失只用 `proj_verts[:,:,:2]`。

## 3. 第三通道（channel 2）现状 —— 明确记录

| 上下文 | channel 2 的值 | 证据 |
|---|---|---|
| BEDLAM 官方投影（训练 NPZ 路径） | **常数 1.0**（齐次坐标） | project_official 校验 `np.allclose(projected[:,2],1.0)`（generate_bedlam_densekp.py:41）；README 段落也写明 |
| 训练 loader 内 | 不归一化、不使用：`proj_verts[:, :-1]` 只作用于 x,y（datasets/utils.py:620-621）；loss 只读 `[:,:,:2]`（densekp_trainer.py:96,102） | — |
| 预测端 | `with_var=True`（configs/hydra/experiment/densekp.yaml）时 channel 2 = log σ | densekp_trainer.py:105-112 |
| 真实历史标注 | **UNKNOWN**（无历史 NPZ 实证；ANNOTATION_AUDIT.md） | — |

即：GT 第三通道是"固定齐次值 1.0"，训练不使用它；预测端第三通道是 log 不确定度，两者含义不同，不要混用。

## 4. 一图流

```text
assets/densekp_regressor_R.npy [138,10475]        ← 138 点的唯一身份定义
        │
SMPL-X vertices [T,10475,3] ──── R @ V ────→ densekp_3d [T,138,3]   (SMPL-X local)
        │                                        │
        │ world transform（分域）                  │ 同一变换逐点应用
        ▼                                        ▼
   camera transform ──────────────────────→ DenseKP 3D in camera frame
        │                                        │
        ▼                                        ▼
   2D projection（persp / omni-radtan / BEDLAM）→ densekp_2d [138,2]
        │
        ├─ depth 对照 → valid / visible（域内容差）
        ├─ joints2d → official_bbox → center / scale
        ▼
   训练 NPZ: imgname / center / scale / proj_verts[N,138,3](xy=原图像素, ch2=1.0)
        ▼
   DatasetKeypoints（共享 crop trans, xy/256-0.5）→ proj_verts_cropped
        ▼
   DenseKP.compute_loss（gt[:,:,:2] vs pred[:,:,:2]，×60）
```

## 5. 阅读顺序（按代码）

1. `tools/annotation/inspect_surface_operator.py` — 看清 R 的每一行是什么；
2. `tools/generate/generate_real_motion_geometry.py:15` — R@V；
3. `tools/render/render_p0_blender.py:165-224` — 3D→2D + 交叉验证 + manifest；
4. `tools/render/finalize_p0_outputs.py` — depth/visibility；
5. `tools/annotation/generate_bedlam_densekp.py` — 训练 NPZ 生成（官方 bbox/投影约定）；
6. `tools/validate/validate_bedlam_densekp_tiny.py` — 闭环验证；
7. `densekp/core/datasets/utils.py::get_example_projverts` — crop 归一化；
8. `densekp/core/densekp_trainer.py::compute_loss` — 消费端。
