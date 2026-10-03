# 跟着一个真实样本读完整链路

主例是保存的街景 **heavy / cam2 / 人物3**：09-29遮挡监督标签复用09-28图像和预测，**不是当前新训练样本**。本页把实际制品接到当前函数；不声称这些旧制品由今天的代码重新生成。B5只读原JSON/路径；原回读记录在 `outputs/acceptance/refactor_reading_20261002/sample_trace.json`。

## 1. 从人物和姿态到几何

路径缩写均相对仓库根：

```text
G = outputs/baselines/temporal_case_aaron_1095
S = outputs/experiments/ds_occlusion_control_20260928
A = outputs/experiments/ds_occlusion_amodal_20260929
```

| 环节 | 当前实现 | 本例的实际产物 |
| --- | --- | --- |
| 人物/动作 | [load_real_smplx_motion.main](../tools/generate/load_real_smplx_motion.py) | G/geometry/canonical_motion.json：BEDLAM rp_aaron_posed_009/1095、male |
| 人体/138点 | [generate_real_motion_geometry.main](../tools/generate/generate_real_motion_geometry.py)、[compute_densekp_3d](../src/densekp_retrain/body/geometry.py) | G/geometry/geometry_manifest.json；vertices.npy [39,10475,3]、faces.npy [20908,3]、densekp_3d.npy [39,138,3] |
| 衣物 | [prepare_p0_dressed_geometry.main](../tools/generate/prepare_p0_dressed_geometry.py) | G/clothing_geometry/clothing_manifest.json及数组，来自既有BEDLAM衣物模拟 |
| 场景与相机 | [world_points](../src/densekp_retrain/body/transforms.py)、[rig_poses](../src/densekp_retrain/camera/rig.py)、[Blender装配](../src/densekp_retrain/rendering/blender/assembly.py) | S/heavy/people.json、rig.json、source/source_manifest.json |
| 成像与标注 | [process_scene](../src/densekp_retrain/imaging/ds.py) | S/heavy/source/cam2.exr、cam2/normal_erp.png、pair_annotations.npz、cam2/boxes.json |
| 框与监督 | [evaluate_ds_occlusion_control.main](../tools/validate/evaluate_ds_occlusion_control.py)、[make_targets](../densekp/core/keypoint_supervision.py) | S/clear/cam2/boxes.json → A/fixed_boxes.json → A/heavy/fixed_box_loader.npz的索引1 |
| 裁剪与模型 | [DatasetKeypoints](../densekp/core/datasets/dataset_keypoints.py)、[DenseKP](../densekp/core/densekp_trainer.py) | A/model_config.yaml、loader_config.yaml、heavy/cam2_person_3.npz |
| 还原与指标 | [restore_crop_xy](../densekp/core/crop_geometry.py)、[evaluation_masks / point_errors](../densekp/core/evaluation.py) | A/report.json及上述逐例NPZ；保存预测、GT和affine，不包含完整输入图像张量 |

几何帧16对应动作切片内帧80；衣物原时间轴为181=start_frame101+80。不要混用切片索引与源帧号。此资产保留source translation；不是所有生成入口都把平移置零。

138点由 `R[138,10475] @ vertices[10475,3]` 得到，不是BEDLAM关节数组。人物位置[1.8,0.3,0]、yaw90°；身体、衣物和点共用 `(x,-z,y)`、身体最低点lift、yaw和location，不能分别贴地。

本例衣物不是当前简单shell配方。另一条当前骑行链是独立照片姿态→角色shape→`body.parameters.parameters_from_pose/evaluate`→`clothing.preparation.prepare_simple_clothing`→`clothing.groom_transfer.transfer_groom`，十款STYLES留在adult_candidate_recipe。已有reference/rest/joints是迁移对照共同输入，不等于本轮重算rest_body；完整来源、UV/groom生产者和许可在 [ASSETS](ASSETS.md)。

## 2. 世界坐标到图像与标签

`urban_street_assets`来自build_urban_street基底与upgrade_urban_street_assets的apartment kit/Hidden Alley道具；`configs/fisheye_scene_urban_street_assets.json`引用场景。入口和资产链见 [场景索引](rendering/SCENE_RECIPE_INDEX.md)。

原生多人入口 [render_equirectangular_source.py](../tools/fisheye/render_equirectangular_source.py) 组织场景、灯光和人物，公共Blender模块创建对象/材质/groom/相机/compositor。cam2/cam4分别生成1536×768源ERP的RGB、Depth、IndexOB，再由process_scene做DS重映射、有效域、标签与人物框。

`rig_poses`给camera-to-world；世界行向量到相机为 `(world-t) @ R`。`camera.erp.project_erp`输出像素和射线距离。DS可见性在**原始源ERP距离/实例图**中做周期邻域检查，不拿Omni重映射鱼眼depth.npy代替；两条路径的距离、插值与可见性规则见 [契约](CONTRACTS.md)。

当前process_scene还会应用真实参考图提取的镜头mask。本例原图早于完整镜头mask更新，不能重新跑当前代码后要求必然像素相同。显式mask和几何域决定in_view；**不能以RGB黑色判无效区**。

## 3. 人体框怎样决定网络输入

本例用clear条件的固定框评价heavy遮挡，不因遮挡再缩框：

```text
原框 [x1,y1,x2,y2] = [1081,325,1183,503]
center = [1132,414]
width=102，height=178
side=max(width,height)*1.2=213.6
scale=side/200=1.068
```

一个loader行是一张图中的一个指定人物；本例loader有5行，索引1是cam2/person3，多行可以共享图像。当前Dataset流程直接可读为：

```text
read_image → determine_crop → transform_crop → _supervision → 返回样本
                 图像与138点共用同一affine
```

定义见 [crop_pipeline](../densekp/core/datasets/crop_pipeline.py) 的CropBox、CropPlan、CropResult。Dataset显式路径和旧ID均可解析；imgname相对image_root，绝对路径不再次拼根。图像按原BGR读取、保留display通道语义，训练输入转RGB并按ImageNet mean/std归一化为float32[3,256,256]。

关闭增强时，此框affine约为：

```text
[[1.1985013, 0, -1228.70350],
 [0, 1.1985020, -368.179832]]
normalized_xy = (affine @ [x,y,1]) / 256 - 0.5
```

**框中心右移，裁剪窗口也右移，同一个原图点在裁剪内左移；图像内容和GT必须一起移动。** 尺度变大覆盖更多原图，同一人体在网络输入里变小。网络只看到裁剪图，不另读bbox、人物ID或GT。ViT内部取中间192列，输出仍对应完整256-square，不能还原时另加32px。

增强改变实际affine后再重算监督，不能只用原plan.box判断。另一条已有固定证据 `outputs/acceptance/refactor_dataset_crop_20261001/walkthrough_sample.json` 来自框挑战val第1行：center=[1110.7600,405]、side=321.6000；固定seed930给scale=.8、rotation=-33.8913°、平移比例约[.00451869,.00184876]，中心成为[1112.2132,405.5946]。原in_view112不变，实际监督112→102、遮挡监督71→68。这是另一数据样例，不是主例的来源；B5没有重跑增强。

## 4. 为什么遮挡点仍监督

本例138点中123个在有效观测域且完整裁剪内：22个表面可见，101个被前景遮挡但仍有可靠几何GT；其余15个在有效域外。新标签用 `in_view & inside_full_crop`，**没有再与visible相交**。

遮挡表示视线到目标表面的路被挡住，不表示没有几何标签。裁剪外或成像域外则没有当前输入契约所要求的观测位置，所以不参与本次监督。增强后的点必须重新检查两轴 `[-.5,.5)`；旧NPZ若原来只给visible权重，仍保持旧语义，不因接口退役悄悄补成amodal。

遮挡评价和可见评价分组报告，GT支持集决定分母。预测落到域外仍计其GT误差并另报拒绝，不能丢掉坏预测改善均值。

## 5. 模型、loss和坐标还原

[DenseKP.forward_step / compute_loss](../densekp/core/densekp_trainer.py) 接收上述图像，head输出138×3。训练目标、加权归一化与with_var分支见 [契约](CONTRACTS.md)；主例没有参与训练，也没有逐样本训练loss制品，不能伪造这一环节的实际结果。当前训练入口为densekp.train_keypoints，来源隔离试验由train_source_isolated_trial组合；命令与计算边界见 [COMMANDS](COMMANDS.md)。

本例点2原GT约[1117.84021,331.96893]，归一化GT约[-.0662911,-.3840406]；保存预测[-.0216906,-.3941702,.5772585]。还原先 `(pred_xy+.5)*256`，再逆affine，约得[1127.36689,329.80526]。第三值.5772585原样保留，不当默认置信度或误差权重。

裁剪像素误差、ERP周期像素误差和两相机各自误差不能直接互换。当前候选入口 [evaluate_rows](../src/densekp_retrain/evaluation/candidate_runner.py) 把输入核查、预测、配对、报告分开；原DS保存预测回放仍有自己的比较与分母规则。回归suite只是组织任务，不统一指标。

## 6. 追溯到哪里停止

`G/geometry/geometry_manifest.json`仍引用已缺失的 `outputs/temporal_case_aaron_1095/geometry/canonical_motion.npz`。当前G下存在canonical_motion.npz，但没有旧副本证明迁移前字节相同；本轮不改历史manifest。原人体/衣物/场景依赖的路径见sample_trace记录，存在不等于源码交付自带。

源渲染没有完整冻结所有贴图/HDRI/外观及权重版本，旧mask与当前规则也有版本差异。因此这个例子足够沿保存制品理解链路，但不是在任意干净机器位级重建同一图像的承诺。原参数、数值/渲染验收及未覆盖范围统一见 [VALIDATION](VALIDATION.md)。当前来源隔离数据、三份权重和效果门槛另见 [DATA_AND_MODELS](DATA_AND_MODELS.md)。
