# DS 镜头成像区域遮罩

> 2026-09-29 后续：[两场景带遮罩图像已完成新的 DenseKP 推理](DS_MASKED_BASELINE_20260929.md)。
> 本页保留遮罩生成阶段的验证记录。

用户确认 20260320 实拍镜头使用 DS 标定，因此继续使用对应 cam2/cam4 的 DS 参数。
本次只接入成像区域遮罩，不切换相机模型，也不改变鱼眼 2880×2880 或试验 ERP 1536×768
分辨率。正式实拍 ERP 为 5760×2880；本次产物仍为低分辨率诊断样例。

## 遮罩来源与规则

实际参考目录为 `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/Output/20260320_riding_dataset3/image/fisheye/`。
使用 `cam2/02214.jpg`、`cam4/02214.jpg`，复用
`tools/fisheye/build_ds_multiperson_pilot.py::lens_mask_from_reference`：

1. 灰度 > 5；51×51 椭圆核闭运算。
2. 取最大外轮廓并填充。没有额外腐蚀，没有复制真实照片 RGB 到合成图。
3. 每台相机独立得到 2880×2880 二值遮罩，占图像 85.0438% / 84.6715%。

这是单帧图像估计的轮廓，不是标定公式给出的数学有效域，也不是精密测量的光学边界。
正式扩量前仍应多帧检查暗背景是否影响外轮廓。生产 `pano-stitch` 的遮罩估计默认另有
8 像素腐蚀，本次鱼眼轮廓没有采用该腐蚀；最终 ERP 仍与原有生产参考有效区取交集。

## 接入位置

`run_ds_scene_pair_pilot.py` 和 `run_ds_occlusion_control.py` 都要求显式传入
`--fisheye-reference-root`，可用 `--reference-frame` 改变参考帧，默认 `02214`。
参考图缺失或尺寸不符会报错。

`process_scene` 用相同 DS LUT 将二值镜头遮罩以最近邻映射到 ERP，和已有 ERP mask
取交集。`point_in_view` 使用该交集判定入视野状态；表面遮挡判定仍来自源 ERP 的深度/实例。
鱼眼 RGB 和实例 ID 同时在镜头外清零，然后分别用线性/最近邻插值展开到 ERP。
最终 RGB、实例图、框和点有效性均遵循新的有效区。

每相机新增 `reference_lens_mask.png`；`report.json` 的 `cameras[].lens_mask`
记录参考图片绝对路径、提取规则、腐蚀量、面积和额外剔除的 ERP 有效像素数。

## 新产物

- 双场景：`outputs/experiments/ds_scene_pair_20260928_lens_mask/{cycling_forest,urban_street_assets}/`。
- 三档遮挡：`outputs/experiments/ds_occlusion_control_20260928_lens_mask/{clear,mild,heavy}/`。
- 每组均有 `{cam2,cam4}/fisheye.png`、`reference_lens_mask.png`、`normal_erp.png`、
  `valid_mask.png`、`annotations.npz`、实例图和框。
- 逐图对比记录：`outputs/experiments/ds_occlusion_control_20260928_lens_mask/lens_mask_comparison.json`。

新目录的 `source/` 软链接到各自旧目录的渲染源；复制了 rig/people 元数据，并用
`--skip-render` 重新执行处理。旧产物和既有预测均未覆盖，Blender 未重跑。
移动或归档新产物时需要同时保留其 source 目标。

## 验证结果

共 5 组、10 路图像完成逐项比较：

- 鱼眼在镜头外全部为零，镜头内像素与原图逐项一致。
- cam2/cam4 每图约 115 万 / 123 万个鱼眼像素被置黑。
- ERP 有效区分别额外剔除 23 / 16 个像素，其中有颜色的像素为 20 / 15 个。
  保留下来的 ERP 有效区域内，RGB 没有任何变化。
- 所有 annotation 数组、人物实例 mask 和 boxes 均与对应旧版本逐项一致。
- 场景测试 5 项通过，包含完全遮掉一路镜头时 RGB、实例、框、点有效性同步失效的测试。
- 两个场景均通过实际 DatasetKeypoints 验证：每场景 6 个人体视图，裁剪坐标还原最大误差
  0.0000192 原图像素，138 点 regressor 检查通过。

本次修正了中间鱼眼图的成像边界。没有重新运行 DenseKP 预测，不能将此次修改表述为
模型精度提升；渲染源分辨率、材质和噪声限制仍然存在。
