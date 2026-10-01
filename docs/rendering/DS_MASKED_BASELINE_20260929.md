# 3D 场景＋鱼眼遮罩：现有 DenseKP 基线

2026-09-29。对两场景的带遮罩图像重新执行真实 GPU 推理，共 12 个给定框视图。
使用已有 `assets_external/densekp.ckpt`，未训练；当前人物仍来自原有动作，尚非骑行动作。

## 输入及检查

- 输入：`outputs/experiments/ds_scene_pair_amodal_20260929/`，引用
  `ds_scene_pair_20260928_lens_mask` 的图像与标注。
- 背景：`assets/scenes/photoreal/cycling_forest/cycling_forest.blend` 和
  `assets/scenes/photoreal/urban_street/urban_street_assets.blend`。源 manifest 均记录
  `lighting=native_scene`；人体和背景在同一 3D 场景中渲染。
- 20260320 DS 内参，两光轴同一水平线、180° 相反；0.18 m 基线为合成试验值。
- 鱼眼 2880×2880，诊断 ERP 1536×768。鱼眼镜头遮罩与 ERP 有效区都实际应用。
- 四路鱼眼遮罩外像素全部为零，四路 ERP 有效区外像素全部为零；没有用 RGB 黑色估计点是否有效。
- 每场景 6 行 loader，622 个有效监督点，其中 292 可见、330 遮挡；本例均在完整裁剪内。
- 12×138×3=4968 个模型输出均有限；预测第三值原样保存，未用于筛选或加权。

这是既有 3D 渲染资产的新推理，不是本轮重新渲染。源图仍有明显采样噪声；旧源 manifest
未保存采样数和随机种子，本轮不补写猜测值。正式扩量时应重新固定、记录渲染参数。

## 结果在哪里

输出：`outputs/experiments/ds_scene_pair_masked_inference_20260929/`。

- [森林预测对照图](../../outputs/experiments/ds_scene_pair_masked_inference_20260929/cycling_forest/prediction_review.jpg)
- [街景预测对照图](../../outputs/experiments/ds_scene_pair_masked_inference_20260929/urban_street_assets/prediction_review.jpg)
- [完整报告](../../outputs/experiments/ds_scene_pair_masked_inference_20260929/report.json)
- [图像和遮罩检查](../../outputs/experiments/ds_scene_pair_masked_inference_20260929/input_checks.json)

图中左列 cam2、右列 cam4，从上到下人物 1–4。绿点为可见 GT，橙点为遮挡 GT，红点
为预测，青色为目标可见实例轮廓。预览放大两倍，误差仍按原始 256-square 裁剪像素计算。
全部 138×3 预测、还原 xy、GT 分组和误差保存在各场景的 `cam*_person_*.npz`。

| 目标 | GT 有效点数 | 森林全部点均值 px | 街景全部点均值 px |
| --- | ---: | ---: | ---: |
| 人物 1 / cam2，边缘截断 | 91 | 8.03 | 7.79 |
| 人物 1 / cam4，边缘截断 | 87 | 12.68 | 10.14 |
| 人物 2 / cam2，边缘截断 | 90 | 3.93 | 4.38 |
| 人物 2 / cam4，边缘截断 | 93 | 5.34 | 5.63 |
| 人物 3 / cam2，后方遮挡 | 123 | 83.39 | 83.22 |
| 人物 4 / cam2，完整对照 | 138 | 4.39 | 4.16 |

人物 3 的 22 个可见点误差分别为 106.79 / 111.62 px，101 个遮挡点为 78.29 / 77.03 px。
可见点也会失败，不能认为可见性能够替代预测准确性。该目标裁剪中前景人物占很大区域，
目标表达有歧义；图像与指标支持“给定框后定位失败”，不足以单独证明身份交换的具体机制。

此处人物 3 用重遮挡可见实例框 `[1085,325,1134,503]`，不同于遮挡控制实验冻结的
clear 框 `[1081,325,1183,503]`。不能将这里的约 83 px 与该实验的约 17.7 px 相减，
归因为遮罩或监督规则的影响。模型权重也没有因修改监督策略而发生变化。

## 双视角对应

先取两边 GT 都在有效视野内的同一人物、同一语义点，再计算该点两边误差的平均与最大值。
表中“较差视角均值”是先逐点取较大误差、再按点平均。预测被镜头遮罩拒绝时仍保留误差。

| 目标 | 共同有效点 | 森林两路均值 / 较差视角均值 px | 街景两路均值 / 较差视角均值 px | 森林 / 街景两路均有效的预测数 |
| --- | ---: | ---: | ---: | ---: |
| 人物 1 | 40 | 10.94 / 14.97 | 8.67 / 11.88 | 34 / 33 |
| 人物 2 | 45 | 4.19 / 5.69 | 4.54 / 6.52 | 40 / 40 |

报告还细分两边可见、仅一边可见、两边均遮挡。这衡量 DenseKP 对应端点的定位质量，
没有运行相机匹配、人体重建或合图。人物 3/4 在 cam4 无实例框，分别是完全被挡住和
不在有效视野；没有为这些缺框视图运行模型，也不计为检测器失败。

## 复现与结论

在 CameraHMR 环境、仓库根目录运行，输出需选择新目录：

```bash
python tools/validate/evaluate_ds_scene_pair.py \
  --pilot-root outputs/experiments/ds_scene_pair_amodal_20260929 \
  --output outputs/experiments/ds_scene_pair_masked_inference_20260929 \
  --device cuda:0
```

使用 RTX 3090、float32、无 autocast、关闭 TF32、种子 0，checkpoint 严格加载。
模型和 loader 配置分别归档。当前链路满足 3D 背景及镜头遮罩要求，定位失败案例也已保留。
只有两个场景、重复身份和原有动作，尚不足以评价泛化或骑行能力。

骑行素材准备与拟合交接见 [骑行输入包](../motion/cycling/CYCLING_FIT_HANDOFF_20260929.md)。
