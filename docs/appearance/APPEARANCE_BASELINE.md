# 最小外观基线

## 实现

在同一段 16 帧真实 motion、同一份 `vertices.npy`、同一相机和同一投影上新增：

- 肤色 Principled BSDF：base color `(0.48, 0.20, 0.11)`，roughness `0.58`；
- 10 m × 10 m 简单地面；
- 非透明深蓝世界背景；
- Key、Fill、Rim 三盏 Area light。

没有加入 HDRI、纹理、衣物、头发或鞋。输出位于 `outputs/baselines/p0_appearance_baseline/`，包括 `rgb/`、`mask/`、`depth/`、`overlay/` 和 `contact_sheet_before_after.png`。

为防止地面污染人体监督，每帧分两遍：第一遍保持原 P0 的透明背景与仅人体几何，写 depth/mask；第二遍只覆盖 RGB，开启地面和非透明世界。DenseKP projection 不重新计算。

## 回归

使用 BLAKE2b-128 指纹及 Blender mesh 回读共同验证几何：

```ini
GEOMETRY_FINGERPRINT_BEFORE = 96e4372d99f5aeab631b56db8ea5731c
GEOMETRY_FINGERPRINT_AFTER  = 96e4372d99f5aeab631b56db8ea5731c
MAX_MESH_SOURCE_DIFF = 0.0
GEOMETRY_UNCHANGED = YES
```

与 `outputs/baselines/p0_real_motion/` 比较，camera manifest、`densekp_2d`、camera-space 3D、DenseKP depth、visibility、人体 depth 和人体 mask 均逐元素一致，最大绝对差为 0。

```ini
PROJECTION_MEAN_ERROR_PX = 0.000010112
PROJECTION_P95_ERROR_PX  = 0.000019480
PROJECTION_MAX_ERROR_PX  = 0.000031241
PROJECTION_REGRESSION = PASS
APPEARANCE_BASELINE = PASS
```

机器可读结果见 `outputs/baselines/p0_appearance_baseline/appearance_regression.json` 与 `p0_render_report.json`。
