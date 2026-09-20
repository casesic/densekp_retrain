# BEDLAM 服装纹理接入

## 确定性选择

目标为 `rp_aaron_posed_009` / animation `1095`。`config/textures_clothing.csv` 声明该 subject 兼容 `texture_01` 至 `texture_10`；animation 本身不绑定唯一纹理。本阶段按用户明确授权采用确定性基线：

```ini
TARGET_TEXTURE_ID = texture_01
TEXTURE_SELECTION_POLICY = USER_AUTHORIZED_DETERMINISTIC_BASELINE
```

来源 shard 为 `clothing_textures.0.tar`（4,870,778,880 bytes），仅提取 `texture_01_diffuse_1001.png`（4,257,702 bytes）与 `texture_01_normal_1001.png`（2,846,492 bytes）。两者位于 `/home/yy/datasets/BEDLAM/clothing_data/tiny/rp_aaron_posed_009/1095/textures/texture_01/`。

## UV 审计与方向

服装拓扑为 13,027 vertices、26,328 triangular faces。NPZ 提供 14,776 个 UV vertex，`faces_uv` 与 `faces` 均为 `(26328, 3)`；几何索引和 UV 索引均合法，逐 polygon loop 的数量匹配。

```ini
U_RANGE = [0.0149999997, 0.7371339798]
V_RANGE = [0.0308400001, 0.9850000143]
UV_INDEX_VALID = YES
```

Blender 按 face-corner 写入 `vertices_uv[faces_uv[face, corner]]`。单帧对照见 `outputs/audits/clothing_texture_audit/uv_original.png` 与 `uv_vflip.png`：原始方向形成完整一致的浅蓝上衣与绿色长裤，V 翻转后上下装纹理区域错配。因此：

```ini
UV_MAPPING_VERIFIED = YES
UV_V_FLIP_REQUIRED = NO
```

## 材质与回归

diffuse 使用 sRGB；normal 使用 Non-Color，经 Normal Map 节点连接；metallic=0、roughness=0.58、opaque，不启用 displacement。正式输出在 `outputs/baselines/p0_dressed_textured/`，共 16 帧。

`texture_regression.json` 验证 body/clothing geometry、camera、DenseKP 2D、camera-space 3D、depth、mask 和 visibility 与无纹理 dressed baseline 完全一致。visibility 为 861 visible、1331 occluded、16 out-of-frame、0 behind-camera；最大投影误差 `3.3463941410797083e-05 px`。平均渲染时间为 `374.133 ms/frame`。
