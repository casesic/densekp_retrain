# 最新执行状态（2026-09-06）

官方 archive 已到位。Body skin 16 帧 PASS；eye 因独立纹理 UV 不匹配而 BLOCKED。已提取资源位于 `/home/yy/datasets/BEDLAM/body_textures/tiny/rp_aaron_posed_009/`。最新证据见 `outputs/body_texture_audit/skin_texture_audit.json` 和 `outputs/baselines/p0_dressed_skin_textured/appearance_v1b_report.json`。以下为资源下载前的历史审计。

# BEDLAM Body / Skin Texture 审计

## 官方资源与选择策略

BEDLAM v1 官方 render 工具明确要求从 BEDLAM 网站下载 `bedlam_body_textures_meshcapade.zip`，解压后导入 `MC_textures_skintones`。材质脚本只把选中的纹理接到 `M_SMPLX` 的 `BaseColor` 参数。官网也将 body textures 列为受 BEDLAM 许可约束的资源，并注明 skin textures 来自 Meshcapade。

body texture 不绑定 subject。sequence generator 依据 subject gender，从 `textures_body.txt` 对应的 male/female 列表随机抽取，并将结果写入 `be_seq.csv` 的 `texture_body`。因此 `rp_aaron_posed_009` 没有固有 skin ID。资源到位后，本项目将选择 male 列表的第一个兼容纹理作为确定性基线，不声称复刻历史 RGB。

```ini
BODY_TEXTURE_ARCHIVE = bedlam_body_textures_meshcapade.zip
BODY_TEXTURE_MAPPING_POLICY = GENDER_FILTERED_RANDOM_PER_SEQUENCE
BODY_TEXTURE_UV_SOURCE = SMPL-X face-corner UV topology
```

本机在 `/home/yy` 与 `/mnt/hdd/yuyue/home/yuyue` 的有界搜索中未找到 archive 或可确认的官方 skin texture。必须登录 [BEDLAM Download](https://bedlam.is.tue.mpg.de/) 并接受许可后手动下载。官方公开页面未给出文件大小，因此不猜测。建议下载后放到 `/home/yy/datasets/BEDLAM/body_textures/bedlam_body_textures_meshcapade.zip`。

## UV 与 topology 审计

本机已有 `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/PEAR/assets/SMPLX/smplx_uv.obj`。`tools/audit_body_uv.py` 对其与当前 16 帧 DenseKP body geometry 做了全量比较：

- body vertices：`[16, 10475, 3]`
- geometry faces：`[20908, 3]`
- UV vertices：`[11313, 2]`
- UV faces：`[20908, 3]`
- geometry faces 逐项完全一致；UV 索引合法且全部 finite
- U 范围：`[0.003845, 0.995247]`
- V 范围：`[0.0073, 0.998814]`

这证明现有 UV OBJ 可无几何修改地附加到当前 10475-vertex SMPL-X mesh。由于 archive 尚未取得，仍需在下载后核对其中 texture 命名、尺寸和方向，才能把“具体 texture 与该 UV”标记为已验证。

## 眼球审计

SMPL-X topology 有 3 个 connected components，顶点数分别为 9383、546、546；后两个与左右眼球 component 相符。但 UV OBJ 没有 group 或 material statement，不能从该文件建立 eye-specific material region。BEDLAM Unreal assets 另有 `SMPLX_eye` texture；当前尚未取得，因此本轮不伪造眼球材质。

## 后续资源计划

- Hair：BEDLAM2 `b2_groom_abc.tar.gz`，Alembic strand groom；还需针对目标 body T-pose/head shape 创建 groom binding。
- Shoes：BEDLAM2 `b2_shoes.tar.gz`，包含 color、normal、displacement；接入前需审计 toeless/sock-feet body 修改，不能改变当前 DenseKP body geometry。
- HDRI：BEDLAM v1 使用 Poly Haven 清单中的 4K HDR，BEDLAM2 使用 8K HDR；后续只按 whitelist 选择并单独下载。

本轮没有下载资源、没有修改 renderer/body geometry，也没有运行 16 帧 skin render。现有 clothing texture baseline 保持不变。
