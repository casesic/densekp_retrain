# Appearance V1 计划

## V1-A：服装纹理

V1-A 已通过。按用户授权选择 `texture_01` 作为确定性基线，只提取 diffuse 与 normal；以 NPZ 的 face-corner UV 建立 UV layer。单帧方向对照确认无需 V flip。diffuse 使用 sRGB，normal 使用 Non-Color 与 Normal Map，metallic=0、roughness=0.58。

相同 body、clothing、相机的 16 帧结果位于 `outputs/p0_dressed_textured/`。几何、投影、depth、mask 与 visibility 回归全部通过，before/after 见 `contact_sheet_before_after.png`。

OBJ/NPZ 都没有 material group，当前衣物是单一 material region；除非 texture archive metadata 给出相反证据，不人为拆分上衣和裤子。

## 后续外观来源（只读审计）

V1-B body/skin 已完成资源与 UV 前置审计：当前 SMPL-X face topology 与本机 UV OBJ 全量一致，但官方 `bedlam_body_textures_meshcapade.zip` 不在本机，须由用户从 BEDLAM 许可下载页面取得。详见 [Body Texture 接入审计](BODY_TEXTURE_INTEGRATION.md)。

| 层级 | 官方来源 | 状态 |
|---|---|---|
| Skin | BEDLAM `bedlam_body_textures_meshcapade.zip`；BEDLAM2 bald-head body textures | 需要受许可下载 |
| Hair | BEDLAM2 `b2_groom_abc.tar.gz`，并需针对 body shape 建 binding | 需要受许可下载 |
| Shoes | BEDLAM2 `b2_shoes.tar.gz`，color/normal/displacement maps | 需要受许可下载 |
| HDRI | 官方脚本指向 Poly Haven 4K/8K HDR；Starter Pack 也含 HDR | 需要下载，本轮禁止 |

下一步为 V1-B skin；hair/shoes/HDRI 不阻塞 DenseKP 数据格式验证。

## V1-B 执行结果（2026-09-06）

Body skin baseline PASS：确定性使用官方 male `skin_m_white_01_ALB`，复用已验证 UV，sRGB、metallic=0、roughness=0.58、opaque。仅提取 body ALB 与 SMPLX_eye 两张资源。16 帧位于 `outputs/p0_dressed_skin_textured/`；geometry/camera/DenseKP/depth/mask/visibility 回归全部通过。

Eye 单独 BLOCKED：通过官方 SMPL-X joint 23/24 权重确认两眼 component，但现有眼球 UV 与独立 SMPLX_eye 图集虹膜位置不兼容，保留原眼材质。不得把该结果描述为完整 body+eye 接入通过。证据见 `outputs/body_texture_audit/skin_texture_audit.json`；结果与 benchmark 见 `outputs/p0_dressed_skin_textured/appearance_v1b_report.json`。

## V1-C 执行结果（2026-09-06）

Hair 与 shoes 均未通过当前 BEDLAM v1 Blender baseline 的 compatibility
gate，因此未生成 full-appearance render。Hair 是 Alembic strand groom，需
对 BEDLAM2 neutral locked-head 目标身体的默认 T-pose GeometryCache 创建
subject-specific Unreal binding。Shoes 不是独立 mesh，而是对 toeless body
施加 color/normal/displacement，并用每鞋 offset 抬高整个人体；会改变当前
body geometry 与 translation。详细证据见
`docs/APPEARANCE_V1C_HAIR_SHOES.md` 和
`outputs/appearance_v1c_audit/report.json`。

下一步为 V1-D environment：使用 BEDLAM whitelist 首项、Poly Haven CC0
的 `abandoned_church` 4K HDR 作为单一确定性基线。

## V1-D 执行结果（2026-09-06）

HDRI environment 16 帧 `PASS`。Renderer 新增 `simple/hdri` 可选模式，默认
行为仍为 simple，且单帧 RGB pixel regression 完全一致。正式参数为
`abandoned_church_4k.hdr`、strength 1.0、rotation 0°、visible background，
保留一个 90 W 弱 fill 与 neutral rough ground。Body/clothing/camera/DenseKP
不变，depth、mask、visibility 回归通过，最大投影误差仍为
`3.3463941410797083e-05 px`。完整结果见
`docs/APPEARANCE_V1D_HDRI.md` 与
`outputs/p0_dressed_hdri/appearance_v1d_report.json`。

## Phase 9.4 eye geometry audit（2026-09-06）

Eye geometry 位置有效，无需改动。两个 546-vertex / 1088-face component
属于同一 SMPL-X body mesh，joint 23/24 权重、rest-space 对称性、head-motion
fit 与 source-model replay 全部通过；没有独立 eye transform、parent 或双重
变换。红褐色“错位感”来自 legacy reddish material 缺少 sclera/iris/pupil
视觉线索。最终使用 neutral material fallback，并完成 16 帧 material-only
回归。详见 `docs/EYE_GEOMETRY_AUDIT.md` 和
`outputs/eye_fix/eye_geometry_fix_report.json`。
