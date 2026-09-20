# P0 real SMPL-X motion regression

## 结论

```ini
REAL_MOTION_AVAILABLE = YES
REAL_MOTION_SCHEMA_VERIFIED = YES
P0_REAL_MOTION = PASS
```

本轮没有下载 motion。使用的是官方 `bedlam_render` Git 仓库随代码提供、已被其 Blender preprocessing 脚本直接消费的 SMPL-X 示例：

```text
third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_animation.npz
```

其 `motion_info` 记录来源为 AMASS/KIT `bend_left01_stageii.npz`，文件已随当前授权使用的官方代码仓库存在。本轮仅使用该本地示例，不访问或复制其上游数据集。

## 搜索与候选确认

在 `/home/yy/Projects` 和 `/mnt/hdd/yuyue/home/yuyue/Projects` 中有界搜索 `motion_seq.npz`、motion/SMPL-X 命名 NPZ。排除以下非 motion：模型文件、SMPL-X/MHR 映射文件、此前 procedural 参数。唯一直接满足“真实、完整 SMPL-X motion 且解释有官方代码依据”的候选是上述 176642-byte 示例。

官方 `third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_anim_to_alembic.py` 调用：

```text
bpy.ops.object.smplx_add_animation(..., anim_format="SMPL-X", target_framerate=30)
```

同目录 README 明确说明 BEDLAM NPZ 是 Y-up、SMPL-X OpenGL coordinate notation，pose corrective 在导出 ABC 时烘焙。本轮不经过 ABC，直接将同一参数送入 Python `smplx`。

## 原始 schema

| key | shape | dtype | 解释 |
|---|---:|---|---|
| `motion_info` | `(6,)` | Unicode | 动作名、索引、AMASS/KIT 来源、原 fps 等元数据 |
| `betas` | `(10,)` | float32 | 静态 male body shape，不是逐帧 betas |
| `poses` | `(126,165)` | float64 | 每帧完整 SMPL-X axis-angle pose |
| `global_ori` | `(126,3)` | float64 | 根 axis-angle；与 `poses[:,:3]` 最大差为 0 |
| `trans` | `(126,3)` | float64 | 根 translation，SMPL-X 米制坐标 |
| `mocap_frame_rate` | scalar | int64 | 30 fps |
| `gender` | scalar | Unicode | `male` |

`poses[165]` 严格拆分为：

```text
[0:3]     global_orient     [T,3]
[3:66]    body_pose         [T,63]
[66:69]   jaw_pose          [T,3]
[69:72]   leye_pose         [T,3]
[72:75]   reye_pose         [T,3]
[75:120]  left_hand_pose    [T,45]
[120:165] right_hand_pose   [T,45]
```

这些字段均为 axis-angle，不是 rotation matrix。文件没有 `expression` 或 `dmpls`，因此 SMPL-X 使用模型默认零 expression，不使用 DMPL。

## Canonical adapter

新增 `tools/load_real_smplx_motion.py`，renderer 不解析 motion schema。adapter 校验 `poses[165]`、`global_ori` 一致性、静态 betas、translation shape、finite 值和 gender，再输出 canonical：

```text
global_orient [T,3]
body_pose [T,63]
jaw/leye/reye pose
left/right_hand_pose [T,45]
betas [10]
transl [T,3]
source_transl [T,3]
source_frame_ids [T]
```

原始 126 帧中选取 16 个分散帧：

```text
render: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
source: 0, 8, 17, 25, 33, 42, 50, 58, 63, 75, 83, 92, 100, 108, 117, 125
```

映射写入 `outputs/baselines/p0_real_motion/geometry/frame_mapping.csv`，同时存在 canonical JSON 与 geometry manifest。

## Translation policy

```ini
SOURCE_TRANSLATION_USED = NO
```

原始 `trans` 全序列范围：

```text
min = [-1.696030, 1.384052, 0.000000] m
max = [ 0.025926, 1.443621, 1.457335] m
```

adapter 将选定的原 translation 保存在 `source_transl`，但向 SMPL-X forward 传入的 canonical `transl` 为零。本轮只验证真实 articulation、root orientation 与 body deformation；不验证 locomotion trajectory。这样固定 P0 camera、object placement 和所有 projection/depth 约定保持不变，且没有静默丢弃原数据。

## Geometry 与单一数据源

使用本机 `SMPLX_MALE.npz` 和 source betas，16 个 canonical 帧生成：

```text
vertices       = [16,10475,3] float32
faces          = [20908,3] int32
densekp_3d     = [16,138,3] float32
densekp_3d[t]  = R @ vertices[t]
```

测试首尾最大顶点变化 0.411166 m。Blender mesh 回读相对 `vertices.npy` 的最大差为 0；DenseKP 重新计算相对落盘数据最大差为 0。

## Renderer regression

复用 P0 的 `tools/render/render_p0_blender.py` 和 `tools/render/finalize_p0_outputs.py`，没有改变 camera、projection、depth、mask 或 visibility 算法。真实 motion 暴露了一个旧代码问题：manifest 的 `avatar_id`/`motion_id` 曾硬编码为 procedural toy。唯一 renderer 修改是从 `geometry_manifest.json` 读取这两个已有 schema 字段；不在 Blender 内解析 motion。

`finalize_p0_outputs.py` 仅新增逐帧 visibility 统计输出，分类算法不变。

## 数值结果

解析投影对 Blender `world_to_camera_view`：

```text
mean = 0.000010112 px
p95  = 0.000019480 px
max  = 0.000031241 px
```

仍远低于 mean 0.1 px、max 0.5 px 门槛，且与 procedural P0 同属浮点噪声量级。

已知 2 m 正视平面再次得到 center/off-axis depth 都为 2.0 m，维持：

```ini
DEPTH_SEMANTICS = CAMERA_SPACE_Z
```

## Visibility regression

总计：

```text
VISIBLE        = 1036
SELF_OCCLUDED  = 1172
OUT_OF_FRAME   = 0
BEHIND_CAMERA  = 0
```

逐帧 `(visible,self_occluded,out_of_frame)`：

```text
0:(74,64,0)   1:(75,63,0)   2:(71,67,0)   3:(70,68,0)
4:(76,62,0)   5:(76,62,0)   6:(69,69,0)   7:(74,64,0)
8:(64,74,0)   9:(57,81,0)  10:(59,79,0)  11:(64,74,0)
12:(54,84,0) 13:(51,87,0)  14:(51,87,0)  15:(51,87,0)
```

随人体转向侧面和弯身/迈步，可见点减少、自遮挡点增加，趋势与 contact sheet 一致。人物完整留在固定 P0 视野内。

## Overlay 与 pyrender

`outputs/baselines/p0_real_motion/overlay/contact_sheet.png` 已视觉检查：头、肩、手臂、躯干、腿在明显 pose/朝向变化中均贴合 mesh；绿色/红色点与可见/自遮挡面基本一致。

pyrender 使用同一 vertices/faces/camera 对 frame 0/8/15 做交叉检查，silhouette IoU 为：

```text
0:  0.998771
8:  0.999155
15: 0.999186
mean: 0.999037
```

## Acceptance

| 项目 | 结果 |
|---|---|
| REAL_MOTION_SCHEMA_VERIFIED | YES |
| REAL_MOTION_GEOMETRY | PASS |
| BLENDER_RENDER_REGRESSION | PASS |
| PROJECTION_REGRESSION | PASS |
| VISIBILITY_REGRESSION | PASS |
| DENSEKP_OVERLAY_REGRESSION | PASS |
| P0_REAL_MOTION | PASS |

下一步可以保持真实 motion geometry 不变，进入 camera-domain 的 near-camera、image-edge、truncation 参数化；鱼眼/ERP、多人、衣服和训练仍不在本阶段范围。
