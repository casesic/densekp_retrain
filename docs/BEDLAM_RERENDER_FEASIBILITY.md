# BEDLAM / BEDLAM2 定制重渲染可行性审计

## 1. 执行结论

**推荐以 BEDLAM2 为基础，重渲染可行。** 它比 BEDLAM v1 多了可复用的 Unreal Starter Pack、更完整的动态相机描述、filmback 控制、人体部位跟踪、VCam 轨迹、头发与鞋，以及相机真值的中心时间子帧记录。第一阶段应坚持透视相机，只验证一个人物、一段动作、一个场景、一个镜头的 RGB 与 DenseKP 138 点严格对齐。

本轮仅审计代码，没有安装 Unreal、下载资产包或执行渲染。官方仓库以浅克隆放在 `third_party/`，没有修改上游文件。

| 结论项 | 结果 |
|---|---|
| 推荐基线 | BEDLAM2 |
| 重渲染 | 可行 |
| BEDLAM 原生鱼眼 | 不支持 |
| BEDLAM 原生 ERP | 不支持 |
| Starter Pack | 存在，推荐用于 P0 |
| P0 复杂度 | 高：主要风险是 Unreal 依赖和跨 Blender/Unreal/离线 SMPL-X 的逐帧对齐，而不是 DenseKP 算子 |

证据来源为官方 [BEDLAM v1 render](https://github.com/PerceivingSystems/bedlam_render)、[BEDLAM2 render](https://github.com/PerceivingSystems/bedlam2_render)、[BEDLAM2 retargeting](https://github.com/PerceivingSystems/bedlam2_retargeting) 和 [bedlam_clothing](https://github.com/PerceivingSystems/bedlam_clothing) 仓库。本地审计版本分别为 `82d6c67`、`5765085`、`6f9154e`、`81d0a33`。

状态词含义：`FACT` 表示由仓库代码或官方说明直接确认；`UNKNOWN` 表示公开代码不足以确认；`NEEDS_ASSET` 表示工具链存在，但执行依赖未下载的授权资产。

## 2. BEDLAM v1 pipeline

```text
SMPL-X v1.1 male/female motion NPZ
  -> Blender: blender/smplx_anim_to_alembic/smplx_anim_to_alembic.py
  -> 带 pose corrective 的人体 Alembic ABC
  -> Unreal: unreal/import/import_abc_smplx.py
  -> 人体 GeometryCache
  + 预模拟衣服 ABC: unreal/import/import_abc_clothing.py
  + 人体/衣服纹理、HDRI
  -> tools/sequence_generation/be_generate_sequences_crowd.py 生成 be_seq.csv
  -> tools/sequence_generation/be_modify_sequences.py 调整相机
  -> unreal/render/Core/Python/create_level_sequences_csv.py
  -> unreal/render/Core/Python/create_movie_render_queue.py
  -> unreal/render/Core/Python/render_movie_render_queue.py
  -> RGB；可选 EXR depth/cryptomatte mask；camera CSV
  -> tools/post_render_pipeline/be_post_render_pipeline.sh
```

- `FACT`：Unreal Engine 5.0.3 / Windows 10+，Blender 3.2.2+。
- `FACT`：人体和衣服均以 `GeometryCache` 导入，SMPL-X pose corrective 已烘焙进 ABC。
- `FACT`：第二遍 Movie Render Queue 可输出 32-bit float multilayer EXR、depth 和 cryptomatte segmentation。
- `FACT`：v1 支持静态、zoom、orbit 等相机修改，但控制面少于 v2。
- `NEEDS_ASSET`：人体、动作、模拟衣服、纹理、HDRI、Unreal 工程资源。

## 3. BEDLAM2 pipeline

```text
locked-head/no-head-bun neutral SMPL-X motion NPZ（16 betas）
  -> Blender ABC: blender/smplx_anim_to_alembic/smplx_anim_to_alembic.py
       -> 带 pose corrective 的渲染人体 GeometryCache
  -> Blender FBX: blender/smplx_anim_to_fbx/smplx_anim_to_fbx.py
       -> 仅用于骨骼/人体部位相机跟踪，不承担最终人体渲染
  -> Unreal imports:
       unreal/import/import_abc_smplx.py
       unreal/import/import_fbx_smplx.py
       unreal/import/import_abc_clothing.py
       unreal/import/groom/import_abc_groom.py
       unreal/import/groom/create_groom_bindings.py
       unreal/import/shoes/*
  -> tools/sequence_generation/be_generate_sequences_crowd.py -> be_seq.csv
  -> tools/sequence_generation/be_generate_camera_animations.py
       -> be_camera_animations.json
  -> unreal/render/Core/Python/create_level_sequences_csv.py
  -> unreal/render/Core/Python/create_movie_render_queue.py
  -> Movie Render Queue + BE_MRQ
  -> PNG / 16-bit multilayer EXR / camera metadata
  -> tools/post_render_pipeline/be_post_render_pipeline.sh
  -> depth / mask / normal / camera CSV+JSON / MP4
```

- `FACT`：Unreal Engine 5.3.2 / Windows 11，Blender 4.0.2+。
- `FACT`：`BE_MRQ` 用来记录正确的中心时间子帧相机真值；普通 temporal sampling 下若缺它，RGB 与相机真值可能不是同一子帧。
- `FACT`：渲染人体使用 ABC；FBX 骨骼体隐藏，仅为 pelvis、spine、neck 等目标提供 socket。
- `NEEDS_ASSET`：`BE_Core`、`BE_IBL`、`BE_EnginePlugin`、`BE_MRQ` 及 Starter Pack 或完整资产。

## 4. Avatar

1. `FACT`：avatar 几何本质由 SMPL-X 模型、betas、pose 和 trans 决定；导入 Unreal 后形状已经烘焙在每个 ABC GeometryCache 中，不是运行时 beta 滑杆。
2. `FACT`：同一动作可以对不同 betas 重新生成 ABC；但这不等于任意交换已烘焙 GeometryCache 的 body shape。
3. `FACT`：v1 使用 SMPL-X v1.1 female/male。v2 渲染输入使用 locked-head、no-head-bun、neutral 模型和 16 个 betas；性别仍用于外观资源映射。
4. `FACT`：皮肤/人体纹理与头发可以按配置选择；头发是 Groom asset，并需对目标 body shape 建 binding。脸部外观受头部形状和人体纹理共同约束，不是无限独立组合。
5. `FACT`：Starter Pack 的 51 个 body shapes 是已经准备成 Unreal 资产的预烘焙人体/动作子集，不需先跑 Blender 和 import scripts。
6. `FACT + NEEDS_ASSET`：可创建新 shape，路径为 `betas + motion NPZ -> locked-head SMPL-X -> Blender ABC -> Unreal GeometryCache`；若使用鞋，还要 toeless v-template、鞋位移/材质；若使用头发，要生成 Groom binding；若穿衣，要有与该 body+motion 对应的 cloth simulation。
7. `UNKNOWN`：公开仓库没有证明任意新 body shape 可自动获得质量合格的服装模拟、头发 binding 和所有外观组合，需小样验证。

## 5. Motion

- `FACT`：v1/v2 的主输入是 SMPL-X `.npz`；v1 的 Blender 脚本也能适配 AMASS 格式。v2 motion 文件含 pose、trans、betas 等参数。
- `FACT`：shape 与 motion 在 SMPL-X 参数层解耦；在发布的 ABC/Unreal 资产层已经烘焙耦合。让同一 motion 驱动新 avatar，需要重新生成 ABC，衣服也通常需对应模拟。
- `FACT`：`bedlam2_retargeting` 使用 Unreal 5.3/5.4 IK Retargeter。实际路径为：NPZ 生成 source/target FBX，导入 Unreal，建立配对 CSV，retarget，再导出 FBX/NPZ，最后 `adjust_floor_npz.py` 校正地面。输出 NPZ 包含 `(frames,165)` pose、trans、betas、neutral、30 fps。
- `PARTIAL`：骑车、弯腰、下蹲、伸手、扭转等自定义动作可加入，只要最终得到合法 SMPL-X NPZ。对非 SMPL-X 骨架，理论上可建立 IK Rig/chain mapping，但官方自动化围绕 SMPL-X 编写，不能视为任意骨架一键支持。
- `NEEDS_ASSET`：SMPL-X Blender add-on、模型文件，以及待处理动作本身。

## 6. Clothing / appearance

- `FACT`：渲染仓库消费的是已经模拟好的 clothing ABC GeometryCache。`bedlam_clothing` 当前 README 明示为 WIP，公开代码可导出人体 motion ABC 和处理鞋，但不包含一套可直接批量生成任意新服装模拟的完整方案。
- `FACT`：衣服几何通常绑定到特定 body + motion + outfit 模拟结果；shirt/pants/dress 不能在运行时任意拼出新几何组合。
- `FACT`：衣服纹理可在 outfit 允许的纹理表内变化；人体/肤色纹理也可替换。**随机换纹理不等于随机换衣服几何。**
- `FACT`：头发为独立 Groom asset，但需 body-shape binding；v1 发布流程不提供可直接复用的商业头发，v2 Starter Pack 包含头发。
- `FACT`：v2 鞋使用特殊 toeless/sock-foot 人体模板加鞋形位移图和材质，并非普通独立刚体鞋模型。
- `FACT`：Starter Pack 包含模拟衣服及纹理、人体纹理、头发、鞋和 HDR 图像。
- `UNKNOWN`：Starter Pack 之外每类外观资源的可自由组合矩阵；应以官方 mapping/whitelist 和一次 Unreal 资产加载测试为准。

## 7. Scene

- `FACT`：官方 pipeline 可使用平面 stage + panoramic HDRI 做 image-based lighting，也使用常规 Unreal 3D level。HDRI 的“panoramic”只描述环境贴图，不代表输出 ERP 图像。
- `FACT`：序列生成可写入 camera root/world location，LevelSequence 创建脚本在当前已打开的 Unreal level 中布置角色、灯光和相机。
- `FACT`：可以替换为自有 Unreal level，前提是该 level 提供 BEDLAM 脚本期待的 actor、目录、插件和资产路径。
- `PARTIAL`：SimWorld 或公司场景只要能正确导入 Unreal，理论上可接入相同 LevelSequence/MRQ 层；其材质、坐标、碰撞、光照和 actor 命名适配量取决于外部实现，本轮无法确认。
- `NEEDS_ASSET`：官方示例工程/场景不在 Git 仓库主体内，需要从授权下载区获取对应 Unreal 资源。

## 8. Camera

BEDLAM2 的 `tools/sequence_generation/be_generate_camera_animations_config.py` 和 `be_generate_camera_animations.py` 实际支持：

| 能力 | 审计结论 |
|---|---|
| position / rotation | `FACT`：world/static、root/local XYZ、theta/yaw、phi/pitch、roll 可配置并形成 keyframe |
| focal length / HFOV | `FACT`：JSON 保存 HFOV，`create_level_sequences_csv.py` 用 `sensor_width / (2*tan(HFOV/2))` 写入 focal track |
| sensor width / height | `FACT`：v2 配置显式提供并写入 CineCamera filmback track；v1 主要读取 CineCamera asset 现有 filmback |
| camera motion | `FACT`：static、follow、dolly、orbit、zoom、shake、人体部位 look-at、VCam capture |
| static camera | `FACT`：`static_world_location`、`follow_static_location` 或相同 keyframe 均可实现 |
| image edge / truncation | `FACT`：可通过 body placement、camera local Y/Z、target offset、rotation 有意制造 |
| very near | `FACT`：distance 与相机位置可设；官方 preset 已有约 180--200 cm 的近景，自定义可更近 |
| collision | `UNKNOWN`：相机生成代码未发现通用碰撞求解；不能假定会阻止穿模 |
| near clipping | `UNKNOWN`：仓库配置/脚本未发现 near-clip 参数，实际值由 CineCamera/引擎资产决定，P0/P1 必须实测 |
| output resolution | `FACT`：MRQ/批渲染参数可设置，示例命令传入 `ResX/ResY` |

要生成“边缘”和“截断”样本，应关闭或绕过以完整取景为目标的 `camera_autodistance`，显式记录相机 pose、filmback、焦距、分辨率和 target offset。

## 9. Fisheye

对两个 render 仓库的脚本、配置和说明搜索 `fisheye`、`lens distortion`、`camera model`、`render target`、`cubemap` 等后，未找到鱼眼成像或标定集成。结论是：

- A（BEDLAM 原生支持）：**否**。
- B（Unreal/插件可扩展、BEDLAM 未集成）：**是，需另行选型和验证**。
- C（自定义 shader/post-process/ray projection）：**可行候选**。
- D（透视或 cubemap 后离线映射）：**可行候选，优先 cubemap 而非单个透视视图**。

严格 DenseKP GT 的要求不是“把二维点看起来弯一下”，而是 RGB 和 3D 点共享同一个明确鱼眼模型，例如 equidistant 或 equisolid：

```text
同一帧、同一光心的场景 RGB（最好为 cubemap）
  -> 按目标 fisheye ray model 重采样 RGB
SMPL-X vertices + body/world/camera transform
  -> R[138,10475] @ vertices
  -> 同一 fisheye ray model 直接投影 138 个 3D 点
```

单透视图离线 warp 只能覆盖源视锥内射线，超广角时会缺失侧后方内容并降低边缘采样质量；cubemap 覆盖更完整。还必须统一主点、有效圆、视场、图像方向、遮挡和落在相机后的点处理。

## 10. ERP

仓库没有 equirectangular、360 输出或 `SceneCaptureCube` 到 ERP 的实现，故 **ERP_NATIVE = NO**。推荐：

```text
固定同一 optical center、同一帧渲染六个 cubemap faces
  -> 每个 ERP pixel 反算球面方向并采样 cubemap
  -> 得到 ERP RGB

DenseKP camera-space 3D
  -> 归一化方向
  -> longitude / latitude
  -> ERP pixel
```

DenseKP 应直接做 spherical projection，不能先把 perspective 2D 点做通用 warp。ERP seam 上同一个点可能落在图像两端，标注与 loss 需显式采用环形横坐标。若输出 depth，建议存光心到表面的径向距离，或者至少把深度定义写进数据格式；不能不加说明地混用 perspective Z。cubemap 六面必须锁定同一帧和光心，否则动态人物会产生接缝。

## 11. GT generation

官方直接输出能力：

| 输出 | v1 | v2 |
|---|---|---|
| RGB | PNG/EXR | PNG/EXR |
| depth | 可选 multilayer EXR | 可选 multilayer EXR |
| segmentation / instance | cryptomatte/layer，可后处理 PNG | cryptomatte/layer，可后处理 PNG |
| normal | 未在主流程确认 | world/camera-space normal 后处理 |
| camera pose/intrinsics | camera CSV | EXR metadata -> CSV/JSON，含 filmback/HFOV/focal |
| body identity | `be_seq.csv`/asset naming | `be_seq.csv`/asset naming |
| 每帧 SMPL-X vertices | 不直接从 Unreal 导出 | 不直接从 Unreal 导出 |
| DenseKP 138 | 不提供 | 不提供 |

严格对齐的后续实现应保存一份不可歧义的 render manifest：source NPZ、body/betas、motion frame、ABC/Unreal asset、sequence start、body world transform、实际相机 GT、resolution 和 projection model。然后：

```text
SMPL-X NPZ 同帧参数
  -> SMPL-X 10475 vertices
  -> 应用与 Unreal 完全相同的 body/world transform
  -> densekp_regressor_R.npy @ vertices
  -> 应用 BE_MRQ 记录的实际 camera transform
  -> perspective / fisheye / spherical projection
  -> DenseKP 138 二维坐标 + 几何可见性/画面内状态
```

`FACT`：NPZ 和 `be_seq.csv` 能保留动作帧与人物变换，BE_MRQ 能保留实际相机真值，所以该链条理论上可严格对应。`NEEDS_VALIDATION`：ABC 时间采样、坐标轴/单位、root transform、motion blur 中心子帧和离线 SMPL-X 重建必须用 P0 overlay 数值验证后才能批量生成。

## 12. Extreme-case generation

| 目标 | 控制方法 |
|---|---|
| Near camera | 缩小 camera-person distance；关闭 auto-distance；检查 near clip 与穿模 |
| Image edge | camera local Y/Z、look-at target offset、body XY、camera yaw/pitch |
| Truncation | 让身体一部分明确落出 view frustum；不要自动重新居中 |
| Extreme pose | 按 motion 元数据/离线关节统计筛动作与帧，或导入自有 SMPL-X motion |
| Scale | distance、focal/HFOV、sensor 和 resolution 联合控制 |
| Multi-person overlap | `be_seq.csv` 控制人物数量和 XYZ/yaw；自定义模式需放松官方防重叠 occupancy/safety zone |

建议配置接口草图：

```yaml
avatar:
  body_id: it_4001_XL
  gender: neutral
  betas: null                 # 已烘焙资产时由 body_id 决定
  body_texture: null
  hair: null
  shoes: null
motion:
  source_npz: null
  motion_id: null
  frame_range: [0, 1]
  start_frame: 0
clothing:
  geometry_cache: null
  texture: null
scene:
  unreal_level: /Game/Bedlam/IBLMap
  hdri: null
camera:
  model: perspective          # perspective | fisheye | erp
  resolution: [1280, 720]
  sensor_mm: [36.0, 20.25]
  hfov_deg: 65.470451
  distance_cm: 300.0
  yaw_deg: 0.0
  pitch_deg: 0.0
  roll_deg: 0.0
  target_body_part: pelvis
  target_offset_cm: [0.0, 0.0, 0.0]
  near_clip_cm: null          # 需从 Unreal 资产读取/显式设定
augmentation_case: baseline
gt:
  densekp_operator: assets/densekp_regressor_R.npy
  camera_source: be_mrq_center_subframe
  projection_model: perspective
```

## 13. BEDLAM v1 vs BEDLAM2 comparison

| 项目 | BEDLAM v1 | BEDLAM2 |
|---|---|---|
| Unreal | 5.0.3 | 5.3.2 |
| Blender | 3.2.2+ | 4.0.2+ |
| avatar | male/female v1.1 ABC | locked-head neutral、16 betas、ABC + tracking FBX |
| motion | SMPL-X NPZ，可适配 AMASS | SMPL-X NPZ；另有官方 retargeting 工具 |
| shape/motion 灵活度 | 可重烘焙，资产层耦合 | 可重烘焙；retargeting 更完整，资产层仍耦合 |
| clothing | 预模拟 ABC | 预模拟 ABC，Starter Pack 可直接用 |
| hair | 流程中无可发布的商业 hair | Groom import/binding，Starter Pack 含 hair |
| shoes | 无完整专门流程 | toeless body + displacement/material |
| scene | HDRI stage / Unreal level | HDRI stage / 多种 Unreal level 接口 |
| camera | 静态、修改、zoom/orbit | JSON keyframe、static/follow/dolly/orbit/VCam/body-part target |
| filmback | 主要来自 CineCamera asset | sensor width/height 显式配置并写 track |
| depth/mask | 支持 | 支持 |
| normal | 未确认 | 支持后处理输出 |
| camera GT | render CSV | BE_MRQ 中心子帧 metadata -> CSV/JSON |
| starter assets | 未发现对应 quickstart | 150 motions / 51 shapes Starter Pack |
| setup | 旧版本、能力较少 | 依赖更多组件，但 Starter Pack 跳过 Blender/import |
| fisheye / ERP | 均未集成 | 均未集成，但相机/GT架构更便于扩展 |
| DenseKP 造数适配 | 可行 | **更合适，推荐** |

推荐 BEDLAM2，不是因为它已解决鱼眼/ERP，而是因为它对相机、资产 quickstart、相机真值与外观的基础设施更完整。

## 14. Minimal prototype plan

### P0：最小验证

目标：1 avatar、1 motion、1 scene、1 perspective camera，渲染极少量帧，并输出准确 DenseKP 138 GT。

1. 经授权仅准备 BEDLAM2 Quickstart 所需的 `BE_Core`、`BE_IBL`、`BE_EnginePlugin`、`BE_MRQ` 和 Starter Pack 组件；使用 Windows 11 + Unreal 5.3.2。
2. 从 Starter Pack whitelist 选一套 body + motion + clothing，生成最小 `be_seq.csv` 和 `be_camera_animations.json`。
3. 渲染 PNG/EXR，并从中心子帧 metadata 提取 camera GT。
4. 用同一 motion NPZ 重建 SMPL-X，应用 `be_seq` body transform 和实际 camera transform，计算 `R @ vertices` 后透视投影。
5. 在 RGB 上画 138 点；逐项验证帧号、单位、坐标轴、主点、filmback、焦距、body root、motion blur 子帧。通过后才定义 P0 完成。

复杂度评估为 `HIGH`：Starter Pack 降低资产准备工作，但 Unreal 插件依赖以及跨系统几何严格对齐仍需实机验证。

### P1：Camera domain

1. 先用透视相机增加 near、edge、truncation，并建立自动 overlay/误差检查。
2. 固定一种鱼眼数学模型；优先试验同光心 cubemap -> fisheye，并让 138 点直接使用同一模型投影。
3. 固定 cubemap -> ERP 规范及 seam/depth 语义；六面锁帧后生成 ERP RGB/depth 和 spherical DenseKP。
4. 对近裁剪、人物穿相机、画外点、相机后点、鱼眼有效圆和 ERP seam 写明确标签约定。

### P2：Domain randomization

在 P0/P1 的几何回归测试不变的前提下，扩大 avatar、motion、预模拟 clothing、纹理、场景、HDR lighting 和多人物遮挡。先只组合官方 mapping 允许的资源；新 shape + 新衣服几何另立 cloth-simulation 子项目，避免与相机域验证耦合。

## 15. Open questions

1. `NEEDS_ASSET`：Starter Pack 各下载组件的精确压缩/解压大小未在 Git 仓库或可公开访问说明中列出，下载前应先查看授权下载页文件清单。
2. `NEEDS_ASSET`：`BE_Core`、`BE_EnginePlugin`、`BE_MRQ` 的实际资产内容与本地安装兼容性必须在 Windows Unreal 5.3.2 验证。
3. `UNKNOWN`：CineCamera 实际 near clip、超近距离 Groom/cloth/skin 的稳定性，以及相机是否会穿模。
4. `UNKNOWN`：官方资产对自定义 fisheye plugin、SceneCaptureCube 和六面锁帧 MRQ 的兼容性。
5. `NEEDS_VALIDATION`：SMPL-X/Blender/Unreal 坐标变换、ABC sample 与 BE_MRQ 中心子帧能否达到逐像素 DenseKP 对齐。
6. `UNKNOWN`：外部 SimWorld level 的 actor、材质、单位和灯光接口；只能确认 Unreal level 是可替换边界。
7. `LICENSE`：BEDLAM2 许可只允许单用户、非商业科研/教育/艺术用途，禁止转授和再分发；项目用途和产物共享方式必须先满足官方许可。

Starter Pack 的官方获取入口是 [BEDLAM2 项目页](https://bedlam2.is.tue.mpg.de/) 的下载区；安装与依赖清单见本地 `third_party/bedlam2_render/unreal/render/unreal_quickstart.md`。官方说明确认其中有 150 段已配模拟衣服的动作、51 个 body shapes、人体/衣服纹理、头发、鞋与 HDR images，并且这些渲染资产已经准备为 Unreal 资产，足以作为第一次 synthetic render demo 的起点。
