# 城市三名骑行者：同一场景逐帧双 rig ROI 数据

日期：2026-10-11。状态：**31 时刻生产、打包、序列旁表和有界 Dataset 回读通过；本批新视频的人工观感待观看。** [并列观看页](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/index.html)提供[边缘 rig](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/rigs/rig_edge/pack/dual_roi.mp4)及[斜前方 rig](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/rigs/rig_front_oblique/pack/dual_roi.mp4)的双路视频。灰色排版区未渲染，黄色线表示各机物理有效域边界；没有生成低清背景或完整 ERP RGB。本批只用于 `integration_only/shared_source_diagnostic`，没有模型前向、参数更新或正式来源划分。

## 冻结场景与三份动作

直接只读复用上一批[城市 attempt5 精确预检成品](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/city_attempt5_cpu_only_correction1/prepared/)。其 31×3×138×3 世界点、31×3×4×4 场景矩阵与 31×3×2×138 有效域数组在本批逐轴核查；旧 `diagnostic_only` 预检双路边缘为 23/31，精确活动头发与逐人 square crop 并集 ROI 已通过。旧严格 CPU 投影遮挡事件未通过的记录保持原状。这次**没有**重选布局、重新抽人物、拟合或 build 缓存；[资产只读复用记录](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/assets_reuse.json)列出权威 manifest 和持久文件。

| 人物 | 身体／动作／衣物／头发 | 原动作来源 | 首个组合 asset 帧 |
| --- | --- | --- | ---: |
| p1（预定主目标） | M01／APD6526／M02_simple／M01_groom | APD，身份／session 未核实 | 0 |
| p2 | M02／participant_1 comfortable／M01_simple／F02_groom | Zenodo participant_1 | 4 |
| p3 | F02／participant_2 straight／F02_simple／F02_groom | Zenodo participant_2 | 19 |

三个角色家族、衣型和 motion ID 均不同；p2/p3 共用 F02_groom，不能称三种发型。 participant_1 与 participant_2 是不同的发布参与者代码及录制路径，现实演员身份未独立核验。三人首／中／末所用身体缓存顶点确实变化，场景不是平移静态网格。每个人自己的原整数 source tick、time_base、量化误差和 asset/source 帧保留，合成 episode 为 0–30 tick、1/10 秒时间基；每 rig 的两个相机消费同一时刻三人状态。原 rig 相对外参和 0.18m **合成**基线不变，后者不是实测真机参数。`scene_episode_id` 中的 `s20261027` 仅是本任务命名空间，**并未**使用 seed20261027 新抽三人布局；实际布局来自 seed20261023 的 attempt5，见[命名澄清](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/provenance_clarification.json)。

## 第二 rig 的冻结与预检

`rig_edge` 两机矩阵精确复制旧合格城市 attempt5。`rig_front_oblique` 只改变整组双机的道路平面位置和 yaw，三人世界轨迹、时间、shape、衣物、头发及环境不动。[版本化配置](../../configs/urban_three_cyclists_multirig_a1_20261011.json)和[24 份先冻结的相机候选](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/frozen_camera_candidates.json)用 camera seed20261028 洗牌、给有界小扰动；另用 seed20261029 仅做计划级变化核查，人物选择逐字段一致且相机表不同，未投影或渲染第二 seed。

按冻结顺序第 0 份候选即通过，另外 23 份没有投影，不能计作拒绝。其整体 rig origin 为 `[-2.74495649, 4.58355702, 0]`m，yaw `-36.34225869°`。31 时刻的目标两机同时在原 `[45°,80°)` 斜前方类别、且至少 8 个共同物理域内 R138 点的最长连续窗口为 **frame7–14，共 8 帧、首末 0.7 秒**；共同物理点≥8 为 **24/31**。三人逐帧 body／cloth／实际活动 groom 的非周期 ERP square crop 支持、逐相机联合 ROI≤700万像素通过，最大仅 1,609,797 像素。静态场景中四个相机中心的 10 方向、0.15m 短射线净空检查通过，但不是完整运动扫掠体碰撞证明。 斜前rig位于本城市道路边带的合成位置；本批未证明真实安装可行性。两 rig 的组内相对矩阵差最大 `2.78e-16`，cam4 没有重复翻转；physical mask 沿用相机局部校准坐标，没有用黑 RGB 或 ROI 边界代替有效域。见[冻结执行清单](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/frozen_execution.json)、[斜前预检](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/prepared_front_oblique/geometry_preflight.json)和[俯视图](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/topdown_rigs.svg)。CPU 几何预检实测 61.61 秒，未使用满 15 分钟上限。

## 公共接口与真实生产

[多 rig 入口](../../tools/experiments/run_multirig_roi_a1.py)现接受版本化 `scene_episode_rig_list_v1`：[本批 rig 清单](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/rig_list_manifest.json)的每项独立记录 `rig_id`、prepared/calibration 引用、用途和 camera seed。旧 `--edge-prepared/--side-prepared` 双 rig 命令、旧单 rig 消费协议与旧河岸样本仍可用。公共[逐帧 Blender 回调](../../tools/render/render_multirig_roi.py)去掉写死的两 rig 名称并核对三人人数轴；[常驻同步 CPU 消费者](../../src/densekp_retrain/runtime/frame_consumer.py)按 `(frame,rig)` 唯一键接收，不允许重复或未知 rig。旧专用 frame0-side 故障续接没有冒充通用恢复；复杂部分完成状态仍须显式核查，直接重跑被拒绝。

本次正常路径**一个 Blender 进程加载城市环境与三人一次、一个 CPU 消费者**。每时刻更新三人的 body／cloth／head-groom 一次，再依序渲染 `rig_edge(cam2,cam4)`、`rig_front_oblique(cam2,cam4)`，整帧四路完成后同步消费两组，再进入下一时刻。[相机事件日志](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/render_calls.jsonl)与[逐单位状态](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/unit_states.json)核对 31 次正常路径几何更新、124 次成功相机调用、62 个已消费 `(frame,rig)` 单位，**无渲染故障重试**。本任务 ROI-only、5760×2880 逻辑 normal ERP、64 samples CUDA、原城市场景灯光/材质/色彩与无降噪设置不变；世界仍参与阴影、遮挡和反射。相机每帧各自规划本 rig 和三人完整 square crop 的联合支持，生产整数 ROI 与预检逐帧一致。

## 数据、覆盖与边界

[全量机器验收](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/acceptance.json)检查 124 张实际 ROI 的尺寸、offset、mask、实例 ID、有限 GT、时间、rig/person 映射、预检 `in_view`、原生框支持及 pack 对应。两组各 31×2×3＝186 人物观察槽，共 **372 槽、323 条合法原生框行**：edge 181 行／5 不可用（4 个完全邻人遮挡无框、1 个原框小于历史 2px 合法边）；斜前 142 行／44 人物物理域外。无框槽位留在旁表，未造黑色训练图、上帧框或零误差。两个 rig 各有独立 [NPZ 与序列旁表](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/rigs/rig_edge/pack/)，都保留相同 `scene_episode_id` 和各自 `rig_id`；只在同 rig 内配对 cam2/cam4。两组序列索引的持久路径和 packed row 一一对应检查通过，真实 dataset3 曝光同步并未因此得到认证。

预先固定每 rig × 三人 × frame0/15 × 双机的 **24 槽**，实际 Dataset 读取 edge 11、斜前 9，合计 **20 行**；其余 4 槽无合法框，不换帧凑数。[各 rig 读回记录](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/rigs/rig_front_oblique/pack/readback.json)验证局部 ROI RGB 裁剪、138 点完整 ERP→256-square 仿射、`_trans`、`img_size=[2880,5760]`及 `in_view & inside_full_crop` 可靠遮挡监督。第三值不是置信度。完整 31 帧视频已编码并解码核对 PTS 0.0–3.0 秒，末帧展示保持 0.1 秒，不复制动作帧。

**edge rig** 的主目标两路同刻双边缘为 **23/31**，共同物理点≥8 为 31/31；**斜前 rig** 的对应值为 **0/31** 和 **24/31**。斜前只作为短窗口视角对照，不能计入双边缘配额，也不能称全段斜前。三人都至少在一台相机持续 31 帧有合法框且≥8个实际可见 R138 点，满足本批连续6帧内容门槛；p3在斜前 cam4 全段域外，已如实保留。两组实际邻人 ID＋range 遮挡证据中的目标 15/15 个百分点加重—恢复三阶段诊断**均未通过**；edge/cam2 和 cam4 仍存在邻人遮挡，但不称事件恢复。严格端点 `boundary_transition` 未作为本批门槛，也未据本结果宣称通过。技术通过、边缘覆盖、斜前窗口、邻人恢复、人眼观感分别记录。[覆盖清单](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/coverage_inventory.json)保留前批失败与接受范围，新增的两个 rig 仍属于**同一个**包含 APD、participant_1、participant_2 的共享来源组，不可拆入拟议 train/val 两侧。

## 计算、测试和下一步

开算前检查 GPU/进程、CPU 内存和磁盘；GPU1 当时有另一 Blender 进程，选择空闲 GPU0 `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，未终止或抢占别的任务。[资源和耗时日志](../../outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/task_wall.json)：Blender 进程约 **1099.77 秒**，其中装配至帧回调约 **48.20 秒**；首时刻回调 **44.13 秒**，后 30 时刻均值 **33.41**、中位 **31.88**、范围 **29.99–41.62 秒**，含同步消费。两 rig 相机调用累计 edge **700.59 秒**、斜前 **189.95 秒**；一个常驻消费者启动 **2.25 秒**，62 单位函数体累计 **61.76 秒**。两次 rig 打包／读回／视频收尾共 **38.24 秒**；生产任务完整 wall **1138.03 秒（18.97 分钟）**，低于本批 45 分钟渲染＋必需消费上限。不同 ROI 面积和首路动态同步使两 rig 相机秒数不可当同负载速度比较。Blender 峰值 RSS 约 12.54GiB、消费者约 0.84GiB、GPU 采样峰值 7917MiB；任务磁盘采样峰值 359.87MB，成功 compact 后 252.66MB、临时残留约0.26MB。GPU kernel、EXR 写出和同步不能再细分。独立 CPU 预检、短射线检查、开发与小型测试不混入生产 wall。

改动相关的 **23 项**定向 CPU 测试通过：相机 seed 与人物随机流分离、三人独立帧轴、旧双 rig 读取兼容、rig ID 冲突与重复消费拒绝、共同窗口、来源时间冲突。没有全库 discover、DenseKP/YOLO 前向或参数更新。验收脚本最初把“栅格有微小原框”误等同于“必有打包行”，与实际 `below_min_side_px` 不可用槽位冲突；仅修正该检查的语义并全量重跑校验，未修改标签、历史 2px 规则或重渲染任何帧。

复用入口：先用 `python tools/experiments/prepare_urban_three_multirig_a1.py plan --output <新冻结候选文件>`，再用 `... preflight --frozen <候选文件> --output <新prepared目录>`；以 `scene_episode_rig_list_v1` manifest 指向已验收的两个 prepared 目录，运行 `python tools/experiments/run_multirig_roi_a1.py --rig-list-manifest <manifest> --output <全新目录> --gpu-uuid <空闲GPU-UUID> --budget-seconds 2700`。本批为在渲染前完成场景短射线净空检查，先调用 `prepare(...)` 建立尚未启动的生产目录，再执行 `--run-prepared`；该选项拒绝已有相机调用或已消费单位。旧资产和旧输出未覆盖、没有 SHA256 或 commit。

仍未覆盖 4–5 人、三岔路口、任意数量 rig 的实测、连续真实相机曝光同步、正式来源独立划分、增强框预留与模型效果。当前公共 Blender 装配上限仍是 3 人；4–5 人需扩展逐人装配/实例 ID、所有人对几何检查并验证联合 ROI 成本。三岔路口需另立道路局部坐标、合理转向动作和轨迹/相机预检，不能把这段直行队伍改名复用。下一批应先评审来源关系和框增强 ROI 支持，再决定数据冻结、原权重基线和受控训练；本批不自动进入这些阶段。
