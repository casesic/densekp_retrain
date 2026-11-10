# 两种种子道路模式：单人骑行经过双相机交界与行走／骑行交会

日期：2026-10-10。**两种模式的配方、31 时刻 CPU 预检和各一段 ROI-only 双路连续数据均已交付。** 数据链与物理边缘覆盖通过；混合段的实际邻人遮挡加重／恢复诊断也通过。本批全部为 `integration_only/shared_source_diagnostic`，没有推理、训练、合图或独立验证集。查看入口：[两段视频与轨迹的统一页面](../../outputs/experiments/seeded_scene_modes_a1_20261010/index.html)；[单人视频](../../outputs/experiments/seeded_scene_modes_a1_20261010/runs/single_cyclist_boundary_pass_s20261010_e00/pack/dual_roi.mp4)、[混合视频](../../outputs/experiments/seeded_scene_modes_a1_20261010/runs/mixed_walk_cycle_crossing_s20261010_e00/pack/dual_roi.mp4)。

本轮代码限定为公共 `mixed_road_pass` 的单／双人预检和过界判定、种子配方入口与已绑定资产池、CPU 投影图和 manifest 汇总入口、单人无邻人事件的统计处理，以及新模式有界 Dataset 抽查元数据；两种模式继续调用原 `plan_road_pass.py`、`run_continuous_roi.py`、ROI 消费、打包和 Dataset，没有第二套渲染器或监督实现。旧 road_pass 默认、历史配方和数值 fixture 未改。

## 实际接通的模式和种子

新增 `seed_scene_modes.py`，输入 `mode`、master seed、episode count、已验证绑定的资产池及参数范围，输出不同 ID 的子episode配置和来源记录。抽样算法版本 `seeded_scene_modes_v1`；资产选择与布局使用分开的局部随机流，候选再按各自子seed取值，不使用全局 RNG。骑行 body／衣物／groom／动作整套选择，不拆开随机重绑；M04 是固定已存在行人组合。每段人物动作起点只从能覆盖全部 31 个合成 10 Hz 时刻的原记录中选，按 `nearest_original_record_ties_earlier` 保存原 source tick/time_base 和量化误差，无循环、插值或复制。M01 的 76 条原观察刚好覆盖 3 秒，合格起点仅 1 个，明确记录为固定；F02 有 30 个合格起点。道路朝向由各资产已核查前向约束为 +Y，不作为独立随机轴；两人、rig 和 0.18 m **合成**基线不随机。

冻结 master seed 20261010、20261011，各模式各一份，完整 31 时刻预检；同 seed 重算四份配置逐项相同，不额外计候选。不同 seed 确实改变了资产（单人 M01→F02）、原动作起点及布局候选，混合两份虽均选 F02，也分别取原起点 3/13、M04 起点 0/1 与不同轨迹。详情：[4 份配方与选中记录](../../outputs/experiments/seeded_scene_modes_a1_20261010/recipes/)、[重现检查](../../outputs/experiments/seeded_scene_modes_a1_20261010/recipe_reproducibility.json)、[来源关系](../../outputs/experiments/seeded_scene_modes_a1_20261010/source_graph.json)。

| 模式／seed | 骑行资产；原记录起点 | 尝试／最多12 | 选中 attempt | 最大单路联合 ROI |
|---|---|---:|---:|---:|
| 单人／20261010 | M01/APD；0 | 8 | 7 | 1,600,225 px |
| 单人／20261011 | F02/bikefit；10 | 1 | 0 | 2,601,769 px |
| 混合／20261010 | F02/bikefit；3，M04/KIT425；0 | 5 | 4 | 4,220,970 px |
| 混合／20261011 | F02/bikefit；13，M04/KIT425；1 | 12 | 11 | 4,264,225 px |

总共尝试 **26/48** 个布局候选；拒绝均留在 [逐候选记录](../../outputs/experiments/seeded_scene_modes_a1_20261010/preflight_summary.json)：单人首 seed 有 6 次未证实穿越、1 次同时边缘不足；混合首 seed 有 2 次近邻顶点、2 次 root 距离不足。第二混合 seed 的 11 次近邻拒绝也保留，没有把失败从分母删去或换资产。对每模式按冻结 seed 顺序只渲染第一份预检通过的配方，未使用模型表现挑选。顶点最近距仍是身体／衣物表面的近似，不是三角面连续碰撞证明。

公共 `mixed_road_pass` 预检现在显式支持 1 或 2 人；旧两人配置及历史默认保留。单人省去邻人检查、仍执行道路、固定骑行高度、逐人物原时间、物理域、完整 ERP 裁剪、逐人 union ROI 及 7M 像素上限。两模式目标均须在**同一时刻** cam2/cam4 各 ≥8 个域内、≥5 个域外可靠 R138 点、共同域内 ≥8；至少 16/31 时刻且连续跨度 ≥0.5 秒。单人另要求首时刻两路各 ≥20 域外点、末时刻各 ≤5，且物理 mask 有符号边界距离的第 10 百分位由负转正，避免把沿边停留叫“经过”。四份选中轨迹的 31 个原动作帧身体前向与道路方向最大偏差均 ≤25°，见 [朝向审计](../../outputs/experiments/seeded_scene_modes_a1_20261010/orientation_audit.json)。预检框是无遮挡几何代理和保守裁剪包络，不是真实可见实例紧框；实际框仅在渲染后产生。普通 ERP 边界／跨接缝裁剪按现有拒绝策略处理，没有实现周期拼接。

## 用户要求的单人过交界

首个合格 seed 的 M01 连续踩踏资产在场景中沿直路以**合成** 0.496 m/s 前进，原 APD 源帧与整数 tick 保留。没有自行车模型，也没有把骑行动作重新拟合或改速。两相机见同一人、同一时刻的不同局部；“交界”按各自鱼眼 physical valid mask 的投影判定，**不是**实体相机间 0.18 m 空隙、ERP 左右接缝或未经标定的最终合图线。实际首／中／末域外 R138 数：cam2 **25→11→1**，cam4 **20→16→2**；两路边界距离低分位数均由负转正。31 时刻中 21 个满足同时双边缘门槛，主连续窗口 frame 0..19、跨度 1.9 秒。31/31 时刻有双路合法输入且共同 crop 监督 ≥8 点，实际共同数从首时刻 93 到末时刻 135。这里证实的是从双侧部分截断进入几乎完整有效域的**一段经过过程**，不要求 3 秒内完成从一相机完全消失到另一相机完全出现。

[俯视道路轨迹](../../outputs/experiments/seeded_scene_modes_a1_20261010/visuals/single_cyclist_boundary_pass_s20261010_e00/topdown.png)标相机和场景行进方向，不虚构地面上的“精确拼接缝”；[首中末双路物理域／ROI 投影](../../outputs/experiments/seeded_scene_modes_a1_20261010/visuals/single_cyclist_boundary_pass_s20261010_e00/physical_projection.png)以绿／红 R138 点区分域内／域外、蓝框表示高质量 ROI。[直接观看单人双路视频](../../outputs/experiments/seeded_scene_modes_a1_20261010/runs/single_cyclist_boundary_pass_s20261010_e00/pack/dual_roi.mp4)显示真实原生框、physical 边界和合成时刻。无邻人，邻人遮挡指标明确为不适用；仍存在原动作自身或场景遮挡。

## 混合交会与真实数据链

首个合格混合配方选 F02 前倾骑行＋不同身份、外观和原动作来源的 M04 行人。F02 场景拥有固定高度与直路速度 0.487 m/s；M04 保留 source-root-relative 真实行走，整段仅一次固定放置与 yaw，未再叠加平移速度。两人 body/cloth/R138 各用自身世界矩阵，groom 用自身 scene×head；F02 原 1/30000 PTS、M04 发布 30 Hz 时间各自保留。双路物理边缘 frame 0..18 共 **19/31**、连续跨度 1.8 秒；具有两路输入和 ≥8 个共同 crop 监督点的时刻 **30/31**。

真实实例 ID＋range 的保守邻人遮挡证据通过原 15/15 个百分点、每阶段 3 个相邻时刻诊断：cam2 前 0..2、中 16..18、后 20..22，加重／恢复各 **46.96** 个百分点；cam4 前 0..2、中 16..18、后 21..23，加重 **21.74**、恢复 **20.99** 个百分点。frame18 的 F02 cam2 完全受邻人遮挡、无原生框，共同 crop 集合为 `null`；frame19 cam2 原生框重现、共同监督 95 点，frame20 为 127 点。该过程并未因无框终止，也没有补上一帧框或把不可评估点对计成误差零。混合段 124 槽位中 99 `observed`、1 `empty`、24 `person_absent`；两路 RGB 共 62 张、有效 packed 行 99。行人有时离开某一路物理域，不能把它误报为整张相机图缺失。遮挡曲线、每人物／相机分母和框变化见 [实际验收](../../outputs/experiments/seeded_scene_modes_a1_20261010/runs/mixed_walk_cycle_crossing_s20261010_e00/pack/multiperson_acceptance.json)及[曲线](../../outputs/experiments/seeded_scene_modes_a1_20261010/runs/mixed_walk_cycle_crossing_s20261010_e00/pack/target_curves.svg)。这里只验收几何遮挡与框重现，没有预测恢复。

两段各一次 Blender 装配、一个常驻同步 CPU 消费者；每段 31 时刻×2 相机各 62 次成功 ROI 调用，合计 124。64 samples CUDA、5760×2880 逻辑 ERP、既有材质／色彩／灯光、无降噪与自适应；没有低清背景、完整 ERP RGB 生产读写或 DS RGB 往返。两段所有 62 个实际整数 ROI 与各自完整 31 时刻预检**逐一相同**，最大单路分别 1.60M／4.22M 像素。真实 Dataset 对首中末及双路做局部 RGB→256-square、R138 仿射、监督集合、`_trans`／完整 ERP `img_size` 核对，通过；单人实际读 **6** 行，混合实际读 **10** 行。两段 NPZ／序列索引一致：总计 186 观测槽位、161 合法行，遮挡可靠 GT 仍按 `in_view & inside_full_crop` 监督。视频各解码 31 帧，PTS 0..3.0 秒、末帧另展示 0.1 秒；灰色排版区是未渲染区域，不是真实背景。全部成功临时源按所有权清理，长期数据不依赖 EXR。详见 [统一验收 JSON](../../outputs/experiments/seeded_scene_modes_a1_20261010/acceptance.json)。

**验收范围偏差：**本轮要求每模式最多 8 条实际 Dataset 读取。单人符合；混合的 prepared recipe 漏传显式 `scene_mode`，收尾沿旧混合规则读了 10 条（超 2），之后才发现。原[回读记录](../../outputs/experiments/seeded_scene_modes_a1_20261010/runs/mixed_walk_cycle_crossing_s20261010_e00/pack/readback.json)保持不改；已修正新模式元数据传递，使后续收尾上限生效，未为此重渲染或追加 Dataset 读取。[执行偏差记录](../../outputs/experiments/seeded_scene_modes_a1_20261010/execution_notice.json)将数据链通过与抽查预算超额分开。不能声称本轮严格满足“每模式 ≤8 次”这一项。定向 CPU unittest **33 项通过**，覆盖 seed/子ID、资产许可和内部接受过滤、过短来源拒绝、无合法道路布局保留拒绝、两类 root／时间、旧 road_pass 入口及 ROI／观测策略；无全库 discover 或模型更新。

## 成本、来源与下一步边界

单人使用启动时空闲的 GPU0（UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`）；随后 GPU0 出现他人计算进程，混合段改用无计算任务的 GPU1（UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`，仅有空闲桌面 Blender），没有终止或争用他人进程。单人一次装配回调 20.43 秒，首帧 Blender block 16.23 秒，后 30 帧均值 9.57 秒，含打包／6 行回读／视频整段 **359.61 秒**；混合一次装配 36.65 秒，首帧 32.35 秒，后 30 帧均值 26.91 秒，完整交付 **956.54 秒**。合计新增渲染及必需消费 **1316.15 秒（21.94 分钟）**，低于 45 分钟。混合更大 ROI 与第二人物负载不同，不以两段差值声称算法加速。采样 GPU 峰值单人 3535 MiB、混合 4508 MiB；Blender RSS 峰值约 7932／12030 MiB。临时源采样峰值约 176／558 MB，保留目录约 75.8／224.9 MB，整个输出约 308.3 MB。相机调用计时含同步与 I/O，不伪造纯 GPU kernel 时间；分项见 [统一验收](../../outputs/experiments/seeded_scene_modes_a1_20261010/acceptance.json)。无渲染重试、额外 episode 或训练计算。

资产池目前只允许已接通且内部接受的 M01/APD、F02/bikefit 与 M04/KIT425 绑定组合；F02 历史冻结 manifest 的旧 pending 字段不被本批覆写，当前用户接受状态由后续交付与对话确认。bikefit 许可为 unknown，只限内部集成诊断；KIT425 是历史验证来源，M04 是混合段共同遮挡者。双路邻帧、换 seed、同角色家族的其它动作衍生版本均不能跨独立 train/val。缺独立行人来源、正式来源图划分、p2/M02 已接受公共适配，以及训练框扰动的预留 ROI 支持；现有 32px 规划余量不等于任意增强可用。队伍 1–5 人、三岔路口、入场延迟、相机随机位姿、随机衣物重绑均**未实现**。8 段/200 步草案仍待评审，不由本报告授权。

复用命令（输出必须是新目录；下列展示入口与配置，渲染仍需显式调用）：

```bash
PYTHONPATH=src:. python tools/experiments/seed_scene_modes.py \
  --config configs/seeded_scene_modes_a1_20261010.json \
  --mode single_cyclist_boundary_pass --master-seed 20261010 --episode-count 1 \
  --output outputs/experiments/<new_root>/recipes/single_s20261010
PYTHONPATH=src:. python tools/experiments/plan_road_pass.py \
  --config outputs/experiments/<new_root>/recipes/single_s20261010/single_cyclist_boundary_pass_s20261010_e00/config.json \
  --output outputs/experiments/<new_root>/preflight/single_s20261010/plan.json --geometry-preflight
PYTHONPATH=src:. python tools/experiments/run_continuous_roi.py \
  --prepared-episode outputs/experiments/<new_root>/preflight/single_s20261010/single_cyclist_boundary_pass_s20261010_e00 \
  --output outputs/experiments/<new_root>/runs/single_s20261010 \
  --observation-policy record_unavailable_v1 --storage-mode roi_only \
  --consumer-mode persistent --gpu-uuid <当时空闲设备UUID> --budget-seconds <本批剩余额度>
```

预检命令的 `plan.json` 上级目录需预先存在；混合模式把 `--mode` 改为 `mixed_walk_cycle_crossing`，其余沿相同公共 planner、renderer、consumer 与 Dataset。默认生成入口不调用 Blender；当前示例不得写回本轮已有目录。完成后停在两模式种子场景交付点，不自动扩大队伍/episode、推理或训练。
