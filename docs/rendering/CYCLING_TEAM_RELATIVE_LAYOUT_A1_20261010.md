# 河岸双人骑行队伍：相对布局与真实数据

日期：2026-10-10。交付目录：[cycling_team_relative_layout_a1_20261010](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/)，[直接观看双路视频](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/riverfront_2/run/pack_correction1/dual_roi.mp4)，[查看页](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/index.html)。本批仅 `integration_only/shared_source_diagnostic`；没有模型推理、训练或正式来源划分。

## 结果与边界

河岸两人段完成31个合成10 Hz 时刻、首末3.0秒、62次双路ROI相机调用。31个时刻两人均继续推进各自原动作；124个观察槽位均有合法原生实例框，打包124行。主目标person1的物理双边缘条件满足21/31时刻，最长连续窗口frame0–19、1.9秒；两路均有部分人体被物理有效域截断。31/31时刻的目标双路共同crop支持至少8点。视频中person2在person1前方沿同向道路骑行，双机都能看到两人；新组合的人眼接受状态**未从旧角色自动继承**，仍应看道路视频。

遮挡覆盖与生产正确性分开：CPU投影三阶段机会在两路均未通过；实际IndexOB/range邻人证据同样显示**person1两路邻人遮挡比例均为0**，原15/15个百分点、每阶段3个相邻时刻的加重—恢复目标未通过。person2被person1明显挡住：共同物理GT口径下，cam2邻人遮挡比例范围0.600–0.847，cam4为0.417–0.797；它没有满足同一三阶段恢复规则。person2在每个时刻都有合法框，cam2每帧可见R138为7–18点，cam4为29–43点，因此它确实进入两路图像而非只存在于配方。没有无框或重现事件，本段不是完全遮挡—重现样例。保留整段作为不同身份、动作和衣型的边缘队伍技术数据，不将遮挡诊断写作成功。

首末域外R138：frame0 cam2/cam4分别27/18，frame30分别1/2。它展示从边缘向域内运动，但旧双机同时首帧各至少20域外的严格端点 `boundary_transition` 未通过；不能把双边缘覆盖等同完整端点过界。目标实际共同物理点frame0/15/30分别93/110/135。边缘曲线、逐人可观测性和真实邻人遮挡见[曲线](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/riverfront_2/run/pack_correction1/per_person_curves.svg)与[俯视轨迹](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/trajectory_topdown.svg)。灰色视频排版区是未渲染区域，不是背景；黄线是标定physical域边界，蓝框是可见实例原生框，不是YOLO框。

## 旧拒绝与冻结的新布局

上批河岸12个候选中，5个ERP几何跨缝、6个保守完整square支持越出全图、1个主目标双边缘不足；旧拒绝保持原样。[代表旧失败证据](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/old_failure_root_causes.json)重算旧attempt0/3/6：attempt0的person2在frame0 cam2投影横跨x≈1.3–5759.9，属于ERP接缝，而非镜头物理域截断；attempt3的person2投影尚未跨缝，但保守square到x=6309、y=2893，越过5760×2880右、下边。旧person2的道路起点靠近rig。保守支持包络覆盖可能出现的可见实例框，**不证明当时真实mask紧框必然越界**。

本批引用已通过河岸单人的rig/person1**相对**轨迹作为锚点，再在道路内对person2用纵向间距、横向错位、速度差构造32格结构化候选，seed 20261023只决定稳定顺序和格内小扰动。[配置](../../configs/cycling_team_relative_layout_a1_20261010.json)、[冻结32个候选](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/frozen_candidates.json)均在本次投影前写出；实际仅投影并选中attempt0，另一个seed仅做不投影的计划级变化检查。选中rig原点`(0,-0.067669)m`、yaw `90.705788°`；person1从`(-0.004427,1.804993)m`以`0.495989m/s`前进；person2从`(0.326203,3.175913)m`以`0.441511m/s`前进，起点相对纵向约`1.371m`、横向约`0.331m`。两人独立固定高度1.2493m/1.1571m，原shape与姿态不变。31帧道路范围、逐人包络间距、朝向、主目标边缘及实际active groom保守ROI检查通过；最大预检联合ROI cam2约165万、cam4约163万像素，均低于700万。间距检查为采样表面包络近似，不是连续三角形零相交证明。

新策略为 `projected_occlusion_policy=diagnostic_only`，旧缺省仍为 `require_event`。前者仅将CPU投影的三阶段事件从**几何拒绝**改成诊断；它仍完成全31时刻的真实active groom、道路、逐人间距、双边缘、非周期ERP接缝、完整方框与联合ROI硬检查。失败记录现在保留已完成逐帧/逐人投影及首个失败位置，未渲染时 `actual_occlusion` 仍为未评估。[城市旧attempt5仅CPU回核](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/city_attempt5_cpu_only_correction1/prepared/geometry_preflight.json)在新诊断策略下通过23/31双边缘和精确groom/ROI；其CPU三阶段事件仍为false，未渲染、未改写旧严格拒绝。

## 资产、时间与实际消费者

仅只读复用旧批两份经配方/依赖/持久文件/数值检查的组合交付，没有build、重绑或studio制作：[缓存检查](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/frozen_asset_reuse.json)。person1为M01/APD6526/M02_simple/M01_groom，原时间起点0；person2为M02/participant_1 comfortable/M01_simple/F02_groom，起点4。两个身体家族、原动作来源组和衣型不同，衣发父引用不自动创造新身份。person3 F02/p2不参与本段。组件版本、父交付与原来源记录进入[episode旁表](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/riverfront_2/run/pack_correction1/episode.json)；每人物原source帧、整数源时间和量化误差进入观察索引，双相机消费同一完整场景状态。scene-owned行进仅施加一次，body/cloth/R138共用每人scene矩阵，groom复合各自head；不循环、不补帧、不把两段动作强写成统一30 Hz。

实际使用公共单Blender进程、常驻同步CPU消费者、`roi_only`、逻辑ERP5760×2880、64samples CUDA。无低清背景渲染或全ERP RGB读写；完整河岸世界仍参与阴影和遮挡。每人先规划完整square支持，再取联合ROI；physical有效域与render coverage分离。可靠GT按`in_view & inside_full_crop`监督，包括遮挡点，Dataset `_trans`/`img_size`仍用完整ERP坐标，第三值不作置信度。序列索引[校验通过](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/riverfront_2/run/pack_correction1/validation.json)：31帧、124 observed槽位、124个合法框、124打包行，未指向已清理的临时EXR。预定首/中/末×双机×两人[12次真实Dataset回读](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/riverfront_2/run/pack_correction1/readback.json)通过局部RGB采样、138点仿射、可靠遮挡监督、offset、`_trans`和全ERP `img_size`；没有为无框槽位造训练行。本段刚好没有无框，但保留`record_unavailable_v1`支持。

首次在沙箱内启动时，runner的`nvidia-smi`子进程返回设备访问错误，未进入GPU计算；沿同一冻结输入在获准的运行环境启动。首次收尾又遇到旧打包器将单人`component_provenance`字典硬套在逐人列表上的TypeError。31帧的渲染与同步消费均已成功，未重渲染；原`finish.log`与不完整`pack/`保留。修复逐人谱系校验后在独立`pack_correction1/`成功打包、回读和编码，[修正记录](../../outputs/experiments/cycling_team_relative_layout_a1_20261010/finish_recovery.json)明确两次收尾，不能宣称一遍完整通过。`pack_correction1`是本批权威pack；没有更改原图、标签、监督或旧产物。

## 成本与使用限制

CPU新河岸预检45.03秒、城市旧候选回核64.76秒、代表旧拒绝回核1.46秒；首次城市回核因旧candidate记录含额外键引发TypeError，其精确CPU秒数未记录，失败现场保留，起点到修正再试间隔约80秒（包含代码修正），未把它算作零。Blender进程493.58秒，其中31个时刻帧循环首帧17.87秒、后30帧均值13.75秒/中位13.64秒/范围12.64–15.61秒。进程墙钟减去帧循环之和约63.16秒，包含加载、装配及未细分同步/退出，**不是纯装配耗时**。cam2首调用13.59秒、后续均值11.32秒；cam4首调用2.38秒、后续均值1.36秒；同步CPU逐帧消费后续均值约0.78秒。首次失败收尾3.42秒，修正打包/回读/视频12.54秒。全部已测新增计算加上失败回核仍远低于45分钟上限；没有重启Blender或增加相机调用。

GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 在启动时空闲，GPU1有他人进程。GPU0设备采样峰值6971MiB包含任务开始后短暂出现的其他Python进程，不能解释为本Blender独占显存；本任务没有终止他人进程。峰值CPU RSS约826332KiB，采样总磁盘峰值约216MB、临时源峰值约173MB，成功收尾后本run长期文件约62.8MB、临时源0。视频31帧PTS为0..3.0秒，末帧展示保持0.1秒另列。视觉判断只针对这一个河岸、两人、背向/近背向队伍；该二人新组合观感仍需用户按视频判断。城市三人仍只有CPU资格；4–5人、遮挡恢复合格队伍、三岔路口、随机框增强、正式来源隔离和训练均未完成。不同seed或同episode的邻帧不能划为独立训练/验证。

## 重放命令

在仓库根目录使用既有Python环境；旧交付和旧strict策略不会被改写：

```bash
PYTHONPATH=src:. python tools/experiments/prepare_cycling_team_relative_layout_a1.py freeze
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/prepare_cycling_team_relative_layout_a1.py river_preflight
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/prepare_cycling_team_relative_layout_a1.py city_recheck
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/run_continuous_roi.py --output NEW_EMPTY_RUN --storage-mode roi_only --consumer-mode persistent --frame-count 31 --budget-seconds 2300 --prepared-episode outputs/experiments/cycling_team_relative_layout_a1_20261010/riverfront_2/prepared --gpu-uuid GPU-d481acc7-106f-29e5-1298-8c0d398261b2 --observation-policy record_unavailable_v1
```

这些命令用于**新空目录/新的冻结任务**；本批现有输出不可覆盖。定向CPU测试22项通过：`tests.test_cycling_team_relative_layout_a1`、`tests.test_cycling_team_a1`、`tests.test_seed_asset_composition_a1`、`tests.test_observation_continuation`。测试包含旧严格拒绝、新策略仍拒绝ERP跨缝、seed重放/不同seed、缓存核验及旧单人/混合兼容；未做全库discover或模型参数更新。
