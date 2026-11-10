# F02＋M04 混合道路场景：布局修复与完整 episode

日期：2026-10-10。状态：**31 时刻双路 ROI-only 数据链通过；双路物理边缘与邻人遮挡恢复诊断通过**。本数据为 `integration_only/shared_source_diagnostic`，没有模型推理、训练或独立验证资格；bikefit 许可状态仍为 unknown。

## 结论与产物

使用原 F02 前倾骑行和 M04/KIT425 行走资产，在一个 Blender 进程中完成合成 10 Hz 的 frame 0..30（首末 3.0 秒），两路共 62 次高质量 ROI 相机渲染。一个常驻同步 CPU 消费者生成 62 张局部 RGB、局部 mask/标签；每时刻两人、两相机的 124 个观测槽位均有记录。实际有 102 个合法原生框/NPZ 行，20 个 `person_absent` 和 2 个 `empty` 槽位保留原因，未补框或伪造图像。打包、序列旁表、31 帧原时钟视频和实际 Dataset 回读均完成，参见 [本轮输出](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/)。

F02 在两个相机**同一时刻**符合各自 ≥8 个物理域内、≥5 个域外 R138 点且共同域内 ≥8 点的条件：frame 0..19，20/31（64.5%），连续跨度 1.9 秒；超过预先冻结的 16/31 与 0.5 秒门槛。实际邻人 ID＋range 证据显示 cam2 的遮挡加重/恢复分别为 26.17/26.17 个百分点（前 0..2，中 16..18，后 21..23），cam4 为 36.52/19.47 个百分点（前 0..2，中 17..19，后 21..23）；两路各阶段均为三个相邻时刻，通过原 15/15 诊断规则。F02 的 cam2 原生框在 frame 18 消失、19 重现；cam4 在 19 消失、20 重现。框缺失是自然观测状态，不是渲染失败。两路共同 crop 监督点 ≥8 的时刻为 29/31；无共同输入的 frame 18/19 不计作误差零。曲线、逐帧分母和原因见 [验收数据](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/multiperson_acceptance.json)、[曲线](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/target_curves.svg)。

## 原拒绝与本轮布局

上轮 12 个候选拒绝的是 `plan_native_crop_region` 的保守支持包络：人体/衣物/groom 几何投影范围内可能实例框的完整 square crop，并加栅格容差和 32 个最终 ERP 像素余量。上轮没有实际实例栅格，不能称作“真实原生框越界”。M04 旧放置在完整轨迹后段过近相机，使保守包络越过完整 ERP 画布或联合 ROI 负载。物理有效域边界、ROI 边界、全 ERP 画布边界分别检查，没有裁短越界 ROI 冒充支持。

本轮将 F02 已找到的目标位置/速度/高度及 rig 固定：平面起点 x=-0.061719 m、y=1.015858 m，场景速度 0.478101 m/s、固定高度 1.10 m。只调整 M04 固定道路放置，选中 x=-1.63 m、y=1.075 m、固定 yaw -90°；M04 原 source-root 轨迹和速度未再叠加场景速度。F02 仍为 pelvis-centered 身体、场景拥有行进。两人各自 body/cloth/R138 共用对应世界矩阵；groom 用该矩阵乘原 head 矩阵。相机、物理 mask、5760×2880 ERP、64 samples CUDA、材质/灯光/色彩均未改。

[冻结候选](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/freeze/candidate_freeze.json)共 12 个明确参数，seed 20261010，没有第 13 个抽样。首次对这同一列表使用纯 XY AABB 分离时 12 个全拒绝；进一步检查发现候选 2 的包围盒只交叠约 9 mm，而整个 31 帧中两人物 body＋cloth 最近顶点至少 0.1078 m、root 至少 1.007 m。于是为**显式新配置**增加 `aabb_then_vertex_3d_v1`：仅当包围盒交叠时以三维顶点最近距离区分包围盒假阳性，原 root 0.9 m 和近邻 0.1 m 门槛照旧；历史默认路径未改变。重新用**同一 12 个冻结候选**预检，前两个因实际近邻/root 距离拒绝，第三个即 attempt 2 通过，未再试余下九个。这是身体/衣物顶点距离近似，不是全三角形连续碰撞认证。

CPU 预检对 31 时刻、两相机规划的最大联合 ROI 为 4,274,556 像素，低于 7,000,000 上限。首、旧失败附近的 9/10、末等时刻分别保存了几何投影、**无遮挡几何代理框**及其 square crop、保守支持包络、两人并集、完整画布和物理域边界；代理框没有冒称 `visible_instance_tight`。见 [支持诊断](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/preflight_narrow/f02_m04_repair_00/support_diagnostics.json) 与 [双路投影图](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/preflight_narrow/f02_m04_repair_00/projection_review/)。最终配方和 31 条逐人物源时间/矩阵在 [冻结执行清单](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/frozen_execution.json) 和 [episode recipe](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/preflight_narrow/f02_m04_repair_00/recipe.json)。F02 保留原 1/30000 PTS 和最近观察映射误差；M04 保留发布 30 Hz 来源和原 100 Hz 不确定性。10 Hz 只是合成 episode 时钟，不是原生视频曝光同步。

## 数据与验收

实际 62 个 ROI 整数边界与预检**逐一完全相同**，最大 4,274,556、均值 3,228,215 像素；见 [逐调用比对](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/roi_plan_vs_actual.json)。同一相机同一时刻多人共用一张局部 RGB；person mask/原生框来自当前 ROI 实例缓冲，外部区域仍是 unknown，physical valid mask 不由黑色 RGB 推断。全部 31 帧的身份、原源帧/时刻、矩阵、有限 GT、mask 尺寸、框坐标、支持区和状态由收尾校验通过；[序列索引](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/sequence_index.json)保留 124 槽位与 102 行映射。`in_view & inside_full_crop` 的可靠 GT 继续监督，包括遮挡点。框策略仍是原生可见实例紧框、deployment 1.0；未做框扰动或检测器预测。

首/中/末实际 Dataset 读取 10 行，再补查 frame 18 cam2 的 M04 有效框和 frame 20 cam4 的 F02 重现框，总计 **12 行**。逐行独立从局部 RGB 与 offset 计算裁剪，与 Dataset 的 256-square `img_disp` 精确相同；R138 完整 ERP→crop 仿射、监督布尔集合、`_trans` 和 `img_size=[2880,5760]` 一致。无框槽位只查旁表，不读伪造训练行。见 [主回读](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/readback.json) 与 [补查](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/supplemental_readback.json)。与改动相关的 20 项 CPU unittest 通过；没有全库 discover、模型前向或参数更新。

[双路视频](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/dual_roi.mp4)为 31 个 1/10 秒 PTS，最后一帧另展示 0.1 秒；标识相机、人物、实际原生框和 physical 边界。灰色排版区及 ROI 外不是真实背景。[中间时刻示意](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/pack/event_frame_015.png)可快速检查交会和两路截断。没有低清背景渲染或完整 ERP RGB 生产读写，ROI 内仍由完整场景参与遮挡、阴影与光照。62 个成功源缓冲在必要消费后按本任务所有权清理，长期数据不依赖 EXR。

## 成本、失败记录与使用范围

GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2` 在启动时空闲；GPU1 有他人进程，没有争用或终止。一次 Blender/场景装配回调 38.64 秒，一个常驻 CPU 消费者；Blender 进程 wall 976.30 秒。首帧 Blender frame block 33.89 秒、CPU 消费 1.91 秒；后 30 帧 block 平均 28.14 秒（中位 27.97，范围 24.34–32.95），CPU 消费平均 1.83 秒。两相机 camera-call combined 总计 cam2 628.49 秒、cam4 205.88 秒，包含同步/写出，不能拆成 GPU 纯 kernel；规划时间包含在 frame block 内，不重复相加。成功收尾的 pack 6.76 秒、初次 10 行回读 4.59 秒、视频 10.40 秒。采样峰值 GPU 6199 MiB、Blender RSS 约 11,814 MiB。当前运行目录约 228.5 MB、整个本轮目录约 233.4 MB，任务临时源目录已无残留；逐帧和成本口径见 [成本记录](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/final_costs.json)。这与旧单人的 ROI 负载不同，不作严格吞吐加速结论。

首次 CPU 收尾发现新鲜 `record_unavailable_v1` episode 被错误要求存在 `resume_manifest.json`，导致打包校验失败；原失败 pack 保留为 `pack_attempt_1`。修正为仅真实续接要求该文件，未重渲染任何帧，只重跑 CPU 收尾。原 runner 记录 985.91 秒（含失败收尾 9.54 秒），CPU-only 重试另约 20.12 秒，累计必要渲染/消费/收尾约 **1006.03 秒（16.77 分钟）**，低于 30 分钟预算。这里的重试未记录内部精确分项，以命令 wall 约值列示；详见 [重试说明](../../outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/run/finish_retry_manifest.json)。成功产物和首次失败证据均保留，没有刷成“一次无错”。

运行入口：先执行 `PYTHONPATH=src:. python tools/experiments/freeze_f02_mixed_repair.py --output <new-root>/freeze`，再执行 `PYTHONPATH=src:. python tools/experiments/plan_road_pass.py --config <new-root>/freeze/config.json --output <new-root>/preflight --geometry-preflight` 完成最多 12 候选 CPU 评审，生成 `frozen_execution.json` 后将 `run_continuous_roi.py --prepared-episode <选中episode目录> --output <new-root>/run` 指向所选 prepared episode，显式使用 `--observation-policy record_unavailable_v1 --storage-mode roi_only --consumer-mode persistent --gpu-uuid <available-uuid> --budget-seconds 1800`。本次精确命令和 GPU 选择见 `run/run_config.json`、`run/blender_process.json`；不得将这次 frozen F02＋M04 配方当作随机批量生成默认。

本轮解决了一个不同外观与动作来源组合的完整双路数据链和遮挡事件，但仅一个共享来源诊断 episode。不能据此认定正式来源隔离、人物多样性、真实双机曝光同步、真实检测框误差或模型效果。后续批量/训练需先评审来源关联、可发布许可、框增强预留 ROI 支持和独立评价方案；本轮停在场景交付点。
