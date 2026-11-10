# 行走 source-root 组合接入与独立来源核查（A2，2026-10-10）

> 后续进展：本报告记录首次接口交付时的资源状态。用户提供既有 BEDLAM2 账号后，A2 已定向取得 KIT/11 行走参数，并制作 M03 组合；见 [KIT11_M03_WALK_ASSET_A2_20261010.md](KIT11_M03_WALK_ASSET_A2_20261010.md)。下文“未取得参数”仅描述本报告原有批次，不代表当前状态。

本批将行走的源 root 策略接入现有 `compose_asset_a2.py`，用同一入口回放 M04＋KIT/425 的 94 帧身体、简单衣物、头发和 R138。新独立行走来源**没有取得参数文件**，因此没有制作 M03 新行人或预览；三条候选仅是官方映射索引线索，不能计为已完成 T5 来源隔离。全部产物仍为 `integration_only`。

## 本批冻结与实现

[冻结输入和保护清单](../../outputs/experiments/walking_composition_a2_20261010/frozen_inputs.json)在改实现前保存；[旧 M04 独立 before 数组](../../outputs/experiments/walking_composition_a2_20261010/before_m04/)包含原参数、body、cloth、head、R138、faces 和原 manifest。原源为 `us_2495_L_2200.npz`，发布 30 Hz，选 source `[9,103)`；原 100 Hz 到发布版本的精确曝光对应仍未证实。新时间表保留 `source_frame=9..102`、`time_base=[1,30]`，`episode_time_tick` 从本组合首帧起算，未宣称补齐原采集 PTS。[原源只读回核](../../outputs/experiments/walking_composition_a2_20261010/m04_raw_source_readback.json)表明 root 相对 frame 9 逐值一致，源 pose 与旧参数的最大浮点差为 2.39×10⁻⁷ rad。

组合配置现在显式区分 `fixed_pelvis_scene_owns_progression` 与 `source_root_relative_to_start_plus_fixed_scene_matrix`。行走要求权威参数中同时有 `source_frame_ids` 和 `source_root_relative`，且 `transl[t]−transl[0]` 与该相对轨迹一致；参考帧必须匹配源首帧。缺字段、重复 root、策略不匹配和未知策略在计划阶段拒绝。新目标形状的 `transl[t] = source_root_relative[t] − target_shaped_rest_pelvis`；局部姿态、原米制轨迹和整数时间保持，目标 shape 全段固定。子窗口从非首帧开始时，资产保留相对共同参考帧的 root，A1 的固定场景矩阵在 episode 起点只减一次该值，后续仍保留行进。身体、衣物、R138 使用同一固定场景矩阵，groom 先用本体 head 矩阵再施加该场景矩阵；没有第二条恒速道路轨迹或逐帧贴地。

`configs/asset_composition_a2_20261010/` 增加 M04、KIT425 motion、M04 简单衣物/groom、显式行走 recipe 与 schema 枚举。CPU 缓存键包括 root 策略、motion 描述、窗口及依赖文件时间/大小；已占用但版本或依赖不同的键拒绝覆盖。旧骑行策略、配置和缓存保持原行为。计划模式不构造 SMPL-X 模型或渲染。

复现命令：

```bash
PYTHONPATH=src python tools/experiments/compose_asset_a2.py --name M04_KIT425_walk_replay_v2 --plan
PYTHONPATH=src python tools/experiments/compose_asset_a2.py --name M04_KIT425_walk_replay_v2 --build
PYTHONPATH=src python -m unittest tests.test_asset_composition_a2
```

第二行是首次构建的命令；产物已存在，重复执行会按 append-only 规则拒绝覆盖。若需再次制作，应另选输出目录，而非覆盖本次结果。

## M04 旧结果独立回放

[回放差异](../../outputs/experiments/walking_composition_a2_20261010/m04_replay_comparison.json)以改动前冻结的旧数组为基准：94 帧所有局部旋转、betas、源帧、`source_root_relative` 和 faces 逐值一致；`transl` 最大差 2.39×10⁻⁷ m，body/joints 为 4.77×10⁻⁷ m，cloth/R138 为 7.16×10⁻⁷ m，head 矩阵为 2.39×10⁻⁷。这些差异来自 `float32` 平移加法与重新求值顺序；没有放宽验证阈值、更换形状或时间。全 94 帧有限、拓扑固定、R138 回读最大差 0。它证明统一入口数值回放兼容，不代表重新验收完整 studio 或道路观感。

[衣物材质与绑定引用回核](../../outputs/experiments/walking_composition_a2_20261010/m04_material_binding_readback.json)表明两个材质 JSON 字节一致，groom 指向同一份 M04 reference；新衣物 rest 缓存是该 reference 的派生，而不是另一件服装。

作为骑行默认行为回归，统一入口在本批独立目录重放了已接受的 F02＋bikefit 120 帧；[与旧权威回放的比较](../../outputs/experiments/walking_composition_a2_20261010/cycling_regression.json)显示 body、cloth、head、R138、两套 faces、参数及时间表全部逐值/字节一致。没有重渲染该旧视频。

[A1 只读读取检查](../../outputs/experiments/walking_composition_a2_20261010/m04_a1_readback.json)实际调用 `dynamic_scene.geometry_paths`、`select_timeline` 和 `person_matrices`；非首帧 source12 的 episode 时间归零，但保留其相对共同 frame9 的 root。两种固定矩阵下 body/cloth/R138/head 皆可读；[单次 root 场景代数检查](../../outputs/experiments/walking_composition_a2_20261010/m04_scene_algebra.json)验证 episode 起点平移仅扣一次、后续源行进不丢失。[统一交付](../../outputs/experiments/walking_composition_a2_20261010/builds/M04_KIT425_walk_replay_v2/delivery_manifest.json)和 [`densekp_scene_asset_v2` 行走读取视图](../../outputs/experiments/walking_composition_a2_20261010/builds/M04_KIT425_walk_replay_v2/a1_walking_view.json)引用持久缓存；后者供 A1 现有 `geometry_paths`/行走参数路径读取。骑行专用 `adapt_proposal` 仍拒绝此 root 策略，不能伪标成骑行。未生成道路矩阵、相机图、ROI、Dataset 行或 `scene_ready` 状态；A1 后续需用实际行走配方消费本视图，按其固定落地策略验收。

## 新普通行走来源：索引候选，均未取得动作

[三条有界候选清单](../../outputs/experiments/walking_composition_a2_20261010/candidate_inventory.json)由本地 BEDLAM2→AMASS 映射 CSV 提取。优先 `KIT/314/walking_medium05_stageii.npz`；另两条是 `SOMA/soma_subject1/walk_001_stageii.npz` 与 `soma_subject2/walk_001_stageii.npz`。它们的源路径不同于 `KIT/425/walking_medium05_stageii.npz`，索引给出发布区间和原始 fps 线索，但**本机没有这三条的姿态参数**；文件名不证明确为稳定直线行走、演员独立、root 连续或实际可用帧率。本地另一小型 BEDLAM `motion_seq.npz` 实为 KIT/883 的 `wipe_arm_dabbing01`，故未充当行走候选。M03 固定 30 维 shape、simple 衣物和 groom 的本地引用存在，但没有拿 KIT425 换名冒充新来源。

作为仅检查接口的反例，M03＋已有 KIT425 的非首帧子窗口可以通过统一入口 `--plan` 的模型/绑定/时间检查；它仍是 KIT425 同源衍生，未执行 `--build`，也不能补 T5 独立行走来源。

[BEDLAM2 官方页面](https://bedlam2.is.tue.mpg.de/)明确要求注册、阅读并接受其[非商业研究许可](https://bedlam2.is.tue.mpg.de/license.html)后才进入下载区；[AMASS 官方页面](https://amass.is.tue.mpg.de/)也要求注册，并有其[独立许可](https://amass.is.tue.mpg.de/license.html)。本批没有现成的、可核实许可与体积的单条直接下载入口，因此没有绕过访问条件或下载大包；实际新增下载量 **0 B**。三条原演员/录制关系及与预训练数据重叠均待补证。取得一条合法、可读取的参数文件与来源元数据后，下一批才可冻结约 3–4 秒窗口、将原始姿态和 root 显式接入当前 schema、制作 M03 全段穿衣预览。若其模型关节/手部表示不同，须先证实转换语义，不能仅凭 165 维直接填入。

[机器来源关系图](../../outputs/experiments/walking_composition_a2_20261010/source_graph.json)分别记录已交付的 M04/KIT425 与未制作的 M03 候选，保持身份家族、原 motion 和衣物/头发父引用分离；共同的标准 SMPL-X 模型本身不等于共同演员。

## 资源、限制与交接

[执行记录](../../outputs/experiments/walking_composition_a2_20261010/execution_record.json)记载计算前 GPU UUID、他人进程、可用内存/磁盘。M04 制作 3.12 秒，接口修正后的缓存回放 0.84 秒，F02 骑行回归 3.81 秒；三项合计约 7.77 秒，使用 CPU，无 GPU 计算或新渲染。目录约 52 MiB，其中独立 before 约 16.2 MB；首次版本和一次缺少 `PYTHONPATH` 的无产物脚本失败均保留。定向测试 6 项通过。用户 TODO、A1 代码、外部项目和旧权威资产未改。

下一步由 A1 决定是否把新的行走读取视图接入真实场景；这与新来源是否取得是两件事。正式 T5 train/val 必须联同目标角色、遮挡者、原 motion 演员/录制及历史使用关系共同冻结。当前 M04/KIT425 仍是已有来源的回放，新 M03 行人未制作，不能称本批已补足独立行人。
