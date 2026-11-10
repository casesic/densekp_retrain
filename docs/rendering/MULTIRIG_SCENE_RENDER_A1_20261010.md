# 同一人物场景、两组双机逐帧 ROI 渲染

日期：2026-10-10。状态：**数据接口与本次两rig连续生产通过；新侧视机位观感待观看。** 全部产物 `integration_only/shared_source_diagnostic`，未运行模型、推理或训练。直接看[并列视频页](../../outputs/experiments/multirig_scene_render_a1_20261010/index.html)、[边缘rig视频](../../outputs/experiments/multirig_scene_render_a1_20261010/rigs/rig_edge/pack/dual_roi.mp4)、[侧视rig视频](../../outputs/experiments/multirig_scene_render_a1_20261010/rigs/rig_side/pack/dual_roi.mp4)。灰色排版区未渲染，不是真实背景。

## 输入与两种用途

只读复用[已通过的河岸人物配方](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront/recipe.json)：M01/APD6526，31个10Hz合成时刻，原动作时间、shape、衣发、固定高度、道路轨迹及所有逐帧 `scene_from_model` 不变。`rig_edge` 的两台相机矩阵逐项复制旧合格rig。`rig_side` 来自[上一批冻结的32份候选](../../outputs/experiments/seeded_side_pass_a1_20261010/frozen_camera_candidates.json)，按“有效共同侧视最长连续段、共同物理点达标时刻数、原attempt编号”的固定顺序核查；第一份原attempt 10就通过本批完整预检，因此没有新增相机候选。其机位是整体平移/旋转双机，组内相对矩阵最大差 `6.94e-16`；0.18m仍是合成基线，不是实测外参。[冻结执行清单](../../outputs/experiments/multirig_scene_render_a1_20261010/frozen_execution.json)与[实际rig清单](../../outputs/experiments/multirig_scene_render_a1_20261010/multirig.json)记录位置、朝向、相机seed和来源。

旧候选是在**侧视且两路同时边缘截断**目标下拒绝，旧拒绝文件未改。本批事前具名 `side_view_control`：道路有符号位置首末跨零；连续至少3个10Hz时刻两机均在原side `[80°,100°)` 且至少8个共同物理域内R138点；全段至少16/31时刻共同物理点≥8。完整身体/衣物/active groom、原生框完整square crop、插值支持、非周期ERP接缝拒绝、每路ROI≤700万像素等硬检查未放宽。实际侧视共同窗口为frame 3–6（4帧、首末0.3秒），共同物理点达标21/31；其双路边缘为0/31，严格首末物理端点过界未通过。frame 4的cam2有138点域内、cam4有25点域内和113点域外，因此“cam2完整、cam4截断”仍有25个共同物理机会，不能计入双路边缘配额。见[物理域及投影图](../../outputs/experiments/multirig_scene_render_a1_20261010/physical_projection_evidence.png)和[俯视轨迹图](../../outputs/experiments/multirig_scene_render_a1_20261010/topdown_rigs.svg)。渲染后补核的静态场景相机中心10方向、0.15m短射线净空检查四机均通过；它不证明几何节点实例的完整扫掠体无碰撞。

## 生产接口与恢复

新增[多rig入口](../../tools/experiments/run_multirig_roi_a1.py)和[Blender逐帧消费者](../../tools/render/render_multirig_roi.py)。一个 `scene_episode_id` 保存人物、世界点与31条时间表；两个 `rig_id` 分别拥有静态双机位姿、physical投影、ROI、标签、框、NPZ和序列旁表，camera_id仍为cam2/cam4。公共装配器每帧更新一次body/cloth/groom，随后按 `frame → rig_edge(cam2,cam4) → rig_side(cam2,cam4)` 调用共用ROI渲染函数；完成该帧四次相机渲染后，同一个常驻同步CPU消费者按rig消费成功才进入下一帧。每rig使用独立输出目录和任务键，不能把跨rig两台相机凑成双路点对。现有单rig入口及帧协议缺省保持。新模式已具备(frame,rig)完成项记录；对其它复杂局部失败拒绝直接套用旧“完整帧前缀”恢复。

首轮frame 0四次ROI已经渲染，`rig_edge`消费成功；`rig_side`因本批预检写出的 `expected_in_view.npy` 少了单人轴（`[31,2,138]`，应为`[31,1,2,138]`）拒绝。保存[首轮故障证据](../../outputs/experiments/multirig_scene_render_a1_20261010/failure_attempt_1/diagnosis.json)，修正本批预检写法与自身数组后，仅从原侧视缓冲重新消费frame 0，再在新Blender进程渲染frame 1–30；成功的frame 0未重渲染。**实际两次Blender加载、两个先后而非并行的CPU消费者进程；正常无故障路径是一次加载、一个消费者。** 原时间与人物没有重置，重试额外进程如实计成本。[124次相机事件日志](../../outputs/experiments/multirig_scene_render_a1_20261010/render_calls.jsonl)和[逐(frame,rig)状态](../../outputs/experiments/multirig_scene_render_a1_20261010/unit_states.json)可核对。所有成功ROI临时EXR已按所属rig消费后清理；新pack不依赖EXR，根目录仅保留约0.17MiB装配manifest。

## 实际验收

两组各31帧、双机共124张ROI；`rig_edge` 62/62观测有合法原生框，双路物理边缘21/31、31/31时刻共同物理点≥8；`rig_side` 56/62合法框、21/31时刻共同物理点≥8、0/31双边缘。侧视cam2的frame 25–30为 `outside_physical_domain_no_instance_box`，保留完整观测槽位且不生成训练行；两rig合计118行。视频PTS为0.0–3.0秒，31帧，末帧展示保持0.1秒。共同crop和训练输入可用性在各自索引中另计，绝不把无框记作零误差。

全124个实际相机输出检查了ROI尺寸/偏移、局部mask、有限GT、原生框支持及rig身份；同一时刻两rig的世界R138、人物矩阵和来源时间一致。`rig_edge`与旧河岸权威31帧×2相机比较，完整ERP上的138点xy最大绝对差 **0像素**、physical `in_view`逐点完全一致。两rig的NPZ与旁表均通过路径/行映射校验。预先固定首/中/末各双路槽位，真实Dataset回读边缘6行、侧视5行；侧视末帧cam2无框，没有换其它行凑数。每个已读取条目的局部图像affine、138点affine、`_trans`、全ERP `img_size=[2880,5760]`与可靠 `in_view & inside_full_crop` 监督检查通过，遮挡可靠GT仍参与。详见[验收机读结果](../../outputs/experiments/multirig_scene_render_a1_20261010/acceptance.json)及每rig `pack/readback.json`。16个定向CPU单测通过，未做全库回归。

## 实际成本与后续界限

旧候选的本批完整CPU预检15.54秒。首轮失败Blender wall 35.55秒；修正续跑Blender wall 379.11秒，随后两rig打包/回读/视频合计25.42秒；两次实际渲染/消费/打包/视频进程wall相加 **440.09秒（7.33分钟）**；独立CPU预检15.54秒、渲染后静态相机净空补核及最终索引校验另列，不把开发或等待时间混入。续跑一次装配到首个帧回调约19.02秒；后续30帧总循环平均11.95秒/时刻、中位11.64、范围10.45–15.32秒，包含同步消费。相机渲染调用合计302.73秒，其中边缘rig 231.72秒（每时刻7.47秒）、侧视rig 71.01秒（每时刻2.29秒）。首个时刻四路调用合计12.25秒，随后消费者遇到形状错误，因此没有可作为正常完成首帧的总耗时；续跑首个完整时刻frame 1为15.32秒。Blender记录的人物顶点/矩阵更新均值0.060秒/时刻，首个边缘cam2调用仍吸收动态场景GPU同步成本，因此71.01秒只是**本场景追加第二组rig的实测相机调用成本**，不是通用线性速度。CPU消费函数体合计41.83秒；两次消费者启动分别2.83与2.23秒。Blender峰值RSS约3.34GiB、GPU采样峰值5483MiB、任务磁盘采样峰值158.50MiB，其中两次运行临时峰值112.90MiB；当前长期目录104.66MiB。采样无法分离GPU kernel与文件写出时间。

本批只验证单人、两组**静态**rig和本河岸场景；不支持移动rig、任意组数或把两rig当独立来源。未接A2新行人衣物/窗口，未预留随机框增强ROI，尚无正式来源分组、原权重基线或受控训练。原8段/200步设计仍`pending_review`，本批不自动进入下一阶段。

复用命令：先用 `tools/experiments/seed_fixed_person_side_pass_a1.py preflight --config configs/multirig_scene_render_a1_20261010.json --frozen <冻结候选表> --output <新side-prepared目录>` 完整核查侧视候选；再在**全新输出目录**运行 `python tools/experiments/run_multirig_roi_a1.py --edge-prepared <edge-prepared目录> --side-prepared <side-prepared目录> --output <新输出目录> --gpu-uuid <空闲GPU-UUID> --budget-seconds 1800`。已有目录拒绝重跑；故障续跑仅对本次明确记录的首帧形状故障提供专用选项，其他部分状态需要先另行核查。
