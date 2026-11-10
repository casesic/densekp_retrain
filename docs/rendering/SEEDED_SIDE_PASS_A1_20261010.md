# 固定人物轨迹、种子化双机位姿：河岸侧向预检

日期：2026-10-10。本批**未生成新的侧向视频**。在投影前冻结的32份河岸道路侧方相机候选全部未满足 `side_pass_edge_coverage`，按事前上限停止；没有启动Blender、消费ROI、打包或读取真实Dataset。完整机器证据见[验收状态](../../outputs/experiments/seeded_side_pass_a1_20261010/acceptance.json)、[候选摘要](../../outputs/experiments/seeded_side_pass_a1_20261010/candidate_summary.json)、[逐帧投影](../../outputs/experiments/seeded_side_pass_a1_20261010/prepared/candidate_checks.json)和[查看页](../../outputs/experiments/seeded_side_pass_a1_20261010/index.html)。该结果不改变任何旧视频、旧通过状态或用户对河岸双人队伍视频的观感接受。

固定输入是已交付的[河岸M01/APD单人prepared配方](../../outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront/recipe.json)：31个合成10 Hz时刻，原asset帧0..75的已选时间映射、shape、M01简单衣物和groom、逐帧`scene_from_model`及世界R138保持只读。新接口区分 `scene_episode_id`（此固定人物场景）与 `rig_id`（一组双机相机子seed）；本批只计划一组实际生产，未有合格`rig_id`进入生产。每个候选整体改变rig平面原点和yaw，`rig_poses`同时旋转两机中心与物理镜头矩阵；两机ERP和物理镜头的相对矩阵均与原合成0.18m基线一致。未改physical mask、相机安装高度、pitch/roll或cam4轴定义。相机seed `20261024`仅决定道路侧方32格的顺序及格内≤0.035m/≤0.2°扰动；yaw由指向固定轨迹中点的方位和原交界基准求出。`20261025`只作计划级变化检查，不投影。

[配置](../../configs/seeded_side_pass_a1_20261010.json)预先固定道路正X侧相机距离2.75/3.15/3.55/3.95m、沿路位置2.25/2.75m与交界yaw偏移−3/−1/+1/+3°的32格组合。[冻结表](../../outputs/experiments/seeded_side_pass_a1_20261010/frozen_camera_candidates.json)在投影前写出；同seed逐字段重放一致，不同camera seed改变相机表且`scene_episode_id`、人物时间和世界数组引用不变。[轨迹与相机俯视图](../../outputs/experiments/seeded_side_pass_a1_20261010/trajectory_topdown.svg)使用冻结候选8说明相机位于道路侧方、人物从其道路横截面前方走到后方；它不是鱼眼物理交界图。

目标分别检查：同一时刻cam2/cam4各≥8个域内、≥5个域外可靠R138且共同域内≥8，整段至少16/31时刻；两机侧视角均在`[80°,100°)`且与双边缘重合连续≥6个10 Hz时刻；人物沿路有符号位置首末跨零。旧`boundary_transition`首末双机同步20→5/5→20域外点仅诊断。32份候选都完成31时刻physical域和视角投影，道路横截面条件通过，**双边缘均为0/31**；每个时刻至少一台相机的域外R138为0。双机同属侧视最多8/31时刻，最长连续仅4帧；与双边缘重合为0。[最接近候选8的曲线](../../outputs/experiments/seeded_side_pass_a1_20261010/closest_candidate_curves.svg)显示cam2与cam4在不同阶段交替完整/截断，而非两路同时部分截断。其相机原点约`(2.750,2.777)m`，yaw约`−180.995°`，道路有符号位置从`−0.972m`变为`+0.516m`；严格双机端点过界也未通过。两组相机的物理有效域关系和人物角尺寸共同造成冲突；目前证据只约束**本次冻结的道路侧方范围**，不能证明所有河岸机位几何上都不可能。

轻量physical投影未达标，按预定顺序没有进入body/cloth/**active groom**的完整square裁剪、ERP接缝、≤700万像素ROI、场景实体相机碰撞或Blender真射线检查；这些门槛**未验证**，不能把本批称为可渲染布局。没有实际mask紧框，未把投影代理写成`visible_instance_tight`。因此本批新侧向数据为0帧、0框、0训练行、0次Dataset回读，用户要求的视频也不存在；[查看页](../../outputs/experiments/seeded_side_pass_a1_20261010/index.html)只读链接**以前的**河岸单人、河岸双人、正面和混合视频，均标明不是本次侧向产物。

接口改动限于[固定人物相机入口](../../tools/experiments/seed_fixed_person_side_pass_a1.py)、[视角目标](../../src/densekp_retrain/data/viewpoint.py)及可选的谱系字段。新配方若通过预检会用唯一的`episode_id/sequence_id`引用共享`scene_episode_id`和该组`rig_id`；标注NPZ、打包行、框、观察索引及episode/camera元数据携带并校验这两个ID。旧配方没有新字段时仍按原接口读取。现有生产循环仍是一段运行消费一组rig：后续多rig同帧生产还需把每帧的rig集合、逐rig ROI和输出命名传入renderer/常驻消费者，按rig扩展观察槽位、打包和失败/断点清单，并验证“更新一次人物几何→依次渲染多组rig”。本批未实现多rig同时生产，也不因换rig或seed制造来源独立划分；同场景所有视角仍属原APD/M01来源组。

定向CPU测试23项通过，包含相机随机流分离、同seed重放、相对矩阵、side窗口/道路通过不能代替双边缘、历史视角/序列兼容、新索引身份拒绝与`synthetic_test_only`带rig列NPZ的实际Dataset读取；这不是本批侧向真实图像回读。未做全库测试或模型计算。预检实际约6.05秒，生成诊断图约1.34秒，最终本任务目录约9.1MiB；GPU渲染/消费预算使用0秒。资源检查在CPU投影之后完成：GPU0/1 UUID分别为`GPU-d481acc7-106f-29e5-1298-8c0d398261b2`/`GPU-6f046d56-acbe-d49b-774a-9d463e768cac`，当时已有其他CPU/GPU作业，内存和磁盘余量见同目录资源记录。**CPU投影前未保存即时资源快照**，这是本批操作记录缺口；没有GPU作业因此未争用设备。

[场景覆盖清单](../../outputs/experiments/seeded_side_pass_a1_20261010/coverage_inventory.json)把技术完成、用户观感、双边缘、严格过界和邻人遮挡恢复分开。最新河岸两人视频按本轮用户反馈标为人工接受的边缘队伍例子，原主目标真实15/15恢复失败与严格端点过界失败保持；城市三人仍仅CPU资格，4–5人和三岔路口未完成，森林不列为连续生产通过。正式数据准备还缺A2新行人材质/窗口道路接入、框增强ROI预留、全体目标及遮挡者的来源分组与配额、原权重基线和受控训练。旧8段/200步仍待评审。

| 模式／环境 | 已完成范围 | 技术覆盖与观感 |
|---|---|---|
| 单人过界／城市、河岸 | 各31时刻双路 | 双边缘21/31；原端点过界通过；共享APD来源 |
| 单人正面边缘／城市 | 31时刻双路 | 正面31/31、双边缘22/31；端点过界未通过；用户已接受该边缘例子 |
| 单人侧向横穿／河岸 | 本批32份CPU相机候选 | 双边缘全为0/31；未渲染、无人眼成品 |
| 骑行＋行人交会／城市 | M01/M04及F02/M04成品 | 有自然无框与恢复例子；两路恢复按各自证据判定，不能代替侧视样例 |
| 两人骑行队伍／河岸 | 31时刻双路，124合法行 | 双边缘21/31、用户接受视频；主目标15/15恢复与严格端点过界未通过 |
| 三人队伍／城市；4–5人及三岔路口 | 三人仅CPU，后两者无成品 | 尚无完整场景技术/观感验收 |

这些行重用旧权威报告，详细seed、组件、原时间起点、动作来源和视频链接在机器清单的`evidence`字段；换相机、环境或seed都不产生新的独立动作来源。

在仓库根目录重放计划与既有冻结表的CPU预检时，须改用**新的空输出路径**；不要覆盖本批记录：

```bash
PYTHONPATH=src:. python tools/experiments/seed_fixed_person_side_pass_a1.py plan --output NEW_FROZEN.json
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/seed_fixed_person_side_pass_a1.py preflight --frozen NEW_FROZEN.json --output NEW_PREPARED_DIR
```

只有`geometry_preflight.json`为`passed`的prepared目录才允许进入既有`run_continuous_roi.py --storage-mode roi_only --consumer-mode persistent --observation-policy record_unavailable_v1`。本批目录状态为`rejected`，不得对它运行渲染。下一批如要得到真实侧向样例，需先评审相机安全放置区域与原rig有效域的几何冲突，再冻结新的有界候选；不得事后把这32次拒绝当成功。
