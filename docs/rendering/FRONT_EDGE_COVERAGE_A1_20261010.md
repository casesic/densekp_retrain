# A1：正面单人双路边缘覆盖

日期：2026-10-10。**`edge_coverage` 已交付；`boundary_transition` 未通过且不作为本片段目标。**[直接观看31帧双路视频](../../outputs/experiments/front_edge_coverage_a1_20261010/run/pack/dual_roi_front_edge.mp4)，或打开[视频与证据页面](../../outputs/experiments/front_edge_coverage_a1_20261010/index.html)。本片段是 `integration_only/shared_source_diagnostic`，不进入 datasets，不是模型效果或正式来源独立验证。上批[被端点策略拒绝的记录](SEEDED_VIEWPOINTS_A1_20261010.md)保持原样。

## 策略与精确候选重放

新增具名 `coverage_goal=edge_coverage|boundary_transition`。旧单人缺字段时仍要求首末过界；旧混合缺字段时仍没有首末过界硬门槛，非法值拒绝。混合模式若显式要求尚未实现的 `boundary_transition` 会明确拒绝，不假称已执行该端点检查。两类都计算物理双边缘、两机实际观察角、逐帧域外R138和 signed boundary、原端点诊断。`edge_coverage` 只将端点诊断从该目标的硬拒绝改为明确的 `boundary_transition_passed=false`；道路、头向速度、source时间、两机rig、GT、完整ROI及实例框支持等硬检查未改变。`scene_mode` 仍标单人/混合，`coverage_goal` 标数据目的，贯通种子入口、prepared recipe、打包摘要、抽查计划和视频标题。

从上一批 seed **20261012** 的 M01/APD 单人原[候选记录](../../outputs/experiments/seeded_viewpoints_a1_20261010/preflight/single_20261012_retry1/single_cyclist_boundary_pass_front_s20261012_e00/candidate_checks.json)按原序重放。新[继承记录](../../outputs/experiments/front_edge_coverage_a1_20261010/replay_manifest.json)验证旧12个候选的布局小数、child seed、相机child seed、整套rig参数和道路方向与新候选生成器逐项相同；**attempt0** 是新策略下第一份完整通过者，没有抽新正面候选、换资产或更改原动作起点。它的目标参数为 x=−0.0212255864m、原抽样 y=1.8031407587m、正速度0.425893323m/s、方向−Y；相机整体 origin=(0.1000048816, 0.0078904640, 0)m、yaw=89.586218°，合成基线0.18m和相对外参不变。由原参数一次推导实际世界起点 y=3.1067174023m，3秒末为1.8290374334m；与旧 `actual_target_start_xy_m` 一致，没有重复加偏移。M01原asset帧0..75、source帧30..105，按10Hz合成时间对原整数时间表最近邻选31条；动作不倒播、不插值、不循环。身体/衣物/R138共用场景矩阵，groom合成头矩阵；行进方向与身体前向保持原门槛。

新[完整预检](../../outputs/experiments/front_edge_coverage_a1_20261010/preflight/single_cyclist_boundary_pass_front_edge_coverage_s20261012_e00/geometry_preflight.json)跑完31时刻的道路、固定高度、投影、按人 square crop、精确 active groom ROI 与7M像素上限，21.24秒，选中attempt0。两路各≥8域内且≥5域外、共同域内≥8的**同一时刻**共22/31（70.97%），连续窗口为时刻11..30、跨度1.9秒；两机同为 `front=[0°,45°)` **31/31**，与边缘重合最长1.9秒。最终相机与世界矩阵重新计算62个角度，和配方保存值逐项相同：cam2全段0.79°–25.57°，cam4为1.65°–29.42°。cam2域外R138首末7→16，cam4为27→53；按原20→5或5→20、两机同步且 signed boundary 翻转的端点定义，**`boundary_transition_passed=false`**。它是运动中的正面边缘截断，不称作完整过界。

## 真实数据与观看

本轮一个Blender进程加载场景/人物一次、一个常驻同步CPU消费者，31个10Hz时刻、双路**62次ROI相机调用**。逻辑normal ERP仍5760×2880，ROI 64 samples CUDA，原灯光/色彩、无降噪/自适应；没有低清背景或完整ERP RGB生产读写。视频31帧PTS为0..3.0秒，末帧展示0.1秒另列。视频中黄色为相机物理有效域边界，青色是实际可见实例紧框；未渲染区不代表真实背景。标题明确写“正面边缘覆盖 · 不是完整端点过界”，两台角度逐帧显示；标题版只从已生成视频再编码，增加Blender调用0、Dataset回读0。另有[俯视轨迹](../../outputs/experiments/front_edge_coverage_a1_20261010/diagrams/topdown.png)和[首中末物理域示意](../../outputs/experiments/front_edge_coverage_a1_20261010/diagrams/physical_projection.png)。

[全帧实际检查](../../outputs/experiments/front_edge_coverage_a1_20261010/full_frame_audit.json)：62/62份图像、local valid/person mask与存储ROI尺寸一致；实际ROI与预检整数边界完全一致，两机camera_to_world、世界R138、原asset/source索引和10Hz tick对应；标签有限，所有62个当前可见实例框由局部mask加offset重算并有完整高质量crop支持。无框/缺失槽位为0，此结果并未取消 `record_unavailable_v1` 的合法无框处理。打包/旁表为31帧、62观测槽、**62合法NPZ行**，共同crop监督≥8为31/31；没有伪造缺失样本。可靠GT依旧按 `in_view & inside_full_crop` 监督，遮挡点参与；两路监督累计cam2 3952点、cam4 3032点，其中遮挡监督分别1344、976点。共同crop点逐时刻范围66–110；不运行DenseKP/YOLO，也没有定位误差结果。

读取前冻结的[抽查行计划](../../outputs/experiments/front_edge_coverage_a1_20261010/planned_readback.json)指定首/中/末三时刻两路六个 observation ID，中段时刻15处于双路边缘窗口。实际[Dataset记录](../../outputs/experiments/front_edge_coverage_a1_20261010/run/pack/readback.json)读取6行（≤8），局部RGB采样、R138 affine、监督、`_trans`和全ERP `img_size` 检查通过；抽查不是全62行Dataset性能或模型效果验收。`readback_plan.json` 在首次 item 读取前落盘。

## 斜前混合CPU支线

按上批冻结的M01＋M04、seed20261012/attempt0的相机和骑行目标先作**一份**目标单独31时刻诊断，未盲排行人。结果cam2的目标138点全段在物理域外、cam4全段域外点为0，双路边缘0/31；前/中/末两机水平观察角分别约11°/38°/51°与8°/35°/47°，并非持续斜前。详见[CPU决策](../../outputs/experiments/front_edge_coverage_a1_20261010/mixed_cpu/decision.json)。按本批顺序规则标 `target_not_qualified_no_walker_layout`：消耗1个目标诊断候选，新增行人布局候选0/最多12，没有混合配方成品、实例框或斜前渲染。上批12个保守ROI失败及人物间距失败仍按原报告解释；本结果没有将保守包络误称实际可见框。

## 计算、限制与复用

渲染前即时检查GPU/进程、内存和磁盘，选GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，未使用GPU1的桌面Blender。runner首次在沙箱内调用 `nvidia-smi` 返回错误9、**尚未启动Blender或产生帧**；按工具权限规则以同一已授权命令重试后成功，实际Blender启动仅1次，未重渲染成功帧。定向CPU unittest **27项通过**，最后的小型种子记录改动又运行相关16项通过；覆盖两目标的不同停止条件、旧默认、seed与相机重放、朝向/双边缘、ROI预算拒绝和原混合行为；无全库discover或参数更新。

[实际成本](../../outputs/experiments/front_edge_coverage_a1_20261010/acceptance.json)：装配回调20.03秒，首时刻14.47秒，后30时刻均值11.69秒、中位11.51秒、范围9.58–14.58秒；Blender含同步消费387.48秒，打包1.78秒、六行回读1.68秒、原视频编码7.53秒，原生产及必需交付总wall **401.52秒（6.69分钟）**，低于20分钟预算。相机调用cam2首11.26秒、后均值8.00秒，cam4首1.41秒、后均值2.38秒；同步消费回调合计24.87秒。GPU采样峰值3543MiB，Blender RSS峰值约8223MiB；输出约96MiB，其中62份ROI PNG约88MiB，NPZ约0.16MiB。临时源采样峰值约200MiB，成功消费后剩余0；长期数据不依赖EXR。上述相机秒数含同步/I/O，不是GPU纯kernel计时。额外标题重编码和独立全帧核查各约1秒，不计作渲染加速。

未来可用 `seed_scene_modes.py --coverage-goal edge_coverage --view-classes front` 为**新**seed配方明确指定目标；不传字段时旧单人端点规则不变。示例：

```bash
PYTHONPATH=src:. python tools/experiments/seed_scene_modes.py \
  --config configs/seeded_viewpoints_a1_20261010.json \
  --mode single_cyclist_boundary_pass --view-classes front \
  --coverage-goal edge_coverage --master-seed 20261012 --episode-count 1 \
  --output outputs/experiments/<new_recipe_directory>
```

本批交付直接使用[冻结候选继承配置](../../outputs/experiments/front_edge_coverage_a1_20261010/config.json)；上面命令演示未来新配方接口，不等同于重放本批精确候选。斜前目标／行人空间与完整ROI支持仍需另批明确布局，完整端点过界仍未获本片段证明。框扰动、独立来源划分、更多模式及训练均未启动。
