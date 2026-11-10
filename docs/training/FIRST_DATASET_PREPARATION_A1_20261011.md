# 首轮数据配方与框增强支持（待评审）

日期：2026-10-11。**本批只有CPU计划、代码和既有ROI样本回读；没有新渲染、资产制作、模型推理或训练。** 用户已接受已展示的城市T形路口三人正／背／侧视频观感；该段仍是共享来源的转弯/多视角诊断，双边缘最长2时刻，15/15邻人恢复未通过，不计入边缘配额。旧4–5人及河岸队伍的技术诊断也不因观感接受改写。

## 入口与阶段

`PYTHONPATH=src python tools/experiments/prepare_first_dataset_a1.py` 默认只列能力；`save --output <path>`冻结配方；`preflight --output <path>`调用现有来源校验器、原整数时间映射和必要时的A2 **plan**，不调用A2 build、SMPL-X模型、Blender或GPU。当前配置为 [first_dataset_preparation_a1_20261011.json](../../configs/first_dataset_preparation_a1_20261011.json)，本批机器产物在 [episode_plan.json](../../outputs/plans/first_dataset_preparation_a1_20261011/episode_plan.json)。两处输出位置的同seed计划逐字段相同；人数、组件、每人时间、布局、相机和框各有稳定子流。路口仅城市已验三人；城市/河岸单骑行、混合交会和2–5人分散骑行可表达，超出模式人数或河岸路口拒绝。组件选择既可在已注册组合中抽样，也可从提议池允许的身体/motion与A2简单衣物、静态groom分别抽样；后者经A2公共`plan`技术检查，未制作时明确`build_required`。本批一个M01/p2＋M02衣物交叉组合CPU计划通过，但没有制作。

这层是**配方调度**，不复制渲染器、投影器或Dataset。跨池衣发父身体不会被自动判作同一身份，但标记`appearance_parent_review_required`并暂不授予正式来源资格；通用衣型的审查结论可另行明确。`plan_valid`、`source_eligible`、`asset_ready`、`geometry_checked`、`rendered`、`dataset_verified`分开；配置的坐标/rig数值是未验证的候选参数，未运行身体投影和道路/ROI几何预检。未来生产必须通过原公共准备器的全段几何检查和ROI支持，不能把这里的`plan_valid`当作渲染许可。全部新配方`pending_review`。

## 首轮来源与规模建议

以[A2提议池](ASSET_SOURCE_POOLS_A2_20261010.md)为起点，机器[来源表](../../outputs/plans/first_dataset_preparation_a1_20261011/source_pools.json)逐组合列身份/shape与纹理来源、motion录制/参与者、衣发父引用、许可、接受窗口、场景消费状态。相同参与者的comfortable与alternate是**两种不同motion ID**，可以分别计动作种类；它们仍共享参与者和录制关系，报告人物或录制独立性时必须如实披露。标准SMPL-X拓扑/算法共享本身不将全部角色合成同一身份。KIT11/KIT425目前只确认不同录制路径，演员是否独立未知；bikefit许可unknown，留诊断池。

推荐先评审4个新episode而非为凑旧8段扩来源：城市train单骑行p2_M01、河岸val单骑行p1_alternate_M02、城市train混合p2_M01＋KIT425_M04、城市val混合p1_alternate_M02＋KIT11_M03_simple。每段31时刻/10Hz，前两段各1组双机，后两段各2组，合计124场景时刻、372次计划ROI相机调用、620个人物/相机/时刻槽位，合法框行只会少于槽位。冻结组件、起点、seed、rig及原整数时间见机器计划。建议训练抽偶数tick的16个时刻/episode，评价及缺失分母保留全31时刻；同轨迹多rig、邻帧、衣发衍生框和换seed不增加独立来源数。

4个计划的元数据与精确时间映射均有效；3个满足当前**提议**来源校验，训练混合段因KIT425_M04权威目录没有完整人工接受范围而未达标。4个均未满足“这个精确组合已被道路场景消费并预留本轮增强ROI”的资产就绪门槛。因此**现在没有一段获准进入正式批量生产**。前两段是优先补场景资格的候选，验证混合段随后；训练混合段先核实KIT425接受范围。现有共享来源的多人与路口旧episode最多引用4段作流程诊断，不能按行随机拆入两侧。每段所有目标和遮挡者必须同侧，A2 `validate_batch`保持整个`scene_episode_id`跨rig不可跨池。[缺口与模式矩阵](../../outputs/plans/first_dataset_preparation_a1_20261011/gaps.json)明确：提议训练侧仅p2一段骑行、一个骑行身体；验证侧p1两种motion但仅M02一个骑行身体。故两侧当前都无法组装不同身份的3–5人正式样例；旧多人/路口保持共享来源诊断。此结论不把“同一参与者”误算作“同一动作”。

[覆盖目标](../../outputs/plans/first_dataset_preparation_a1_20261011/coverage_targets.json)将双路物理边缘、单侧截断、邻人ID＋range遮挡、完全/轻遮挡、无框重现、较小一路人体尺度、正背侧和城市/河岸分别计，所有比例注明人/机/时刻分母；维度可重叠。CPU投影只预测物理机会与尺度，真实可见框、邻人遮挡和可用输入要等渲染后核验。无框保留在计划时刻分母，定位误差为null。[预算](../../outputs/plans/first_dataset_preparation_a1_20261011/budget.json)按同类负载给出CPU预检5–20分钟、四段渲染消费收尾约30–80分钟、临时峰值200–600MB、长期约400–1200MB的**待评审工程范围**；资产补齐时间未知，不能以旧单人最优吞吐承诺多rig。原8段/200步只是历史草案，未执行、未授权。

## 渲染前预留与训练时开关

新增[box_augmentation.py](../../src/densekp_retrain/data/box_augmentation.py)的`bounded_native_box_v1`，默认候选概率为原生50%、中心25%、scale15%、单侧截断10%；中心每轴≤父框最大边3%，scale0.9–1.1，截断≤该侧10%。幅度可在冻结上限内缩小，概率可重配且和为1。这是工程假设，不是YOLO误差拟合。原生框仍从**渲染后的可见实例mask**取得；预先只用投影body/cloth/active groom包络规划所有可能父框及其增强方形crop。令包络最大边为`M`，增强半边≤`0.55M`，中心扰动≤`0.03M`，单侧截断中心移动≤`0.05M`；每方向额外`0.63M`再计栅格2px、浮点余量和原32px边界余量。按人规划再取联合ROI，原全图边界、ERP接缝和700万像素限制保持；超限拒绝，不补黑或缩框。预检与`render_rig_roi`读同一可选策略字段，旧配方缺字段行为不变。

新生成pack在每条父行保留`parent_bbox_xyxy`与`box_augmentation_reserved_policy`，序列旁表保留原父框/box ID、rig/来源；下游索引合并保留这两个字段。Dataset默认关闭，按旧中心/scale与顺序读取；打开时先对**所有父行**检查训练幅度不超过数据声明、整族crop均有ROI支持，再以独立`seed|epoch|parent_index|explicit_visit_index`抽取实际框，在真实256-square仿射前验证支持并由原`in_view & inside_full_crop`重算监督，遮挡点仍监督。无支持、非法框、ERP越界直接报错，不重抽、丢行或回退。当前不支持框扰动叠加旋转、预滤波、flip或Albumentations；这些组合显式拒绝。`DataModule.set_box_augmentation_epoch`需在下一轮实际训练迭代器创建前调用；重复访问缺省visit0，若要重复访问独立变体须由调用方显式设visit索引，不能依赖worker先后。评估`is_train=False`始终禁用随机框。

待评审的[关闭配置](../../configs/hydra/experiment/first_trial_box_aug_off_a1_20261011.yaml)与[开启配置](../../configs/hydra/experiment/first_trial_box_aug_on_a1_20261011.yaml)只在`DATASETS.BOX_AUGMENTATION.enabled`不同；[完整对照参数off](../../configs/first_trial_box_aug_off_a1_20261011.json)／[on](../../configs/first_trial_box_aug_on_a1_20261011.json)亦只差此开关。两组应锁定同一原始权重、父样本集合/顺序、曝光次数、200次待评审更新、优化器和其它增强；不展开额外训练行。原生框评价与冻结框挑战输入在两份权重上完全相同。现阶段未启动任何训练。

## CPU验收及未覆盖

定向7项新测试、18项ROI相关和10项序列索引测试通过。12条事先冻结的城市/河岸、单人/多人历史父行做了66次真实Dataset回读：每行off与旧原包逐字段一致；6条整族增强支持通过并各做3个确定性on变体及固定评估读取，图像/仿射/138点/监督/全ERP `img_size`与重复读取一致；另6条旧窄ROI整族支持不足，按原因拒绝，off仍可读。小型CPU画布独立比较完整图与非零offset ROI的同一增强仿射与像素。明细在[验收记录](../../outputs/acceptance/first_dataset_preparation_a1_20261011/checks.json)和[逐行支持](../../outputs/acceptance/first_dataset_preparation_a1_20261011/box_support_readback.json)。所有定向测试和回读命令累计远低于20分钟CPU预算；没有Blender、GPU、模型或训练调用。

下一批评审需先决定：正式来源与许可/预训练暴露允许的独立性口径，KIT425接受窗口，哪些精确组合先完成道路消费，四段还是更小的规模/预算，以及固定挑战框与主评价配额。随后才可用公共生成器为新数据预留ROI、冻结共同父样本资格并考虑原权重基线和受控训练。
