# F02前倾骑行＋M04行走：公共道路链接入预检

日期：2026-10-10。**结果：资产和逐人物时间映射已进入公共mixed planner，12个冻结候选均未通过完整ROI支持预检；本批没有渲染episode、打包行或Dataset回读。** 这是有界停止，不是F02视觉质量或既有M01＋M04场景的回退。详细拒绝见 [preflight_decision.json](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/preflight_decision.json)，训练小试草案另见[设计报告](../training/FIRST_TEMPORAL_TRIAL_DESIGN_A1_20261010.md)。

## 输入及可复现接口

固定目标是A2已接受穿衣动态的`cycling_F02_v3_bikefit_XBLMidr4dps_8662_8782_simple_v1`，引用其只读[a1_asset_adapter.json](../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/a1_asset_adapter.json)及120条原[timeline](../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/timeline.json)；行人是既有`cycling_M04_v1_local_walk`、KIT/425 walking_medium05绑定。它们为不同角色家族、不同衣物/groom及不同motion来源，M04原外观限制保留。没有使用M01身体、矩阵或APD时间假设，也未修改两个资产。

入口为`python tools/experiments/prepare_f02_mixed_scene.py --output <new-plan-dir> --seed 20261010`，再由公共`plan_road_pass.py --config <new-plan-dir>/config.json --geometry-preflight --output <new-plan-dir>/plan.json`作CPU预检；若通过才可调用既有`run_continuous_roi.py --prepared-episode <episode-dir> --observation-policy record_unavailable_v1 --storage-mode roi_only --consumer-mode persistent ...`。本次实际只运行前两层，没有冻结可执行episode，也没有运行Blender。准备器只写A1目录，公共规划器仍按旧单源路径读取旧配方。

合成episode时钟为tick0…30、time_base=1/10；F02用其实际PTS步长1001、base1/30000作最近原记录映射，M04用其发布30 Hz记录，各自严格递增。抽查：tick0/1/15/30的F02 asset帧为0/3/45/90、source帧8662/8665/8707/8752；对应M04 asset帧0/3/45/90、source帧9/12/54/99。F02 tick1实际PTS8673665，时间误差+0.0001秒；tick30误差+0.003秒，未把它当30 Hz/APD微秒，也未复制/插值。两人物记录和[31条映射及资产引用](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/asset_adapter_result.json)均保留。此处只验证选帧规则，未声称相机曝光同步。

F02资产已pelvis-centered且global_orient在身体顶点内。只做一次Y-up→Z-up，依据选段模型+Z朝前的中位水平向量推导固定道路yaw约−72.2555°，由场景负责沿+Y以0.44–0.48 m/s的合成常速前进；未额外叠相机→资产旋转。固定放置高度1.10m，使所选身体最低点相对道路约0.149–0.328m，区别于M01的1.20m和行人脚底策略。M04仍是原source-root加一次固定放置，未再叠场景速度。参数及依据在[orientation_and_height.json](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/plan/orientation_and_height.json)；世界矩阵的逐帧实际渲染一致性因无合格配方尚未验收。

## 几何检查及停止原因

保持原urban_street、双机相对外参、0.18m**合成**基线、标定与physical masks。主seed20261010只产生12个有界候选，随机项限骑行/行人道路起点及骑行速度；相同配置和seed的候选参数重复一致。旧M01完整混合场景的遮挡/恢复结论只读保留。为F02细身形缩小目标在道路交界带的起点范围；前置目标单独投影不是完整配方候选，可见[示意图frame9](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/projection_review/target_only_009.png)。绿色/红色为138点在physical mask内/外，黄色为物理域边，浅蓝为body＋cloth凸包代理，绿色方框是目标预计完整crop，橙框为目标单人ROI；灰/黑只是mask示意，不是场景RGB。

候选0的**目标单独几何**同时双边缘20/31时刻、frame0–19连续1.9秒；frame9 cam2为108域内/30域外，cam4为119域内/19域外，共同physical点89。说明cam4不再像上一批失败布局那样始终完整。但这不是正式episode通过：尚须行人支持、联合ROI及实际栅格遮挡。四张[前/中/后示意](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/projection_review/)和逐时刻[target_only_edge.json](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/projection_review/target_only_edge.json)明确标`target_only`。为确定目标交界带，正式多人候选前另计算了21＋15＋16＝52个**目标单独**位置探针；它们不含行人、不是完整配方候选，但比原指令期待的必要投影核查更宽，后续应改为从已有几何区域直接限定范围。正式完整人物组合仍只有12个。

首次预检发现旧多人ROI算法把两人全部几何先合大框，再用**跨人总宽度**扩成单个方形，首帧即纵向越过2880画布。已加显式`per_person_square_union_v1`：每人独立用原有完整crop＋32px余量算法，确认各自区域有效后再取一个矩形并集；旧配方缺省仍走原算法。CPU单元测试覆盖旧整体框越界而按人并集合法、单人区域真正越界仍拒绝。新选项同时接入preflight和公共renderer；**尚未经真实Blender验证**。

同一12个候选改用新ROI算法，全部在前7–9时刻超过本配方7M像素/相机联合ROI预算。为区分预算与硬边界，另用**同一12个参数**做一次只读诊断，把面积限临时提高到16M像素，未用于生产或更改最终配置；12个均在frame9–10因**行人自己的完整crop越出全ERP画布**而拒绝，例如候选0 frame9 cam2行人body投影y约778…2572，加完整方形crop/余量后超过2880。放大联合ROI预算也不能修正这个单人支持错误。保留[原集体算法记录](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/plan/)、[按人7M记录](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/plan_grouped/)及[16M诊断](../../outputs/experiments/f02_mtb_mixed_scene_a1_20261010/plan_diagnostic_unlimited/)；没有抽第13个布局、降低分辨率或把ROI外像素补黑。

`record_unavailable_v1`仍用于已渲染帧的完全遮挡/无框等合法观测，但不能把规划阶段**有潜在训练crop却越界**伪装为正常无观测。若后续另批调整行人固定起点/远近，应仍按预设候选上限重新冻结配置，并复核行进朝向、道路分离、两路物理边缘和遮挡预筛；本批不偷偷重摆或跨到生成阶段。

## 验收、资源与未完成项

本批定向CPU八项测试通过：其中四项针对F02整数PTS与行人独立时间、原root接口、seed局部重现及按人ROI合法/越界，另四项为现有连续ROI边界/接缝/几何行为；5个改动Python文件AST解析通过，计划JSON可严格解析。没有全库测试、模型构造或参数更新。CPU预检三次同候选集合分别约2.31、3.84、4.34秒；投影评审和文档时间不冒充渲染成本。新目录约4.5 MiB（写报告前采样），只含配方、拒绝记录和投影图。

资源初查：GPU0 RTX3090 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`约18MiB占用；GPU1 UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`有桌面Blender约407MiB，未干预。RAM约188GiB总/140GiB可用，仓库盘约12TiB可用。因前置未通过，没有启动GPU/Blender；预算30分钟中新增渲染/消费为0，0次相机调用、0张ROI、0训练行、0次Dataset回读、0个视频。自然无法报告本次双边缘**实际栅格**、遮挡15/15恢复、真实输入affine或吞吐；这些都标未验收，而不是失败画质的推断。

下一批A1的最小前置是冻结新的**行人远近/轨迹**及按人并集ROI面积上限，在不超过新批允许候选与时间预算时预检完整31帧；只有双边缘及全部本地crop支持都成立才启动同一公共ROI-only流水线。A2仍只需按约定交付其它动作绑定，本批没有依赖或修改其BVH工作。来源隔离与训练决策见独立设计草案；本次不得称F02已经公共场景消费通过。
