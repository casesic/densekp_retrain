# M04行走＋M01骑行：双路同时物理域边缘（A1，2026-10-09）

本轮一个urban_street混合episode，沿用ROI-only公共链，不渲染背景或完整ERP RGB，不运行模型或训练。**结论：不同外观/动作的两人物已实际接通；同一骑行目标在0.0–1.8秒连续19个时刻同时跨越两路物理域边缘。但1.8秒时目标在cam2被行人完全遮住，原生框为空，共同crop门槛失败，已停止。31时刻整段和遮挡恢复未完成。** 上一批两名M01的结果保持不变：多人流程通过，但cam4截断不足、遮挡恢复未通过。

产物：`outputs/experiments/mixed_walk_cycling_edge_a1_20261009/`；冻结执行：`frozen_execution.json`；CPU配方：`mixed_m04_m01_road_pass/recipe.json`；实际生成：`run/`。所有数据均`integration_only/shared_source_diagnostic`，不移入datasets。

## 已有资产与本轮使用

|person_id|角色与来源|外观证据|状态边界|
|---|---|---|---|
|1，稳定目标|cycling_M01_v2_APD6526_30_106_simple_v1，APD/5724486/6526|固定M01 shape、白色皮肤贴图、short-casual groom、原简单运动衣物|用户已接受身体/衣物/头发及单人道路效果；本轮不修改参数、绑定或材质|
|2，行人/遮挡者|cycling_M04_v1_local_walk，KIT/425/walking_medium05|独立30维目标shape、indian_03皮肤、curly groom、酒红上衣/深色裤子及自身绑定|已有行走缓存和连续ROI消费；`human_dynamic_review=pending`及原groom贴合限制保留，不提升为正式外观验收|

两者不是换person_id的M01副本。`appearance_evidence.json`保存实际皮肤、groom源、衣物材质及shape引用；`asset_audit.json`记录路径和缓存shape。M04身体/衣物/head/R138有94条缓存（source9..102），本轮使用source9..99。没有接入A2的新山地车、F02或BEDLAM人物。

## 两种源时间和root

全局合成时钟31条，`episode_time_tick=0..30`，`time_base=[1,10]`，首末3秒。它不是原视频曝光同步。

- M01按APD原整数微秒tick最近记录映射，asset0..75选31条，source30..105；不以frame/25替代微小不均匀间隔。最大绝对量化误差0.019999秒；单位仍是按25Hz推断，非官方曝光标定。
- M04按发布30Hz记录映射，asset0,3,…90，source9,12,…99；量化误差0。原100Hz采集映射仍未知，不伪造。
- 最近记录距离使用`Fraction`精确有理数，平局选更早记录；必须源/asset帧严格递增、在原范围内，误差不超过所选记录朝向目标时刻的相邻采样间隔的一半。无插值、重复、循环、步频改变。
- `episode_timeline.json`、每人`personN_source_timeline.json`与`personN_timeline.json`保留原时钟、来源原点、记录索引、有符号误差和局部间隔；每路标签写全局tick及逐人物源字段。相机消费同一完整人物状态。
- M01沿用pelvis-centered身体，由场景唯一拥有+Y行进，固定高度1.2m，原轴转换和朝向不变。本轮速度0.577934m/s是显式缓慢合成行进参数，不是APD真实速度，也没有改变动作播放速度。
- M04身体已含source-root-relative-to-start，场景只施加一个恒定矩阵（固定转向-90°、横穿放置和整段固定地面抬升）；不再加一条场景平移，不逐帧贴地。衣物/R138与身体共用各自矩阵；groom用scene×本人的head。

`person_time.py`与`dynamic_scene.person_frame_record`增加显式`per_person_nearest_v1`。未选择该版本的旧单人、旧行人和同源多人仍走原严格时间分支。旁表重新从各自完整原timeline计算映射，校验标签、矩阵列、person和packed row，不能用APD字段填行人。

## 冻结布局与CPU结论

保持rig、0.18m合成基线、标定目标20260320-riding-dataset3和两路physical mask不变。先用12个固定空间截面解释交界带（`domain_cross_sections.json`，不是渲染候选或模型选择），再冻结小范围：目标x[-0.025,0.02]m、y[1.85,1.95]m，前进速度[0.45,0.60]m/s；行人固定起点x[-1.8,-1.55]m、y[1.05,1.15]m，沿自身root轨迹横穿。

3个固定seed 20261009/10/11，各最多12候选；实际每个seed第0候选即通过，未扩搜索。选择顺序中的第一个：seed20261009、attempt0，目标起点(-0.014096,1.869990,1.2)m，行人固定道路起点(-1.628567,1.075043)m。相同seed的时间映射、抽样和矩阵已复核；不承诺跨Blender版本RGB位级一致。

硬门槛：同一时刻，两路各≥8域内且≥5域外可靠R138点，共同域内≥8；31时刻至少16个满足且连续窗口≥0.5秒；双边缘时刻实际共同crop监督点≥8。统计先按时刻求两路交集，不用各路计数相加。距离单位为完整5760×2880物理mask像素，64px只作距离诊断，没有沿用旧宽松通过条件。

CPU结果：20/31（64.52%）同时满足，连续frame0..19，即0.0–1.9秒；两路目标身体轮廓均跨physical边界。目标计划原生crop共同点与共同域内点相同。两人XY身体/衣物AABB最小分离0.4855m，身体顶点质心XY距离最小1.6529m（候选JSON沿用`minimum_root_distance_m`字段，实际本分支的距离参考是身体顶点质心，不是骨盆）；这是保守包络检查，不是全网格相交或自行车接触认证。

|时刻|cam2域内/外|cam4域内/外|共同域内/计划crop|cam2/cam4身体与衣物顶点域内比例|
|---|---|---|---|---|
|0.0s|108/30|122/16|92/92|0.852/0.884|
|1.5s|127/11|122/16|111/111|0.932/0.902|
|1.9s|132/6|124/14|118/118|0.972/0.916|
|3.0s|138/0|138/0|138/138|1.000/1.000|

`projection_review/`保存完整ERP示意、frame0/15/19/30局部身体+衣物轮廓、域内/域外R138、ROI与计划完整square crop；不是全景RGB。边界由physical mask生成，不是ROI边缘或ERP接缝。cam4改进证据是连续同时间交集与轮廓截断，不能用上一批“3/76任意点距边64px、零截断”描述本轮。

邻人遮挡预筛：cam2投影高阶段15..17、恢复22..24；cam4高阶段16..18、恢复22..24。凸包加前景深度包络只表示机会。实际采用原`central_ray_consensus_v1`：不可见点的中心射线为另一已知person、较GT近>4cm，且3×3至少5/9同ID近点证据；分母为目标共同physical域内GT，不随visible框删点。固定15个百分点加重及恢复，前/中/后各3个相邻输出时刻，实际结论另列；不以自遮挡增加替代互遮挡。

## 执行与接口

```bash
mkdir -p <新的独立目录>
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/plan_road_pass.py \
  --config outputs/experiments/mixed_walk_cycling_edge_a1_20261009/config.json \
  --output <新的独立目录>/plan.json --geometry-preflight

/home/yuyue/anaconda3/envs/camerahmr/bin/python tools/experiments/run_continuous_roi.py \
  --prepared-episode <新CPU计划目录>/mixed_m04_m01_road_pass \
  --output <不存在的独立run目录> --storage-mode roi_only \
  --consumer-mode persistent --frame-count 31 --budget-seconds 1800 \
  --gpu-uuid <即时核查为空闲的GPU-UUID>
```

本次GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`；GPU1已有其他任务，未争用。一次Blender、一个常驻同步CPU消费者；每帧全部人物同时在场，每相机一个联合ROI，RGB不按人物复制。64 samples/CUDA、无降噪/自适应、原材质灯光和颜色设置不变。只存ROI RGB/局部mask与独立高分range/IndexOB产生的标签；物理mask按段共享。区域外unknown。框是可见实例紧框、deployment1.0，不是YOLO框；仍监督可靠`in_view & inside_full_crop`，包括遮挡点。

`frozen_execution.json`记录阈值、参数、完整时钟、版本/dirty状态、输入及代码size/mtime、62次成功相机调用上限、1800秒预算和最多一次针对性修正。没有计算SHA256。新公共改动仅涉及planner/时间、现有renderer/consumer、sequence packing/index及收尾；Dataset、模型、loss和A2实现未修改。

## 验收与剩余范围

- 21项定向CPU测试通过（0.411秒）：`tests.test_mixed_road_pass`7项＋`tests.test_dynamic_scene`7项＋`tests.test_multiperson_road_pass`7项。覆盖非均匀原时钟、确定性平局、重复/越界/错误误差拒绝、每人物身份时间、两种root、同时间双边缘/持续时间、多框共享RGB与无框槽位、旧同源严格检查。
- 旧M01及旧行人各首中末世界点/矩阵有界对照通过；旧M01和旧ROI行人各1个实际Dataset读取通过。未重渲染历史数据或全量回放。
- 原冻结抽查frame0/15/30；实际停止后保留前两者，并以失败端点18替代未生成的30作取证，未按模型或效果选样。最多12个槽位，实际10个有效crop回读通过；2个无框槽位保留。RGB和138点使用完整ERP→256-square同一affine，`_trans`/`img_size`/监督语义不变。
- 视频使用10Hz合成时钟，保留逐人物源引用；固定每路全ERP显示窗口，黄色physical边界，灰色区明确未渲染，末帧展示时长单列。
- 本轮不认证M04正式外观质量、真实曝光同步、真实骑行速度/自行车接触、多人泛化或训练效果。APD与KIT425均有既有来源关系；整个混合episode按包含全部人物的来源集合分组，不构造独立train/val。

实际结果如下。本轮停在部分交付点，不自动改框策略、扩数据或训练。


## 实际结果：物理边缘落实，原生框支持在完全遮挡处失败

一次Blender（PID1479190）和一个常驻消费者（PID1479222）装配并更新了frame0..18，38次ROI相机调用全部完成；frame0..17正常消费，frame18消费因冻结覆盖门槛失败。没有重启Blender、重渲染成功帧或追加episode。

`failure_diagnosis.json`证据：frame18/1.8秒，cam2目标130个域内R138点全部不可见，130个均有邻人实例ID＋近于GT的range证据，可见实例mask为空、原生框null；cam4仍有目标框。目标共同物理域内114点，但没有成对模型crop，共同crop计数为0，配对统计按无框记null。没有把遮挡GT改为无效，也没有改用固定/跟踪框来继续。

|验收项|本轮实际结论|
|---|---|
|混合人物/时钟/root接口|已生成19个时刻，逐人物来源、两路同状态、矩阵/GT有限性和实例归属检查通过|
|同一目标双路同时物理边缘|0.0–1.8秒，19/31计划时刻（61.29%），连续1.8秒；其余12时刻未生成，布尔序列为null，不推测为true/false|
|双边缘窗口的实际共同crop|frame0..17为92–115点，连续1.7秒；frame18完全遮挡无框，冻结的逐双边缘时刻≥8条件失败|
|互遮挡加重与恢复|有明确邻人遮挡加重；完整三阶段15pp/15pp、各3邻时刻规则未通过，恢复段未生成|
|完整生产流程|partial_stopped；不能把前缀打包成功称为31时刻整段通过|
|真实Dataset/旁表|60行NPZ、124槽位旁表校验通过，10次新数据实际getitem的图像affine、138点affine与监督集合精确一致；不代表数据覆盖或模型效果通过|

cam4改进不是视频裁切：已生成19时刻每时刻有5–16个域外目标R138点，cam2为7–30个；两路逐同一时刻交集均为true。头部、手臂及身体侧部的跨域轮廓见CPU边界图与实际视频。上一批cam4零截断的结论保持原样。

|时刻|共同物理域内点|cam2邻人遮挡/共同点|cam4邻人遮挡/共同点|共同crop|
|---|---:|---:|---:|---:|
|0.0s|92|0/92（0%）|0/92（0%）|92|
|1.5s|111|2/111（1.80%）|0/111（0%）|111|
|1.6s|114|16/114（14.04%）|7/114（6.14%）|114|
|1.7s|115|61/115（53.04%）|13/115（11.30%）|115|
|1.8s|114|114/114（100%）|61/114（53.51%）|null：cam2无框|

冻结metric的描述字符串沿用了旧`three-native-record windows`；本批函数实际输入为10Hz合成输出序列，窗口始终是3条相邻输出记录，不是3条连续APD原帧。`time_wording_clarification.json`单列这个文字遗留，保留原冻结记录；阈值、选帧、统计实现未变。

这是同源时序变化，不是同姿态/同位置因果对照。完整计数、邻人语义点列表与证据代码在`run/pack/per_person_time_statistics.json`；目标框中心、宽高、scale、共同点曲线在`target_curves.csv/svg`。cam2有效框宽135–339px、高531–793px，scale2.655–3.965，随后消失；cam4宽150–309px、高526–792px，scale2.63–3.96。框变化不构成模型误差的证明。

## 部分打包与缺失、拒绝的区别

原31条全局时钟和各人物完整选帧映射未被截短或重编号。实际渲染的源范围：M01 asset0..45中选19条（source30..75），M04 asset0,3,…54（source9..63）。实际31条计划中没有复制姿势或补帧。

旁表保留124槽位：60 observed有效框；15 person_absent（行人早期cam4域外且无实例框）；1 empty（frame18 cam2骑行目标完全遮挡）；48 camera_missing（frame19..30在停止后未生成，理由明确）。一个ROI由同相机多人物框共享，38张RGB不按人复制。

frame18的已写成品经过失败取证后以`rejected_frame.json`登记为**有栅格、质量门槛拒绝**，不是伪造成功的`report.json`；其余3个人物/相机槽位有合法原生框，纳入诊断NPZ，失败槽位保留GT/原mask及无框原因。`frame_acceptance_status`与episode级`execution_status=partial_stopped`保留拒绝状态。`sequence_index.validation=passed`只说明索引/文件/packed row一致，不撤销本轮覆盖失败。

为处理这个真实停止点，公共打包与收尾增加显式`partial_delivery.json`支持；保持全部计划槽位，对已产生但拒绝的帧保留证据，对未生成时刻标missing，视频只编码真实连续前缀，不自动恢复生产。另运行混合拒绝帧打包与旧同源多人打包2项定向测试，0.366秒通过；没有重复全库验收。

原冻结之后的改动单独记在`preview_label_amendment.json`（缩略视频ID可读性）与`partial_finalization_amendment.json`（只处理停止后的打包/回读/视频）。原生产renderer、ROI、consumer标签/监督、冻结配方及门槛没有改动。失败帧原始文件size/mtime复核未变。

实际视频`run/pack/dual_roi.mp4`有19帧，PTS0.0..1.8秒、10Hz合成采样，末帧展示0.1秒，容器时长1.9秒；没有拉长到3秒。两路使用各自固定完整ERP显示窗口，黄色为physical边界，灰区为未渲染排版区域。顶部source显示目标p1源帧，两人的精确源时间各自保存在旁表和readback的逐人物记录中。`event_frame_000/015/018.png`为视频的代表画面。

## 实测成本与保护

本轮GPU为RTX3090，UUID见前文；没有争用GPU1或终止其他进程。38次相机调用成功、1次消费质量拒绝、0次渲染重试。预算停止原因是数据门槛，不是耗时耗尽。

|范围|实测|
|---|---:|
|场景/人物一次装配combined|33.519s|
|常驻CPU消费者启动|2.334s|
|首个成功时刻循环|39.343s|
|后续17个成功时刻|均值34.853s，中位35.501s，范围30.793–38.199s|
|cam2后续18次ROI调用（含拒绝帧的已完成渲染）|均值20.982s|
|cam4后续18次ROI调用|均值10.031s|
|逐时刻ROI规划，19时刻|均值1.227s|
|正常消费函数/同步往返，18时刻|均值2.519s/2.525s，其中局部保存1.454s|
|原Blender进程wall，包含失败|703.748s|
|原任务至失败记录|703.762s|
|部分收尾CPU进程|17.270s（函数内打包2.493s、统计＋回读4.210s、视频7.236s）|
|累计已测渲染＋必要消费/收尾|721.032s，约12.02分钟；失败计入，另有约0.10s取证读取|
|生成启动至部分收尾完成wall|1167.229s，约19.45分钟，含中途诊断和补充部分收尾代码的人工/开发间隔|

GPU调用计时含动态同步、Cycles、compositor及写出，不能解释为纯kernel；失败消费尾段未单独计时，不编造拆分。CPU投影预检52.413s另列，阅读/开发不纳入30分钟渲染与必要消费额度。

两路ROI平均4.679M/4.480M像素，范围3.591–5.373M/3.789–5.000M；近距离混合人物联合ROI远大于上一批远距离M01约0.35M/路，且外观/负载、执行设备有变化，不作严格等负载速度比较，也不展开优化。

- ROI RGB合计185,564,722B（176.97MiB）；run成品及元数据不含失败缓冲约191,868,647B（182.98MiB，记录时点值）。
- 成功帧自有临时源共4,328,383,191B（4.03GiB）已在消费及必要回读通过后清理。
- frame18失败临时源223,243,140B（212.90MiB）完整保留；加失败缓冲的run约415,111,787B（395.88MiB）。未启动时刻的小型所有权目录也保留，未冒充成功清理。
- 1秒采样临时磁盘峰值650.03MiB；实测GPU峰值4071MiB，Blender RSS峰值约11.37GiB，消费者峰值932,832KiB。没有全图RGB生产读写，也没有模型前向。
- 输入资产inventory及失败帧原文件复核未变。本任务没有写TODO；最终检查发现它在21:56出现外部编辑，已原样保留，见`concurrent_TODO_change_observed.diff`。当前训练计划也未修改。未改A2成果、外部项目、旧产物/fixture，未提交commit或计算SHA256。

## 后续缺口与交付点

本轮证明了**不同外观的行走＋骑行可共用公共链，且同一目标可持续同时处于两路物理有效域边缘**；还不能交付满足全部门槛的完整遮挡恢复episode。CPU可见身体投影框没有模拟真实互遮挡后的紧框消失，这是本次预筛与成像之间暴露的具体差异。

下一批若继续，需要在新冻结配方中限制遮挡峰值以保住两路原生crop，或另行评审完全遮挡时的固定/跟踪框策略；不能在本批事后替框或放宽门槛。还需实际完成恢复段、保持来源集合隔离规则，M04正式外观接受状态也仍待定。不会自动重启剩余12时刻、追加episode、接新资产或训练。


必要代码清单：`data/person_time.py`、`data/mixed_road_pass.py`新增精确逐人物时间与有界混合配方；`data/road_pass.py`和`plan_road_pass.py`仅增加显式版本分派；`dynamic_scene.py`保留旧分支并提供逐人物记录；既有`render_road_pass_sources.py`、`run_continuous_roi.py`、`roi_consumer.py`消费每人物状态；`sequence_packing.py`、`sequence_index.py`、`multiperson_acceptance.py`、`validate_persistent_roi.py`接通旁表及部分收尾；`preview_mixed_road_pass.py`输出CPU物理域图；`tests/test_mixed_road_pass.py`为独立人工测试。没有另建Dataset、人体生成器、投影或监督算法。
