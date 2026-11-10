# 首轮四段候选数据执行记录（2026-10-11）

本批按[原四段候选计划](../../outputs/plans/first_dataset_preparation_a1_20261011/episode_plan.json)执行，保持人物组合、分侧、31个10 Hz时刻及各人的原整数时间映射。四段中三段完成ROI-only生产，e00在有界几何预检后停止。产物仍在`outputs`，均为内部集成候选；未训练、未运行模型、未批准正式train/val或发布。

| 段 | 候选侧与内容 | 结果 | 实际相机ROI | 观察槽/原生合法行 | 双路物理边缘/共同crop≥8 | 视频 |
|---|---|---|---:|---:|---|---|
| e00 | train；城市p2/M01单骑行 | 几何拒绝 | 0 | 0/0 | 无合格配方 | 无 |
| e01 | val；河岸p1 alternate/M02单骑行 | 技术生产完成，观感待评 | 62 | 62/62 | 11/31；31/31 | [双路视频](../../outputs/experiments/first_four_episodes_a1_20261011/e01/run/pack/dual_roi.mp4) |
| e02 | train；城市p2/M01＋KIT425/M04 | 技术生产完成，观感待评 | 124 | 248/173 | 两rig各0/31；back 0/31、front 31/31 | [背面](../../outputs/experiments/first_four_episodes_a1_20261011/e02/run/rigs/rig_back/pack/dual_roi.mp4)／[正面](../../outputs/experiments/first_four_episodes_a1_20261011/e02/run/rigs/rig_front/pack/dual_roi.mp4) |
| e03 | val；城市p1 alternate/M02＋KIT11/M03 simple | 技术生产完成，观感待评 | 124 | 248/200 | 两rig各0/31；back 29/31、front 31/31 | [背面](../../outputs/experiments/first_four_episodes_a1_20261011/e03/run/rigs/rig_back/pack/dual_roi.mp4)／[正面](../../outputs/experiments/first_four_episodes_a1_20261011/e03/run/rigs/rig_front/pack/dual_roi.mp4) |

[统一观看页](../../outputs/experiments/first_four_episodes_a1_20261011/index.html)、[机器验收](../../outputs/experiments/first_four_episodes_a1_20261011/acceptance.json)、[来源记录](../../outputs/experiments/first_four_episodes_a1_20261011/source_review.json)分别保存观看、实绩和来源。五段实际视频均为31帧、10 fps，0–3.0秒取样，末帧再展示0.1秒；未渲染的排版/ROI外区域不是完整场景RGB。

## 几何、覆盖与首次消费

e00的原12套布局中，1套目标未进入身体有效域，11套被同一目标双路同时边缘条件拒绝。随后冻结一个**既有人物布局**，只细化100组位置/yaw机位，计入每rig上限，总计112组机位尝试，仍为0个同时双边缘时刻；没有渲染、换人或降低门槛。[冻结机位表和拒绝](../../outputs/experiments/first_four_episodes_a1_20261011/e00/rig_refinement1/)保留。该缺口需要后续另行评审布局/相机几何，不能把本批计为4段完成。

e01先保留原提案，再对同一12个候选做场景引用修正后的逐项回放，选第4个。实际11/31个双边缘时刻（连续tick0–10、跨度1.0秒），满足**本批预先冻结的7/31诊断下限**；历史消费者结果中仍显示旧16/31汇总门槛，两者不应混读。[批次验收](../../outputs/experiments/first_four_episodes_a1_20261011/acceptance.json)显式列出7和11。真实原生框高度中位数436个完整ERP像素，范围350–600。

e02/e03以混合交会与不同机位为用途，未把双边缘设成生产门槛。两组rig消费同一人物世界状态，分别检查投影和ROI；实际e02的back共同crop≥8为0/31、front为31/31，e03为29/31和31/31。e03背面rig的真实邻人ID/range证据通过原15/15、每阶段连续3帧诊断：cam2加重/恢复各16.7个百分点，cam4为60.2/16.8个百分点；e02两rig及e03正面rig未达到。CPU投影机会与实际遮挡分列，缺少共同输入没有记成0定位误差。严格端点过界、正式覆盖配额均未由此宣称通过。

每个已生成段一次Blender装配、一个同步CPU消费者；e02/e03同一时刻更新一次两人的状态，再处理两组双机。生产用5760×2880逻辑ERP、64 samples CUDA、原场景光照色彩与物理mask，只有高质量联合ROI，没有低清背景或完整ERP RGB。三段共310次相机ROI调用；实际整数ROI与精确active-groom预检逐次相同，差异0；所有31帧、实例ID、时间、世界矩阵、有限GT、mask与持久索引的公共验收均通过。[独立收尾核查](../../outputs/experiments/first_four_episodes_a1_20261011/final_integrity.json)再次确认每人原帧严格递增、两路源时间相同、同一混合场景两rig的世界矩阵逐值相同。自然无框和域外保留槽位及原因，不造训练行。

源时间由逐人物原timeline映射：e01 p1 source9615–9687；e02 p2 source5719–5809、KIT425 source10–100；e03 p1 source9608–9680、KIT11 source9–99。合成10 Hz不表示这些来源具有同一帧率。KIT425本批使用的`M04_KIT425_walk_replay_v2`与旧已观看道路片段的`cycling_M04_v1_local_walk`并非同一缓存版本；旧片段覆盖KIT source9–99，本批10–100。旧观看接受可作为动作/角色参考，**不能升级为新94帧缓存全部已人工接受**。本批两段混合视频交付后统一待评。

首次道路画面抽看显示骑行姿态人物，但没有自行车道具；这是观感/语义限制，不能用标签链路通过替代用户对骑行场景的观看判断。当前只引用既有技术资产，不修改其接受记录。

## 框增强支持、回读与来源

渲染前按`bounded_native_box_v1`从body/cloth/active groom投影构造**所有允许可见父框**的保守square crop支持；原生框渲染后由真实实例mask产生，再逐行核验整族支持，区别于把一个预测框当作已知。概率为原生50%、中心25%、尺度15%、单侧10%；平移每轴≤最大边3%、尺度0.9–1.1、单侧≤10%，deployment扩张1.0。生产中没有永久随机框、没有另渲一套on图像。实际435/435个合法原生父行整族支持通过，0个支持缺陷；三个段的ROI都低于700万像素上限。若未来变更增强范围、预处理或crop方式，必须重新证明支持。

[共同父样本清单](../../outputs/experiments/first_four_episodes_a1_20261011/common_parent_rows.json)列出全部435行，供训练时同一份原始RGB/GT/框分别读取on与off；开关仍只在Dataset取样时生效。[固定挑战框](../../outputs/experiments/first_four_episodes_a1_20261011/fixed_box_challenge.json)是17个有支持的确定性中心扰动输入，未运行模型；[训练偶数tick与全段评价索引](../../outputs/experiments/first_four_episodes_a1_20261011/training_candidate_ticks.json)仍是pending_review。四段预定最多24个真实Dataset父槽位，其中e00的4槽无数据；其余槽位无框时不替换，有效槽位经原生、on和固定原生评价共70次真实`Dataset`读取，仿射、138点、监督、`_trans`、完整ERP `img_size`及来源ID通过。未支持的flip、旋转、预滤波、Albumentations仍拒绝。可靠GT仍按`in_view & inside_full_crop`监督遮挡点。

候选划分只声称骑行目标家族及发布动作参与者/录制不同，行人KIT11/KIT425仅已确认**录制路径不同**；演员独立与DenseKP原始预训练暴露未知。整个episode含遮挡者，全部rig、邻帧、增强框成组；e00缺失导致当前train侧只有混合e02，不能作为完整的正式对照。e01的河岸与train城市又存在环境混杂。p1两个动作片段关联，但不同片段仍可作为不同动作，不能称为独立参与者。所选骑行动作记录CC BY4.0，KIT/BEDLAM2和人物纹理涉及非商业/单独条款；只限当前内部候选用途，许可与发布资格没有扩大。

## 成本与下一步

CPU几何预检含e00追加机位与e01同候选修正回放累计365.65秒（6.09分钟，20分钟上限内）。三段装配、渲染、同步消费、打包/内置回读/视频实际分别421.47、1154.90、1245.81秒，合计2822.18秒（47.04分钟，80分钟上限内）。整批有界计算约53.13分钟，未发生渲染故障重试。所用空闲GPU UUID为`GPU-d481acc7-106f-29e5-1298-8c0d398261b2`；运行前资源快照各在`run/resources_before.json`。输出目录现占约739.5 MiB，其中e00预检日志34.3 MiB；长期数据与诊断均保留，未测得可信临时峰值，不据收尾残量推称峰值。

[定向检查](../../outputs/experiments/first_four_episodes_a1_20261011/targeted_checks.json)的20项单测及脚本语法检查通过。下一步先观看本批视频，决定无自行车道具的骑行姿态和两套首次混合消费是否可作为小试候选；再评审e00几何缺口、来源/许可限制及正式分侧，冻结on/off共同父样本与评价协议，之后才授权原权重基线和训练。本批没有下载/安装、重制资产、修改源.blend、运行DenseKP/YOLO或更新权重，也没有移入`datasets`。
