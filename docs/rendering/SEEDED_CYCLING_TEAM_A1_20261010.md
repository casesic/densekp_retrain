# 种子骑行队伍：组件与CPU几何接入结果

日期：2026-10-10。**本批未生成队伍视频或训练数据。** 河岸2人是规定的首段，冻结的12个不同布局全部未通过完整ERP裁剪/接缝或双路物理边缘预检；城市3人的第6候选曾通过边缘与ROI几何，但未通过原任务要求的投影遮挡加重—恢复机会预筛，完成12个冻结候选后也没有合格布局。按执行顺序与硬条件均未启动渲染。没有降低5760×2880、64 samples、7M像素联合ROI上限，或再抽第13个候选。[机器验收](../../outputs/experiments/seeded_cycling_team_a1_20261010/acceptance.json)明确记录0次Blender启动、0次ROI相机调用、0行NPZ、0次Dataset回读、0个视频。本结果是**组件和队伍CPU接口接入**，不是两环境队伍生产验收。

## 冻结范围与实际组件

[配置](../../configs/seeded_cycling_team_a1_20261010.json)固定master seed `20261022`、先河岸2人后城市3人、每段最多4套组件计划与12个布局、3秒/31个合成10Hz时刻、最多7M像素/相机的联合ROI。独立的身体、动作、衣物、头发、时间和布局子随机流按稳定排序及条件过滤选择，缓存命中不会重抽。组件源先限APD6526、participant_1、participant_2；bikefit许可unknown未扩大使用范围。人数只支持2或3，未实现4–5人和入场前伪帧。配方与[三人的选择证据](../../outputs/experiments/seeded_cycling_team_a1_20261010/frozen_team_plan.json)均已冻结。

实际仅为每个人冻结1套组合、每段分别复用前2套／前3套，没有启动更多组件计划或build。条件选择时person2、person3分别跳过4、79个**仅内存枚举且明显违反身份/来源/衣型不重复条件**的四元组；它们不是另外制作的组合候选或额外seed试验。

| 人物 | 身体／原动作来源 | 衣物／头发 | 原时间起点 | 组合状态 |
|---|---|---|---:|---|
| person1，预定目标 | M01／APD6526 | M02_simple／M01_groom | 原记录0 | 本任务新制作，数值检查通过 |
| person2，河岸与城市遮挡者 | M02／participant_1 comfortable | M01_simple／F02_groom | 原记录4 | 本任务新制作，数值检查通过 |
| person3，仅城市 | F02／participant_2 straight | F02_simple／F02_groom | 原记录19 | 本任务新制作，数值检查通过 |

三人身份家族、原动作来源组、衣型互不相同；person2/3头发父组件相同，未声称三种发型。前两套在实现错误后的同任务重放命中了**本任务刚生成的**已核验交付，并非A2旧缓存命中；[缓存记录的时序说明](../../outputs/experiments/seeded_cycling_team_a1_20261010/component_build_evidence.json)避免将重放状态误读为首次命中。组合通过A2公共`composition.plan/build`，未修改A2算法、注册表、绑定或既有交付。新组合只获技术合格的`integration_only`使用资格；已展示旧组合的用户观感接受不自动转移。

新[队伍模块](../../src/densekp_retrain/data/cycling_team.py)为公共渲染器准备逐人原整数timeline、scene矩阵、真实R138世界点、两机rig、physical mask与逐人square-crop联合ROI。原动作分别按最近观察、平局取早、量化误差约束映射到统一合成时钟；没有把APD/p1/p2写成统一30Hz，也没有复制person1的asset frame。每人固定身体/衣发和放置高度，scene沿+Y正时间常速推进；不循环、倒播或逐帧贴地。身体、衣物、R138共用该人物scene矩阵，groom的预检支持按scene×head计算；公共生产器仍负责实际栅格。三人相邻人物对执行道路及包络/根距检查。这只是有限近似，不是全三角形连续碰撞证明。

## 有界布局及生产门槛

每候选检查全部31时刻、目标同一时刻双路各≥8域内且≥5域外R138、共同物理点≥8；通过需至少16/31并有≥0.5秒连续窗口。物理mask来自标定，不从黑RGB、ROI边界或ERP接缝推导。每人物先独立算完整square支持再合并，两路均拒绝接缝跨越、越过完整画布和>7M像素；接受候选再用当前实际active groom曲线复核。朝向对道路速度≤25°、固定骑行支撑与逐对距离另查。原生框必须待渲染后由可见实例mask产生，CPU预检不把几何代理当真实框，也不把预投影重叠叫邻人遮挡GT。

| 环境 | 实际CPU布局 | 拒绝与结论 |
|---|---:|---|
| riverfront，2人 | 12/12 | 5个ERP几何跨缝、6个完整裁剪越画布或为空、1个同时双边缘不足；[最终候选](../../outputs/experiments/seeded_cycling_team_a1_20261010/riverfront_2_overlap_gate_correction2/prepared/candidate_checks.json)。无合格首段，不启动渲染。 |
| urban_street_assets，3人 | 12/12 | 双边缘不足4、越画布3、接缝2、人体包络过近2、投影邻人遮挡加重—恢复不足1；[最终候选](../../outputs/experiments/seeded_cycling_team_a1_20261010/urban_3_overlap_gate_correction2/prepared/candidate_checks.json)。第6个仅通过边缘与ROI几何，目标双边缘23/31，但**未通过遮挡机会预筛**，不列为可执行完整配方。 |

首次预检还暴露新实现把逐人有效域数组自身追加、导致维度不齐的错误；下一轮定向检查发现城市第6候选没有按原要求执行CPU投影三阶段遮挡机会预筛。这两处实现纠正的原结果分别保留在`riverfront_2`/`urban_3`和`*_implementation_correction1`目录，最终`*_overlap_gate_correction2`只重放**同一批**冻结候选，没有增加独立布局或覆盖失败事实。河岸尤其是person2起点相对rig较近，接缝/越界是当前冻结范围下的明确限制；未把计划ROI裁短以取得通过。城市第6候选的部分几何证据仍可复核：三人不同源时钟、31×3×4×4矩阵、31×3×138世界GT已落盘，三人3秒道路位移分别约1.576/1.392/1.406米，两路精确groom联合ROI为每帧约0.799–3.683M／0.815–3.881M像素；但它不是完成全部队伍目标的可渲染配方。还没有实际IndexOB/range或邻人遮挡15/15个百分点证据。两环境均**不声称遮挡加重与恢复通过**，无框自然继续策略也只能在真实渲染中验收。

本批没有调用Blender、视频编码或Dataset，因此不存在可直接观看的新队伍视频、合法训练行、框消失/重现、裁剪监督或真实吞吐数字。原河岸单人链和旧城市多人/行人链保持只读。旧第二环境`environment_selection.json`是在森林运行中的决策快照，记1个已完成调用；其最终`acceptance.json`记录森林frame0双路共2个成功调用，以后者为最终计数，旧文件均未改写。

## 核验、资源和后续限制

定向CPU unittest 5项通过：seed重放/实际组件变化、2/3人身份与原来源、候选上限、城市部分几何证据中的三人源时钟/矩阵/R138及≤25°朝向、最终有界拒绝、旧单人和旧混合recipe读取。未运行全库测试或模型。计算前GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`占用18MiB、GPU1占用407MiB，可用RAM约138GiB、卷约12TiB；无人任务被终止。实际输出约103MiB，主要为三套组合几何与CPU证据。组合/预检的CPU总wall未在本轮入口独立计时，不能伪造精确吞吐；渲染与消费均为0，60分钟上限未被渲染消耗。

复用命令（新目录、相同版本配置）：

```bash
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/seed_cycling_team_a1.py plan --config configs/seeded_cycling_team_a1_20261010.json --output outputs/experiments/<new_team_run>
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/seed_cycling_team_a1.py prepare --config configs/seeded_cycling_team_a1_20261010.json --output outputs/experiments/<new_team_run>
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python -m unittest tests.test_cycling_team_a1 -v
```

下一批若继续河岸队伍，需先评审**新的、有界**person2/rig相对布局范围，使所有实际完整crop远离ERP接缝与全图边界；城市还需能产生投影前景交叠上升再回落机会的有限间距/速度范围。不能将本批失败候选改写为成功。随后才能运行公共`run_continuous_roi.py`、实际IndexOB/range邻人事件、打包/旁表、最多12/18行Dataset回读和视频。不应从本批推断其渲染速度或多人观感。全部数据及组合仍是`integration_only/shared_source_diagnostic`；共享人物家族/动作录制、邻帧和换seed不能拆成独立train/val。框增强、队伍4–5人、路口、来源独立划分、推理与训练均未启动。
