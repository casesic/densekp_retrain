# A2 资产来源与首轮候选池（待评审）

日期：2026-10-10。**本文件及所有池均为 `proposed/pending_review`；原资产仍为 `integration_only`。** 这里只核对元数据、运行纯 CPU 组合计划与规则反例；没有生成正式样本、分配训练／验证、运行模型或批准许可。机器清单见 [source_catalog.json](../../outputs/plans/asset_source_pools_a2_20261010/source_catalog.json)、[pool_proposals.json](../../outputs/plans/asset_source_pools_a2_20261010/pool_proposals.json)、[relations.json](../../outputs/plans/asset_source_pools_a2_20261010/relations.json)。

## 结论与可声称范围

现有资产足以**提出**一个骑行目标角色家族和已发布参与者／录制分开的首轮候选池，且不需新增人体或动作。混合行人只能声称目标角色家族和 KIT 录制路径分开；KIT11 与 KIT425 的原演员关系未证实。Zenodo participant_1 与 participant_2 是不同的发布 participant code 和录制文件，但未独立核验现实演员身份。两个骑行窗口 `p1_comfortable`、`p1_alternate` 属于同一 participant_1 录制，只能算一个来源组。原 DenseKP 预训练数据与这些来源是否重叠未知，所以目前不能称“从未见过的演员测试”。

首选池尚不可直接投入生产：M04/KIT425 的 94 帧统一回放数值通过，但没有证据将 A1 道路短段的观感接受扩成整段 studio 穿衣接受；p2/M01 与 M03/KIT11 原组合尚无公共场景实际消费；BEDLAM2 和 Meshcapade 资源使用条件仍需按拟议训练／再分发用途核准。已被人眼检查或用于调参的历史片段只能作开发或诊断，不能改名为全新测试集。

## 核查依据与四层状态

角色以 [组件注册](../../configs/asset_composition_a2_20261010/components.json)和 M01、M02、F02、M03、M04 各自的 `manifest.json` 为准：五者有各自固定 shape、皮肤和脸部父引用，动作的原演员并非合成目标角色。历史 [split_v2](../../outputs/experiments/source_isolated_pilot_20260930/split_v2.json)与[可能来源审计](../../outputs/experiments/source_isolated_pilot_20260930/identity_source_audit/report.json)提示保守脸部细节组 M01/M04 与 M02/M03；这表示可能共用底层外观细节，**不等于**已证明共享真实扫描人。原 `face_source_group` 字段和保守审计应同时看，不单凭一个字段断言独立。共同使用标准 SMPL-X、R138 或绑定算法不是人物同源证据。共用通用衣型／发型模板属于外观重叠，限制“全外观独立”的说法，但不自动合并所有演员组。

五段骑行 motion 为 APD6526、bikefit 视频、p2 一段及 p1 两段；权威交付和接受范围见[能力清单](../../outputs/experiments/garment_render_contract_a2_20261010/capability_and_sources.json)。APD 的 M01/F02 衍生与 A1 的 M02 衍生共用 APD track，身份／session 未知，邻窗和换外观不新增来源。bikefit 视频已接受身体和 F02 穿衣动态，但原视频许可仍 unknown；公开可看和内部重建不构成正式训练或发布授权。p1/p2 的发布参与者及片段见[动作库存](../../outputs/audits/cycling_motion_library_a2_20261009/inventory.json)，Zenodo 10668611 的本地官方记录标注 CC BY 4.0；仍需保留署名和模型／人物材质各自的限制。

行走方面，M04/KIT425 与 M03/KIT11 的[原配映射和许可说明](../data/bedlam/BEDLAM_BEDLAM2_RESOURCE_REFERENCE_20261009.md)证明 `KIT/425/walking_medium05_stageii.npz` 与 `KIT/11/WalkingStraightForwards05_stageii.npz` 是不同路径／录制，不能证明演员独立。M03 简单衣物和已接受的 [BEDLAM 转移衣物](../appearance/BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md)共享目标 M03、KIT11 motion 和时间，后者仅锁定 M03＋KIT11＋M03 groom＋权威完整 `[0,120)` 缓存。合法 `[15,106)` 子窗口引用原资产帧与完整 root 参考，不创造新人物或动作，也不解锁换件；见[衣物窗口契约](../appearance/GARMENT_RENDER_CONTRACT_A2_20261010.md)。BEDLAM2 非商业科研条款和 Meshcapade CC BY-NC 纹理条款并不等同于“所有训练及再分发均已批准”。

机器目录分别保存 `compatible`、`built`、`human_accepted_scope`、`scene_consumed`、`eligible_under_proposed_policy`。例如 M04＋p1 的一份具名 `composition.plan` 通过 21 个依赖检查，仍未 build、未观看、未道路消费；M04/KIT425 的 A1 短场景消费不提升其完整 94 帧 studio 人工状态。A1 已做的 M01/APD＋M02/p1 河岸队伍、F02/bikefit＋M04 混合段和跨衣发组合均有真实场景证据，但它们把若干拟议池来源放在同一 episode，全部保留 `diagnostic_only`。[河岸队伍](../rendering/CYCLING_TEAM_RELATIVE_LAYOUT_A1_20261010.md)、[F02 混合段](../rendering/F02_MTB_MIXED_SCENE_REPAIR_A1_20261010.md)、[种子组件场景](../rendering/SEEDED_ASSET_COMPOSITION_A1_20261010.md)分别说明实际消费范围。

历史 [BOX_ROBUSTNESS_400](BOX_ROBUSTNESS_TRIAL_400_20261001.md)使用 `split_v2`：M01/M04 为历史训练目标家族，M02/M03/F02 为历史验证目标家族；KIT425 也是历史验证 pose 来源。若续训该微调权重，历史 train/val 的参数暴露和模型选择均需计入。若从原始 `densekp.ckpt` 重启，过去微调的参数暴露不在新权重中，但原权重预训练来源重叠未知，既有验证观察仍影响方案选择；不能据此宣布新测试集未见过。

## 首选：目标家族与录制路径隔离候选

| 侧别 | 骑行 | 行走 | 当前证据与门槛 |
| --- | --- | --- | --- |
| `train_candidate` | `p2_M01`，120 帧 | `KIT425_M04`，94 帧 | 两套已制作；p2 穿衣已接受但原组合道路未验，M04 完整 94 帧穿衣观感范围仍待明确 |
| `val_candidate` | `p1_comfortable_M02`、`p1_alternate_M02`，各 96 帧 | `KIT11_M03_simple`，120 帧 | 三套已制作且有穿衣人工接受；p1 两窗口仅一个参与者／录制，M03 原组合道路未验 |

这五套已制作组合覆盖四个目标家族、骑行三个窗口但仅两个发布 participant／录制组、行走两条路径。M01/M04 与 M02/M03 的保守外观风险组分侧；现有发布动作元数据可支持**骑行发布参与者／录制与目标家族隔离的开发性验证**，不支持更强的现实演员或预训练全独立声明。混合行人只能主张 KIT 录制路径隔离，演员独立为 unknown；整套混合指标必须按这一较弱层级报告，不将骑行子集的证据外推给行人子集。M03 换 BEDLAM 衣物的缓存是同源派生，首轮先留诊断；其原生 UV 材质的 A1 道路消费者也尚未接通。

`APD_M01`、`APD_F02`、`bikefit_F02`、已生成的 A1 跨衣发／混合 episode、M03 固定缓存换衣版本保持 `diagnostic_only` 或未解决。任何与正式两侧来源有交集的旧 episode 不纳入拟议正式池。一个 episode 中所有 target 和 occluder 必须同侧；双机、多 rig、邻帧、不同 seed、框扰动、速度／相位／循环衍生沿用同一 `scene_episode_id` 和来源关系。此约束比按单个 target 文件分池更严格。

## 备选：现成场景流程诊断，不作独立验证

可借用已完成的 M01/APD 与 F02/bikefit 道路结果，以及**同来源**旧 M04/KIT425 道路资产的流程证据，规划 `APD_M01＋KIT425_M04` 对 `bikefit_F02＋KIT425_M04` 的内部对照。这里登记的 `KIT425_M04` 是 A2 的 94 帧统一回放，**并非**那份已道路消费的 A1 旧资产，不能把旧 scene 状态转给新 manifest。拟议三套唯一已制作组合涉及两个骑行动作素材、**同一个**行走录制和共享 M04 遮挡者；bikefit 许可 unknown，APD 演员／session unknown。机器校验明确报 `cross_pool_identity`、`cross_pool_motion`，此方案只作内部 process diagnostic，不能当正式 train/val、更不能靠换 rig、人物外观或种子消除交集。它与首选方案的区别在于利用已有相近场景的流程证据，以来源隔离作明确让步。

## 校验与 A1 消费契约

独立 [校验模块](../../src/densekp_retrain/assets/source_pools.py)只读 JSON，无模型或渲染导入；[CLI](../../tools/validate/check_asset_source_pools_a2.py)读取[episode 示例](../../outputs/plans/asset_source_pools_a2_20261010/example_episodes.json)。`scene_episode_id` 表示合成事件，`rig_id` 与 seed 仅表示观测变体；每个 person 明列 role、四组件、原 asset 窗口和声明来源组。所有人共同校验，遮挡者不豁免；A1 必须从冻结 episode 提供 `expected_person_count` 并与真实人物表比对，校验器无法从一份自报 JSON 发现被完全遗漏的人。固定 M03 缓存只可取原连续整数子窗口，头发／人体／motion 必须保持锁定；已制作及接受范围不能由组件 `plan` 或前 60 帧预览推导。

定向 CPU 检查 8 项通过；[validation.json](../../outputs/plans/asset_source_pools_a2_20261010/validation.json)记录首选关系图无内部跨池冲突、备选共享 M04 冲突、有效／无效 episode 例子及 0 个缺失的已列权威引用。A/B/C 组合虽有全 120 帧数值成品，人工只接受前 60 帧；校验器不让这项接受覆盖后 60 帧。`p2_M01` 元数据在首选池合格，`KIT425_M04` 因完整人工范围未定而不合格；这只是拟议规则的资格，不是训练许可、道路、Dataset 或最终 split 批准。已有 [source_split.py](../../src/densekp_retrain/data/source_split.py)服务旧冻结划分；本模块只输出新提案，不改旧默认。A1 交接字段与未满足前置条件见 [a1_handoff.json](../../outputs/plans/asset_source_pools_a2_20261010/a1_handoff.json)。

运行：

```bash
PYTHONPATH=src python -m unittest tests.test_source_pools_a2 -v
PYTHONPATH=src python tools/validate/check_asset_source_pools_a2.py --catalog outputs/plans/asset_source_pools_a2_20261010/source_catalog.json --pools outputs/plans/asset_source_pools_a2_20261010/pool_proposals.json --proposal-id primary_recording_and_character_isolated
PYTHONPATH=src python tools/validate/check_asset_source_pools_a2.py --catalog outputs/plans/asset_source_pools_a2_20261010/source_catalog.json --pools outputs/plans/asset_source_pools_a2_20261010/pool_proposals.json --episode outputs/plans/asset_source_pools_a2_20261010/example_episodes.json --example-id train_cycling_p2
```

给训练总计划／首轮草案的建议更新段落：**T5 现有组合可形成发布参与者和目标家族隔离的骑行候选；混合行人的 KIT11/KIT425 仅录制路径隔离，演员 unknown。M04 完整穿衣人工范围、p2/M01 与 M03/KIT11 道路消费、资源用途及 episode 全员来源冻结后，再评审是否生成首轮训练数据。所有历史混合／跨件场景仍属诊断；T6 未获批准。** 本批没有修改这些共享文档，也没有新增资产制作。最小补齐以验收与消费者接入为主；若最终任务要求演员级独立混合验证，还需可靠的 KIT 演员映射或另一条演员可证的行走录制。

[执行记录](../../outputs/plans/asset_source_pools_a2_20261010/execution_record.json)列出 1 次纯 CPU `composition.plan`、8 项定向测试、权威引用回读和零模型／渲染／GPU／训练调用；各元数据命令观测耗时均不足一秒，累计远低于本批 10 分钟计算上限。阅读与编写时间单列，不将其算作计算。原有权威 manifest、接受记录、注册表、A1 文件和用户 TODO 均未改动。
