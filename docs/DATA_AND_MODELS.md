# 数据、权重与活动版本

当前目标是 `20260320-riding-dataset3` 的跨相机人体语义点对应。检测漏检和最终合图不在本项目范围；检测框引起的偏心、截断、邻人干扰在范围内。框增强改善部分合成鲁棒性，但**未获得部署替换认可**，不自动替换原始模型。

以下路径相对仓库根。outputs 和外部资产被 Git 忽略，不随源码交付；本页依据选定索引、status 和 comparison JSON，不重新评价模型。

## 哪些数量代表什么

| 数据根与索引 | 行数/用途 | 不能据此推断 |
| --- | --- | --- |
| `outputs/experiments/source_isolated_pilot_20260930/train_index.json` | 198 个训练父样本 | 不是198个人或独立姿态 |
| 同目录 `val_index.json` | 79 个验证父样本 | 不是独立真实测试集 |
| `outputs/experiments/box_robustness_400_20261001/data_v2/train_index.json` | 400 次训练暴露；参考/中心/尺度/单侧框配额200/100/50/50 | 不是400个独立父样本，也不等同训练步数概念 |
| 同目录 `val_index.json` | 18 个父观测的143个挑战框；144个提案中1个因单侧支持不足拒绝 | 不是143个独立人物 |
| `outputs/experiments/cycling_ten_adults_candidates_20260930/candidate_index.json` | 24 行旧骑行回归，另排除3行 | integration_only，不属于正式来源隔离划分 |
| `outputs/experiments/mixed_riders_pedestrians_20260930/candidate_index.json` | 27 行旧骑行/行人回归，5个视图无框 | 固定 mixed 制作 CLI 已退役，已有制品仍可消费 |

`source_isolated_pilot_20260930/split_v2.json` 定义9角色的6 train / 3 val：

- train：cycling_F01_v2、F03_v1、F05_v2、M01_v2、M04_v1、M05_v1（均带 cycling_ 前缀）。
- val：cycling_F02_v3、M02_v3、M03_v1。F04 全家族排除，不能只换外观版本重新纳入。
- 训练来源为 COCO 292226/person1330527 与 KIT 314/bend_left01；验证为 COCO 306800/person225160 与 KIT 425/walking_medium05。

实际来源检查覆盖同场景所有人物和遮挡者，包含人物、姿态、外观衍生关系、同图和相邻动作帧；规则见 [契约](CONTRACTS.md)。已记录来源交集为空不等于皮肤供体或模型预训练资料独立。森林/街景共享，未设置独立场景测试集。4个姿态来源组、10个离散源姿态及36个组合检查不能写成36个独立动作。独立照片不拼接成共享 shape 的连续 motion。

## 活动版本的唯一入口

读取 `outputs/experiments/box_robustness_400_20261001/status.json` 的 `active`。当前已完成并停止、实际400次更新，部署接受为 false。

| active 字段 | 相对该实验目录的值（report 除外） |
| --- | --- |
| data | `data_v2` |
| crop_context | `actual_crop_context_v2.json` |
| indices_with_context | `data_context_indices.json` |
| training | `training` |
| evaluation | `evaluation_v2` |
| case_metadata | reviewed_case.json alongside predictions; original case.json retained |
| comparison | `comparison_final/report.json` |
| audit | `final_audit_v2.json` |
| report | 仓库中的 [框增强实验原报告](training/BOX_ROBUSTNESS_TRIAL_400_20261001.md) |

status 中 `no_commit` 是实验当时的记录，不表示当前源码未提交。不要自动选“最新”目录。早期 `data`、空的 `evaluation`、旧 crop_context/final_audit 都保留，但不是活动版本。最初上下文 mask 使用 int16 warp 的错误由 v2 的 float32 核查替代；部分展示缓存、邻人字段也经复核修订。保存预测、原 case 和失败制品未覆写，不能把不同版本统计拼成一份结果。

## 三份权重与结论

| 标记 | checkpoint | 配置/结果 |
| --- | --- | --- |
| A 原始 | `assets_external/densekp.ckpt` | 构造仍需 ViTPose 初始化资产 `assets_external/vitpose_backbone.pth` |
| B 普通400步 | `outputs/experiments/source_isolated_trial_400_gpu1_20261001/step_0400/checkpoint.ckpt` | 同根 `model_config.yaml`、`comparison/report.json` |
| C 框增强400步 | `outputs/experiments/box_robustness_400_20261001/training/step_0400/checkpoint.ckpt` | training/model_config.yaml；活动 comparison_final/report.json |

B、C 是有限小试：AdamW、lr=1e-6、weight_decay=1e-4、batch1/accum1、seed930、bf16训练/FP32验证、with_var=False；这些是历史实验条件，不是推荐追加训练。B 的四项效果门槛只过1项。C 过3项，完整对照仍失败。

| C 的原四门槛（完整256裁剪像素） | A | B | C | C 判定 |
| --- | ---: | ---: | ---: | --- |
| rear | 19.4202 | 18.4844 | 17.3960 | 通过（上限18.449216） |
| occluded | 12.3647 | 11.9685 | 11.3533 | 通过（11.746464） |
| complete | 3.6567 | 4.2387 | 4.3527 | 失败（4.156706） |
| visible | 4.8558 | 4.8949 | 5.1212 | 通过（5.355821） |

143框挑战的共同GT ERP像素均值 A/B/C 约13.9247/14.3309/11.0049；相对参考框漂移约12.2918/13.5381/9.5787；严重失败框43/48/35。共同GT计12532次点暴露，不能与18个父观测的1568个唯一共同点混为同一分母。原79行验证 supervised 集为8974点（其中4703遮挡），也不等于 all_in_view 的9193点。

旧24行骑行回归裁剪均值约6.8913/6.5623/7.0746，旧27行混合回归约7.0253/6.1523/6.4192。C 的框挑战改善与旧骑行退化同时成立。历史22.5 ERP像素移框例改善仍明显劣于参考框；它不是实际 YOLO 误差分布。已有真实帧核查仅涉及02214的一帧5框，没有完整真实GT或跨帧关联，不能推导部署效果。

失败的 `outputs/experiments/source_isolated_trial_400_20261001/` 是GPU0中断版本：262个更新前记录至少对应261次完成更新，最后一步不确定。它不是 B 的400步结果；100/200步权重或择优300步不能替代上述固定对照。

## 历史数据仍可读取

旧 NPZ 第三值继续按其原标签权重解释；新标签的显式监督策略另见 [契约](CONTRACTS.md)。旧 Python tuple/评分 CLI 退役不等于删除数据格式。早期裸身、integration_only、失败版本和正式 train/val 不混用。精简没有重算实验结果或改变评价门槛。

资产和许可见 [ASSETS](ASSETS.md)；按需检查和已有证据见 [VALIDATION](VALIDATION.md)。上述小型索引与报告在本轮只读核实；权重只核查路径及元数据，没有读取内容。

更早的裸身双照片SMPL-X小试曾执行8步全模型更新：森林监督均值8.26→18.08 px，街景7.52→16.96 px，流程通过但定位效果失败。它与后来的穿衣、来源隔离及框增强数据不是同一资格或对照；原制品在 `outputs/experiments/cycling_smplx_20260929/` 及对应训练记录，完整路径见 `git show d8bd387:docs/motion/cycling/CYCLING_SMPLX_PILOT_20260929.md`。不将该失败从当前结论中抹去，也不继续使用其退役制作命令。
