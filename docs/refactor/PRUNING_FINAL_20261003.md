# DenseKP 精简最终交付

2026-10-03。实际起始HEAD为 `d8bd387b9862f202f15a527430b603101d891854`，已提交B1–B4；起始仅用户docs/TODO.md修改。本轮B5仅文档归并与只读交付核查，成果尚未提交，停在最终评审点。

现在从 [README](../../README.md) → [真实样本](../SAMPLE_WALKTHROUGH.md) → [契约](../CONTRACTS.md)/[代码指南](../READING_GUIDE.md) 开始；日常使用看 [命令](../COMMANDS.md)。数据权重、资产许可、验证及现存限制分别由四个主题页承接，无需先读历史阶段报告。

## 实际成果与保留能力

| 批次 | 已交付结果 | 固定源码基线/边界 |
| --- | --- | --- |
| B1 | 必要测试辅助归tests；退役10个旧loss/训练辅助/导出包装文件 | 8db7ca5包含B1；原数值fixture保留，旧完整对象pickle兼容不再承诺 |
| B2 | loop父配置改名、八预设退役、具名裁剪消费者、旧裁剪/两个评分CLI删除、限定常量/_C/default_config/三个misc函数清理 | 48b0749；累计16个整文件退役（含旧父文件），不是数据格式或BEDLAM能力退役 |
| B3 | 参数/简单衣物/groom/crossings公共化，STYLES留配方；两suite薄入口 | ada2295；4个脚本退役，旧mixed资产/回归消费保留 |
| B4-D | 按完整依赖组退役失败制作、一次性工具及专用测试/AST fixture | 57d19df；53个文件；当前数值fixture、生产者与长期检查保留 |
| B4-P | 两个默认展示的长期runtime入口替代历史驱动 | d8bd387；再退役2驱动；有限真实验收不等于新效果认可 |
| B5 | 当前知识归并、历史Markdown退役、导航与只读交付核查 | 本轮未提交；没有改运行代码、配置、测试或数字期望 |

两个包继续分工：densekp管模型及直接数据/目标，densekp_retrain管生成、来源、应用评价和回调；tools保留具体资产/场景/实验配方。138点、人体衣物共同变换、DS/Omni差异、完整256裁剪/内部192列、遮挡监督、来源隔离和指标分母保持。框增强权重仍未获部署替换认可。

旧tuple裁剪、旧评分CLI、固定mixed制作、BVH/BABEL骑行拟合输入、失败衣物修复/union/settling/polo、指定sweep/riverfront/pyrender、迁移capture/AST等能力已明确退役，不加兼容包装。保留当前制作/成像/数据/训练/评价/回归及按需真实检查；细目见 [KNOWN_BEHAVIOR](../KNOWN_BEHAVIOR.md)。

## 本轮清单及保留文件

先写逐文件知识迁移清单，后修改导航并删除。清单与字节保护证据位于 `outputs/acceptance/pruning_b5_20261003/` 的inventory.json、execution_manifest.json；只保存必要文档/源码/fixture快照，没有复制资产或历史验收库。删除候选均为d8bd387中已跟踪Markdown，不删除未知用户文件。

112份历史报告、重复架构/数据流/旧README导航归并后删除；新增6份主题/交付文档。原许可附件、资产manifest、独立数值JSON报告及所有outputs原位保留，不建立archive目录。

以下5份历史依据暂留：

| 文件 | 保留原因 |
| --- | --- |
| docs/training/BOX_ROBUSTNESS_TRIAL_400_20261001.md | 活动status.active.report指向它；不改活动记录来清理文档 |
| docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md | finalize_adult_candidates仍把此路径写入组合来源报告 |
| docs/appearance/COMMUTER_CLOTHING_REPAIR_20260930.md | 同上；保留失败结论而非恢复制作能力 |
| docs/data/bedlam/BEDLAM_OPERATOR_AUDIT.md | 含独有437/138算子及官方来源统计；当前契约提取必要约定，未重读算子 |
| docs/data/bedlam/BEDLAM_TINY_FRAME_MAPPING.md | 独有帧对应和neutral兼容未决证据；真实tiny输入当前缺失，不能用受控输入替代 |

用户TODO原字节保留；其旧计划/历史路径不是现行导航，不据此重启已退役工作。注册表key、to_lower、专用透视时序/相机压力/BEDLAM工具和两DS evaluator间依赖继续暂缓，均不阻断本轮文档交付。

## 证据与覆盖限制

[VALIDATION](../VALIDATION.md)列出各批原execution_manifest及长期检查。B4-P只验原始一权重、两输入、两加载路线与batch/item，前后16次前向、DenseKP零更新；CUDA仅success/resumed/save_failure共8次小网络更新，其余七场景仅派发。B5没有重跑这些检查。

本轮只读检查本地Markdown链接、源码AST符号、推荐参数、选定status/index/报告和资产文件存在/元数据；不执行推荐命令或测试。不查询GPU、不读权重内容、不安装依赖、不训练/推理/制作/渲染。源码副本旧检查不等于空环境安装；小几何/真实Blender代表场景不等于所有外观通过。

原样本canonical旧路径缺失、mask/外观版本未完全冻结、默认courtyard缺资产、皮肤供体不完全可追溯、非独立真实测试集等限制已写入主题页。无法只读补证的事项继续保留，不补实验或重录期望。保全结果与文档规模见末节。

## 历史全文如何取回

本轮所有删除报告可稳定读取：`git show d8bd387:<原路径>`。例如：

```bash
git show d8bd387:docs/refactor/PRUNING_PLAN_20261002.md
git show d8bd387:docs/refactor/PRUNING_B4_RUNTIME_20261003.md
```

112项原路径、独有信息、主题落点、引用更新与证据路径都记录在本轮inventory；各主题页也给出无需inventory即可理解和使用的关键事实。上述命令只定位已跟踪历史，不能恢复ignored资产/权重/outputs；本轮没有为它们建立备份。旧报告“未提交/下一步”保持为历史语境，不再作为当前任务状态。

## 本轮只读核查与规模

起始128份受管Markdown，当前22份（含原TODO与5份专项历史依据）；新增6、修改10、删除112，净少106文件。统计只含root/docs及配置、数据、fixture说明，不把运行代码或ignored证据算作减少量。

本轮静态核查162个本地链接、61个关键AST符号、13个命令入口参数；315个保护文件字节一致，含用户TODO、运行代码/配置/测试与原始数值fixture。五份暂留报告亦字节不变；删除集合精确等于预先清单，新增Markdown未被忽略。命令静态说明补准Omni的clothing-geometry选项及Blender比较的before/after位置参数，没有改实现或运行它们。

不含TODO的文档从16971行减到2105行，净少14866行；行数包含仍保留的历史附录。精确字节和实际文件集合见execution_manifest.json、document_counts.json；本记录不把旧测试或桩检查冒充本轮执行。当前只剩本批文档成果和原TODO修改，未提交。不存在阻断本批完成的未解决引用；已知资产/复现缺口继续保留，不恢复训练。
