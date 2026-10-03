# B4 整体交付检查表

2026-10-03。B4-D 已提交 `57d19df`；B4-P 已完成本批限定验收，未提交，停在整体评审点。
用户原有 docs/TODO.md 保留。没有启动 B5。

| 范围 | 完成内容 | 证据与限制 |
| --- | --- | --- |
| B4-D | 原白名单53项退役；BVH骑行、失败衣物、一次性诊断/迁移等整组闭包处理 | [B4-D记录](PRUNING_B4_DELETE_20261003.md)；原验收与失败制品保持，不承诺重制退役配方 |
| B4-D测试 | 现行裁剪/来源/成像/归一化/指标、数值fixture保留；专用capture和AST约束退役 | 当时CPU及Blender合成/导入记录保留，本批未重跑其数值/Blender流程 |
| B4-P推理 | 新 check_inference_runtime；退役 verify_refactor_inference | [B4-P记录](PRUNING_B4_RUNTIME_20261003.md)，原始权重、两输入、两strict路线、batch/item；前后各8前向，差0，DenseKP无更新 |
| B4-P回调 | 新 check_fixed_validation_runtime；退役 verify_fixed_validation | 公共FixedValidation + 原run_case，success4/resumed2/save_failure2，累计8次小网络更新；358数组及原元数据精确一致 |
| B4-P默认安全 | 标准库打印计划；显式execute才查设备/计算，显式输入/期望/新输出/预算 | 8方法在原仓及无资产副本两cwd通过；其余7回调场景只派发检查，未冒充CUDA验收 |
| 保全 | TODO、数值fixture、公共算法、活动status、资产/权重/历史输出不变 | 受保护小文件字节对照，大权重只读及size/mtime记录；ignored制品不因Git而备份 |

B4累计退役55个文件（B4-D53+B4-P2），逐项以各批白名单和实际删除为准。
B4-P新增两个运行入口与一个安全测试；工具407行替换旧351行，测试166行；没有把减少的整文件数冒充净代码减少。
本批运行代码仅为长期检查入口，模型、公共推理、回调及原fixture实现未重组。

## 当前长期检查在哪里

- 推理：`tools/validate/check_inference_runtime.py`，明确原始归一化输入、独立期望、checkpoint/model-config/backbone、路线/profile/设备/全新output。默认不计算；execute只前向。
- 回调：`tools/validate/check_fixed_validation_runtime.py`，明确source/before/scenario/设备/output/更新预算，resumed需原step2。依赖源码checkout中的tests，不承诺脱离tests的wheel使用。
- 真实DenseKP生命周期：`verify_densekp_model_lifecycle.py` 继续按需保留，本批未运行。
- Blender装配、制品比较与后处理：`verify_blender_assembly.py`、`compare_blender_acceptance.py`、`verify_blender_postprocess.py` 继续保留，本批未运行。
- 两套DS evaluator、现行 regression suite、参数/衣物/groom/crossings、来源/打包/裁剪/成像链保持。
- audit_skin_texture、inspect_bedlam2_groom、prepare_source_trial_config及当前制作生产者保持。

使用方式与计算边界统一由 [COMMANDS](../COMMANDS.md) 导航。
默认展示不读取权重、不查询GPU、不创建输出；真正执行必须显式execute并即时核查设备/进程。
物理设备1本批可用不表示GPU0恢复；不自动换卡/重置/终止进程。
旧capture、旧类路径采集、四入口AST抽取不迁入新工具。历史代码可用 `git show 57d19df:<path>` 查看。

## 未覆盖和后续边界

- 本批未重新验证普通400步/框增强权重、完整数据集或其它evaluator，也未修改其公式与输入规则。
- 七个未实测CUDA回调场景仍可显式选择，但本批只有参数派发覆盖；不声称十场景全部通过。
- 没有SMPL-X/人物制作/外观验收/Blender/渲染；源代码交付检查不是新环境安装验收。
- 小网络8次更新只用于既有回调契约验收，不是DenseKP效果训练。框增强权重仍未获部署替换认可。
- B5历史报告归并、当前知识导航和文档精简尚未实施；历史报告正文保留当时状态。

本批无阻断退役的未解决依赖，可提交B4整体验收；不自动提交commit或继续B5。
