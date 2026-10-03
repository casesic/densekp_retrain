# B2 最后一批：限定符号清理

日期：2026-10-03。**实施及定向 CPU 验收完成，未提交；停在 B2 整体验收点。**

[执行清单](../../outputs/acceptance/pruning_b2_symbols_20261003/execution_manifest.json) · [编辑前逐符号核查](../../outputs/acceptance/pruning_b2_symbols_20261003/symbol_audit_before.json) · [冻结删除名单](../../outputs/acceptance/pruning_b2_symbols_20261003/deletion_symbols.json) · [精确对照](../../outputs/acceptance/pruning_b2_symbols_20261003/comparison.json) · [B2 整体检查表](PRUNING_B2_COMPLETION_20261003.md)

## 起始状态与实际范围

实际 HEAD `8db7ca5ec7583d1749db2e5f33e96abf3eff5a1e`，分支 master。已有 B2a/B2b 未提交成果及用户 `docs/TODO.md` 修改；完整 status 在 starting_state.json。没有重置、覆盖或重复实施前批。

运行代码仅修改以下三个文件，不删除整文件、不迁移消费者：

| 文件 | 实际删除 | 原样保留 |
| --- | --- | --- |
| densekp/core/constants.py | 37 个已规划旧常量名，对应40处赋值（含3处重复定义） | VITPOSE_BACKBONE、TRANSFORMER_DECODER、NUM_DENSEKP_SMPL=138、NUM_JOINTS=44 的原表达式及 os import |
| densekp/core/configs/__init__.py | `_C` 构造树44处赋值及 default_config | dataset_config、to_lower、所有必要 imports、路径基准和完整注册表 |
| densekp/core/utils/misc.py | instantiate_callbacks、instantiate_loggers、get_metric_value；独占 hydra/List/Callback/Logger imports | task_wrapper、extras、save_file、close_loggers、log_hyperparameters、装饰器及 log 初始化表达式 |

删除常量的完整名字如下（不删除 YAML 同名字段）：

- 路径：CHECKPOINT_PATH、CHECKPOINT_PATH_SMPLX、CAM_MODEL_CKPT、DENSEKP_CKPT、SMPL_MEAN_PARAMS_FILE、SMPL_MODEL_PATH、SMPLX_MODEL_PATH、DETECTRON_CKPT、DETECTRON_CFG、SMPL_to_J19、JOINT_REGRESSOR_TRAIN_EXTRA、SMPLX2SMPL、DOWNSAMPLE_MAT、REGRESSOR_H36M、SMPLX_MODEL_DIR、SMPL_MODEL_DIR。
- 尺寸/参数：IMAGE_SIZE、IMAGE_MEAN、IMAGE_STD、NUM_POSE_PARAMS、NUM_BETAS、NUM_BETAS_SMPLX、NUM_PARAMS_SMPL、NUM_PARAMS_SMPLX、NUM_JOINTS_SMPLX、NUM_PARAMS_SMPLX_HANDS。
- 映射/均值：H36M_TO_J17、H36M_TO_J14、body_permutation、extra_permutation、smpl_to_openpose、FLIP_KEYPOINT_PERMUTATION、JOINT_NAMES、JOINT_IDS、JOINT_MAP、MANO_LEFT_MEAN_6D、MANO_RIGHT_MEAN_6D。

仅更新 docs/READING_GUIDE.md 的当前导航，新增本报告与 B2 完成检查表。未新增长期“符号不存在”测试。历史报告正文不变。

## 依赖依据

按原 inventory 限定符号逐个核查，先写 deletion_symbols.json，再采集 before 和编辑。检查直接、相对、通配 import、模块别名与属性、getattr/import_module 路由、配置字符串/命令及当前文档；输出保留同名非依赖记录，不把字符串匹配当调用。

实际活动导入：densekp_trainer 使用 VITPOSE_BACKBONE，smpl_head_keypoints 使用 TRANSFORMER_DECODER/NUM_DENSEKP_SMPL，dataset_keypoints 使用 NUM_JOINTS。train_keypoints、smoke/lifecycle 工具和边界测试使用 dataset_config；loading 与 Dataset 测试使用注册表；训练入口使用 task_wrapper/log_hyperparameters。模型及生命周期入口只读源码，未执行。

没有仓内消费者使用删除符号。旧常量之间的 JOINT_NAMES/permutation/H36M 引用属于本次一起删除的内部闭包。cfg.MODEL.IMAGE_SIZE、同名 YAML 值等是配置字段；不是 constants 属性。动态测试路由、训练审计/固定验证/安装检查的模块表没有这些退役符号；普通报告字段不映射为模块。

没有发现需要超范围迁移才能删除的符号。未知外部旧 Python 导入按已授权边界停止支持；不新增别名或兼容包装。DATASET_FOLDERS/DATASET_FILES 和 to_lower 的暂缓是明确范围选择，不阻碍本次验收。

## 独立基线与验证

新验收目录 `outputs/acceptance/pruning_b2_symbols_20261003/` 保存482份现有源码、配置、说明和原 fixture 字节快照，以及四批具名 execution_manifest 原字节。before_gate 在源文件仍未编辑时通过。没有扫描资产或整个 outputs，没有读取权重或七真实样本图片。

现有解释器 `/home/yy/conda_envs/camerahmr/bin/python`；一次性 acceptance.py 明确列出17个方法 ID。阻断 Module/Trainer/Optimizer 构造、Trainer.fit、optimizer.step、torch.load、CUDA 初始化、SMPL-X/Blender 导入及 Hydra instantiate；禁止训练 logger 构造，写出限新验收目录。misc 导入保留其普通 Python 日志对象，未调用 Lightning logger 工厂、任务装饰器或其写出函数。

| 检查 | 结果 |
| --- | --- |
| 四常量 | 定义 AST/表达式和 import 相同；环境变量未设置、空字符串、显式相对路径三种情况下的值、类型、字典顺序全部相同；未改变路径解释 |
| configs | 排除获准删除的树/函数后，全模块 AST 相同；dataset_config 返回类型、dump、冻结状态、new_allowed、冻结赋值异常及独立实例行为完全一致 |
| 注册表 | DATASET_FOLDERS 共151次键定义/150个最终键，含 coco-val-smpl 重复覆盖；DATASET_FILES 两表分别11/138个键；全部 key、值表达式、出现顺序、运行值与类型精确一致 |
| misc | 五个保留函数（含嵌套函数、装饰器、异常/finally/日志顺序）AST 相同；仅删独占 imports 并更新模块说明。before/after 实际导入成功，未调用这些函数 |
| 配置/评价 fixture | 原两个 audit YAML 和19组指标的值、类型、异常及独立配置实例检查通过；未重录期望 |
| 当前裁剪 | 原31个合成 Dataset、11个低层 fixture 用例仍通过，另选两项图像/标签共同变换与 flip 拒绝检查；没有重跑七样本完整回放 |
| capture 拒绝 | 前批 Dataset/geometry/metrics 拒绝检查继续通过；geometry verify 仅测试中断后的分支可达性，没有真实几何回放 |
| 测试结果 | 17项前测、17项后测全部通过，无 skip；只运行列明方法，无 unittest discover、Objective/Lifecycle 或整模块训练测试 |
| 无资产交付 | 新建360文件源码副本，无 assets/outputs/权重；双包及三个目标模块从副本导入，同一17方法通过，无 skip |

before/runtime.json 与 after/runtime.json 逐字节相同，无过滤字段、无容差调整。一次性 AST 检查、引用检查及驱动均只在新验收目录。

源码副本使用现有依赖，**未做 editable 重装或空环境安装**。misc 静态/导入检查不证明完整训练生命周期；常量检查不证明模型可运行。未运行模型、真实权重、推理、任何参数更新、SMPL-X 实例化、渲染、GPU、下载或依赖安装升级。未计算 SHA256、未提交。

## 保全与结论

最终478份范围外保护文件字节不变，包括15份原 fixture、所有 B2a 配置、前批代码与说明（仅本次获准更新的阅读指南除外）、TODO。四批原验收清单字节不变；没有新增整文件删除。保留源码/配置无已删符号依赖，两份新报告不被忽略。

没有未解决依赖或行为差异。本批限定符号清理已完成，可结合 B2 整体检查表进行收尾评审；注册表 key、to_lower 和 B3–B5 不在本批继续处理。所有成果保持未提交。
