# B2 整体交付检查表

日期：2026-10-03。**B2 授权范围实施完成，建议收尾评审；尚未提交，不进入 B3。**

当前 HEAD 仍是 `8db7ca5ec7583d1749db2e5f33e96abf3eff5a1e`（master）。B2a 两批、B2b 两批及限定符号清理均在工作区；用户 docs/TODO.md 原修改保留，不属于本次成果。当前完整 status、文件/符号清单和最终检查见 [本批执行清单](../../outputs/acceptance/pruning_b2_symbols_20261003/execution_manifest.json)。以下结论沿用各批独立证据；没有为收尾重新跑全量配置、图片、几何或模型。

## 分批完成与证据

| 阶段 | 实际交付 | 已核查证据及边界 |
| --- | --- | --- |
| B2a 父配置 | default_hmr.yaml 原字节迁 loop.yaml，四继承者只改父名；旧父选择明确拒绝 | [报告](PRUNING_B2A_20261003.md) / [清单](../../outputs/acceptance/pruning_b2a_20261003/execution_manifest.json)：两 cwd 的80组合；运行配置一致，仅父来源标识变化。72项 task 解析成功、8项既有缺 GENERAL 异常保留；完整 Hydra help 占位异常保留 |
| B2a 预设 | 删除五 data 组、两 experiment 组和一个 JSON 示例；保留组不变 | [报告](PRUNING_B2A_PRESETS_20261003.md) / [清单](../../outputs/acceptance/pruning_b2a_presets_20261003/execution_manifest.json)：80保留组合相等，原异常不修复；七退役组在两 cwd 共14次 MissingConfigException，无别名回退 |
| B2b 裁剪消费者 | visualize、BEDLAM validator 和测试直引用公共定义，CropResult 具名取值；Dataset capture/original=True 提前拒绝 | [报告](PRUNING_B2B_CROP_MIGRATION_20261003.md) / [清单](../../outputs/acceptance/pruning_b2b_crop_migration_20261003/execution_manifest.json)：七固定样本及增强、31合成Dataset/11低层、可视化图片报告精确对照；真实BEDLAM tiny缺失，仅两行受控输入调用对照 |
| B2b 接口退役 | 公共 geometry 动态路由；删除七个旧裁剪/评分/专用测试文件及 legacy_tuple；geometry capture 提前拒绝 | [报告](PRUNING_B2B_RETIREMENT_20261003.md) / [清单](../../outputs/acceptance/pruning_b2b_retirement_20261003/execution_manifest.json)：现行裁剪和几何数组精确一致；Omni保存数值回归通过、无skip；24项后测、23项无资产副本检查通过。未运行真实SMPL-X回放 |
| B2 限定符号 | 三文件中删除37旧常量名、_C/default_config、三个 misc 辅助函数；四活动常量/注册表/保留函数不变 | [报告](PRUNING_B2_SYMBOLS_20261003.md) / [清单](../../outputs/acceptance/pruning_b2_symbols_20261003/execution_manifest.json)：AST与独立配置结果精确对照；17项前测、后测和无资产副本检查分别通过，无skip。没有训练/模型执行 |

配置组合不等于 CPU/GPU/DDP/MPS 设备运行，更不等于模型可运行；完整 Hydra 容器并未全部解析成功。geometry verify 的中断分支检查与 AST 核查不等于真实人物几何回放。受控 BEDLAM 输入不等于真实数据验收。

## 实际文件与接口退役清单

B2 累计删除16个已跟踪文件；没有新增 archive，也没有删除原 fixture、历史输出或资产。

| 类别 | 删除路径 |
| --- | --- |
| 原父配置（改名） | configs/hydra/trainer/default_hmr.yaml；内容保留于 loop.yaml |
| 五个 data 预设 | configs/hydra/data/train.yaml、densekp_train.yaml、bedlam_v2_v1.yaml、eval.yaml、eval_smplx.yaml |
| 两个 experiment 预设 | configs/hydra/experiment/camerahmr.yaml、bedlam2.yaml |
| JSON 路径示例 | configs/datasets.example.json |
| 旧裁剪实现/包装 | densekp/core/datasets/utils.py、densekp/legacy/crop.py、densekp/legacy/__init__.py |
| 旧评分 CLI | tools/validate/evaluate_omni_densekp.py、tools/validate/evaluate_densekp_npz.py |
| 旧裁剪专用测试/辅助 | tests/test_legacy_cleanup.py、tests/legacy_cleanup_fixture.py |

文件内删除：CropResult.legacy_tuple、旧 geometry 导出身份测试；constants 的37个旧名字（40处赋值）；configs 的 `_C` 构造树（44处赋值）及 default_config；misc 的 instantiate_callbacks、instantiate_loggers、get_metric_value 及其独占 import。[逐符号名单](../../outputs/acceptance/pruning_b2_symbols_20261003/deletion_symbols.json) 与 [限定符号报告](PRUNING_B2_SYMBOLS_20261003.md) 给出全部名字。

明确停止支持：旧 default_hmr 父组名、七个退役预设组名、旧 JSON 示例；get_example/get_example_projverts 旧导入/tuple 以及旧 SMPL/HMR 裁剪能力；旧 Omni 自动造框/ERP 居中/三相机评分和旧注册 ID 评分 CLI；Dataset/geometry 迁移工具的旧 capture 与辅助 original=True；本次删除常量及三个工厂/取指标、旧默认配置树的 Python 接口。

B1 metrics capture 的提前拒绝继续有效。本次没有为退役接口新增别名。旧 Omni 与 NPZ 评分的过滤、分母、报告各异；不宣称当前 candidate evaluator 是等价替代。历史命令通过 Git/报告可追溯，不保证退役接口仍能在当前树执行。

## 保留系统与边界

- `densekp` 继续负责模型、Dataset、裁剪、监督、loss 和相关训练/推理；`densekp_retrain` 负责生成、实验组织与应用评价。双包根保持，自动发现不再包含 densekp.legacy。
- Hydra 保留六个 data 配置（unselected 与五个 tiny）、default/densekp experiment、现行 trainer/loop/paths；默认选择、override 和参数值保持。tiny 不是旧全量配方的等价替代。
- 全部 DATASET_FOLDERS/DATASET_FILES 的 key、值表达式、定义顺序和重复覆盖行为保留。显式 annotation_path/image_root 与旧 ID 解析不变；旧 NPZ 格式和标签权重继续支持，BEDLAM 处理能力并未整体删除。
- 当前具名裁剪保留读图/插值/dtype/增强/RNG/原地变异规则；固定138点、辅助NUM_JOINTS=44、完整256-square与内部192列约定不变，无额外32px偏移。遮挡可靠GT仍按有效域和完整裁剪参与监督；不按RGB黑色判断无效，不把预测第三值默认作为置信度。
- Omni/DS 生成、成像、公共相机/ERP与历史数值回归保留；当前候选、严重遮挡、框偏移及历史固定样例的必要评价入口保留。人物/姿态/衣物、几何和来源隔离规则不变；真实目标仍为20260320-riding-dataset3。
- 模型构造、state_dict、优化器和训练步数未改，原始/普通400步/框增强权重未重评。框增强权重仍未获部署替换认可。

所有原 fixture 保留，legacy_cleanup_v1.json 的归一化期望仍由当前测试消费。有效行为测试未随旧接口清理而丢弃。新增长期检查仅限必要 capture 拒绝；本批限定符号验证全部使用一次性脚本，没有增加“符号不存在”长期测试负担。

## 限制、暂缓与停止点

本轮收尾复核未运行训练、参数更新、真实模型构造/权重加载/推理、SMPL-X实例化、Blender/渲染或GPU；未下载、安装升级依赖或计算SHA256。无资产交付在已有依赖环境内运行源码副本，没有重新做 editable 安装，更不是空环境安装全部依赖。

所有前批失败/限制记录保留，包括父配置上下文校准、首次几何数组验收保护钩子误挡纯公式、缺失真实 BEDLAM 小数据，以及未覆盖真实几何/设备路径；没有放宽原容差、改期望或修复算法来通过精简验收。

注册表 key 删除、to_lower、人物制作链、其余兼容工具与 B3–B5 明确暂缓。portrait/closeup、mask 取整、既有报告/算法差异等行为修复仍单列，不混入 B2。上述范围选择不阻碍已完成 B2 的收尾评审。

没有未解决的仓内删除依赖或新增行为差异。**建议验收 B2 授权范围；保持未提交，停在此处，不自动开始 B3。**
