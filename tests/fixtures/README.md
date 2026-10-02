# 指标与配置的独立基线

`evaluation_contract_v1.json`、`audit_*_v1.yaml` 于 2026-10-01、HEAD
`a3a5e7bb641bf1dee774d015127fee6b39c28bda` 的旧实现仍在原脚本中时采集。
捕获先于函数迁移；不得从兼容包装或新实现重新生成期望值来使测试通过。

19 组指标用例覆盖空集合、NaN、正负无穷、严格异常与失败/null 两种策略、
不同点数的按点/按人汇总、空行计数、可选分组元数据、裁剪边界、两视角分组、
预测拒绝与 1536/5760 ERP 周期。输入数组记录 dtype/shape，非有限值用字符串编码。
两个 YAML 固定未声明和显式 `supervision_weight` 的全部配置值。

采集入口：`tools/validate/verify_refactor_metrics_config.py --phase capture`。
原实现快照、完整七样本 NPZ 和来源记录在
`outputs/acceptance/refactor_metrics_config_20261001/before/`；该目录被忽略。
这些小型 fixture 和测试保留在非忽略路径，不依赖本地大资产即可运行数值契约测试。
完整输入回读需要原有图片、标签与保存预测；它没有重新运行模型。

## 训练审计基线

`training_audit_v1.json` 和原始 NPZ 在 `CheckUpdates`、`probes`
仍独立定义于 `train_cycling_smoke.py` 时采集。此后不从新模块或重导出重新生成期望值。
测试使用 `training_audit_arrays_v1.json`，由原始 NPZ 无损转换，保留 dtype/shape/value。
NPZ 被现有规则忽略，因此正文与数值 JSON 放在非忽略路径，不改 `.gitignore`；
原始 NPZ 仍保留在本地与 outputs，不依赖它运行单元测试。
采集代码为 `tools/validate/verify_training_audit.py`，原脚本和采集代码快照位于
`outputs/acceptance/refactor_inference_audit_20261001/before/source_snapshot/`。

仅替换大网络为 CPU 小网络（含 Dropout，使 Torch RNG 可观察），保留实际 DenseKP
loss、AdamW、Lightning hooks、保存和恢复。2 步保存、恢复至 4 步，共 18 个事件和
174 个数组，记录参数/梯度、优化器状态及参数顺序、采样行序、模式、Torch RNG、
NumPy 状态向量和 Python RNG；另有 6 个拒绝梯度用例。
独立探针测试补充完整 NumPy RNG 状态、4096 上限、混合 dtype、detach/clone 和模式约定。
参考环境为 Python 3.10.20、Torch 2.0.0+cu118、Lightning 2.0.2、NumPy 1.26.4；
同环境要求精确一致。本 fixture 不证明真实 DenseKP 预测或 GPU 训练生命周期已通过。

## 历史兼容清理基线

legacy_cleanup_v1.json 在本轮修改前从原 utils/train_utils/losses 捕获，含数值、RNG 和拒绝行为；legacy_import_bodies_v1.json 是 12 个历史脚本的冻结原执行体 AST。
B1 保留这两份原始 fixture 的全部字节。tests.test_legacy_cleanup 继续检查尚未退役的裁剪值、RNG 和拒绝行为；tests.test_image_normalization 直接使用其中 denormalize 的独立期望。tests.test_import_boundaries 保留12脚本导入无副作用和冻结执行体检查，并验证已退役 capture 在导入数据依赖或写文件前拒绝。旧 loss/train_utils 的接口断言随 B1 退役，不再保证完整对象 pickle。此前 before/after 和 pickle 恢复证据仍保留在 outputs/acceptance/refactor_cleanup_20261002。

来源打包的输入构造位于 tests/source_packing_fixture.py，成像逐制品比较位于 tests/artifact_comparison.py；它们只迁移必要辅助，source_packing_v1.json、imaging_pipeline_v1.json 等独立期望均未重新生成。B1 记录见 docs/refactor/PRUNING_B1_20261002.md。
