# 指标与配置的独立基线

`evaluation_contract_v1.json`、`audit_*_v1.yaml` 于 2026-10-01、HEAD
`a3a5e7bb641bf1dee774d015127fee6b39c28bda` 的旧实现仍在原脚本中时采集。
捕获先于函数迁移；不得从兼容包装或新实现重新生成期望值来使测试通过。

19 组指标用例覆盖空集合、NaN、正负无穷、严格异常与失败/null 两种策略、
不同点数的按点/按人汇总、空行计数、可选分组元数据、裁剪边界、两视角分组、
预测拒绝与 1536/5760 ERP 周期。输入数组记录 dtype/shape，非有限值用字符串编码。
两个 YAML 固定未声明和显式 `supervision_weight` 的全部配置值。

原采集入口：`tools/validate/verify_refactor_metrics_config.py --phase capture`（capture 在 B1 退役，整个迁移工具在 B4-D 退役；仅供定位 Git 历史）。
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
历史采集代码为 `tools/validate/verify_training_audit.py`（B4-D 已退役），原脚本和采集代码快照位于
`outputs/acceptance/refactor_inference_audit_20261001/before/source_snapshot/`。

仅替换大网络为 CPU 小网络（含 Dropout，使 Torch RNG 可观察），保留实际 DenseKP
loss、AdamW、Lightning hooks、保存和恢复。2 步保存、恢复至 4 步，共 18 个事件和
174 个数组，记录参数/梯度、优化器状态及参数顺序、采样行序、模式、Torch RNG、
NumPy 状态向量和 Python RNG；另有 6 个拒绝梯度用例。
独立探针测试补充完整 NumPy RNG 状态、4096 上限、混合 dtype、detach/clone 和模式约定。
参考环境为 Python 3.10.20、Torch 2.0.0+cu118、Lightning 2.0.2、NumPy 1.26.4；
同环境要求精确一致。本 fixture 不证明真实 DenseKP 预测或 GPU 训练生命周期已通过。

## 历史兼容清理基线

legacy_cleanup_v1.json 来自原 utils/train_utils/losses，仍保留全部原字节，
当前 tests.test_image_normalization 直接使用其中 denormalize 独立期望。
B4-D 明确退役 legacy_import_bodies_v1.json 和 scene_recipe_bodies_v1.json 两份程序全文 AST，
以及只验证旧工具 capture 拒绝的测试；没有放宽 AST 比较或重录期望。
普通 Python 导入安全仍覆盖三个当前工具；三个场景脚本在真实 Blender 中单独检查。
历史 AST 和原代码可从 ada2295 查阅；此前 before/after/pickle 证据保留在 outputs/acceptance/refactor_cleanup_20261002。

来源打包的输入构造位于 tests/source_packing_fixture.py，成像逐制品比较位于 tests/artifact_comparison.py；它们只迁移必要辅助，source_packing_v1.json、imaging_pipeline_v1.json 等独立期望均未重新生成。原B1记录可用 `git show d8bd387:docs/refactor/PRUNING_B1_20261002.md` 查看。

## 当前裁剪调用迁移

dataset_crop_v1.json 的 31 个 Dataset 与 11 个低层用例保持原字节和全部断言。
tests/dataset_crop_fixture.run_lowlevel 现在直调 crop_example，并仅在测试辅助内将 CropResult
具名字段编码成原 result_N 布局；return_trans_flag 用例保留原期望，公共具名接口无需此无效参数。
test_named_outputs_mutations_failures_and_rng 是原低层测试的新名称，不是重新录制期望。

B4-D 移除 run_dataset 的 original 参数和旧 capture/verify CLI；当前 31 个 Dataset / 11 个低层用例、
输入变异、异常与 RNG 对照继续有效。旧七样本独立基线仍在 outputs/acceptance/refactor_dataset_crop_20261001，
本批不删、不重录；日常契约使用原 JSON 与临时小输入，不依赖本机 outputs。

## 几何公共路由与 capture 退役

geometry_camera_v1.json 的独立数值期望、拒绝记录和容差不变。geometry_camera_fixture.TARGETS
仅列公共模块和符号，erp_points 直引 densekp_retrain.camera.erp；四项有效几何契约继续保留。
B4-D 移除 api 的 original 参数和整个历史 geometry 回放工具；公共 TARGETS/合成输入/调用顺序和编码保持。
旧真实 SMPL-X 迁移回放记录继续保留；当前小数组检查不等于真实人物几何重算验收。
scene_jobs_v1.json 的命令、顺序、路径与失败期望继续原样保留。

## Blender 小型相交契约

`tests/test_commuter_intersections_blender.py` 保留公共 crossings 的四项解析断言：穿面、分离、共享边过滤、
coplanar unresolved=None。DQS/scan 参考及后三测试随失败制作能力退役，没有迁入公共运行包。

真实 Blender 场景导入检查由 tests.test_scene_recipes.check_blender_recipe_imports 显式调用，
允许 bpy/mathutils，禁止制作 main、场景访问、资产 IO、写出、扫描和进程；不会由普通 CPU unittest 自动触发。
普通 Python 的 tests.test_import_boundaries 使用三个显式脚本路径，保留原无副作用/RNG/sys.path 约束。
这些检查不读取人物资产、不渲染、不代表衣物质量；历史 B3.3 数值和 AST 证据仍可追溯。

B4-P已由check_inference_runtime与check_fixed_validation_runtime替代历史驱动；后者复用原fixed_validation_fixture。只实测三场景累计8次小网络更新，其余七场景仅派发检查；相关数值/生命周期fixture保持。使用边界见 [VALIDATION](../../docs/VALIDATION.md)。
所有其它原数值 fixture 不修改、不重录、不放宽容差。
