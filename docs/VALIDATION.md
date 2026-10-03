# 验证方式与已有证据

日常检查、按需真实计算和历史验收是三件不同的事。B5只读核查文档、源码符号、选定JSON与路径，没有运行本页命令、测试或模型。下列“通过”均指已有验收记录及其原范围，不能当作当前环境重新执行通过。

## 日常与按需检查

| 检查层 | 入口/测试 | 依赖、计算及限制 |
| --- | --- | --- |
| 源码/配置 | `scripts/audit_workspace.sh --no-tests`，内部check_workspace显式范围 | 不扫描整个outputs/纹理/权重；静态检查不证明模型可运行 |
| CPU契约 | audit_workspace默认基础契约；显式`--model-contracts`增加右列数据契约 | 按显式测试列表运行；数据裁剪、来源/同场景遮挡者、框规则、成像、指标/config、归一化、公共几何、场景命令、人物准备小输入及suite桩；无真实权重/人体资产 |
| 导入安全 | `tests/test_import_boundaries.py` | 普通Python只检查audit_skin_texture、prepare_procedural_eyes、blender_headless_smoke_test，不借模拟bpy通过 |
| Blender合成/导入 | `tests/test_commuter_intersections_blender.py`、`tests.test_scene_recipes.check_blender_recipe_imports` | 真实Blender中的四项crossings及三个场景脚本受控导入，禁止main/资产读取/制作/渲染；不会由普通CPU测试自动验收 |
| runtime默认安全 | `tests/test_runtime_check_entries.py` | 小输入和桩验证计划/派发；不等于真实CUDA或权重加载 |
| 推理runtime | `tools/validate/check_inference_runtime.py` | 默认标准库计划；显式execute后检查当前手工/Lightning strict路线。独立输入/expected，atol=1e-5、rtol=0；batch/item模式分别保持 |
| 回调runtime | `tools/validate/check_fixed_validation_runtime.py` | 默认计划；显式execute使用当前FixedValidation及tests.fixed_validation_fixture.run_case，有小网络更新预算。依赖源码tests，不承诺纯wheel |
| 真实模型生命周期 | `tools/validate/verify_densekp_model_lifecycle.py` | 会构造/更新DenseKP及保存恢复；不属于日常CPU检查 |
| Blender真实制品 | verify_blender_assembly、compare_blender_acceptance、verify_blender_postprocess | 分别装配/有限渲染、比较、已有EXR后处理；显式资产和新输出，不能因存在工具就默认执行 |

不要用unittest discover扫入训练生命周期；`test_training_audit`、`test_trial_callbacks`及部分边界方法含更新或重型依赖，执行前核查具体方法。保留测试源码不代表授权运行。入口参数、输入输出和设备边界见 [命令](COMMANDS.md)。后续GPU执行须即时检查设备与进程，不沿用旧GPU1可用结论；GPU0异常未在B5检查。

## 独立期望与保护

[fixture说明](../tests/fixtures/README.md) 记录原捕获来源。现行测试保留原字节，不通过新实现重新生成期望。原31个Dataset/11个低层用例包含增强参数、图像/GT/affine/mask、输入变异、异常及RNG；仅测试辅助把具名结果编码为历史result_N。

数值JSON/YAML随源码；原training_audit NPZ与大样本位于被忽略目录，不作为日常无资产依赖。几何fixture是小数组算法契约；不能替代真实SMPL-X模型、groom外观或全制作质量。两份迁移期整段AST fixture和旧capture入口已退役，不再维护无入口的拒绝测试。

## 已有证据索引

以下均相对仓库根，目录中的 execution_manifest.json/比较记录限定各自验收。outputs被忽略，**未随Git交付或自动备份**；缺失时不能凭报告声称可重放。

| 证据根 | 已验证范围与边界 |
| --- | --- |
| `outputs/acceptance/refactor_metrics_config_20261001/` | 指标/config及七样本回读，复用保存预测；不是新模型前向 |
| `outputs/acceptance/refactor_dataset_crop_20261001/` | 七固定样本完整输入和启用增强的独立前后对照；walkthrough_sample.json解释框变化 |
| `outputs/acceptance/refactor_reading_20261002/` | sample_trace.json的真实样本链接/回读证据；B5只读引用，没有重新加载Dataset |
| `outputs/acceptance/pruning_b1_20261002/` | 必要测试辅助解耦和旧包装退役；前测曾误执行4次CPU小网络更新，原scope_deviation记录保留，并非“从未计算” |
| `outputs/acceptance/pruning_b2a_20261003/`、`pruning_b2a_presets_20261003/` | 两cwd配置组合/类型/解析异常对照，父名变化与七退役组拒绝；不证明cpu/gpu/ddp/mps实际设备可运行 |
| `outputs/acceptance/pruning_b2b_crop_migration_20261003/`、`pruning_b2b_retirement_20261003/`、`pruning_b2_symbols_20261003/` | 裁剪/增强与旧fixture、限定符号；BEDLAM仅受控小输入，不是真实tiny验收；geometry verify分支核查不是真实SMPL-X重放 |
| `outputs/acceptance/pruning_b3_1_20261003/`、`pruning_b3_2_20261003/` | 参数/四角色简单衣物及选定真实绑定数组groom迁移；保存reference/rest为共同输入，没有重做模型rest_body或制作main |
| `outputs/acceptance/pruning_b3_3_20261003/` | Blender5.1.1小型crossings/DQS/scan独立数值；后来DQS/scan参考能力随失败制作退役，不说明衣物质量 |
| `outputs/acceptance/pruning_b3_4_20261003/` | suite任务argv/cwd/env/gate/失败顺序的桩对照；没有实际运行evaluator/summary/compare |
| `outputs/acceptance/pruning_b4_delete_20261003/` | 53项删除闭包、当前CPU契约、真实Blender四项crossings及三个场景导入；没有人物资产或渲染 |
| `outputs/acceptance/pruning_b4_runtime_20261003/` | 下述有限真实前向/CUDA回调及默认安全/源码副本检查 |
| `outputs/acceptance/pruning_b5_20261003/` | inventory逐文件知识去向，字节保护、链接/符号/命令静态核查；没有执行计算 |

较早真实验收保留于原位置：`outputs/acceptance/refactor_inference_audit_20261001/`、`refactor_reporting_callbacks_20261001/`、`refactor_geometry_camera_20261002/`、`refactor_imaging_20261002/`、`refactor_blender_20261002/`。真实渲染结论限Blender5.1.1、CPU及代表场景，不能扩展到未测分支。仍保留的 `docs/refactor/densekp_crop_parity_20260928.json`、`densekp_real_acceptance_20260928.json` 是原数值记录，不是新的运行许可。

## B4-P真实计算的准确范围

- 一份原始DenseKP checkpoint，两个完整归一化输入val_baseline_002/001，explicit/lightning两条strict路线，batch/item两个profile。迁移前8次、迁移后8次，**总16次前向，DenseKP更新0**；496个state项的key/shape/dtype及固定预测/模式/RNG按原规则检查，报告最大差0。
- CUDA回调只实测success 4次更新、resumed从独立step2恢复后2次、save_failure 2次，**累计8次替代小网络更新**。358数组及原元数据、优化器/RNG/state_key/异常和部分写出比较；另7场景仅派发桩验证。
- 没有重评普通400步/框增强权重、完整数据集或制作/外观；不产生新训练效果或部署认可。

两runtime默认不读权重、不查GPU、不写目录或起进程；文件存在性校验不证明输入数值有效。源码副本验证使用已有依赖，不等于editable重装或空环境安装。此前离线editable验收及依赖冲突见 [安装](environment/INSTALLATION.md)。

## 如何找历史全文

B1–B4最终已包含在固定基线 `d8bd387`。本轮删除的历史Markdown均可只读取回，例如：

```bash
git show d8bd387:docs/refactor/PRUNING_B4_RUNTIME_20261003.md
git show d8bd387:docs/training/OCCLUDED_SUPERVISION_20260929.md
```

B4-D前源码用 `ada2295`，B4-P旧驱动用 `57d19df`；更早删除的代码按原报告所列提交查阅，不能假设d8bd387还含已删工具。逐文件历史文档去向见本轮inventory；当前知识优先阅读主题页，而非顺序阅读所有阶段报告。
