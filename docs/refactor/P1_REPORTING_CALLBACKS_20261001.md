# 第三批：评价流程、展示与有界验证回调

日期：2026-10-01。分支 `master`，HEAD `a3a5e7bb641bf1dee774d015127fee6b39c28bda`。
本批完成 **P1.5、P1.6；P1 整体完成，整个仓库重构未完成**。
没有开始 P2 Dataset/裁剪重构，没有修改模型、loss、Dataset 接口或渲染链，没有提交 commit。

证据根目录为 `outputs/acceptance/refactor_reporting_callbacks_20261001/`，以下称 A。
初始工作树、前两批未提交代码及用户修改保存在 `A/starting_state.json`；
原始源码与采集工具在 `A/before/source_snapshot/`。保留所有失败版本与运行输出。
`.gitignore` 和 `docs/TODO.md` 的用户原有 diff 与规划基线逐字一致。

## 实际代码与阅读路线

candidate CLI 现在按以下顺序组织，活动调用方直接使用公共定义：

```text
tools/validate/evaluate_candidate_baseline.main
  → candidate_inputs.prepare_inputs：读参考和冻结索引
  → 入口执行来源隔离验证 → write_selection
  → candidate_runner.load_evaluation_model：seed、精度、配置、严格加载
  → candidate_runner.evaluate_rows
       read_candidate → compare_reference → predict_candidate
       → score_candidate → save_candidate
  → candidate_reporting.write_evaluation_report
       summarize_observations → pairs.collect_pair_metrics
       → visualization.save_contact_sheets → 检查 checkpoint/分母 → JSON
```

| 定义位置 | 本批职责 |
| --- | --- |
| [candidate_inputs.py](../../src/densekp_retrain/evaluation/candidate_inputs.py) | 冻结输入和参考、保留拒绝与复制顺序 |
| [candidate_runner.py](../../src/densekp_retrain/evaluation/candidate_runner.py) | 加载、逐样本核查、预测、评分和制品保存；每个步骤独立命名 |
| [records.py](../../src/densekp_retrain/evaluation/records.py) | 在步骤之间传递具名数据，说明完整输入、GT 和 mask 的形状/坐标 |
| [pairs.py](../../src/densekp_retrain/evaluation/pairs.py) | 成对视角的共同 GT 集合及 mean/worse endpoint；保持两种历史分组键 |
| [candidate_reporting.py](../../src/densekp_retrain/evaluation/candidate_reporting.py) | 按批次汇总、空 action、最差样本排序、报告字段和最终检查 |
| [visualization/keypoints.py](../../src/densekp_retrain/visualization/keypoints.py) | 点图与 contact sheet；保留 JPEG 重读后的拼图路径 |
| [io/json.py](../../src/densekp_retrain/io/json.py) | 严格 JSON 写入；保留评价的 Unicode 原文和训练报告的 ASCII 转义两种策略 |
| [training/trial_callbacks.py](../../src/densekp_retrain/training/trial_callbacks.py) | FixedValidation 调度、FP32 逐行验证、审计、报告及 checkpoint |

`evaluate_candidate_baseline.write/render_panel/summary/aggregate` 保留旧导入；`render_panel`
也保留 `pred=` 关键字签名。`train_source_isolated_trial.FixedValidation/write` 保留兼容导出。
报告恢复入口直接引用公共 `collect_pair_metrics/write_json`，不再依赖 evaluator。
公共模块不反向导入 tools；来源隔离脚本调用仍在 CLI，Dataset 注册表 patch 暂留，等 P2 再改。

训练入口的密集单行语句已展开，main 使用 args、data_root、checkpoint_stat、row_index、
initial_probes、expected_predictions 等名称。按逆向名称映射比较 AST，main、RunAudit、IndexedRows、
dataset、Tee 与失败包装保持原有表达式和执行顺序；没有增加随机调用或更改 400/100、seed930 等配置。

## 独立 before 与保存预测验收

先冻结旧实现，再执行旧 CLI 保存 `A/candidate_before/`，随后才拆分代码。
采集工具运行的是源码快照中的旧 main、画图及写入定义，绝非新旧别名共用新实现的比较。
选择文件 `A/selected_case_ids.json` 记录 11 个已有观测：

- 原始 val 的 `002/001/004`：完整、遮挡、截断样本。
- 框增强 challenge 的 `000/001/005/007` 及对应相机 `024/025/029/031`：四种成对框观测。

仅复用保存的 prediction_raw；实际 Dataset、GT、affine、mask、参考图片检查仍运行。
少量子集索引标为 integration_only，不冒充新的来源隔离划分。
finite、XY NaN/第三值 Inf、第二行错误 shape 三种情形都执行，覆盖缺失配对、空组、失败分母和异常制品。

[最终回放](../../outputs/acceptance/refactor_reporting_callbacks_20261001/candidate_after_final/result.json)：
**37 份 JSON、438 个数组、78 张图片一致**。数组的 key 顺序、dtype、shape、值精确；
图片的文件字节与解码像素均精确；配置文件字节相同。
报告中的指标、分母、case/字段顺序、null、失败状态及来源路径保持。
允许变化的项目只有：

- `selection_frozen.utc`；
- `checkpoint_load.seconds`；
- `report.seconds`、`report.checkpoint.seconds`；
- 本次运行输出根目录的路径前缀。

另用旧源码及旧 writer 单独冻结报告恢复入口 `A/recovery_before/`。
迁移后的 [recovery_after](../../outputs/acceptance/refactor_reporting_callbacks_20261001/recovery_after/result.json)
验证 val 3 行、challenge 8 行，两目录共 17 份 JSON 一致。
恢复入口沿用不含 box_variant 的历史配对键；candidate 保留包含 box_variant 的键，不能悄悄统一。
恢复报告中原有“79 predictions”文字也保持，归入后续行为修复，不借本批修正。

非忽略路径的 [小型 fixture](../../tests/fixtures/candidate_reporting_v1.json) 从独立旧源码的配对代码段
和 render_panel 生成，覆盖有限/非有限/空交集、缺失视角以及可见/遮挡点颜色、非有限预测和完整方形坐标。
[持久测试](../../tests/test_candidate_reporting.py) 不依赖 outputs 或真实模型。

## FixedValidation：真实 GPU 生命周期验收

旧回调仍定义于训练脚本时，先采集 `A/callback_before/`，然后迁移。
运行前检查 GPU 1 和进程，记录 `gpu1.txt`、`processes.txt` 及每轮 `gpu_preflight.txt`。
GPU 1 为 RTX 3090，使用 UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac` 与
`CUDA_DEVICE_ORDER=PCI_BUS_ID`。沙箱内 CUDA 不可访问；获准后在沙箱外运行。
初次沙箱内 `gpu1.txt` 的驱动访问失败记录保留；成功的设备/进程核查在实际 GPU 运行目录内。
没有使用 GPU 0，也没有假定它已经恢复，没有干预已有其他进程。

[GPU fixture](../../tests/fixed_validation_fixture.py) 使用真实 Lightning、AdamW、Dropout、
DenseKP loss 和三份保存的完整输入。小网络以 BF16 mixed 更新，回调逐行 FP32 forward，
真实执行 CUDA RNG 和 autocast。**没有更新真实 DenseKP 参数，也没有重跑三权重推理或效果实验。**

| 情形 | 保持的结果 |
| --- | --- |
| 成功 4 步 | 在 2、4 步验证；样本各按 0/1/2；eval → FP32 → 审计 → 报告 → checkpoint → train |
| 恢复 | 新类加载独立旧 `step_0002/checkpoint.ckpt`，继续到 4；本轮 reports 仅含 4 |
| NaN、错误 shape | 写 failed_row 数组，抛原 ValueError，不跳样本、不生成成功报告 |
| CPU RNG、CUDA RNG 被消耗 | 写预测后抛 `Validation changed RNG state` |
| parameter version、global_step 被改变 | 写预测后抛原参数/更新计数错误；注入计数异常时 global_step=5 的原行为也保留 |
| checkpoint 保存失败 | 预测和 validation.json 已存在、reports 已追加，随后停止，模型留在 eval |
| 目标目录已存在 | mkdir 阶段直接失败，尚未 eval，保留训练模式 |

[最终 GPU 对照](../../outputs/acceptance/refactor_reporting_callbacks_20261001/callback_after_v2/result.json)：
**10 种情形、742 个数组精确一致**，包括参数、优化器 tensor 状态、CPU/CUDA RNG、checkpoint 模型状态和验证预测。
事件序列、更新数、参数 version、优化器参数顺序、样本顺序、FP32/autocast/inference_mode、报告和失败制品清单一致。
JSON 比较只忽略验证报告耗时和输出目录前缀；优化器 betas 在两端统一为 JSON 表示，未丢弃字段。
另从前后两轮的保存轨迹检查每次回调前后状态，
[参数、优化器和 RNG 不变量](../../outputs/acceptance/refactor_reporting_callbacks_20261001/callback_state_invariants.json)
均精确不变，保存 checkpoint 时为 eval，正常返回后为 train；此检查只读已有数组。

类定义路径从 tools 迁到公共包，**Lightning `state_key` 仍为 `FixedValidation`**。
前后 `state_dict()` 都是 `{}`，实际 checkpoint 的 callbacks 字典也是 `{}`；reports 从来没有恢复成历史列表。
旧类名导入及旧 pickle GLOBAL 类路径仍能解析，[接口测试](../../tests/test_trial_callbacks.py) 固定这些事实。
CPU 测试只覆盖兼容与不触发的 0/1/99/101 步；它们不替代上述真实 GPU 验收。
外层 failure.json 包装以 AST 比较保持；完整训练入口发生异常时，traceback 的源码路径/行号自然随迁移变化，
未宣称整个 400 步入口的运行日志字节相同。

## 其他验收与中途记录

[CPU 回归](../../outputs/acceptance/refactor_reporting_callbacks_20261001/cpu_tests.json)：
74 项通过，0 失败、0 跳过。沿用前批 68 项并增加 6 项报告/接口测试；不运行 Blender。
[结构检查](../../outputs/acceptance/refactor_reporting_callbacks_20261001/structure.json)
确认训练入口的计算 AST、模型文件、用户原有 diff，以及公共模块的依赖边界。
[旧命令检查](../../outputs/acceptance/refactor_reporting_callbacks_20261001/cli_compatibility.json)
在 `/tmp`、无 PYTHONPATH 下执行三个入口的 `--help`，文本与退出码一致。
恢复入口因解除模型导入链不再产生旧 timm FutureWarning，其余两个 stderr 相同。

上一批真实推理复核工具已适配新的 candidate 定义位置；提取出的加载/预测片段经过 CPU 小模型执行检查，
本批没有因此重跑三权重验收，上一批 GPU 结果仍保留原记录。

中途失败未删除：`candidate_after/` 记录画图参数重命名漏改导致的 NameError，修复后最终完整回放通过；
`callback_after/` 首次比较器将 optimizer betas 的 tuple 与 JSON list 直接比较误报，保存的 JSON 实际相同。
修复比较器后运行全部情形得到 `callback_after_v2/`，没有改写旧 before 或放宽数值标准。

## 复核命令与后续边界

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  /home/yy/conda_envs/camerahmr/bin/python -B -m unittest \
  tests.test_candidate_reporting tests.test_trial_callbacks
```

已有基线复核使用 `verify_candidate_reporting.py --phase verify --acceptance A --output 新目录`；
加 `--recovery-only` 复核报告恢复入口。A 替换为文首完整路径，所有输出必须为新目录。
真实回调使用 `verify_fixed_validation.py --phase verify --acceptance A --output 新目录`，
先检查 GPU/进程并设置上述设备 UUID；before 是独立旧实现，禁止从兼容导出重新采集覆盖。
GPU 新环境中的非精确结果需先诊断，不可将这批同环境精确比较改为宽松容差。

本批无阻碍实施的未完成验收。下一批从 P2 显式 Dataset 数据源与裁剪路径开始，仍需各自独立 before。
图像旋转、mask 取整、报告文案/字段等行为修复继续单列；来源隔离/打包、生成与 Blender、安装和历史清理仍待后续。
