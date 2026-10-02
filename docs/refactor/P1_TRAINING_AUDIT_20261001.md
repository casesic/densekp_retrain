# 第二批：真实推理与训练审计迁移

日期：2026-10-01；分支 `master`，HEAD `a3a5e7bb641bf1dee774d015127fee6b39c28bda`。
**P0.2、P1.3、P1.4 本批已完成。** 三个真实 checkpoint 的独立 GPU before/after 通过，
CPU 小网络训练审计通过。没有修改 FixedValidation、Dataset 接口或渲染链；整个重构尚未完成。

## 设备核查及中途失败

证据根目录：`outputs/acceptance/refactor_inference_audit_20261001/`，下称 A。
先检查 GPU 与进程：全局 nvidia-smi 对 GPU 0 报 Unknown Error；单独 GPU 1 正常，
RTX 3090 约占用 540/24576 MiB、利用率 0%，仅见已有桌面/IDE/Blender 进程。
没有停止、修改其他进程或修复 GPU 0。

数字选择 `CUDA_VISIBLE_DEVICES=1` 在沙箱内外都失败；随后指定 GPU 1 的确切 UUID：
`GPU-6f046d56-acbe-d49b-774a-9d463e768cac`，配合 `CUDA_DEVICE_ORDER=PCI_BUS_ID`，
沙箱外小张量计算成功。相同 UUID 在沙箱内的基线尝试仍失败，因此真实前后对照均在获准后于沙箱外执行。
**最终 GPU 1 实际推理可用；不能把早先数字选择的失败解释为 GPU 1 硬件故障。**

前面的阻塞记录保留：`gpu_all.txt`、`gpu1.txt`、`processes.txt`、`cuda_probe.json`、
`cuda_probe_outside_sandbox.json`、`gpu_capture/`、`before/inference/`。
成功证据为 [UUID 探针](../../outputs/acceptance/refactor_inference_audit_20261001/cuda_probe_uuid_outside_sandbox.json)
及两个 GPU 运行目录的 `gpu_preflight.txt`；每次运行前重查设备/进程，没有假定 GPU 0 恢复。

## 独立模型基线

真实推理的 before 在公共推理模块写入前完成，位于 `A/before/inference_gpu1/`。
[采集工具](../../tools/validate/verify_refactor_inference.py) 从冻结的旧入口 AST 中提取实际加载、forward、
转换及检查语句；before 不调用新公共实现。原始源码、实际配置、七份完整输入、代码片段、
严格加载 schema、模式、CPU/CUDA RNG 与原始预测均保存。捕获工具源码另存为 `capture_tool.py`。
已存在的 before 和输出目录均拒绝覆盖。

| 权重 | 路径 | checkpoint step |
| --- | --- | ---: |
| 原始 | `assets_external/densekp.ckpt` | 55000 |
| 普通 400 步 | `outputs/experiments/source_isolated_trial_400_gpu1_20261001/step_0400/checkpoint.ckpt` | 400 |
| 框增强 400 步 | `outputs/experiments/box_robustness_400_20261001/training/step_0400/checkpoint.ckpt` | 400 |

先做过三个 checkpoint 的 [CPU 手动严格加载](../../outputs/acceptance/refactor_inference_audit_20261001/before/checkpoint_loads/result.json)，
随后 GPU 基线与 after 都覆盖 **手动 payload、Lightning 两条加载路线**。
每条路线保持其历史构造顺序，包括 DenseKP 构造时读取 backbone 权重；没有为了减少 IO 跳过初始化。
三权重各 496 个 key，checkpoint/model 的顺序、shape、dtype 相同，严格加载无 missing/unexpected。
配置取各权重已有 model_config，保存 size/mtime；未计算哈希。

七输入为首批的 `val_baseline_002/001/004`、`challenge_000/001/005/007`，
完整 float32 `[3,256,256]` 输入保持原值。每个输入分别执行 candidate、scene、occlusion、box、smoke
五种历史预测代码片段。因此每轮共 3×2×7×5 = 210 组固定输入预测，不是全量模型评价。

[GPU before](../../outputs/acceptance/refactor_inference_audit_20261001/gpu_capture_outside_sandbox/result.json) 与
[GPU after](../../outputs/acceptance/refactor_inference_audit_20261001/gpu_verify/result.json)：
**全部预测最大绝对差为 0**，满足 `atol=1e-5, rtol=0`；输入、schema、模式、CPU/CUDA RNG 精确一致。
使用同一 GPU UUID、Torch 2.0.0+cu118、CUDA 11.8、FP32 highest、无 autocast、cudnn TF32 关闭。
NVML 警告仍出现，但未阻止 UUID 指定下的实际 CUDA 分配和 forward。

## 实际代码迁移

| 公共入口 | 责任与保留语义 | 活动调用方 |
| --- | --- | --- |
| [inference.py](../../densekp/core/inference.py) `load_checkpoint_state_dict` | 先读 payload，再构造 DenseKP，再严格加载；返回模型与原报告需要的 metadata；不改模式/设备 | candidate evaluator |
| 同文件 `load_lightning_checkpoint` | 保留 Lightning 构造/hooks 和 CPU strict=True 路径；不替换为手动加载 | DS scene、occlusion、box evaluator |
| 同文件 `predict_batch` | NumPy 完整 batch → 原始 tensor；inference_mode，不改变 train/eval、precision 或 autocast，不解释第三值 | 四个评价入口 |
| 同文件 `predict_item` | 保留 smoke 的 eval 且不恢复模式、有限 138×3 检查、float32 `[1,138,3]` 返回与原异常 | smoke/source trial 直接使用；smoke 的旧 predict 名兼容导出 |
| [training_audit.py](../../densekp/core/training_audit.py) | `CheckUpdates`、`probes` 的原算法与状态行为 | smoke/source trial 直接导入；smoke 兼容导出 |

公共代码没有反向导入 tools。四个评价入口各自的 shape/nonfinite 拒绝、失败数组保存和异常文本保留；
candidate 继续记录非有限输出并保留指标分母，不能改为统一拒绝。精度和 mode 切换仍由原调用方控制。
旧模型类和 checkpoint key 不移动。训练入口自己的模型加载/优化器/FixedValidation 流程没有迁移。

CheckUpdates 仍要求 backbone/head 各至少存在一个梯度，检查所有存在的梯度；不擅自要求每个参数都有梯度。
`attempts` 记录 pre-step global_step，是当前运行的更新尝试，不是完成计数或恢复历史；默认空 callback state_dict 不变。
probes 仍按 backbone/head 顺序取首个参数最多 4096 个 float32 CPU 标量，detach/clone，不修改模式或 RNG。

## CPU 审计与兼容验收

先在 CheckUpdates/probes 独立定义于 smoke 时采集，再迁源码。before 为 `A/before/audit_contract/`；
原代码与 harness 快照为 `A/before/source_snapshot/`。小网络含 Dropout，使 Torch RNG 的变化可观察，
其余使用实际 DenseKP loss、AdamW、Lightning hooks、保存与恢复。

| 检查 | 已实测结果 | 证据 |
| --- | --- | --- |
| 独立生命周期 | 2 步保存、恢复至 4 步；18 个事件、174 数组精确一致 | [before](../../outputs/acceptance/refactor_inference_audit_20261001/cpu_capture/result.json)、[after](../../outputs/acceptance/refactor_inference_audit_20261001/cpu_verify/result.json) |
| 更新语义 | 参数/梯度/优化器状态、参数顺序、更新计数、采样行序、train/val 模式、Torch RNG、NumPy 状态向量和 Python RNG 一致 | `before/audit_contract/` 与 `cpu_verify/observed.*` |
| 失败路径 | backbone/head 各 missing/NaN/Inf 六例，异常类型/文本/attempts 保持，参数不变 | 同上 metadata |
| 探针边界 | 4096 截断、混合 dtype→float32、独立 clone、detach、两种模式和完整 Python/NumPy/Torch CPU RNG 不变 | [审计测试](../../tests/test_training_audit.py) |
| 推理策略 | 保留整幅输入、raw dtype/非有限输出、模式与 RNG；item 的 eval 副作用及原拒绝；两加载路线严格性和调用顺序 | [推理测试](../../tests/test_inference.py) |
| CPU 回归 | **68 项通过，0 失败、0 跳过**；包含原有训练入口保存/恢复及新审计/推理测试 | [JSON](../../outputs/acceptance/refactor_inference_audit_20261001/cpu_tests_final.json)、[日志](../../outputs/acceptance/refactor_inference_audit_20261001/cpu_tests_final.log) |
| 结构 | 评价入口转换/异常代码片段除 forward 调用外 AST 相同；RunAudit、FixedValidation、trial main 不变；模型文件不变 | [最终结构核查](../../outputs/acceptance/refactor_inference_audit_20261001/structure_final.json) |
| 旧命令 | 四个评价和两个训练入口，在 `/tmp`、无 PYTHONPATH 下的 `--help` 与 before 输出/退出码一致 | [最终 CLI 检查](../../outputs/acceptance/refactor_inference_audit_20261001/cli_compatibility_final.json) |

CPU 数值要求精确一致，没有放宽容差。参考环境为 Python 3.10.20、Torch 2.0.0+cu118、
Lightning 2.0.2、NumPy 1.26.4；CPU 测试设置 `CUDA_VISIBLE_DEVICES=''`。
训练审计期望值保存在非忽略的 `tests/fixtures/training_audit_v1.json` 与 `training_audit_arrays_v1.json`；
数组 JSON 由原始 NPZ 无损转换，保留 dtype/shape/value，未修改 `.gitignore`。
原始 NPZ、本批早期 64 项记录和沙箱失败产物都保留。

## 复现与剩余范围

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/densekp-refactor-mpl PYTHONPATH=src:. \
  /home/yy/conda_envs/camerahmr/bin/python -B -m unittest tests.test_training_audit tests.test_inference
```

GPU 复核须先检查 GPU/进程，在具备 CUDA 权限的环境设置上述 UUID 与 PCI_BUS_ID，运行
`tools/validate/verify_refactor_inference.py --phase verify`，参数为：
`--acceptance A --baseline A/before/inference_gpu1 --output 新目录`（A 替换为文首实际路径）。
CPU 独立对照工具为 `verify_training_audit.py --phase verify --baseline A/before/audit_contract --output 新目录`。
禁止覆盖 before 或从新公共实现重建期望值。

当前批次没有未完成的验收阻碍；GPU 0 仍未恢复，不更改历史命令的默认设备语义。
下一批为 P1.5 应用报告/画图，随后再审查 P1.6 FixedValidation；Dataset、生成链、包装/安装和历史清理仍待后续。
没有进行 DenseKP 效果训练、新 400 步实验、渲染、下载或 commit；用户修改和历史产物保留。
