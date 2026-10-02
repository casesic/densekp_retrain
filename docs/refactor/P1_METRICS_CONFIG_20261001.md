# P0.1 + P1.1 + P1.2 实施记录

日期：2026-10-01。分支 `master`，HEAD 仍为 `a3a5e7bb641bf1dee774d015127fee6b39c28bda`。
本批交付实际代码：完整输入基线、公共评价及配置工厂、活动调用方更新、旧导入兼容和阅读路线。
**本批已完成；全仓库重构未完成。** 后续任务见 [执行清单](IMPLEMENTATION_TASKS_20261001.md)。

## 代码与依赖方向

| 公共定义 | 本批职责 | 历史兼容导出 |
| --- | --- | --- |
| [densekp/core/evaluation.py](../../densekp/core/evaluation.py) | `point_errors`、`evaluation_masks`、`pair_masks`、`prediction_retention`、严格 `summarize` | `tools.validate.occlusion_metrics` |
| [evaluation/candidates.py](../../src/densekp_retrain/evaluation/candidates.py) | 失败状态 `summary`、候选 `aggregate` | `tools.validate.evaluate_candidate_baseline` |
| [datasets/configuration.py](../../densekp/core/datasets/configuration.py) | `audit_cfg`，每次构造独立 YACS 配置 | `tools.validate.validate_bedlam_densekp_tiny` |

上表代码链接从仓库位置阅读；公共模块无 `tools` 导入。活动训练、冻结、评价、对比、标注调用方
已直接导入公共定义，包括 `evaluate_ds_box_control` 原来经另一个 evaluator 间接导入的三个函数。
现有三个测试文件也改为公共导入。旧名称在原模块重导出，CLI 本体继续保留各自流程。

汇总算术、dtype、遍历与归约顺序保留。`summarize` 遇到非有限值仍抛原异常；候选 `summary`
仍返回 `nonfinite_failure`/null 并保留总计数。空组、空行、按点/按人两种分母、可选元数据分组保持。
没有把预测第三值用作权重。新增 docstring 说明数组形状、完整 crop 坐标和 ERP 像素单位。

`summarize_source_occlusion_pairs` 显式加入现有 root/src 启动路径，移除对旧 evaluator 隐式注入 src 的依赖。
本批没有改 Dataset 构造接口、推理、训练回调或优化器。`summarize_saved_candidate_predictions` 的 `write`
和固定框脚本的绘图等其他脚本依赖留到后续职责迁移；这里没有宣称已清除所有脚本间依赖。

## 独立 before 与已通过验证

before 位于 `outputs/acceptance/refactor_metrics_config_20261001/before/`。
在旧函数仍定义于原脚本时先完成采集（UTC 09:15:47），之后才迁代码。保存八份源码快照、
两份实际模型配置、两个 audit 配置 dump、样本来源/行号、七个完整 NPZ、19 组指标 fixture、cases 与汇总。
工具的 capture 检查函数定义模块，并拒绝覆盖目录；新旧名称转发后不允许再冒充旧实现采集。
小型指标 fixture 和配置 dump 同步放入非忽略的 `tests/fixtures/`，供普通回归测试使用。

| 验证 | 实测结果 | 证据 |
| --- | --- | --- |
| 七样本完整回读 | 每样本 29 字段，包括 float32 `[3,256,256]` 归一化输入、中心 `[3,256,192]` 输入、GT、affine、mask；值/shape/dtype 精确一致 | [capture](../../outputs/acceptance/refactor_metrics_config_20261001/before/capture.json)、[after](../../outputs/acceptance/refactor_metrics_config_20261001/after.json) |
| 19 组指标边界 | 空集合、NaN/±inf、严格异常/失败-null、不同分母、mask 边界、两视角、拒绝计数、1536/5760 ERP 周期全部精确一致 | [fixture](../../tests/fixtures/evaluation_contract_v1.json) |
| 七样本应用汇总 | 144 个分组的计数、浮点值、null 字段和 JSON 顺序精确一致 | [before aggregate](../../outputs/acceptance/refactor_metrics_config_20261001/before/aggregate.json)、after |
| 配置 | 未声明/显式 supervision_weight 的 dump 完全相同，配置独立可变、new_allowed 保留 | [契约测试](../../tests/test_evaluation_contract.py) |
| CPU 回归 | 60 项通过、0 失败、0 跳过；原 56 项加 4 项新契约测试（含 19 子用例） | [JSON](../../outputs/acceptance/refactor_metrics_config_20261001/cpu_tests.json)、[日志](../../outputs/acceptance/refactor_metrics_config_20261001/cpu_tests.log) |
| 调用/结构 | 22 个已改旧文件中的保留函数/类，除 import 外 AST 不变；未见活动代码从 tools 导入已迁符号 | [structure](../../outputs/acceptance/refactor_metrics_config_20261001/structure.json) |
| 命令兼容 | candidate、bedlam、DS box、compare box 四入口旧 HEAD/当前 `--help` 输出及退出码一致；在 `/tmp` 且无 PYTHONPATH 验证 | [CLI](../../outputs/acceptance/refactor_metrics_config_20261001/cli_compatibility.json) |
| 独立启动 | paired summary 在 `/tmp`、无 PYTHONPATH 可导入且不再引入 torch | 同上 CLI 记录 |

七样本为 `val_baseline_002/001/004` 与 `challenge_000/001/005/007`，分别覆盖完整人体、
遮挡监督、截断、框中心偏移、收紧和镜头边缘。本批重用保存预测重算误差，**没有重新运行模型**。
所有本批浮点对照要求精确一致，无新增宽松容差；原 CPU 几何测试仍使用其已有容差。
已有 timm 弃用警告不影响测试结果。本地运行环境保持 Python 3.10.20 / NumPy 1.26.4 / Torch 2.0.0+cu118，
`CUDA_VISIBLE_DEVICES=''`；没有 GPU 计算。

## 复现方式

从仓库根目录、现有 camerahmr 环境运行。验证报告需使用新的文件名，不能覆盖旧记录：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  MPLCONFIGDIR=/tmp/densekp-refactor-mpl PYTHONPATH=src:. \
  /home/yy/conda_envs/camerahmr/bin/python -B tools/validate/verify_refactor_metrics_config.py \
  --phase verify --baseline outputs/acceptance/refactor_metrics_config_20261001/before \
  --report outputs/acceptance/refactor_metrics_config_20261001/recheck.json

PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' PYTHONPATH=src:. \
  /home/yy/conda_envs/camerahmr/bin/python -B -m unittest \
  tests.test_evaluation_contract tests.test_candidate_baseline_metrics tests.test_occluded_supervision
```

60 项的完整选择列表保存在 `cpu_tests.json.selected`。没有运行全量 discover：
`LifecycleTests.test_loader_and_training_entrypoint_save_and_resume` 会训练小网络，本批排除；
Blender 测试同样排除。这两项是后续相关路径的必需验收，并非已通过。

## 保留边界与下一批

用户原有 `.gitignore`、`docs/TODO.md` 的 diff 与规划前记录完全相同；历史规划与产物保留。
没有加载真实 checkpoint、训练、渲染、下载、计算 SHA256 或提交 commit。
报告字段、portrait 旋转、mask 取整等行为修复仍独立挂起。

当前没有首批代码交付阻碍。下批继续公共推理和训练审计：先按 P0.2 冻结真实加载/预测基线，
按 P1.4 冻结训练生命周期，再修改对应路径；随后推进 P2 Dataset 和 P3 生成链。
GPU 健康和真实模型/Blender 等价性尚未在本批核实，不能用本批 CPU 通过代替。
