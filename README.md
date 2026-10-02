# DenseKP 重训练

DenseKP 对指定人体框预测固定的 138 个语义点，用于跨相机同一人体的点对应。本项目包含数据生成、Dataset、模型训练与应用评价；检测漏检、最终合图和部署代码由外部项目负责。

目前暂停效果优化。P1–P3 已完成并提交于 `aaf08bd`；P4 安装、入口和历史清理已整体通过评审。P4/P5 改动随本次提交交付；P5 最终阅读与交付核查已完成，待评审后决定整个重构收尾。见 [最终交付报告](docs/refactor/P5_READING_DELIVERY_20261002.md)。
最近的框增强权重仍是诊断结果，未获部署替换认可。

## 从这里开始

先读 [一个实际样本的旅程](docs/SAMPLE_WALKTHROUGH.md)，再按 [代码阅读指南](docs/READING_GUIDE.md) 打开主流程。无需先读重构报告。执行前查 [命令与计算边界](docs/COMMANDS.md)，区分只读检查和实际计算。

```text
姿态与人物来源 → 人体/衣物 → 场景与相机 → 图像、138 点、人体框
                                              ↓
                     读取 → 确定裁剪 → 变换图像和标签 → 计算监督
                                              ↓
                          DenseKP → 固定样本评价与报告
```

| 你要做什么 | 推荐入口 |
| --- | --- |
| 看框怎样影响网络输入 | `densekp/core/datasets/dataset_keypoints.py` → `crop_pipeline.py`；完整坐标仍为 256-square，网络内部取中间 192 列 |
| 看标签与 loss | `densekp/core/keypoint_contract.py`、`keypoint_supervision.py`、`keypoint_objective.py`；可靠 GT 的遮挡点继续监督 |
| 检查标签 NPZ | `python tools/annotation/inspect_annotation.py /absolute/path/to/labels.npz` |
| 查模型/训练入口 | `densekp/core/densekp_trainer.py`、`densekp/train_keypoints.py`；有界实验在 `tools/experiments/train_source_isolated_trial.py` |
| 评价固定候选 | `tools/validate/evaluate_candidate_baseline.py` → `densekp_retrain/evaluation/`；先用 `--help` 核对输入 |
| 找哪个配方生成哪个资产 | [场景配方索引](docs/rendering/SCENE_RECIPE_INDEX.md)；多人原生与历史单人路径分别保留 |
| 安装和跨目录运行 | [安装、依赖、路径与资产](docs/environment/INSTALLATION.md) |
| 查兼容代码或旧实验 | [代码归属与历史保留表](docs/refactor/LEGACY_RETENTION_20261002.md)；旧 loss 不等于当前 DenseKP loss |

核心实现放在 `densekp/`（模型及直接数据处理）和 `src/densekp_retrain/`（生成与应用评价）。
`tools/` 负责组合具体实验和制作配方；旧导入包装为历史调用保留，不是新增代码的默认入口。

## 日常检查

在已准备相应 Python 依赖的环境，从仓库根运行：

```bash
bash scripts/audit_workspace.sh
bash scripts/audit_workspace.sh --model-contracts
```

默认检查源码、配置语法和 9 项无资产几何/相机契约；第二条增加裁剪、来源、评价与兼容测试，共 32 项。
这些检查不训练、不加载真实权重、不渲染，也不遍历 outputs 或纹理目录。可用 `PYTHON_BIN=/path/to/python` 指定解释器。
检查本地 regressor 是显式选择：`--surface-operator /absolute/path/to/densekp_regressor_R.npy`。

## 开发安装与数据准备

独立虚拟环境中，已有构建依赖时可离线安装双包：

```bash
python -m pip install --no-index --no-deps --no-build-isolation -e /absolute/path/to/densekp_retrain
python -m densekp.train_keypoints --cfg job trainer=cpu
```

第二条只打印配置，不开始训练。默认仍为 `data=unselected`；训练必须明确数据来源、监督语义和权重。
普通脚本在根目录用 `python tools/.../entry.py`，其他 cwd 用脚本绝对路径；用户相对路径按各入口原规则解析。
依赖分组及现有环境冲突见安装说明，未验证从空环境安装全套依赖，不修改共享环境来消除冲突。

真实权重、ViTPose 初始化、SMPL-X、138 点 regressor、场景/贴图与完整数据均不打包进源码。
真实目标为 `20260320-riding-dataset3`：鱼眼 2880×2880、ERP 5760×2880；合成实验的 1536×768 ERP 不可混用。
数据要求见 [数据契约](docs/data/DATASET_REQUIREMENTS_20260929.md)，配置角色见 [配置索引](configs/README.md)。

## 项目记录

[文档导航](docs/README.md) 用于按主题查找；[任务清单](docs/refactor/IMPLEMENTATION_TASKS_20261001.md) 用于查看重构进度与验证边界。
历史阶段说明、上游来源与早期资产状态保留在 [旧 README 快照](docs/history/README_BEFORE_P4_CLEANUP_20261002.md)，不代表当前状态。
