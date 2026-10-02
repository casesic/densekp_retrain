# P4：双包 editable 安装、路径与运行入口

执行日期：2026-10-02。P1–P3 已通过用户整体评审；本批安装与入口实现、验收完成，待评审。
P4 历史清理与 P5 最终导读仍待执行，整个重构没有标记完成。本批未训练、真实模型推理、运行 Blender 或自动提交。

## 实际基线与实施

HEAD `aaf08bd14162f6ad0f5f57bbf3d9c75fab2ec81b`，分支 master；开始时只有用户原有 `docs/TODO.md` 修改。
公共数据包、docs/data 忽略问题已在这个提交修复，现有双包 Python 源码已跟踪；本批不重复修复它们。
新证据目录：[refactor_installation_20261002](../../outputs/acceptance/refactor_installation_20261002/execution_manifest.json)。

修改前保存了 Git 状态、TODO 文本、已跟踪源码/配置/fixture 快照、实际依赖版本、被忽略的交付候选，
以及 7 条 CLI × 两 cwd 的独立旧输出和两份 CPU Dataset 小样本、完整配置、图像/数组 before。
旧执行使用明确 PYTHONPATH 加入双包源码；新执行移除 PYTHONPATH，靠目标虚拟环境中的 editable finder 导入。

实际修改：

- `pyproject.toml` 显式声明两个 package-dir，保留 distribution 名称/版本及 Python 范围；增加 data/model/geometry/exr 依赖分组，关闭自动携带非代码 package-data。
- `setup.py` 只从 src 下的 densekp_retrain 和根 densekp 发现包，不遍历资产或整个 outputs，不 import 运行实现。避免手写子包列表随公共模块变化遗漏。
- `.gitignore` 增加已有 Hydra data YAML 的通用例外及 egg-info 排除。六份旧 YAML 原内容保留，默认 data=unselected 不变；未重新解释历史配置。
- `tools/validate/verify_installation_runtime.py` 提供 stdlib/light/data 三模式，验证真实安装位置、轻量导入、Hydra compose、合成单样本 Dataset、路径和遮挡监督。不注入 sys.path，不实例化 DenseKP。
- [安装与环境说明](../environment/INSTALLATION.md) 与 [实际版本清单](../environment/DEPENDENCIES_20261002.json) 进入非忽略路径；当前任务清单更新 P3 已完成、aaf08bd 已提交、公共数据忽略问题已修复。

未修改任何已有模型、Dataset、损失、公共几何、成像、Blender 或实验入口的 Python 实现；原路径、初始化、导入顺序和异常处理继续使用原代码。

## 独立环境与验收

构建使用 `pip install --no-index --no-deps --no-build-isolation -e <checkout>`。所有 pip 目标都为 `/tmp` 下独立 venv，未安装或升级共享环境依赖，没有联网下载。
纯净 venv 使用 ensurepip 和本地 wheel/packaging 副本；CPU venv 只读继承既有依赖。第三个 venv 指向仅含可交付文件的独立 checkout。
构建日志保留在 outputs，包括 pip 自身常规 wheel 元数据；未另做资产校验和计算。

| 核查 | 已有证据与结果 |
| --- | --- |
| 双包安装 | `install.json` 及 `install_final_*.log`；纯净/CPU/交付目录环境均成功 |
| 隔离导入 | `isolated_import.json`、`installed_metadata.json`；纯净环境 `python -I` 不借 cwd/PYTHONPATH，找到两个源码根，distribution 只提供 densekp 与 densekp_retrain |
| 轻量依赖边界 | `runtime_comparison.json`；stdlib 入口和选定相机/评价/成像/任务模块导入后无 Torch、smplx、bpy、DenseKP 模型模块；Dataset 单独归模型依赖环境 |
| 旧命令 | `before_cli.json` / `after_cli.json`；7 入口 × root/独立 cwd，共 14 次 stdout/stderr/退出码精确一致 |
| 真正仓库外 cwd | `external_runtime_repository`、`external_runtime_delivery` 与 `final_runtime_*`；使用 `/tmp/.../external_cwd`；源码根定位、配置和 Dataset 均成功 |
| 配置 | default Hydra compose 前后 YAML 精确；`-m densekp.train_keypoints --cfg job trainer=cpu` 在 root 和另一 cwd 输出精确，不调用 train |
| CPU 数据流程 | 合成 96×128 RGB、138 点、部分 in_view/visible → 实际 256-square Dataset。每次 9 数组含完整输入、GT、affine、mask、shape、dtype 与 before 精确；72 监督点含 48 遮挡点；相对/绝对路径精确一致 |
| 实际 annotation CLI | `annotation_inspection.txt` / `final_runtime.json`；读取生成的 NPZ，root 和真正仓库外 cwd 结果精确 |
| 无资产交付目录 | `delivery.json`；366 文件（源码、配置、fixture、构建描述），没有 assets、assets_external 或既有 outputs；另装 editable，7 入口 help、真实 Dataset、配置通过 |
| 小型契约测试 | `delivery_cpu_tests.log`：20 项通过，无训练；覆盖 Dataset/crop、来源打包、场景任务、评价合约，读取交付 JSON fixture |
| 共享环境与交付 | `shared_environment_unchanged.json`、`final_audit.json`；记录的已装依赖版本未变，TODO/HEAD 不变；新增文件尚待后续提交 |

修改前的第二 cwd 在验收目录内但不同于仓库根；后续明确补了 `/tmp` 外部 cwd，以及不继承 cwd/PYTHONPATH 的 `-I` 检查，未混淆这三种证据。
比较只在交付目录 CLI 中归一化 checkout 路径；数组、配置值、PNG、计数不放宽容差。
最后排版验证工具、固定 OpenEXR 依赖版本后，三个环境重新安装并复验，记录为 final 文件，保留前序记录。

## 交付与真实限制

六份原先被忽略的配置为 `bedlam_v2_v1.yaml`、`densekp_train.yaml`、`eval.yaml`、`eval_smplx.yaml`、`omni_multiperson_tiny.yaml`、`train.yaml`。
本批使其可交付，并不启用缺失的历史数据或改变默认行为。`tests/fixtures/training_audit_v1.npz` 仍保留本地，测试用无损 JSON 版本，不依赖它。

当前依赖环境 `pip check` 的两项失败如实记录：OpenCV 4.13.0.92 要求 NumPy>=2，而当前为 1.26.4；rtmlib 缺 onnxruntime。
这些是原有共享环境问题，没有擅自处理。实际 OpenCV 导入为 4.13.0，同时还存在 headless 4.11 元数据。
extras 使用单一 headless provider 与 NumPy<2 的约束避免推荐混装；**没有从空环境重新解析/安装全模型依赖，不能将这组 extras 称为已验收 lockfile**。
后续若另建完整依赖环境或替换 OpenCV provider，需要再次执行输入精确回归。

真实权重、ViTPose 初始化、138 点 regressor、SMPL-X、场景/贴图、训练数据及完整历史输出继续是外部输入；选定引用存在性见 `external_assets.json`。
原默认 outdoor_courtyard.blend 仍缺失；本批不制造替代资产。OpenEXR 3.5.1 仅在本地 ignored python_deps 下，部分旧后处理入口依赖它，普通环境须另行准备。
这些不阻塞本批双包和无资产 CPU 流程，但阻止把结果扩大为“新机器全部业务开箱即用”。

## 停止点

本批安装与入口准备完成，等待评审。P4 历史清理随后逐项核查调用与复现用途；P5 再做完整阅读路线验收。
不移动/删除历史输出、不重新评价三权重、不修改外部项目、不提交 commit。回退只涉及本批构建/忽略描述、新工具与文档，不影响 P1–P3 已验收运行实现。
