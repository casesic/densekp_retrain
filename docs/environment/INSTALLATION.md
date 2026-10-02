# 开发安装、依赖与运行入口

更新：2026-10-02，P4 安装与入口批次。支持保留完整 checkout 的开发和 editable 安装；不承诺 wheel 携带配置、场景或权重。
本批离线验收见 [实施报告](../refactor/P4_INSTALLATION_20261002.md)。历史 requirements.upstream.txt 保留作上游来源，不能当作当前环境锁文件。

## 双包安装

一个 distribution `densekp-retrain-staging` 提供两个包：根目录的 `densekp` 与 `src/densekp_retrain`。
`setup.py` 只扫描这两个源码目录，构建时不 import 运行模块，不扫描 outputs 或资产；`pyproject.toml` 显式保留两种目录布局。
普通 Python 支持范围仍为 3.10–3.11，本批实际使用 3.10.20。

已有依赖环境的离线开发安装方式（使用你自己的独立虚拟环境解释器）：

```bash
/path/to/venv/bin/python -m pip install --no-index --no-deps --no-build-isolation -e /absolute/path/to/densekp_retrain
```

该命令要求目标环境已有 setuptools>=61 和 wheel，不下载缺失项；依赖不足应先准备本地依赖，再安装。
不要直接在共享训练环境执行升级。`--no-deps` 明确跳过依赖安装，不表示运行时无需依赖。
普通 editable 安装不设置 PYTHONPATH；可从任意 cwd 导入双包，源码仍来自 checkout。
源码、配置必须随 checkout 一起保留，移动 checkout 后应在独立环境重新安装。

本批用了两种隔离程度不同的环境：

- 纯净 venv：不继承 system-site-packages，只有 ensurepip 自带构建基础及从本地复制的 wheel/packaging，验证安装、双包与 stdlib runtime 导入；`python -I` 也通过。
- CPU venv：继承已存在依赖供只读使用，editable 元数据只写入自己的 site-packages。验证实际 Dataset、配置和 CLI；另建只含可交付文件的 checkout 和第三个 venv 做交付验证。

因此已证明安装不依赖当前 cwd 或额外 PYTHONPATH；**没有宣称从空环境离线重装整套 Torch/OpenCV 依赖已经通过**。

## 依赖按职责选择

基础包元数据没有自动运行依赖，保证只查看版本、runtime 路径或在 Blender 使用仓库源码时不强拉 Torch。
可选 extras 是能力分组，不是完整版本锁；本批未执行 extras 的依赖安装。

| 环境/能力 | 声明与典型直接依赖 | 导入和资产边界 |
| --- | --- | --- |
| stdlib 路径/任务组织 | 无 extra；pathlib/json/subprocess | 双包顶层、runtime、rendering.jobs 无模型权重或 bpy 加载 |
| 普通 CPU 数据、相机、指标、图片 | `[data]`：NumPy、SciPy、OpenCV headless、Pillow、PyYAML（配置语法检查） | 不需要 Torch、SMPL-X、Blender；`imaging` 读取 EXR 时另需 `[exr]` |
| Dataset、训练/推理命令与配置 | `[model]`：以上图像依赖、Torch/torchvision、Lightning、timm、einops、YACS、Hydra/OmegaConf、pyrootutils、Rich、Matplotlib、TensorBoard、scikit-image、Albumentations | Dataset 本身依赖 Torch/Lightning 日志；不能把真实 Dataset 称作无 Torch 的轻量模块。模型实例化才读取大权重 |
| 人体重算、衣物绑定 | `[geometry]`：NumPy、SciPy、Torch、SMPL-X、trimesh | 普通 body 顶层不 import smplx；绑定模块属于重型几何环境，模型文件须单独提供 |
| 已有 EXR 后处理 | `[data,exr]`；OpenEXR 3.5.1 | OpenEXR 在读取函数内导入；本地被忽略的 `assets_external/python_deps` 不是可交付安装依赖 |
| Blender 装配/渲染 | Blender 自带 Python、bpy、mathutils、NumPy；源码直接供旧 `--` 脚本引用 | 不向 Blender 安装 `[model]`，不从普通 Python 安装 bpy；Blender Python 的版本不受普通 Python 包安装范围承诺覆盖 |
| 专用历史流程 | pyrender/OpenGL、ffmpeg、外部重建项目等 | 由对应配方文档说明，不因为 editable 成功就认为已安装 |

实测版本完整记录在 [DEPENDENCIES_20261002.json](DEPENDENCIES_20261002.json)。主要版本：NumPy 1.26.4、SciPy 1.15.3、Torch 2.0.0+cu118、torchvision 0.15.1+cu118、Lightning 2.0.2、timm 1.0.12、SMPL-X 0.1.28、Hydra 1.3.2、YACS 0.1.8、Pillow 12.2.0、Albumentations 1.3.1。
Blender 5.1.1 的实测来自 P3，P4 没有重新执行 Blender。

实际共享依赖环境有两项既有 `pip check` 失败，未擅自修复：

- 同时存在 opencv-python 4.13.0.92 和 opencv-python-headless 4.11.0.86，实际 `cv2.__version__` 为 4.13.0；前者的元数据要求 NumPy>=2，与现有 NumPy 1.26.4 冲突。
- rtmlib 0.0.15 缺 onnxruntime；该包不属于此项目的 DenseKP 核心路径。

当前数值验收是在这一既有环境完成，不能把通过运行等同于依赖解析无冲突。
extras 选择单一 headless provider `<4.12` 并保留 `numpy<2`，避免把上述混装写成建议；这是待准备新依赖环境时的约束，并非已验证的新环境锁。
任何 OpenCV/NumPy provider 切换后必须再跑输入与裁剪精确对照，不能拿本批结果代替。

## 承诺的入口与路径

普通 tools 仍是仓库脚本，不安装为 `tools` 包或新增通用命令分发器。
仓库根可使用 `python tools/.../entry.py`；其他 cwd 使用脚本绝对路径。安装后的模型模块支持 `python -m densekp.train_keypoints`。

| 入口 | 本批实际执行 | 真正执行业务时的外部依赖 |
| --- | --- | --- |
| `tools/annotation/inspect_annotation.py` | 两 cwd `--help`；生成的小 NPZ 另做实际检查 | 显式 NPZ；不需要权重 |
| `tools/generate/generate_p0_toy_motion.py` | 两 cwd `--help` | 显式 SMPL-X model-dir、138 点 regressor；本批不生成人体资产 |
| `tools/validate/evaluate_candidate_baseline.py` | 两 cwd `--help` | 权重、ViTPose 初始化、候选索引/图像、来源 manifest |
| `tools/experiments/train_source_isolated_trial.py` | 两 cwd `--help` | 已有来源隔离数据、初始化权重；本批不执行其默认审计或训练 |
| `tools/render/run_temporal_case.py` | 两 cwd `--help` | appearance/geometry、HDRI、Blender；默认命令行为未改 |
| `tools/fisheye/run_fisheye_camera_p0.py` | 两 cwd `--help` | 既有 geometry/衣物/appearance，EXR；重渲染另需 Blender |
| `densekp/train_keypoints.py` 及 `-m densekp.train_keypoints` | 旧脚本 `--help`；安装模块 `--cfg job trainer=cpu` | 真实运行需要明确数据和监督语义；仅加 trainer=cpu 不构成可运行训练配置 |
| `tools/validate/verify_installation_runtime.py` | stdlib/light/data 三模式，真实 CPU Dataset 流程 | 在全新 --output 生成小型验收 PNG/NPZ；不使用已有数据或模型 |

例如，从任意 cwd 验证安装与数据处理（输出目录必须不存在）：

```bash
/path/to/venv/bin/python /absolute/path/to/densekp_retrain/tools/validate/verify_installation_runtime.py --output /tmp/densekp-install-check
/path/to/venv/bin/python -m densekp.train_keypoints --cfg job trainer=cpu
```

上述配置打印不会调用 train。默认 `data=unselected`、空数据列表和未声明监督语义仍然保留；没有启用此前无效或被拒绝的配置。

| 路径类别 | 既有解析基准（本批保持） |
| --- | --- |
| `runtime.repo_root()`、工具 ROOT | 按源码 `__file__` 定位 checkout；不跟随 cwd |
| Hydra 配置目录 | `densekp/train_keypoints.py` 的 pyrootutils 从源码查找 `.git`/pyproject，设置 PROJECT_ROOT 并保留 dotenv 行为；不自动 chdir |
| Hydra data/config 值 | 沿用 YAML 和注册表；配置选择不改路径语义；`paths.output_dir` 依赖 Hydra runtime，单纯 compose 保留未解析字段 |
| Dataset 显式 annotation_path、image_root | 相对 cwd；相对 imgname 拼 image_root，绝对 imgname 覆盖 image_root，空 root 保留；不按 NPZ 所在目录重定位 |
| 新增数据 ID 兼容映射 | 本地 tiny ID 多数按源码根定位；历史 BEDLAM 路径仍有 cwd 相对条目，不统一 |
| candidate 索引、图片引用、用户输出参数 | 各入口原 Path/read_text/resolve 基准保留；跨 cwd 用绝对路径最明确 |
| VITPOSE_BACKBONE | 环境变量 DENSEKP_VITPOSE_BACKBONE 在 constants 导入时读取，否则 `assets_external/vitpose_backbone.pth` 相对 cwd；完整 checkpoint 加载仍执行原初始化读取 |
| appearance manifest | 显式相对路径按 cwd、支持 expanduser；默认路径按 repo_root；其内部字符串不自动按 manifest 目录改写 |
| SceneAssetSpec.blend_path | 保留 JSON 原值，相对 cwd；不是相对 JSON 文件目录；上层 renderer 是否指定 cwd=root 按其原命令 |
| Blender executable | CLI → BLENDER_BIN → PATH；相对候选按调用 cwd 检查；启动任务的 cwd 另由 wrapper 指定 |
| 两条历史 perspective jobs | eye baseline 相对 root，部分纹理/camera 相对 cwd；temporal resolve 时机保持，详见场景索引和独立 fixture |

## 外部资产与交付范围

源码不携带以下资产，安装成功不代表它们存在：

| 资产 | 使用者与要求 |
| --- | --- |
| `assets_external/densekp.ckpt` 与 `vitpose_backbone.pth` | 模型加载；ViTPose 不能因为已有完整 checkpoint 而省略，保持原初始化顺序 |
| `assets/densekp_regressor_R.npy` | 固定 138 语义点 regressor；不能替换成另一版本映射 |
| SMPL-X 男/女/neutral 文件 | 几何入口通过 model-dir 指定，本机外部 pano-stitch 模型目录只读；不复制入包 |
| 场景 `.blend`、贴图、人体/衣物/groom 几何 | [配方 → 资产 → 消费入口](../rendering/SCENE_RECIPE_INDEX.md)；本机默认 outdoor_courtyard.blend 仍缺失，森林/街景代表资产不等于该缺项已修复 |
| datasets/、outputs 中的索引、NPZ、图像、固定预测/验收 before | 真实运行与历史回放的输入；小型测试用 JSON fixture 可独立交付，完整历史验收仍需保留本地输出 |
| `assets_external/python_deps` 的 OpenEXR | 若普通环境未安装 OpenEXR，部分历史入口依赖此本地路径；新环境须显式准备匹配依赖，不打包该目录 |
| 真实骑行标定/图像 | 目标仍为 20260320-riding-dataset3，真实鱼眼 2880×2880、ERP 5760×2880；不混入 031/0311 或合成 ERP 分辨率 |

六份此前被忽略的 `configs/hydra/data/*.yaml` 已解除忽略、保留内容并列入交付；这不改变默认选择或承诺历史 BEDLAM 数据在本机可用。
Python 源码和测试 fixture 的必要性以无资产交付目录实际检查为依据，`tests/fixtures/training_audit_v1.npz` 仍是可选原始记录，测试读取已交付的无损 JSON。

日常检查现统一为 `bash scripts/audit_workspace.sh`（源码/配置和 9 项无资产契约），`--model-contracts` 扩展为 32 项。表面算子改为显式 `--surface-operator`，默认不扫大资产或 outputs。
