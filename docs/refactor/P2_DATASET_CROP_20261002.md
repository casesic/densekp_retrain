# P2.1/P2.2：显式数据源与裁剪流程

完成日期：2026-10-02（Asia/Shanghai）。P1 已通过用户评审；本批实现和本地验收完成，供评审。
只完成 P2.1/P2.2；P2.3–P2.5、生成链及安装整理仍待执行。

## 1. 基线与边界

分支 `master`，HEAD `a3a5e7bb641bf1dee774d015127fee6b39c28bda` 未变。
在已有 P1 未提交实现上继续，保留用户 `.gitignore`、`docs/TODO.md` 修改和全部历史产物。
本批从 10 月 1 日开始，故证据目录仍为
[refactor_dataset_crop_20261001](../../outputs/acceptance/refactor_dataset_crop_20261001/)。
开始时完整 diff、状态和文件快照分别见 `starting_state.json`、`before/source_snapshot/`。

先在旧 Dataset / utils 仍独立执行时捕获 before，再修改生产代码。
`before/data/` 保存 45 用例、1407 个数组；`before/lowlevel/` 保存另外 11 个旧低层接口用例。
旧代码没有显式路径参数，显式路径的独立 oracle 是**把相同 annotation/root 绑定到旧 ID 后运行旧代码**；
不是迁移后让新旧接口都转发到同一实现来生成期望值。
`verify_dataset_crop.py --phase capture` 检查旧签名，拒绝用新 Dataset 重建 before。

本批仅 CPU 数据处理、保存预测回放和已有 CPU 回归；没有 GPU 运算、真实模型推理、
DenseKP 参数更新、效果训练、新 400 步实验、渲染、资产下载、commit/reset/clean。
80 项回归中包含已有小型 CPU 网络保存恢复测试，不是 DenseKP 效果训练。

## 2. 当前入口与职责

| 文件 | 本批职责 |
| --- | --- |
| [dataset_keypoints.py](../../densekp/core/datasets/dataset_keypoints.py) | Dataset 主线：取框和标签 → 读图 → 确定裁剪 → 变换图像/标签 → 重算监督 → 返回原字典 |
| [loading.py](../../densekp/core/datasets/loading.py) | 显式路径和旧 ID 解析，选取 NPZ 字段，图片路径拼接 |
| [augmentation.py](../../densekp/core/datasets/augmentation.py) | `Augmentation` 具名采样结果，以及原有极端人体裁剪规则 |
| [crop_pipeline.py](../../densekp/core/datasets/crop_pipeline.py) | `CropBox`、`CropPlan`、`CropResult`；`determine_crop` 和 `transform_crop` 组合各阶段 |
| [image_processing.py](../../densekp/core/datasets/image_processing.py) | 读图、原预处理、OpenCV 裁剪、颜色归一化、标签 affine 变换 |
| [utils.py](../../densekp/core/datasets/utils.py) | 旧 `do_augmentation` tuple、`get_example_projverts` 十项 tuple 及辅助函数导出兼容；其他历史路径保留 |

`DatasetKeypoints(cfg, dataset, use_augmentation, is_train)` 的旧位置参数仍有效。
新增 keyword-only `annotation_path`、`image_root`；两者均给出时无需注册 ID，省略名称默认 `explicit`。
每个未给出的路径分别查询旧注册表；`image_root=""` 有效，不等同于缺省。
显式 Path 转为文件系统字符串，不规范化到 NPZ 目录或其他绝对根目录。
相对 imgname 用原 `os.path.join` 规则，绝对 imgname 覆盖根目录。
名称继续用于日志、返回字段和 portrait/closeup 分支；旧 `img_dir` 属性仍决定读图位置。

15 处活动 `patch.dict` 改为显式参数，涉及 source trial、smoke、框增强准备、候选/遮挡评价及冻结脚本，
另外移除 BEDLAM 审计入口的直接注册表赋值。每个旧名称、配置、NPZ/root 表达式保留。
完整文件列表在 `caller_migration.json`；15 个调用方的其余计算 AST 一致见 `caller_structure.json`，
BEDLAM 单独见 `bedlam_caller_structure.json`。DataModule 的配置 ID 路线保留，不改其构造接口。
生产公共模块没有反向依赖 tools；测试中的 registry patch 仅用于验证旧路线。
Dataset 直接调用公共阶段，兼容 tuple 包装没有成为新主路径。

## 3. 保持的计算顺序

1. 框仍由 `expand_to_aspect_ratio(scale * 200).max()` 得到；不改变框采样配方。
2. 不支持的 138 点水平翻转仍在读图/随机采样前拒绝；保留构造和取样时的检查。
3. `cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION`、portrait/closeup 图像旋转及数组输入规则保持。
4. 增强随机调用顺序不变：采样参数 → 校验 → 极端裁剪 → 平移；原抗锯齿/Albumentations 在 warp 之前。
5. OpenCV 插值和边界策略保持。颜色仍按原 BGR→RGB、float32 CHW、逐通道倍率/clip/标准化顺序。
6. 辅助关节与 138 点使用同一 affine；先把 affine 像素坐标写回 float32 标签，再归一化 XY。
   两次赋值的中间舍入保留；不加 32 px，不把完整 256 坐标改成中间 192 列坐标。
7. 在实际 affine 和标签变换之后重算 `in_view & inside_full_crop`；遮挡可靠 GT 仍监督。
   旧标签权重和 `visible`/`visibility_known` 元数据策略保持。
8. 原字典顺序、类型、shape/dtype、`img_disp`、原图 GT 和 `_trans` 保持。
   `box_size` 仍是极端裁剪后的宽度、尚未乘随机 scale；不悄悄改成实际采样边长。

27 个保留或迁移的历史辅助函数计算 AST 一致（忽略文档字符串）；
`crop_geometry.py`、`keypoint_supervision.py`、`keypoint_contract.py`、`datamodule_keypoints.py`
相对本批 before 字节不变，见 `structure_final.json`。

## 4. 已执行验收

| 检查 | 已核查结果及证据 |
| --- | --- |
| 独立 Dataset 前后对照 | 45 用例 × 旧 ID/显式两路线 = 90 次执行，2814 数组值、shape、dtype、字段顺序精确一致；`after_final/result.json` |
| 七个实际样本 | val_baseline_002/001/004、challenge_000/001/005/007；每个关闭及启用增强均比较完整归一化输入、显示图、原 GT/裁剪 GT、affine、监督及可见性；包含遮挡、偏框、尺度变化、裁剪截断、有效区边缘和已有 ERP 样本 |
| 增强与随机状态 | 平移、缩放、旋转、颜色、混合、普通/激进极端裁剪、抗锯齿、replicate、Albumentations 固定种子；比较采样参数、warp 参数、输出及 Python/NumPy/Torch RNG 前后状态；不是只验证关闭增强 |
| 路径与历史分支 | 相对/绝对图片路径、portrait/closeup、旧权重标签、长宽比、部分路径回退、缺省名称、可变 img_dir、验证禁用增强、未验证 flip 拒绝 |
| 旧十项 tuple | 11 用例、156 数组精确，含 RGB/RGBA/浮点图、空 dense GT、无归一化、无效尺寸/类型/缺图及 flip-before-IO；输入标签原地修改、类型、异常和 RNG 保持；`lowlevel_after/result.json` |
| 活动候选入口 | 复用 P1 独立预测回放：37 JSON、438 数组、78 图片精确；有限、非有限、错误 shape 三情形；`candidate_integration/result.json` |
| CPU 回归 | 80 项通过、0 失败、0 跳过；`cpu_tests_final.json` 和同名 log 保留逐项结果及所选模块；Blender 集成未运行 |
| 旧 CLI 入口 | 六个评价/训练入口从 `/tmp`、无 PYTHONPATH 运行 `--help` 退出 0；`cli_checks.json`。这只证明导入和参数入口可用，不代表执行训练 |

数据、增强和低层接口对照不使用数值容差。候选回放只允许既有 P1 白名单：
`selection_frozen.utc`、`checkpoint_load.seconds`、`report.seconds`、`report.checkpoint.seconds`
及本次输出根路径前缀；其余报告字段和分母保持。

固定增强种子是代表路径验证，不宣称穷尽全部 Albumentations 或极端裁剪组合。
本批未改变 ERP 周期算法、模型或训练回调，没有重复真实三权重预测或 CUDA 验收。

可移植测试在 [test_dataset_crop_contract.py](../../tests/test_dataset_crop_contract.py)，
独立旧结果保存在非忽略的 [dataset_crop_v1.json](../../tests/fixtures/dataset_crop_v1.json)：
31 个合成 Dataset 用例和 11 个旧 tuple 用例，含压缩数组及完整 metadata，不依赖 outputs 或真实资产。
仅临时目录字符串替换为 `<INPUTS>`；其中一个绝对文件名数组的 Unicode 宽度随临时路径派生。
数值数组及 dtype 从不替换；固定本地路径的 2814 数组对照没有此例外。
最初可移植测试曾因路径长度改变字符串 dtype 失败，已修正测试路径替换，保留 `cpu_tests.json`/log 失败记录，
未重建数值期望值；最终 80 项结果在 `cpu_tests_final.json`。

### 复核命令

在仓库根目录，使用现有环境；`--output` 必须是新的目录，不能覆盖 before/after：

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-refactor-mpl PYTHONPATH=src:. /home/yy/conda_envs/camerahmr/bin/python -B tools/validate/verify_dataset_crop.py --phase verify --acceptance outputs/acceptance/refactor_dataset_crop_20261001 --output /tmp/densekp-p2-review-new
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-refactor-mpl PYTHONPATH=src:. /home/yy/conda_envs/camerahmr/bin/python -B -m unittest tests.test_dataset_crop_contract -v
```

环境：Python 3.10，Torch 2.0.0+cu118、NumPy 1.26.4、OpenCV 4.13、skimage 0.21、Albumentations 1.3.1。
升级这些库可能改变插值或随机增强，不在本次精确一致性承诺中。

## 5. 阅读路线、遗留项和回退

[样本导读](../SAMPLE_WALKTHROUGH.md) 第 4 节连接当前公共函数，第 4.1 节给出实际 challenge_001 框的
中心、scale、增强参数、affine 和单点变换。其固定增强使监督 112→102，遮挡监督 71→68；
这是同一输入启用原增强规则后的裁剪变化，不是本批新旧行为差异。
[阅读指南](../READING_GUIDE.md)、README 和 [任务清单](IMPLEMENTATION_TASKS_20261001.md) 已同步。

以下继续单列行为修复，不在本批调整：portrait/closeup 只旋转图像的历史几何问题、mask 取整、
报告字段，以及无效极端裁剪 level 的历史异常分支。旧 `get_example` 等其他路径暂保留，
未证明无调用及复现用途前不删。两包安装/资产发布留 P4。

没有阻碍本批验收的真实问题。下一批为 P2.3–P2.5：来源划分/打包、框构造和监督 adapter；
仍须先独立冻结当前实现的行序、来源衍生关系、框配方及拒绝行为。
如需回退，以本批 `before/source_snapshot` 中文件为界人工恢复本批改动，保留已评审 P1、用户修改和所有证据；
新模块先解除调用，不使用 reset/clean 或删除历史输出。本批结束等待评审，不开始后续实现。
