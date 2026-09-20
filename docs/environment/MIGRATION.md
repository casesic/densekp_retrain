# 迁移至用户 `yy`

## 安全边界

该暂存工作区仅在 `yuyue` 的项目目录中创建，未读取或修改 `/mnt/data1/yy/home/yy` 下的任何内容。下面的命令仅作为操作说明，应由 `yy` 手动执行。

## 立即复制的内容

复制完整的暂存目录树，包括源代码子集、配置、文档、小型表面算子和验证工具。其中不包含训练图像、检查点、ONNX/TensorRT 文件或拟合代码。

```bash
mkdir -p /mnt/data1/yy/home/yy/Projects/densekp_retrain
rsync -a --info=progress2 \
  /mnt/hdd/yuyue/home/yuyue/Projects/densekp_retrain_staging/ \
  /mnt/data1/yy/home/yy/Projects/densekp_retrain/
```

## 单独处理的内容（大型或受限资源）

| 资源 | 当前来源 | 大致大小 | 使用场景 |
|---|---|---:|---|
| ViTPose 主干网络 | `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/CameraHMR/data/train-eval-utils/vitpose_backbone.pth` | 2.4 GB | 按上游方式从头构建模型时必需 |
| DenseKP 检查点 | `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/CameraHMR/data/pretrained-models/densekp.ckpt` | 7.5 GB | 可选，用于初始化基线或微调 |
| SMPL-X 模型 | `tools/CameraHMR/data/models/SMPLX`（目前在本地仅为占位目录/空目录） | 外部资源 | 生成标注 |
| SMPL-X 到 SMPL 的映射 | `tools/CameraHMR/data/train-eval-utils/smplx2smpl.pkl` | 552 MB | 仅在转换过程明确需要时使用 |

不要将这些资源放进代码的 rsync 复制范围。应由 `yy` 在符合许可证和权限要求的前提下，将其复制或下载到 `assets_external/` 目录，再配置相应路径。这里有意不记录文件哈希；应通过实际加载权重并运行冒烟实验进行验证。

## 不要复制的内容

- `pano-stitch/src/densekp_delivery`，以及任何 `dist/`/`artifacts/` 交付目录；
- `tools/CameraHMR/deploy`、`.onnx`、`.onnx.data`、`.plan` 文件；
- `tools/CameraHMR/output`、演示图像、检测器检查点或 Detectron2；
- CamSMPLify、SingleFitting、`cpp/staged_lm`、HMR2/CameraHMR 拟合流程；
- 数据集、缓存、日志、`__pycache__` 和已有训练输出；
- 完整的 pano-stitch 或 CameraHMR 仓库，除非后续审计证明某项依赖确有必要。

## 需要下载或生成的数据

- 首先准备一个很小的官方 BEDLAM/BEDLAM2 子集，其中包括 RGB、人体参数、相机元数据，最好还包括深度/实例分割；
- 仅当继续采用上游验证方案时才准备 EMDB 标注和图像；否则，创建一个采用相同 138 点格式、独立留出的合成验证集；
- 按照 [`DATA_FORMAT.md`](../data/DATA_FORMAT.md) 的说明生成逐人物 DenseKP NPZ 文件；
- 只有在小规模转换通过投影叠加、数据结构验证和单批次过拟合后，才准备完整的 BEDLAM/BEDLAM2 图像。

## Conda/环境依赖

建议使用 Python 3.10。首先安装与 CUDA 兼容的 PyTorch/torchvision 组合，然后安装 PyTorch Lightning 2.0.x、Hydra/OmegaConf、YACS、NumPy、OpenCV、scikit-image、albumentations、einops、timm、loguru、pyrootutils 和 rich。仅在标注生成环境中添加 `smplx`。审计涉及的训练循环不需要 Detectron2、DensePose、Pyrender 或 Chumpy。

保留 `requirements.upstream.txt` 是为了记录上游依赖，而不是建议不加甄别地直接安装：其中包含推理/拟合依赖，以及未固定版本的 Detectron2 Git 安装项。

## 建议的最终目录结构

```text
densekp_retrain/
  configs/                 # Hydra 配置 + 本地数据集清单
  densekp/                 # 模型/训练器/加载器子集
  tools/                   # 数据结构检查与监督信号生成工具
  docs/
  assets/                  # 纳入版本控制的小型 138 点算子
  assets_external/         # ViTPose、可选 DenseKP 检查点、SMPL-X（不提交至 Git）
  data/                    # 仅存放符号链接/清单；大规模数据放在其他位置
  outputs/                 # 检查点/日志
```

## 迁移后检查

```bash
cd /mnt/data1/yy/home/yy/Projects/densekp_retrain
bash scripts/audit_workspace.sh
python tools/annotation/inspect_annotation.py /path/to/tiny_labels.npz
```

随后更新复制后的数据集注册表（或将其替换为清单适配器），把 `DENSEKP_VITPOSE_BACKBONE` 指向 ViTPose 权重文件，并运行下面的最小实验。

## 最小可运行实验

使用 8–32 个人体裁剪样本，至少留出两张图像/两个人体实例用于验证；设置 `num_workers=0`、批量大小 1–4、FP32，禁用翻转、旋转和 albumentations，并用 100–500 个优化器步数进行过拟合测试。使用 ViTPose 权重初始化；也可以对比从 `densekp.ckpt` 开始微调的结果。

验收标准如下：训练前标注能够正确叠加；损失为有限值且快速下降；模型能够过拟合训练像素；预测点 ID 始终与第 0..137 行保持对应。满足这些条件后，才能扩大数据规模或引入鱼眼/ERP 样本。
