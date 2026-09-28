# DenseKP 小样本训练链路复现

本机现有 24 张 BEDLAM 原始图像，但没有与它们配齐的 SMPL-X motion，不能据此生成可信的 138 点监督。因此本次使用已验收的 `outputs/baselines/p0_dressed_eye_background_fixed/`，其中 16 帧 RGB、人体 mask 与 DenseKP 原图像素投影逐帧对应。

## 生成数据

```bash
/home/yy/conda_envs/camerahmr/bin/python tools/annotation/build_rendered_tiny.py
```

生成到 `datasets/p0_dressed_tiny/`：16 张 RGB、`train.npz`（12 帧）、`val.npz`（4 帧）和 `manifest.json`。bbox 由 person mask 外接矩形扩张 1.2 倍得到。`proj_verts` 为 `[N,138,3]`，前两列是渲染器输出的原图像素坐标，第三列固定为 1。每四帧留一帧作验证；它们都来自同一动作序列，验证集仅用于检查加载链路。

训练数据名为 `p0-dressed-tiny`，在 `densekp/core/configs/__init__.py` 注册，Hydra 数据配置为 `configs/hydra/data/p0_dressed_tiny.yaml`。真实 `DatasetKeypoints` 已验证读出 12/4 个样本、`img[3,256,256]`、`proj_verts_cropped[138,3]`；裁剪坐标复算最大误差约 `2.97e-8`。

## CPU 烟雾验证

```bash
/home/yy/conda_envs/camerahmr/bin/python tools/validate/smoke_densekp_training.py
/home/yy/conda_envs/camerahmr/bin/python tools/validate/smoke_densekp_training.py \
  --full-model --steps 1 \
  --output outputs/experiments/densekp_tiny_smoke/full_model_report.json
```

第一条冻结 ViTPose 主干，检查模型头的两次更新：损失 `334168.6875 → 130514.1640625`，参数确实变化。第二条使用原始 `DenseKP.configure_optimizers()`，更新完整主干与模型头：一步损失 `330525.75`，所检查主干参数最大变化约 `1.00e-5`。这两条都使用原始 `DenseKP.compute_loss()`。报告位于 `outputs/experiments/densekp_tiny_smoke/`。

上述结果来自隔离命令环境中的 CPU 验证；该环境没有暴露 GPU 设备。本机有两张 RTX 3090，训练入口 `densekp/train_keypoints.py` 固定要求 GPU。可通过 `--device cuda:0` 在可访问 GPU 的进程中执行相同烟雾检查。本次只确认一小批数据能执行完整参数更新，不代表模型已收敛或在新场景上有效。

## GPU 与原始 Lightning 循环

在可访问 GPU 的本机终端执行：

```bash
/home/yy/conda_envs/camerahmr/bin/python tools/validate/smoke_densekp_training.py \
  --full-model --steps 2 --device cuda:0 \
  --output outputs/experiments/densekp_tiny_smoke/gpu_full_model_report.json
/home/yy/conda_envs/camerahmr/bin/python tools/validate/smoke_densekp_lightning.py
```

第一条在 RTX 3090 上对完整主干和模型头更新两次，损失 `316985.28125 → 125153.7890625`，所检查的主干参数确实变化，显存峰值约 16.55 GB。第二条调用原始 `DataModule`、`DenseKP.training_step`、`compute_loss` 和 `configure_optimizers`，由 Lightning 以 batch size 1、`16-mixed` 执行两步训练及验证：`completed_steps=2`，`val_loss≈1.0991`，显存峰值约 6.89 GB。为避免生成约数 GB 的临时检查点，该诊断脚本禁用 checkpoint；训练模块本身没有改动。

因此本次验证覆盖了 CameraHMR 的 DenseKP 训练实现及其真实优化循环。`densekp/train_keypoints.py` 的长期训练入口尚未用于本次两步试验；它仍使用自己的默认 checkpoint、日志和 GPU 配置。两步损失变化只证明优化链路可运行，不能当作收敛或泛化结果。

## 自造数据接入

按 `docs/training/TRAINING_DATA_FORMAT.md` 生成逐人样本。训练 NPZ 至少需要 `imgname[N]`、`center[N,2]`、`scale[N]` 和 `proj_verts[N,138,3]`；`imgname` 相对图像根目录，`center/scale` 对应原图 bbox，`proj_verts` 前两列必须是同一张 RGB 的原图像素坐标。138 点顺序固定为 `assets/densekp_regressor_R.npy` 的行顺序。将新数据集的图像根目录和训练/验证 NPZ 注册到 `densekp/core/configs/__init__.py`，再建立 `configs/hydra/data/<name>.yaml`。当前损失不读取 visibility；生成时需要单独审计遮挡和出画点对监督的影响。
