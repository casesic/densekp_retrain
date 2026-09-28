# 经纬图 / 鱼眼多人小样本闭环

本页记录 2026-09-28 的**链路验证**，不是已完成的鱼眼多人重训。现有两个人体共用一条动作序列，只取不同帧；三个相机视角的 6 个逐人裁剪有高度相关性，不能用作独立泛化结论。

## 已实现

- `tools/fisheye/render_equirectangular_source.py` 支持 `--people-config`：每人指定 ID、动作帧、世界平移、朝向；Blender `IndexOB` 将人体和衣服标为同一实例 ID。未提供配置时保留原单人模式。
- `tools/fisheye/run_fisheye_multiperson_pilot.py` 用同一份 3D DenseKP 点生成经纬图与鱼眼图逐人 `[P,138,2]` 投影、出画/可见标记和逐人 mask。可见性同时检查深度和实例 ID，避免把另一人的表面误判为目标人物。
- `tools/annotation/build_omni_multiperson_tiny.py` 根据实例 mask 生成逐人裁剪样本，跨经度接缝时将图像和点一起循环平移。NPZ 的 `proj_verts[...,2]` 为可监督标记；`DenseKP.compute_loss` 与验证误差现在按它加权。旧数据通道恒为 1，保持旧损失语义。
- `densekp/train_keypoints.py` 的 `init_checkpoint` 只初始化模型权重，重新开始优化器与步数；`ckpt_path` 仍为完整 Lightning 状态续训。

## 本机复现

使用已找到的 Blender 5.1.1，不需 sudo 或再下载 Blender。`OpenEXR` 位于项目忽略目录 `assets_external/python_deps/`，供渲染后处理读取多层 EXR，不修改 `camerahmr` 环境。

```bash
/home/yy/conda_envs/camerahmr/bin/python tools/fisheye/run_fisheye_multiperson_pilot.py \
  --blender-bin /mnt/data1/khb/app/blender-5.1.1-linux-x64/blender \
  --output outputs/experiments/fisheye_multiperson_hires \
  --width 2048 --height 1024 --samples 8

/home/yy/conda_envs/camerahmr/bin/python tools/annotation/build_omni_multiperson_tiny.py \
  --source outputs/experiments/fisheye_multiperson_hires \
  --output datasets/omni_multiperson_tiny

/home/yy/conda_envs/camerahmr/bin/python tools/validate/evaluate_densekp_npz.py \
  --dataset omni-multiperson-tiny --checkpoint assets_external/densekp.ckpt

/home/yy/conda_envs/camerahmr/bin/python tools/validate/smoke_densekp_lightning.py \
  --dataset omni_multiperson_tiny --checkpoint assets_external/densekp.ckpt \
  --steps 2 --output outputs/experiments/fisheye_multiperson_hires/finetune_smoke.json
```

本次输出在 `outputs/experiments/fisheye_multiperson_hires/`；训练 NPZ/图像在 `datasets/omni_multiperson_tiny/`。高分辨率样本共 6 个，训练 4、验证 2；每人每视角经纬图可监督点数 70–94。原版 DenseKP 用真实实例 mask 提供的裁剪框测试，6 个样本上可见点平均误差为 3.35–4.06 像素（256×256 模型裁剪图坐标）。2 步 GPU 微调链路完成，未保存新权重。测试使用的是理想框，**不包含检测失败**。

## 正式重训之前还缺什么

1. 本页的 Omni-Radtan 小样本尚未对齐真实相机。后续已建立与真实 DenseKP ERP 输入相符的 Double Sphere 验证管线，见 [DS_REAL_ERP_PILOT.md](DS_REAL_ERP_PILOT.md)；它仍不是足够多样的正式训练集。
2. 需扩大人物身份、动作、服装、场景、距离、遮挡程度，并独立按动作/场景划分训练和验证。当前同一动作跨相机的验证集只适合烟雾测试。
3. 将真实失败案例按“检测框错误、多人重叠、严重畸变、经度接缝”分组，用原始模型先建立可比较的失败基线，再决定直接鱼眼裁剪是否优于现有经纬图裁剪。

正式微调入口可以设 `data=omni_multiperson_tiny experiment=densekp init_checkpoint=assets_external/densekp.ckpt`，但**现在不应对仅 6 个样本运行长期训练**。
