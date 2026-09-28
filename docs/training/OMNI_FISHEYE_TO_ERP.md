# 公司 Omni-Radtan 鱼眼 → 经纬图小样本

这条链路实际经过生成的 Omni-Radtan 鱼眼图，不是直接拿 Blender 原始经纬图当训练输入：

`Blender 全景 + 逐人 3D 标注 → Omni-Radtan 鱼眼 RGB/mask/关键点 → Omni 逆映射经纬 RGB/mask/关键点 → DenseKP NPZ`。

使用 `configs/fisheye_near_camera.json` 和 `configs/fisheye_near_people.json`，相机高度 1.65 m，三人距离约 1.3–2.1 m；比先前 3.4 m 外的两人样本更接近近景多人，但仍是同一动作源、同款衣服和简化地面，不是正式训练集。公司提供的 Omni-Radtan 标定是 3000×3000、标称 200°；它与这批 2880×2880 真实原始鱼眼对应的 DS 转换不能直接互换。后者的实测匹配和有效区记录见 [DS_REAL_ERP_PILOT.md](DS_REAL_ERP_PILOT.md)。

```bash
/home/yy/conda_envs/camerahmr/bin/python tools/fisheye/run_fisheye_multiperson_pilot.py \
  --cameras configs/fisheye_near_camera.json \
  --people-config configs/fisheye_near_people.json \
  --output outputs/experiments/fisheye_near_multiperson \
  --width 2048 --height 1024 --samples 8 \
  --blender-bin /mnt/data1/khb/app/blender-5.1.1-linux-x64/blender

/home/yy/conda_envs/camerahmr/bin/python tools/fisheye/build_omni_erp_pilot.py \
  --source outputs/experiments/fisheye_near_multiperson \
  --output outputs/experiments/omni_near_erp

/home/yy/conda_envs/camerahmr/bin/python tools/annotation/build_omni_multiperson_tiny.py \
  --source outputs/experiments/omni_near_erp \
  --output datasets/omni_near_erp_tiny --val-person-id 3
```

鱼眼图在 `outputs/experiments/fisheye_near_multiperson/fisheye_center/fisheye/rgb.png`，转换后的经纬图在 `outputs/experiments/omni_near_erp/source/fisheye_center_preview.png`，逐人 mask 与 138 点在相邻目录。鱼眼可见点数为 75/66/68，转换后仍为 75/66/68；与 Blender 直接投影的对应可见点比较，转换标注的最大偏差约 0.00034 个 5760×2880 像素。`datasets/omni_near_erp_tiny` 的 2 训练 + 1 验证是同一场景的烟雾测试拆分，不能用于泛化结论。

原权重在理想逐人框上的可见点平均误差分别为 3.54、3.61、4.28 像素（256×256 DenseKP 裁剪图坐标）。这不包含检测器失败，也不足以证明多人和畸变问题已改善；基线报告及逐人叠图位于 `outputs/experiments/omni_near_erp/baseline.json`、`baseline_overlays/`。

`camerahmr` 环境下，用原权重初始化、3090 GPU 混合精度完成 2 步 DenseKP Lightning 训练烟雾测试，未保存新模型；详情见 `outputs/experiments/omni_near_erp/finetune_smoke.json`。

清理时只移除了旧低分辨率 Omni 试验的三个 `fisheye/depth.npy` 缓存（约 105 MiB）；原始 EXR、RGB、mask、标注和报告保留。需要时可从原始 EXR 重建深度缓存。
