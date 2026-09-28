# 真实 ERP 坐标系下的 Double Sphere 多人小样本

这是合成管线与真实输入对齐的验证，不是可用于性能结论的训练集。真实参考图位于 pano-stitch 的 `data_process/tmp/densekp_phaseA_erp_input/`；本项目只复制其中的 `valid_masks.npz` 到忽略目录 `assets_external/real_erp_valid_masks.npz`，没有复制真实图像到训练集。

## 关键坐标约定

这批真实输入使用的是 `pano-stitch/src/densekp_delivery/camera/fisheye_equirect.py`：`ROTATION_MATRIX=I`，cam2 外参为 `I`，cam4 外参为 `diag(-1,1,-1)`。同仓库 `src/lib_pano/camera/fisheye_equirect.py` 使用**不同**旋转，虽然 DS 内参相同，但不能用来复现这批 ERP。用后者生成的图有约四成真实有效区域变黑；改用 delivery 约定后，投影有效域覆盖真实 mask 的全部有效像素（抽样与完整输出均检查）。

`src/densekp_retrain/camera/double_sphere.py` 实现 DS 投影、逆投影及两个方向的重映射。`tools/fisheye/build_ds_multiperson_pilot.py` 将 Blender 完整经纬源图先转换成合成鱼眼，再走 delivery 的鱼眼转 ERP LUT，并应用真实有效 mask、同步变换逐人 mask/关键点，输出 `outputs/experiments/ds_multiperson_pilot/`。cam2/cam4 使用独立的水平朝向来得到可见人物，因此**不是同一时刻的双目配对**。

```bash
/home/yy/conda_envs/camerahmr/bin/python tools/fisheye/build_ds_multiperson_pilot.py
/home/yy/conda_envs/camerahmr/bin/python tools/annotation/build_omni_multiperson_tiny.py \
  --source outputs/experiments/ds_multiperson_pilot \
  --output datasets/ds_multiperson_tiny --val-camera cam4
/home/yy/conda_envs/camerahmr/bin/python tools/validate/evaluate_densekp_npz.py \
  --dataset ds-multiperson-tiny --checkpoint assets_external/densekp.ckpt \
  --output outputs/experiments/ds_multiperson_pilot/baseline.json
```

当前仅 4 个逐人样本，cam2 训练 2 个、cam4 验证 2 个；每相机的两个人各有约 93/69 个有效标注点。跨相机仍是同一动作和场景，验证只检查管线，不反映泛化。实例 mask 提供理想裁剪框；还没有测试人检出器，也没有解决严重畸变、遮挡、多样身份/动作或训练后的性能提升。正式重训须另造更大、更多样且按场景/动作独立划分的数据集，并保留当前小样本作几何回归检查。

原权重在这 4 个理想框上的 256×256 裁剪坐标平均可见点误差为 3.30–3.61 像素，说明当前中心人物样本仍偏容易，**不能**拿它证明目标失败场景已改善。GPU 0（RTX 3090）上完成 2 步 AMP 训练链路检查，未保存新权重；报告分别在 `baseline.json` 和 `finetune_smoke.json`，逐人叠图在 `baseline_overlays/`。

另有靠有效区边缘的压力样本在 `outputs/experiments/ds_multiperson_boundary/`，用 `--cam2-roll-source-px -550 --cam4-roll-source-px -500` 生成，并转成 `datasets/ds_multiperson_boundary_tiny/`。四人的有效点数依次为 53、69、82、29，原权重在理想框上的平均误差为 3.54–4.71 像素，报告和叠图也在该输出目录。它验证了截断标注路径，但仍未构成足量的严重畸变/多人遮挡训练样本。

## 原始鱼眼外观核对

对照真实 `20260320_riding_dataset3/image/fisheye/{cam2,cam4}/02214.jpg` 后发现，先前的 `synthetic_fisheye.png` 只用了 DS 模型的数学可投影域，没裁掉真实传感器镜头外的区域，所以画面几乎填满方形，**不应称为真实外观匹配的鱼眼图**。脚本现可用 `--fisheye-reference-root` 从真实图仅提取二值镜头轮廓，应用于合成鱼眼和逐人 mask；同时按经度周期采样经纬源图，修复画面中的竖直接缝。真实 RGB 不会复制进训练图。带镜头轮廓的新版图片在 `outputs/experiments/ds_multiperson_boundary_lens/`：

```bash
/home/yy/conda_envs/camerahmr/bin/python tools/fisheye/build_ds_multiperson_pilot.py \
  --cam2-roll-source-px -550 --cam4-roll-source-px -500 \
  --fisheye-reference-root /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/Output/20260320_riding_dataset3/image/fisheye \
  --reference-frame 02214 \
  --output outputs/experiments/ds_multiperson_boundary_lens
```

新图的真实 ERP mask 有效像素被提取的镜头轮廓覆盖 100%。但模拟场景仍是远处的两个人、平坦地面和另一处建筑，不像真实骑行画面中的近景多人、车辆与穿戴式相机位置。真实感改进还需要重新布置 Blender 相机和人物及更合适的街景，不能靠镜头轮廓解决。
