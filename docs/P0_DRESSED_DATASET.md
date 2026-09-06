# P0 穿衣合成数据集状态

## 状态

```ini
P0_DRESSED_DATASET = BLOCKED
```

阻塞原因不是 renderer：真实 motion、人体几何、相机、RGB/depth/mask、DenseKP 和最小场景都已通过。唯一阻塞是没有经过验证的 body × motion × clothing cache 组合。

本轮没有创建 `outputs/p0_dressed_dataset/`，因为该目录名应只用于真正满足配对和对齐验收的数据；没有用静态 garment 或不匹配缓存伪造结果，也没有下载大型资产。

## 可保留的回归基准

现有 body-only real-motion 基准：

```ini
BODY_ONLY_RENDER_MS_PER_FRAME = 143.763
PROJECTION_MAX_ERROR_PX = 0.000031241
```

当前没有 clothing render，因此 `BODY_CLOTHING_RENDER_MS_PER_FRAME` 不可测，alignment、RGB、scene depth、person mask、clothing-aware visibility 和 overlay 均不能宣告通过。

## 获得匹配资产后的目标结构

正式数据应写入 `outputs/p0_dressed_dataset/`，包含 geometry、RGB、full scene depth、person mask、DenseKP 2D/3D/depth/valid/visible、manifest、overlay 和 contact sheet。manifest 必须逐帧记录 render/source motion/clothing frame、avatar、motion、clothing、body model、gender、相机、路径、人体几何指纹及服装来源。

验收时必须证明：body vertices 与 DenseKP 3D 加衣前后逐元素一致，analytic projection 最大误差小于 0.5 px，服装没有整体漂移、尺度错误、明显帧偏移或严重穿模。
