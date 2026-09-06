# P0 depth 与 visibility

## 1. Depth pass 实测语义

P0 使用 Blender Eevee Render Layers 的 `Depth` pass，经 compositor 写为 32-bit multilayer EXR，再由独立 Python 环境中的 OpenEXR 3.4.15 提取为 float32 NPY。

已知几何测试将相机置于 identity，放置一张与像平面平行、camera local `Z=-2 m` 的平面：

| 测试量 | 数值 |
|---|---:|
| 图像中心 Depth | 2.0 m |
| `(384,256)` 离轴像素 Depth | 2.0 m |
| 该离轴点 camera-space Z 预期 | 2.0 m |
| 该离轴 ray 欧氏距离预期 | 2.0323913 m |

因此：

```ini
DEPTH_SEMANTICS = CAMERA_SPACE_Z
```

背景值接近 camera far clip；转为 NPY 时将非有限值设为 0，但当前 Eevee 输出背景主要是 far depth。visibility 仅在人体 mask/局部有效表面附近比较，不能把 far background 当人体表面。

## 2. Mask

Blender 5.1 Eevee 未在 Render Layers compositor socket 中暴露旧版 `IndexOB`，P0 又只有一个不透明人体，因此采用同一渲染的 `Alpha` pass 作为人体 mask。RGB 使用透明 film，Alpha 阈值 `>=0.5` 后保存：

```text
background = 0
human      = 255
```

16 帧 mask 均为二值，非零像素范围为 26931--27113。pyrender 对 frame 0/8/15 独立渲染的 silhouette IoU 分别为 0.999074、0.999152、0.999146；边缘微差来自两个 rasterizer 的抗锯齿规则。

## 3. DenseKP point depth

因为 Z pass 是 camera-space Z，DenseKP 点深度必须使用同一度量：

```text
densekp_depth = -densekp_camera_xyz[..., 2]
```

禁止改用点到光心的欧氏距离。鱼眼/ERP 阶段必须重新验证其 depth pass，不能沿用本结论。

## 4. Visibility 分类

P0 编码：

```text
0 = VISIBLE
1 = SELF_OCCLUDED
2 = OUT_OF_FRAME
3 = BEHIND_CAMERA
```

流程：

1. 检查 camera `Z`：`Z>=0` 为 `BEHIND_CAMERA`；
2. 检查解析像素是否落入 `[0,width) × [0,height)`，否则为 `OUT_OF_FRAME`；
3. 对候选点在 `floor(u),floor(v)` 周围 3×3 邻域取 `|render_depth - point_depth|` 的最小值，降低三角形 rasterization 与亚像素落点影响；
4. depth error 不超过容差为 `VISIBLE`，否则为 `SELF_OCCLUDED`。

本轮先统计候选误差：

```text
p50 = 0.003916 m
p75 = 0.074005 m
p90 = 0.141211 m
p95 = 0.217673 m
p99 = 0.251495 m
```

高分位包含真正的背面/自遮挡点，不能直接把全体 p95 当容差。容差按“两个像素的角足迹”定义，并设 2 mm 数值下限：

```text
tolerance = max(0.002 m, 2 * median_candidate_depth / fx)
          = 0.0090451 m
```

16×138 点得到：

```text
VISIBLE        = 1229
SELF_OCCLUDED  = 979
OUT_OF_FRAME   = 0
BEHIND_CAMERA  = 0
```

这符合 P0 人体完整入画、固定前视相机的设置。contact sheet 中绿色可见点落在正面，红色点主要对应背面或自遮挡 surface point，基本合理。

## 5. 局限

- DenseKP 是 surface operator 输出，不是某个 mesh vertex；亚像素投影与三角面曲率会影响 depth error。
- Alpha mask 足够支持 P0 单人体；多人/物体遮挡阶段必须引入 instance/object/triangle ID。
- 更严格 visibility 可在后续加入 CPU/GPU ray-triangle test，作为 depth-buffer 分类的交叉验证。
- 当前不修改训练 loss，也不把 toy motion 产物转换成训练 NPZ。
