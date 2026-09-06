# BEDLAM 与 DenseKP Surface Operator 审计

## 结论

```ini
BEDLAM_OPERATOR_SHAPE = [437,10475]
DENSEKP_OPERATOR_SHAPE = [138,10475]
SAME_OPERATOR = NO

BBOX_CONVENTION_VERIFIED = YES
PROJECTION_CONVENTION_VERIFIED = YES
THIRD_CHANNEL_MEANING = 齐次图像坐标常数 1

READY_FOR_DENSEKP_BEDLAM_GENERATOR = YES
```

BEDLAM 官方的相机、bbox、投影和 NPZ convention 可以复用，但官方 `proj_verts` 是 437 个抽样顶点，不能直接作为当前 DenseKP 网络的 138 点监督。DenseKP generator 必须用 `densekp_regressor_R.npy` 替换官方 `downsample_mat_smplx.pkl`：

```text
densekp_3d = densekp_regressor_R @ smplx_vertices
densekp_proj = project(densekp_3d, camera_translation, camera_intrinsics)
```

不需要也不应修改 DenseKP 网络结构。

## 来源

本机限定范围内没有找到 `downsample_mat_smplx.pkl`。本轮只从 [BEDLAM 官方 GitHub 仓库中的原文件](https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/downsample_mat_smplx.pkl) 获取了该 17.5 MB utility，保存为：

```text
assets/downsample_mat_smplx.pkl
```

预处理逻辑以官方源码为准：

- [BEDLAM v1 SMPL-X preprocessing](https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/df_full_body.py)
- [BEDLAM2 SMPL-X preprocessing](https://github.com/pixelite1201/BEDLAM/blob/master/data_processing/df_full_body_smplx_bedlam2.py)
- [官方 data processing 说明](https://github.com/pixelite1201/BEDLAM/tree/master/data_processing)

本轮没有下载 BEDLAM 图像、动作或完整数据集。

## Surface operator 统计

### DenseKP 138 点 operator

文件：`assets/densekp_regressor_R.npy`

| 属性 | 实测结果 |
|---|---|
| Python 类型 | `numpy.ndarray` |
| 存储形式 | dense |
| shape | `[138,10475]` |
| dtype | `float32` |
| nnz | 204 |
| 每行非零数 min/max/mean | `1 / 3 / 1.4782608695652173` |
| row sum min/max/mean | `0.9999999403953552 / 1.0 / 1.0` |
| 是否存在负权重 | 是，共 3 个；最小值 `-3.9740850414887296e-15` |
| 是否全部 finite | 是 |
| 非零权重 min/max | `-3.9740850414887296e-15 / 1.0` |

3 个负值只有约 `1e-15`，属于浮点级数值残差；结构上仍应原样保留，不能在 generator 中擅自裁零或重新归一化。

### BEDLAM 官方 437 点 operator

文件：`assets/downsample_mat_smplx.pkl`

| 属性 | 实测结果 |
|---|---|
| Python 类型 | `torch.Tensor` |
| 存储形式 | dense，`torch.strided`；不是 scipy sparse，也不是 torch sparse |
| shape | `[437,10475]` |
| dtype | `torch.float32` |
| nnz | 437 |
| 每行非零数 min/max/mean | `1 / 1 / 1.0` |
| row sum min/max/mean | `1.0 / 1.0 / 1.0` |
| 是否存在负权重 | 否 |
| 是否全部 finite | 是 |
| 非零权重 min/max | `1.0 / 1.0` |

该矩阵每行恰好有一个值为 1 的元素，因此它是从完整 10,475 顶点 SMPL-X 网格中选择 437 个顶点的 selection matrix，不进行重心加权。

## Operator 对比

两者 shape 不同：

```text
BEDLAM:  [437,10475]
DenseKP: [138,10475]
```

因此无需进行同 shape 情况下的逐元素误差比较：

```ini
max_abs_diff = N/A（shape 不同）
mean_abs_diff = N/A（shape 不同）
nnz_pattern_identical = NO（shape 不同）
SAME_OPERATOR = NO
```

进一步检查 support 后得到：

- BEDLAM operator 选择 437 个互不重复的顶点；
- DenseKP operator 的 204 个非零系数涉及 204 个互不重复的顶点；
- DenseKP 使用的 204 个顶点中，44 个也出现在 BEDLAM 的 437 顶点集合中；
- 138 个 DenseKP 行中，44 行至少与 BEDLAM 顶点集合有一个 support 重合；
- 其中只有 34 行是 BEDLAM operator 也选择的完全相同 one-hot 顶点；
- 只有 34 个 DenseKP 行的全部 support 都包含在 BEDLAM 的顶点集合中。

这说明两者存在少量顶点集合重合，但 DenseKP 138 点不是 BEDLAM 437 点的简单完整子集，也不能从官方 437 点投影结果无损恢复全部 DenseKP 点。必须从完整 SMPL-X 顶点重新应用 DenseKP operator。

## BEDLAM bbox convention

BEDLAM v1 和 BEDLAM2 官方 SMPL-X preprocessing 使用相同的 `get_bbox_valid()` 逻辑，常量 `SCALE_FACTOR_BBOX=1.2`。

主人体 bbox 的输入是：

```text
joints2d[:22]
```

即 SMPL-X 输出投影后的前 22 个关节。每个关节按以下条件过滤：

```text
x > image_width
y > image_height
x < 0
y < 0
```

满足任一条件的关节被排除。严格按源码，`x == image_width` 或 `y == image_height` 不会被过滤，尽管它们位于通常的像素索引范围之外。generator 若以“复现官方”为目标，应保留这一边界行为，或将任何有意修正单独记录为偏差。

对保留下来的关节，官方计算：

```text
bbox = [
    min(valid_joint_x),
    min(valid_joint_y),
    max(valid_joint_x),
    max(valid_joint_y),
]

center = [
    (xmax + xmin) / 2,
    (ymax + ymin) / 2,
]

scale = max(xmax - xmin, ymax - ymin) / 200
scale *= 1.2
```

结论：

- `center` 是 tight valid-joint bbox 的中心，使用原图像素坐标；
- `scale` 是每个人一个 scalar；
- `scale * 200` 是 tight bbox 最大边乘 1.2 后的正方形裁剪边长，不分别表示 width/height；
- 少于 1 个有效关节时返回无效 bbox；
- preprocessing 进一步要求至少 8 个前 22 关节位于图像范围内，否则跳过该样本；
- 当前 `DatasetKeypoints` 在 `BBOX_SHAPE=None` 时计算 `bbox_size=(scale*200).max()`，再以相同 `bbox_size` 作为 width 和 height，与官方 scalar square-bbox convention 一致。

```ini
BBOX_CONVENTION_VERIFIED = YES
```

## BEDLAM projection convention

官方 v1 和 BEDLAM2 preprocessing 均执行：

```text
vertices3d, joints3d = SMPLX(...)
joints2d = project(joints3d, camera_translation, camera_intrinsics)

vertices3d_downsample = downsample_mat_smplx @ vertices3d
proj_verts = project(
    vertices3d_downsample,
    camera_translation,
    camera_intrinsics,
)
```

`project()` 的数学实现为：

```text
points_camera = points + camera_translation
normalized = points_camera / points_camera[:, 2:3]
projected = K @ normalized
```

相机内参为：

```text
K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

因此返回 shape 与输入点数一致，最后一维为 3：

```text
BEDLAM proj_verts shape per sample = [437,3]
```

前两个通道为 preprocessing 所针对图像方向下的原始 RGB 像素坐标。普通场景使用 `1280x720`；portrait/closeup 场景会交换图像宽高并要求图像顺时针旋转 90°后再与投影对齐。它们不是裁剪坐标，也没有归一化到 `[-0.5,0.5]`。

```ini
PROJECTION_CONVENTION_VERIFIED = YES
```

## 第三通道的真实数学含义

设相机坐标点为 `(X,Y,Z)`。除以最后一维后：

```text
(X,Y,Z) / Z = (X/Z, Y/Z, 1)
```

再左乘上述标准内参矩阵：

```text
(u,v,w) = (fx*X/Z + cx, fy*Y/Z + cy, 1)
```

所以官方 `project()` 返回的第 3 通道是**齐次图像坐标 `w=1`**。它不是 depth、confidence 或 visibility；真实深度 `Z` 在逐分量除法时已经消去。

```ini
THIRD_CHANNEL_MEANING = 齐次图像坐标常数 1
```

## DenseKP generator 决策

本次结果属于 Case B：

```ini
SAME_OPERATOR = NO
```

后续 generator 应复现 BEDLAM 官方的：

- SMPL-X 参数到相机坐标顶点的过程；
- 相机平移和内参投影 convention；
- 前 22 个 joints、图外过滤、至少 8 个有效关节的样本筛选；
- scalar `scale=max(width,height)/200*1.2`；
- `imgname`、`center`、`scale`、`gtkps`、相机参数等 NPZ convention；
- portrait/closeup 图像方向处理。

唯一必须替换的主监督步骤是 surface operator：

```text
# 不用于 DenseKP 主监督
bedlam_437_3d = downsample_mat_smplx @ smplx_vertices

# DenseKP 主监督
densekp_138_3d = densekp_regressor_R @ smplx_vertices
densekp_proj = project(densekp_138_3d, camera_translation, camera_intrinsics)
```

保存的 DenseKP `proj_verts` 应为 `[N,138,3]`，其中 xy 是原图像素坐标，第 3 通道按相同投影数学过程保存为 1。不要从 BEDLAM 的 437 个 `proj_verts` 二次插值生成 138 点。

相机、bbox、投影和替换 operator 的规则现已完全明确，因此具备实现小规模、可审计 DenseKP BEDLAM generator 的条件。仍应先生成极小样本并完成图像叠加与 crop-transform 数值验证，再扩大规模。
