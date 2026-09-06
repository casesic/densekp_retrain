# Tiny BEDLAM DenseKP 监督生成器

## 当前结论

本机限定搜索范围内没有找到可用于生成监督的 BEDLAM RGB、逐帧 SMPL-X 顶点和对应相机记录。因此本阶段完成了生成与验证接口，但没有创建伪造 NPZ、图片或验证报告，也没有启动训练。

```ini
NEED_BEDLAM_SAMPLE_DATA = YES
READY_FOR_TINY_OVERFIT = NO
```

## Inputs（输入）

生成器是 `tools/generate_bedlam_densekp.py`。它有意不在内部猜测不同 BEDLAM 发布版本的参数字段，也不负责调用 SMPL-X；输入必须先整理为一个小型几何 NPZ：

| 字段 | shape | 含义 |
|---|---:|---|
| `imgname` | `[N]` | 相对 `--image-root` 的 RGB 路径，也可为绝对路径 |
| `vertices` 或 `smplx_vertices` | `[N,10475,3]` | 与相机参数对应的完整 SMPL-X 顶点 |
| `joints3d` 或 `body_joints3d` | `[N,J,3]`，`J>=22` | SMPL-X 关节；bbox 只使用前 22 个 |
| `camera_translation` 或 `cam_trans` | `[N,3]` | BEDLAM 投影约定中的相机平移 |
| `camera_intrinsics` 或 `cam_int` | `[N,3,3]` | 每帧相机内参矩阵 |
| `orientation` | `[N]` | v1 只接受明确标记的 landscape；包含 `portrait` 或 `closeup` 的记录会跳过 |

所有数组必须有相同的首维长度。生成器拒绝形状不符、非 finite 顶点、零深度投影和不足 8 个图内 body joints 的记录。默认最多接受 32 个样本，至少需要 8 个有效样本才会写数据集。

示例命令：

```bash
python tools/generate_bedlam_densekp.py \
  /path/to/bedlam_tiny_geometry.npz \
  --image-root /path/to/bedlam/rgb \
  --min-samples 8 \
  --max-samples 32
```

## Coordinate chain（坐标链）

输入 `smplx_vertices` 和 `joints3d` 必须是 BEDLAM 官方 SMPL-X 调用输出、已应用原始 pose、betas、global orientation 后的三维结果，但尚未加 projection 函数中的 `camera_translation`。生成器不接受世界坐标顶点，也不自行推断或改变 world-to-camera extrinsics。完整链路为：

```text
BEDLAM pose/betas/global orient
→ SMPL-X 输出三维顶点与关节
→ 加 BEDLAM camera_translation 得到 camera coordinates
→ 除以 camera Z 并左乘 K 得到 original RGB pixels
→ DatasetKeypoints 的 affine transform 得到 crop pixels
→ /256 - 0.5 得到 normalized DenseKP coordinates
```

## DenseKP surface operator

每个样本严格使用：

```text
densekp_3d = densekp_regressor_R @ smplx_vertices
points_camera = densekp_3d + camera_translation
normalized = points_camera / points_camera[:, 2:3]
proj_verts = camera_intrinsics @ normalized
```

DenseKP operator 是 `assets/densekp_regressor_R.npy`，shape 为 `[138,10475]`。BEDLAM 官方 `[437,10475]` downsample operator 只用于理解官方流程，不用于生成 DenseKP 主监督。`proj_verts[...,0:2]` 是原 RGB 像素坐标，`proj_verts[...,2]` 是齐次图像坐标常数 1，不是深度、置信度或可见性。

生成器会打印并人工复算第 `0、1、10、50、100、137` 行。当前 operator 的非零 support 为：

| 行 | SMPL-X vertex IDs | 权重 |
|---:|---|---|
| 0 | `9050, 9052, 9082` | `0.7344180346, 0.1050590649, 0.1605228931` |
| 1 | `1425, 1431, 1432` | `0.1929576844, 0.2945336103, 0.5125086904` |
| 10 | `3291` | `1.0` |
| 50 | `1336, 9279, 9285` | `-3.974085e-15, 0.6882301569, 0.3117698133` |
| 100 | `6842` | `1.0` |
| 137 | `8672` | `1.0` |

行 50 的极小负值属于 operator 原始浮点残差，生成时原样保留。真实样本到位后，debug NPZ 会记录上述行的直接矩阵乘法与逐 support 加权和之间的最大绝对误差。

## BBox convention（bbox 约定）

先投影 `joints3d[:22]`，保留满足 `0 <= x <= image_width` 且 `0 <= y <= image_height` 的 finite 关节。严格保留 BEDLAM 官方对右、下边界使用等号的行为。有效关节至少为 8 个，然后计算：

```text
center = [(xmin+xmax)/2, (ymin+ymax)/2]
scale = max(xmax-xmin, ymax-ymin) / 200 * 1.2
```

这里的 `scale` 是每个人一个 scalar，`scale*200` 是方形 crop 的边长。

## NPZ schema（输出格式）

成功生成后，默认目录为 `datasets/densekp_bedlam_tiny/`：

```text
images/
annotations/bedlam_densekp_tiny.npz
annotations/bedlam_densekp_tiny_debug.npz
README.md
```

主 NPZ 只含训练读取所需字段：

| 字段 | dtype / shape |
|---|---|
| `imgname` | string `[N]` |
| `center` | `float32 [N,2]` |
| `scale` | `float32 [N]` |
| `proj_verts` | `float32 [N,138,3]` |

debug NPZ 保存原始索引、原始图名、前 22 个投影关节、有效关节数、tight bbox、图像尺寸、DenseKP 图内/扩张 bbox 内数量，以及 surface operator 复算误差。图外点只统计，不过滤，也不生成 visibility 权重。

## Verification（验证）

真实 tiny 数据生成后运行：

```bash
python tools/validate_bedlam_densekp_tiny.py datasets/densekp_bedlam_tiny
```

脚本会通过仓库内真实 `DatasetKeypoints` 读取 NPZ，将 scale、rotation、translation、颜色、Albumentations、extreme crop 和 horizontal flip 全部关闭。输出到 `outputs/bedlam_generator_audit/`：

- `original/`：原图、黄色 bbox、蓝色前 22 joints、红色 138 点，DenseKP 每 10 点标 ID；
- `cropped/`：256×256 crop 上的 138 点；
- `verification.json`：逐样本和整体 crop-transform 最大/平均绝对误差、归一化坐标范围、图外点统计和 operator 复算误差。

数值核对公式为：

```text
expected_xy = affine_transform(original_xy) / 256 - 0.5
```

它与 `DatasetKeypoints` 返回的 `proj_verts_cropped[:,:2]` 逐点比较。第三通道保持为 1，不参与二维仿射变换。

## Known limitations（已知限制）

- 当前没有真实 BEDLAM 样本，因此输出 schema 尚未经过真实文件落盘验证；
- 已检查 `/home/yy/datasets`、当前仓库数据目录及有限的已知 CameraHMR/BEDLAM 路径；可用 sequence、RGB、metadata 和 sample 数量均为 0；
- 尚无原图 overlay、crop overlay、归一化范围或 transform error 的真实数值；
- v1 明确排除需要 90° 旋转的 portrait/closeup，避免隐式方向错误；
- generator 不修改网络、head、loss，也不添加 visibility loss；
- tiny 数据通过视觉和数值检查之前，不得进入 overfit 或训练阶段。
