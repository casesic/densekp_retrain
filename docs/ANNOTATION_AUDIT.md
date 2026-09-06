# DenseKP 现有标注实证审计

## 审计结论

截至 2026-09-05，尚未在允许只读检查的旧 `yuyue` 工作区中找到真实的 DenseKP 训练/验证 annotation NPZ。因此，本阶段仍不能完成依赖真实 ground truth 的 bbox 可视化、`proj_verts` 原图叠加和裁剪变换数值验证。

```text
HISTORICAL_DENSEKP_ANNOTATION_AUDIT_COMPLETE = NO
```

上述结论只表示“真实历史 DenseKP annotation 尚未完成实证”。后续 operator 专项审计已经直接从 BEDLAM 官方 preprocessing 验证了生成端的 bbox、相机和投影 convention，并确认应以 DenseKP 138 点 operator 替换官方 437 点 operator。因此，虽然历史 annotation 实证状态仍为 NO，但新 generator 的接口定义已经解除阻塞：

```ini
READY_FOR_DENSEKP_BEDLAM_GENERATOR = YES
```

完整依据见 [`BEDLAM_OPERATOR_AUDIT.md`](BEDLAM_OPERATOR_AUDIT.md)。不能用重新运行 DenseKP 推理代替缺失的历史 ground truth：模型预测不是训练 ground truth，也不能用来证明原训练 annotation 的实际数值分布。

## Verified facts

以下结论已通过当前代码、旧 CameraHMR 代码、Git 历史或实际文件验证：

- DenseKP 训练功能由旧 CameraHMR 提交 `0ca750f` 引入；该提交包含训练加载器、训练器、模型头、训练配置和推理 demo，但不包含标注生成器。
- 训练加载器 `DatasetKeypoints` 从单个 NPZ 读取 `imgname`、`center`、`scale` 和 `proj_verts`，并将 `part` 或 `gtkps` 仅用于极端裁剪决策。
- 当前训练配置使用 `IMAGE_SIZE=256`，没有配置 `BBOX_SHAPE`；`DO_FLIP=False`、`FLIP_AUG_RATE=0.0`。本轮没有启动训练。
- 旧配置将训练集名称映射至 `data/training-labels/bedlam-labels/*.npz`，将验证集 `emdb-smpl` 映射至 `data/test-labels/emdb_test.npz`；对应 RGB 根目录位于 `data/training-images/...` 和 `data/test-images/EMDB`。
- 旧 CameraHMR 工作区内的 `data/training-labels`、`data/test-labels`、`data/training-images` 和 `data/test-images` 均不存在，配置中指向的真实数据没有落在这些位置。
- 在旧项目、旧用户家目录有限层级、常见数据目录和文件定位数据库中，没有找到配置所列的 DenseKP 训练标注。
- 用户指出的目录 `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/Output/20260311_riding/camerahmr/densekp_npz` 中有 384 个逐人物 NPZ。抽查文件的 keys 一致，但它们属于 DenseKP 推理输出，不是训练 annotation。
- 实际检查的推理文件 `cam2/00513_0.npz` 包含：
  - `keypoints_crop_xy: (138,2), float32`
  - `keypoints_orig_xy: (138,2), float32`
  - `keypoint_value: (138,), float32`
  - `bbox_xyxy: (4,), float32`
  - `bbox_center: (2,), float32`
  - `box_size: (), float32`
  - `image_size: (2,), int32`
  - `crop_trans: (2,3), float32`
  - `crop_inv_trans: (2,3), float32`
- 旧项目 `src/densekp_delivery/pipeline.py` 的保存逻辑直接把推理结果写为 `keypoints_orig_xy` 和 `keypoint_value`。该 schema 与 `DatasetKeypoints` 的输入 schema 不兼容。
- `get_example_projverts` 在无增强时使用恒等增强参数：缩放 1、旋转 0、不翻转、不做极端裁剪、颜色倍率 1、平移 0。
- `get_example_projverts` 使用同一个仿射矩阵 `trans` 变换普通二维关节点和 `proj_verts[:, :2]`，然后执行 `xy / patch_width - 0.5`。
- DenseKP 当前 position loss 只读取 ground truth 的前两个通道；annotation `proj_verts[...,2]` 不参与该损失。

## Unknowns

由于缺少真实训练 annotation，以下内容仍为 **UNKNOWN**：

- 任一真实 DenseKP 训练/验证 NPZ 的完整路径、实际 RGB root、样本数、字段 dtype 和附加字段；
- `imgname` 在真实文件中是相对路径还是绝对路径；
- `scale` 在真实 DenseKP 文件中是 `[N]` 标量还是 `[N,2]` 向量；
- `scale * 200` 在 annotation 生成端的来源、是否已经包含 bbox expansion factor，以及其与人体可见区域的实际覆盖关系；
- `proj_verts[..., :2]` 是否确实为原始 RGB 像素坐标；源码按该约定消费它，但尚无真实数据叠加证据；
- `proj_verts[...,2]` 的实际 dtype、分布和语义；没有找到生成源码，不能将其解释为 confidence、visibility 或 depth；
- 遮挡点、背面点和越界点在真实标注中的编码方式。

历史 DenseKP NPZ 的实际 dtype、数值分布和可视化仍为未知；但 BEDLAM 官方生成端的 bbox 定义、xy 投影坐标系和第三通道数学含义已在 operator 专项审计中由官方源码确认。后续 generator 可以先实现极小样本，再用叠加图和当前 crop 工具完成端到端验收。

## NPZ schema

### 训练加载器要求的 schema（仅由消费代码确认）

| 字段 | 代码要求 | 当前实证状态 |
|---|---|---|
| `imgname` | 第一维长度为 `N`，与 RGB root 拼接 | dtype、路径形式 UNKNOWN |
| `center` | 每个样本使用前两个值作为 `(center_x, center_y)` | 预期 `[N,2]`；dtype 和真实数值 UNKNOWN |
| `scale` | 每个样本执行 `scale * 200` | `[N]` 或 `[N,2]` 尚未实证 |
| `proj_verts` | 每个样本应有 138 个点和 3 个通道 | 预期 `[N,138,3]`；dtype 和真实数值 UNKNOWN |
| `part` | 可选，用于极端裁剪 | 非 DenseKP 主监督 |
| `gtkps` | `part` 缺失时的可选备用字段 | 非 DenseKP 主监督 |

这里不能把上述“加载器要求”写成“真实 NPZ 已验证 schema”。

### 已检查的推理输出 schema

推理目录的 schema 已实证，但不符合训练加载器要求，不能作为本阶段 ground truth。其 `keypoint_value` 来自模型输出的 raw score/第三通道，不能用于推断训练 annotation 的 `proj_verts[...,2]`。

## Coordinate systems

消费代码定义了以下三层坐标变换，但 A 的真实数据语义尚待 ground truth 可视化确认。

### A. 原始图像像素坐标

加载器把 `proj_verts[:, :2]` 直接作为输入图像坐标送入仿射变换：

```text
p_original = (u, v)
```

### B. 裁剪图像像素坐标

```text
p_crop = trans_points2d_parallel(p_original, trans)
```

其中 `trans` 是 `generate_image_patch_cv2` 返回的 `2x3` 仿射矩阵。

### C. DenseKP 归一化坐标

在当前 `IMAGE_SIZE=256` 且正方形 patch 下：

```text
x_norm = u_crop / 256 - 0.5
y_norm = v_crop / 256 - 0.5

u_crop = (x_norm + 0.5) * 256
v_crop = (y_norm + 0.5) * 256
```

新增可视化工具会分别通过加载函数和独立公式计算 C，并报告 `max_abs_error` 与 `mean_abs_error`。由于真实训练 NPZ 缺失，本轮没有伪造输入来给出误差数值。

## BBox convention

以下是代码中已确认的逻辑：

```text
raw_extent = scale * 200
aspect_extent = expand_to_aspect_ratio(raw_extent, BBOX_SHAPE)
bbox_size = max(aspect_extent)
get_example_projverts(center_x, center_y, bbox_size, bbox_size, ...)
```

- 无论原 `scale` 是标量还是二维向量，最终传给 `get_example_projverts` 的 `width` 和 `height` 都相同，均为 `bbox_size`，因此最终裁剪框被强制为正方形。
- 当前 DenseKP 配置没有设置 `BBOX_SHAPE`，所以 `expand_to_aspect_ratio` 原样返回 `scale * 200`；随后 `.max()` 选取最大尺寸。
- 若未来配置二维 `BBOX_SHAPE=(w_t,h_t)` 且输入 `scale` 为二维，`expand_to_aspect_ratio` 只增大较短方向以达到目标宽高比；紧接着 `.max()` 再将结果变为正方形边长。
- `DatasetKeypoints` 在 `scale * 200` 与裁剪调用之间没有固定的额外 bbox expansion factor。
- 训练增强开启时，`do_augmentation` 可生成随机 scale factor，并由仿射矩阵额外缩放裁剪范围；这不是 annotation 中 `scale` 的静态定义。本轮可视化工具强制关闭该增强。

真实 `scale` 的 shape、生成端定义和人体覆盖合理性尚未通过真实 annotation 与 RGB 验证。

## Third channel

训练代码只能证明：

- ground-truth `proj_verts[...,2]` 会随数组保留下来；
- 仿射和归一化只修改 `proj_verts[..., :2]`；
- position loss 只使用 ground truth xy；
- 模型预测的第 3 通道按 log sigma/位置不确定性优化。

没有真实历史 annotation 统计可以证明其实际存盘值；但 BEDLAM 官方生成端的 `project()` 已提供直接数学证据：标准相机内参左乘 `(X/Z,Y/Z,1)` 后，第 3 通道为齐次图像坐标常数 1。详见 [`BEDLAM_OPERATOR_AUDIT.md`](BEDLAM_OPERATOR_AUDIT.md)。

```text
历史 DenseKP NPZ 的实测值 = UNKNOWN
BEDLAM 官方 preprocessing 的生成值 = 1（齐次图像坐标）
```

## Flip audit

1. **图像与变换：FACT。** `generate_image_patch_cv2` 在 `do_flip=True` 时先执行 `img[:, ::-1, :]`，再把中心改为 `img_width - center_x - 1`。随后生成的 `trans` 对“已经镜像后的图像坐标系”执行普通正向仿射；矩阵本身不包含把原图点 x 坐标镜像到新坐标系所需的负向 x 映射。
2. **普通 keypoints：FACT。** `get_example_projverts` 直接用 `trans` 变换传入的普通 keypoints，未在 `do_flip` 分支中调用 `fliplr_keypoints`，也未使用 `flip_kp_permutation`。
3. **138 个 `proj_verts`：FACT。** 它们同样直接通过 `trans`，没有 x 镜像，也没有 point ID permutation。
4. **风险结论：FACT。** 若启用 horizontal flip，图像会翻转，但普通 keypoints 和 138 个表面点不会执行对应几何镜像/左右语义置换，存在明确的监督错位风险。继续保持 horizontal flip 关闭。

本阶段只记录风险，不修改实现。

## BEDLAM generator requirements

BEDLAM 官方 preprocessing 和两个 operator 的专项审计已经明确了 generator 接口：

- `imgname`：`[N]`，复现官方相对于对应 RGB root 的场景/图像路径；
- `center`：`[N,2]`，前 22 个投影关节经过图外过滤后的 tight bbox 中心，使用原图像素坐标；
- `scale`：`[N]` scalar，`max(xmax-xmin,ymax-ymin)/200*1.2`；
- `proj_verts`：`[N,138,3]`，由 `densekp_regressor_R @ smplx_vertices` 后按官方相机投影；xy 为原图像素坐标，第 3 通道为齐次坐标 1；
- `part` 或 `gtkps`：仅当需要复现极端裁剪行为时提供，不能替代 DenseKP 主监督；
- 与审计有关的相机、可见性、人体模型和表面算子元数据应使用独立字段保存；不得把 `proj_verts[...,2]` 当作 visibility、confidence 或 depth。

可以进入极小规模 BEDLAM generator 实现阶段；在扩大数据规模前，仍必须完成至少 5 个生成样本的原图/bbox/点位叠加和裁剪变换误差验证。

## 本轮产物

- 扩展 `tools/inspect_annotation.py`：支持 keys、shape、dtype、样本数、路径示例、center/scale/proj_verts 统计、第三通道 percentile/unique、NaN/Inf、sentinel 和 `--index`。
- 新增 `tools/visualize_annotation.py`：复用当前 `get_example_projverts`，支持原图 bbox + 138 点、256x256 无增强裁剪 + 反归一化点、point ID、指定 index、坐标误差报告。
- 未生成 `outputs/annotation_audit/original/sample_*.png` 和 `cropped/sample_*.png`，因为没有可信真实训练 annotation；避免用推理预测冒充 ground truth。
