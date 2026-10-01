# 跟着一个样本读完整条链

配合 [阅读地图](READING_GUIDE.md) 使用。这里讲的是已运行的街景遮挡样本，数值来自现存
NPZ 和真实 Dataset 读取结果；本轮没有重新训练或生成图片。

**2026-09-29 更新：遮挡点也参与监督和验证。** 下文的原始样本路径及坐标数值保留，
它们来自 09-28 的可见点版本。新版标签和重新评分在
`outputs/experiments/ds_occlusion_amodal_20260929/`，模型预测没有改变。
完整规则和结果见 [遮挡点监督说明](training/OCCLUDED_SUPERVISION_20260929.md)。

## 0. 先确定我们要跟踪谁

选择 **heavy 条件、cam2、人物 3**，即被前景人物遮挡的后方目标。

```text
原图：outputs/experiments/ds_occlusion_control_20260928/heavy/cam2/normal_erp.png
标签：outputs/experiments/ds_occlusion_control_prediction_20260928/heavy/fixed_box_loader.npz
预测：同一 prediction 目录下 heavy/cam2_person_3.npz
```

此 loader NPZ 有 5 行，当前人物是第 2 条样本（索引 1）：cam2 的人物 2/3/4，以及 cam4
的人物 2/3。**一个样本是一张图中的一个指定人物**，因此多个样本可以共享同一张原图。
cam4 的人物 4 在 clear 条件下没有实例框，没有对应样本。

## 1. 138 个标签从哪里来

先看 [body/geometry.py](../src/densekp_retrain/body/geometry.py) 的 `compute_densekp_3d`：

```python
densekp_3d = np.einsum('kv,tvc->tkc', regressor, vertices)
```

这句话表示：每一帧，用 `R[138,10475]` 把 SMPL-X 的 `V[10475,3]` 映射为 138 个 3D 点。
`R` 每一行决定一个语义点由哪些人体网格顶点构成；它与 BEDLAM 的关节数组不是同一种索引。

当前几何来自 `outputs/baselines/temporal_case_aaron_1095/geometry/`：

| 数组 | 当前形状 | 含义 |
| --- | --- | --- |
| `vertices.npy` | `[39,10475,3]` | 39 帧人体网格，最后一维是 xyz |
| `faces.npy` | `[20908,3]` | 每个三角形的三个顶点索引 |
| `densekp_3d.npy` | `[39,138,3]` | 每帧 138 个语义点 xyz |

人物 3 使用几何帧 16，`motion_params.npz` 中对应原始动作帧 80。此资产的 manifest 记录
`source_translation_used=true`。当前 adapter 支持 poses `[T,165]` 的完整 SMPL-X
axis-angle；是否使用 source translation 由生成时的参数决定，不可概括为总是置零。

接着看 [world_points](../tools/fisheye/run_fisheye_multiperson_pilot.py)：几何点从人体的轴
约定转换到 Blender 世界轴，按身体最低点抬到地面，再应用 yaw 和位置。渲染器给 body
和 clothing 应用对应的物体变换。本例人物位置 `[1.8,0.3,0]`、yaw=90°，三个条件不变。

这些 3D 点是生成标注用的。DenseKP forward 本身没有输入这些点或 SMPL-X 参数。

## 2. 3D 世界怎样变成带标注的图片

看 [render_equirectangular_source.py](../tools/fisheye/render_equirectangular_source.py)
和 [process_scene](../tools/fisheye/run_ds_scene_pair_pilot.py)。

1. Blender 打开街景，把固定的人体放入场景；两相机分别从各自光心渲染完整 ERP。
2. 渲染出 RGB、Depth 和 IndexOB。IndexOB 是人物实例 ID，用于区分身体属于谁。
3. `process_scene` 用同一世界点计算投影，并按各自 DS 相机做 ERP→鱼眼→normal ERP。
4. 图像再应用该相机的有效区域 mask。黑色有效区之外没有可用观测。
5. 同时保存各相机 xy、可见状态、实例 mask 和由实例像素得到的框。

上述样例描述的是原有产物。当前 `process_scene` 还在鱼眼阶段应用真实参考图提取的
镜头遮罩：同时裁 RGB 和实例图，并将其映射至 ERP，与已有有效区取交集后再判定
`in_view` / `visible`。新产物目录和对比结果见 [DS 镜头遮罩更新](rendering/DS_LENS_MASK_20260928.md)。

ERP 是经纬度展开的二维图片；这里 normal ERP 是当前 DenseKP 使用的图像栅格。
物理 DS 相机光轴相反，但两路 normal ERP 的虚拟轴向对齐。cam4 的翻转在 DS 投影映射中
处理，不能再额外翻一次图片或坐标。此处的 0.18 m 光心间距是合成试验参数。

区分三个状态：

- `in_view`：投影满足模型域、传感器范围及有效区条件；不说明是否被别人遮挡。
- `surface_visible`：用深度和人物实例 ID 检查表面可见性。
- `visible = in_view & surface_visible`：近似可见点，用于把误差分组；新版同时评价遮挡点。

现有表面检查是源 ERP 的 5×5 邻域、4 cm 距离容差；它是近似规则。本例 cam2 人物 3
有 123 个点落入有效视野，但 heavy 条件只有 22 个被判为可见。

## 3. 从图片和标注，构造一行 Dataset 输入

看 [evaluate_ds_occlusion_control.py](../tools/validate/evaluate_ds_occlusion_control.py)
的 `fixed` 和 `targets` 构造。它从 clear 条件取框，之后冻结：

```text
原图框 xyxy（右下不包含） = [1081, 325, 1183, 503]
center = [(1081+1183)/2, (325+503)/2] = [1132, 414]
方形裁剪边长 = max(102, 178) × 1.2 = 213.6 原图像素
scale = 213.6 / 200 = 1.068
```

这里的 `scale` 是“像素边长除以 200”的存储约定，不是置信度，也不是增强时的随机倍率。
实例框转为 center/scale 是这个 adapter 做的；Dataset 接收时已经是 center/scale。

NPZ 有这些字段：

| 字段 | 整个 NPZ 形状 | 人物 3 样本（索引 1）的意义 |
| --- | --- | --- |
| `imgname` | `[5]` | `cam2/normal_erp.png`，相对图像根目录 |
| `center` | `[5,2]` | `[1132,414]`，normal ERP 像素 |
| `scale` | `[5]` | `1.068` |
| `proj_verts` | `[5,138,3]` | 原图 x、y 和明确生成的二值监督 mask |

文件名 `proj_verts` 容易误导：此处只有 138 个语义点，不是全部 10475 个网格顶点。
人物 ID 用于脚本对应行和实例；这个 NPZ 没把 ID 当作网络特征输入。

原始 09-28 NPZ 的二值 mask 选择可见点。新版额外保存标量
`supervision_policy='in_view_full_crop'`、`in_view[5,138]`、`visible[5,138]`
以及人物/相机 ID。其第三列选择 **有效视野内且落在完整裁剪中的 GT**，遮挡不使其归零。
本例 cam2 人物 3 的监督点由 22 个变为 123 个，其中 101 个被判为遮挡。

## 4. Dataset 怎样同时裁图和变换标签

打开 [DatasetKeypoints.__getitem__](../densekp/core/datasets/dataset_keypoints.py)。
它取出一行、恢复像素边长，然后调用 `get_example_projverts`。

在 [datasets/utils.py](../densekp/core/datasets/utils.py) 中：

```text
cv2.imread → 原图 BGR
generate_image_patch_cv2 → 裁剪图 BGR + affine[2,3]
BGR→RGB、HWC→CHW、颜色标准化 → img[3,256,256]
同一个 affine 作用于标签 xy → 裁剪像素 → xy/256 - 0.5
```

本例 affine 约为：

```text
[[1.1985013, 0, -1228.70350],
 [0, 1.1985020, -368.179832]]
```

实际数值的微小差异来自 float32 框及 OpenCV affine 构造。记 `A` 为这个 2×3 矩阵：

```text
crop_pixel_xy = A @ [original_x, original_y, 1]
target_normalized_xy = crop_pixel_xy / [256,256] - 0.5
```

图像颜色归一化另有含义：`(RGB/255 - mean) / std`。不要把它与坐标归一化混淆。

返回字典最值得关注的字段：

| 字段 | 单样本形状 | 用途 |
| --- | --- | --- |
| `img` | `[3,256,256]` | RGB 标准化后的模型输入 |
| `img_disp` | `[256,256,3]` | 未标准化 BGR 裁剪，用来画对照图 |
| `proj_verts` | `[138,3]` | 原图标签副本 |
| `proj_verts_cropped` | `[138,3]` | 归一化裁剪 xy，以及实际裁剪后的监督 mask |
| `_trans` | `[2,3]` | 原图到完整裁剪的 affine |
| `visible` | `[138]` | 已知的表面可见状态，不是 loss 权重 |
| `visibility_known` | `[138]` | 是否有明确的可见性元数据 |

affine 只变换 xy。旧 NPZ 的第三字段保持原样；声明了新版策略的 NPZ，则由 Dataset
根据**增强后的坐标**重新计算 `in_view & inside_full_crop`。因此平移或缩放后移出裁剪的点
不监督，移入的有效点可以恢复监督。完整裁剪的归一化范围是两轴 `[-0.5,0.5)`。
旧数据没有可见性元数据时，`visibility_known=false`，不能把它当作“全部遮挡”。

当前增强由 `is_train and use_augmentation` 决定；本例明确关闭。训练时尺度、平移、旋转
通过同一 affine 影响图像和点，颜色/降质增强作用于图像。水平翻转目前有保护性报错，
因为尚无验证过的 138 行左右置换；44 关节的置换不能代替它。

## 5. 模型怎样知道我们想预测谁

打开 [DenseKP.forward_step](../densekp/core/densekp_trainer.py)。核心只有这几步：

```python
x = batch['img']                         # [B,3,256,256]
features = self.backbone(x[:,:,:,32:-32]) # 输入 [B,3,256,192]
output = self.head(features)             # pred_keypoints [B,138,3]
```

当前 forward 没有读取人物 ID、bbox 数值、目标 mask、相机内参、另一视角或 GT。
目标的选择体现在**以谁为中心、包含怎样上下文的裁剪图像**里。这也是多人裁剪可能产生
歧义的地方。不能仅凭一个框来自人物 3 的实例 mask，就认为网络收到了“人物 3”这个身份。

模型继续向下：

| 阶段 | 张量形状（当前结构） | 做什么 |
| --- | --- | --- |
| ViT 输入 | `[B,3,256,192]` | 编码裁剪图像 |
| ViT 输出 | `[B,1280,16,12]` | 空间特征图，形状由当前 patch embedding 和 forward 推出 |
| head 展平 | `[B,192,1280]` | 16×12 个图像特征 token |
| head 查询 | `[B,1,1]` 的零 token | 经 TransformerDecoder 读取图像特征 |
| decoder 输出 | `[B,1,1024]` | 一个向量汇总目标信息 |
| `dec_kp` | `[B,414]` → `[B,138,3]` | 直接回归点值，没有热图 argmax |

无需第一遍精读 attention 的矩阵运算，先理解这一层的输入/输出边界。
每个输出点的 xy 仍按完整 **256-square** 定义，裁掉左右 32 列没有把输出坐标系改成 192 宽。
head 没有把输出限制在 `[-0.5,0.5]`；超出裁剪范围的预测不是数组格式错误。

## 6. 三种 `[138,3]`，第三个数不是同一种含义

| 所在位置 | 三列含义 |
| --- | --- |
| `densekp_3d` | 3D 的 x、y、z |
| 本诊断的 `proj_verts` | 2D 的 x、y、生成器明确写出的监督 mask |
| `pred_keypoints` | 预测的归一化 x、y、原始第三输出 |

本地训练在 `with_var=True` 时，把预测第三值作为 log sigma 放进 loss；这不能自动证明
外部权重输出是经过校准的置信度，更不能拿其数值大小给本诊断加权。预测第三值一直原样保存。

读 [keypoint_contract.py](../densekp/core/keypoint_contract.py) 可以看到 Dataset 对标注
语义的要求；读 [keypoint_objective.py](../densekp/core/keypoint_objective.py) 可以看到实际
loss。设 `e²` 为归一化 xy 的平方距离，`w` 为**标注**第三值，`s` 为预测 log sigma：

```text
with_var=False: L = 60 × sum(w × e²) / max(sum(w), 1)
with_var=True:  L = 60 × sum(w × (500 × e² × exp(-2s) + 2s)) / max(sum(w), 1)
```

实现把有效 s 下限限制在 log(1e-6)，并在算差之前移除零权重位置；计算用 float32。
上述是训练目标。验证 `keypoint_error_totals` 算每个人的加权 L2，再按有效人数平均，单位是
归一化裁剪坐标；诊断输出的像素误差则乘 256，在所选 GT 集合内按点等权。
新版分别报告有效视野内全部点、可见点、遮挡点及裁剪外点。预测落在无效镜头区域时，
记录拒绝率，但仍保留该 GT 的预测误差，不能靠删除错误预测降低平均误差。
不要把 `val_loss` 名字理解成与训练 loss 完全相同的公式。

## 7. 用一个真实点检查坐标还原

选择 DenseKP **第 2 行（从 0 开始的语义索引）**，不是第 2 个人。该行在当前样例可见。

```text
GT 原图 xy              = [1117.84021, 331.96893]
GT 归一化裁剪 xy        = [-0.0662911, -0.3840406]
模型原始输出             = [-0.0216906, -0.3941702, 0.5772585]
还原后的预测原图 xy       = [1127.36689, 329.80526]
```

`0.5772585` 不参与 xy 还原或误差加权。看
[restore_crop_xy](../densekp/core/crop_geometry.py)：

```text
pred_crop_pixels = (pred_xy + 0.5) × [256,256]
pred_original_xy = inverse_affine(pred_crop_pixels)
```

这里没有再加 32，也没有再次应用相机旋转。得到的是本例 normal ERP 的坐标；如果外部
应用另有 ROI、旋转或 ERP roll，还需要按它自己的外层变换还原，不能在此处猜测。

GT xy 可以通过相同逆变换回到原图，是几何检查。预测与 GT 的距离是模型误差，两者要分清。

## 8. 训练如何接上这条链

当前这个样本只用于诊断，没有在本轮参与训练。若看正式训练代码，调用关系是：

```text
train_keypoints.train(cfg)
  → validate_training_config + seed + 保存配置
  → DataModule：训练集合拼接、验证集合列表
  → DenseKP：初始化 backbone/head，可加载初始权重
  → Lightning Trainer.fit
      → training_step(joint_batch)
          batch = joint_batch['img']
          forward → compute_loss → zero_grad → manual_backward → AdamW.step
      → validation_step → 汇总每人误差
      → ValidationCheckpoint：在验证后按 val_loss 保存
```

`joint_batch['img']` 是训练 loader 包装出来的**样本字典**；其中 `batch['img']` 才是
`[B,3,256,256]` 图像张量。诊断直接传第二层字典，所以看起来少了一层。

训练 loader 会丢弃不满 batch 的尾批，验证 loader 不丢；多个验证集分别累计统计，
当前 `val_loss` 取 loader 0。`init_checkpoint` 只初始化权重，`ckpt_path` 恢复训练状态。
新版另外记录 `val/crop_l2_visible_0` 和 `val/crop_l2_occluded_0`，只对有可见性元数据的
监督点分组；某组没有有效样本时不输出该组误差，记录空样本计数。`val_loss` 仍取总体。
模型构造仍会先加载 ViTPose backbone 权重；它与完整 DenseKP checkpoint 是两份资产。

实际部署和跨相机重建不在这条训练调用链里。本次两个视角是分别跑单图模型，之后由
诊断脚本按相同人物/语义行汇总误差，没有在模型内部融合双视角。

## 9. 第一遍读完应记住什么

1. 数据生成器决定正确的 138 行语义和 GT，Dataset 不会替你把 BEDLAM 索引转换正确。
2. 一行样本对应一个目标框，框通过图像内容向网络表达目标。
3. 图像与 GT xy 共用 affine；颜色标准化与坐标归一化是两件事。
4. 网络看 192 宽，输出仍是完整 256-square 坐标。
5. 标注 mask、预测第三值、3D 的 z 必须分开理解。
6. 可见点减少、输入框移动、模型误差增加是不同变量，诊断需要分别控制。

再回看最近的框实验就容易理解：两种方形裁剪边长相同，但中心移动 22.5 个原图像素，
模型接收的目标位置和邻人上下文发生变化。实验确认输出对这种变化敏感；内部 attention
为何选择某个人，尚不能仅从误差曲线推出。
