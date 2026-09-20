# 训练数据接口（TRAINING_DATA_FORMAT）

> 从 `densekp/train_keypoints.py` 反向追踪到每个 batch 字段。
> 只记录代码中真实存在的字段，不虚构。

## 1. 反向追踪链

```text
python densekp/train_keypoints.py data=densekp_train experiment=densekp
  └─ @hydra.main(config_path=configs/hydra, config_name=train.yaml)     [train_keypoints.py:103]
  └─ dataset_cfg = dataset_config()  → densekp/core/configs/__init__.py（冻结空壳）  [:47]
  └─ DataModule(cfg, dataset_cfg)    [datamodule_keypoints.py:10]
       └─ train_dataset_prepare(): cfg.DATASETS.DATASETS_AND_RATIOS.split('_')
            → 每个 ds 一个 DatasetKeypoints(cfg, ds) → torch ConcatDataset      [:24]
       └─ val_dataset_prepare(): cfg.DATASETS.VAL_DATASETS.split('_') → is_train=False
  └─ model = DenseKP(cfg)            [densekp_trainer.py:22]
  └─ trainer.fit(model, datamodule, ckpt_path=cfg.get('ckpt_path'))
```

数据集名称 → 路径的映射在 `densekp/core/configs/__init__.py`：

```python
self.img_dir = DATASET_FOLDERS[dataset]                    # dataset_keypoints.py:33
self.data   = np.load(DATASET_FILES[is_train][dataset])    # [1]=train, [0]=val    [:34]
```

当前 `configs/hydra/data/densekp_train.yaml` 选择 28 个 BEDLAM v1 SMPLX 数据集 + val `emdb-smpl`。

## 2. NPZ 标注字段（DatasetKeypoints 实际读取）

| 字段 | 必需性 | shape / dtype | 坐标空间 | 含义（代码语义） |
|---|---|---|---|---|
| `imgname` | 必需 | `[N]` str | 相对路径 | `os.path.join(img_dir, imgname)`（dataset_keypoints.py:63） |
| `scale` | 必需 | `[N]` 或 `[N,2]` float | 边长/200 | `bbox_size = expand_to_aspect_ratio(scale*200, BBOX_SHAPE).max()`（:59；BBOX_SHAPE 未配置 → None） |
| `center` | 必需 | `[N,2]` float | 原图像素 (x,y) | crop 中心（:57-58） |
| `proj_verts` | 必需 | `[N,138,3]` float | xy=原图像素 | DenseKP 监督（:46）；ch2 不参与 loss |
| `part` | 条件 | `[N,J,3]` | 像素+置信度 | 仅极端裁剪决策用（:40-41） |
| `gtkps` | 条件（part 缺失时） | `[N,J,≥3]` | 同上 | 备选来源（:42-43）；都没有则全零 → 极端裁剪不触发 |

注意：`data['imgname']` 等按 NPZ 键名读取；`center`/`scale` 按 `self.scale[index]`、`center[0]` 用法看是**逐样本**（`[N,2]` 与 `[N]`）。

## 3. __getitem__ 输出 item 字段（dataset_keypoints.py:51-97）

| 字段 | shape | dtype | 坐标空间 | 来源 |
|---|---|---|---|---|
| `img` | `[3,256,256]` | float32 | ImageNet 归一化 RGB（BGR→RGB） | get_example_projverts → convert_cvimg_to_tensor |
| `img_disp` | `[256,256,3]` | uint8 | BGR（未归一化） | img_patch_cv |
| `proj_verts` | `[138,3]` | float32 | **原图像素**（未 crop，ch2 原样） | `self.proj_verts[index].copy()`（在 trans 变换**之前**另存副本） |
| `proj_verts_cropped` | `[138,3]` | float32 | crop 中心原点，xy∈[-0.5,0.5]；ch2 原样（GT 为 1.0） | get_example_projverts 内 trans 变换 + `xy/256-0.5` |
| `box_center` | `[2]` | — | 原图像素 | `[center_x, center_y]`（可能含 tx/ty/extreme-crop 平移） |
| `box_size` | 标量 | — | 原图像素 | bbox_w（= bbox_size，方形） |
| `img_size` | `[2]` | float32 | — | `[H,W]`（'closeup'/'portrait' 数据集名会 90° 旋转后取值） |
| `_scale` | 同 NPZ | — | — | 原始 scale |
| `_trans` | `[2,3]` | float32 | — | 本次 crop 的仿射矩阵（cv2.getAffineTransform 输出） |
| `imgname` | str | — | 绝对路径 | join 后的路径 |
| `dataset` | str | — | — | 数据集注册名 |

关键点：**图像和 proj_verts 使用同一个 `trans`**（utils.py:601-621）——这是 RGB 与 DenseKP GT 几何一致性的核心保证。

## 4. model / loss 侧消费

```text
batch['img']  → DenseKP.forward_step: x[:,:,:,32:-32] → backbone → head → pred [B,138,3]
batch['proj_verts_cropped'] → compute_loss: gt[:,:,:2] vs pred[:,:,:2]
   with_var=True: mean(1000·d²/(2σ²)) + mean(log σ²)；否则 mean(d²)；再 ×60   [densekp_trainer.py:85-123]
val: batch['proj_verts'](原图) → trans_points2d_parallel(batch['_trans']) → /256-0.5   [:159-161]
优化: AdamW(backbone+head)，cfg.TRAIN.LR / WEIGHT_DECAY   [configure_optimizers]
```

## 5. synthetic generator 必须产出什么

要让当前 `DatasetKeypoints` 正确读取，生成器（如 `tools/annotation/generate_bedlam_densekp.py`）必须：

1. **NPZ**（路径写入 `DATASET_FILES[1]` 注册表，或新增注册项）：
   - `imgname`：相对 `DATASET_FOLDERS[ds]` 根目录的路径字符串；
   - `center [N,2] float`：原图像素 bbox 中心；
   - `scale [N] float`：`max(bbox_w,bbox_h)/200`（决定是否含 1.2 扩张由生成端定，需与 center/scale 语义一致）；
   - `proj_verts [N,138,3] float32`：`R @ V` 后经与 RGB 相机一致的投影得到的**原图像素 xy**，第三通道写 1.0（BEDLAM 官方约定）；
   - 可选 `part`/`gtkps`（若不提供，极端裁剪增强静默退化为全零关节）。
2. **图像**：放在 `DATASET_FOLDERS[ds]` 指向的目录。
3. **注册**：数据集名加入 `densekp/core/configs/__init__.py::DATASET_FOLDERS`（[1] 训练 dict 的 `DATASET_FILES`）与 `configs/hydra/data/*.yaml` 的 `DATASETS_AND_RATIOS`。
4. **不要在数据集名中使用 'closeup' / 'portrait' 子串**，除非确认标注坐标对应旋转 90° 后的图像（get_example_projverts 会按数据集名旋转 RGB，utils.py:540-541）。
5. 建议同时输出非训练用的审计数组（如 camera-stress 的 densekp_valid/visible/depth），方便 `tools/annotation/inspect_annotation.py` 与 overlay 验收 —— 但这些字段不会被 loader 读取。
6. flip 增强：当前配置 `DO_FLIP=True` 且 `FLIP_AUG_RATE=0.5` 在 yacs 默认里，但 densekp.yaml 的实验数据集为 BEDLAM 集合——上游审计确认 138 点镜像置换未定义，**训练前必须确认所用配置不触发 flip**（TRAINING_AUDIT.md 警告；见 POTENTIAL_ISSUES #4）。

## 6. 现成参考实现

- `tools/annotation/generate_bedlam_densekp.py` — 最小可运行生成器（248 行，含校验与 debug NPZ）；
- `tools/validate/validate_bedlam_densekp_tiny.py` — 用真实 `DatasetKeypoints` 走一遍 crop 并可视化；
- `tools/annotation/inspect_annotation.py` / `tools/annotation/visualize_annotation.py` — 人工核对工具。
