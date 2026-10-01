# 骑行合成的穿衣版本（2026-09-29）

用户指出裸露上身不适合作为当前骑行训练外观。因此在两个已转换的独立人体上新增
基础贴身骑行上衣和短裤，旧版输出和八步训练报告保留。

## 服装与标签

旧 Aaron 服装来自特定 BEDLAM 人物/动作的衣物序列，不能直接套用到新姿态。
本次使用 [prepare_cycling_clothing.py](../../tools/experiments/prepare_cycling_clothing.py)：

- 根据每个人的 SMPL-X shape，在 rest pose 定义上衣、袖口、腰线和短裤区域。
- 从该人的 posed body 生成约 4 mm 外偏的服装表面；对髋部和腋下局部折叠修正方向。
  两人的最大顶点外偏为 5.38 / 5.42 mm，衣物面心相对对应人体面的法向间隔至少约 0.4 mm。
- 两人分别使用红色/蓝色上衣、浅色横条、深色短裤。它是独立几何表面，参与场景光照、
  阴影、深度和实例渲染；不是在渲染后的图片上涂色。
- 服装属于简化贴身原型，没有布料动力学、真实衣服褶皱或完整碰撞检测，领口和袖口仍
  受人体网格分辨率限制。本次未补鞋袜或恢复原图服装。

人体网格及其 `R @ V` 138 点保持不变。衣物与人体使用同一物体变换，包括**按人体最低点
计算的抬升量**。不能给衣物单独落地。Blender 保存实际世界衣物网格，验证器检查它与
相同变换的数值一致性；渲染前也检查服装对应的源人体顶点没有变化。

人物实例 mask 包含身体和衣物。可见性仍沿用 5×5 源 ERP 邻域、同实例和 4 cm 深度容差，
因此这里的 `visible` 是近似的穿衣人体表面可见性，不是“裸皮肤是否暴露”。
遮挡、镜头无效、裁剪排除的监督规则保持原约定；重新渲染后重新计算状态、框和 loader。

## 产物

- 服装：`outputs/experiments/cycling_clothing_20260929_v2/`，每个人独立目录，包含
  `clothing_manifest.json`、网格、材质、人体对应索引/参考坐标以及四方向预览。
- 穿衣场景：`outputs/experiments/cycling_ds_dressed_20260929/`。
- 每场景 `validation.json` 保存实际渲染及 loader 验证数值，`crop_review.jpg` 用于外观和 GT 检查。
- 旧裸身输出 `cycling_ds_20260929/` 继续保留。原 `cycling_clothing_20260929/` 是最初
  法向偏移被局部向内检查拦截的空目录，不是可用服装包。

复用原两个人体、场景、布局、镜头遮罩和相机设置；原生灯光、32 samples、seed 29，
鱼眼 2880×2880、ERP 1536×768。**没有针对穿衣版重新训练或推理**，之前八步训练的
指标不能当作穿衣版本的指标。

两场景实际渲染和 loader 验证均通过，各 7 个给定框视图、684 个监督点，其中 300 个遮挡点。
与旧版逐项比较，八个人物实例的 Blender 世界人体网格、全部 138 点世界坐标、2D 投影和
`in_view` 均相同；可见性由新深度重新计算，不复用旧状态。
森林/街景衣物变换最大差分别为 `3.96e-7 / 8.18e-8 m`，裁剪往返最大差 `1.25e-5 px`。

- [森林穿衣裁剪预览](../../outputs/experiments/cycling_ds_dressed_20260929/cycling_forest/crop_review.jpg)
- [街景穿衣裁剪预览](../../outputs/experiments/cycling_ds_dressed_20260929/urban_street_assets/crop_review.jpg)
- [与裸身版的几何/状态比较](../../outputs/experiments/cycling_ds_dressed_20260929/comparison_to_unclothed.json)

复现时使用新目录：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-cycling-mpl
PY=/home/yy/conda_envs/camerahmr/bin/python
$PY -B tools/experiments/prepare_cycling_clothing.py \
  --converted outputs/experiments/cycling_smplx_20260929 \
  --output outputs/experiments/cycling_clothing_repeat
CUDA_VISIBLE_DEVICES=0 $PY -B tools/experiments/render_cycling_ds.py \
  --converted outputs/experiments/cycling_smplx_20260929 \
  --clothing-root outputs/experiments/cycling_clothing_repeat \
  --output outputs/experiments/cycling_ds_dressed_repeat \
  --fisheye-reference-root /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/Output/20260320_riding_dataset3/image/fisheye
```

相关测试 15 项通过，含衣物不改动人体、衣物共用人体抬升，以及既有相机、独立资产和
遮挡监督检查。首次新增测试的 `1e-12 m` 比较不适用于保存的 float32 衣物；误差为
`6.88e-8 m`，随后改为 `2e-7 m`，仍可检测毫米级错误的独立抬升。

## 目录约定

当前仍放在 `outputs/experiments/`。DenseKP loader 接收 NPZ 和图片根目录，技术上不要求
图片位于 `datasets/`；目录名称也不代表数据质量通过。

外观、标签和样本选择确定后，再打包为 `datasets/<名称_版本>/`：固定按源人物/图片的
训练与验证划分，保留来源和监督策略，处理图像、mask、几何等依赖与相对路径，再跑实际
loader 验证。现有目录含绝对路径，不能直接 `mv` 后假定仍然可用。
原实验、审查图和模型运行记录留在 outputs。见 [datasets 约定](../../datasets/README.md)。
