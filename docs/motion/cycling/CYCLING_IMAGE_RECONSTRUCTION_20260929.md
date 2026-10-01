# 公开骑行图片 → SMPL 姿态试验（2026-09-29）

> 后续已核验本页结果，并完成两个候选的 SMPL-X 转换、带遮罩场景合成和 8 步训练。
> [后续验收报告](CYCLING_SMPLX_PILOT_20260929.md)：流程通过，定位效果下降，未替换基线。
> 下文“尚未转换”保留为本页原始阶段状态。

## 已完成什么

从公开 V-COCO/COCO 图片中下载 12 个骑行人物，使用本地 4D-Humans 代码和作者发布的
HMR2 权重，实际完成 12 次 GPU 重建。每人保存独立 SMPL 参数、网格、原图叠加和侧视图。
本轮没有运行 SAM 3D Body，没有继续 BVH 拟合，也没有修改 DenseKP 生产代码。

**这是 12 个独立估计姿态，不是连续 motion，也不是经过验证的 3D 真值。**
目前尚未转换为 SMPL-X，尚未放入两套场景渲染，不能直接交给现有 138 点生成器。

先看结果：

- [原图与目标人物框](../../../outputs/motion_inputs/cycling_images_20260929/image_review.jpg)。
- [全部重建：输入裁剪 / 网格叠加 / 侧视图](../../../outputs/experiments/cycling_hmr2_20260929/reconstruction_review.jpg)。
- [优先候选：竞速骑行](../../../outputs/experiments/cycling_hmr2_20260929/coco_292226_person_1330527/review.jpg)。
- [优先候选：城市骑行](../../../outputs/experiments/cycling_hmr2_20260929/coco_306800_person_225160/review.jpg)。

## 数据选择与来源

| 数据源 | 本轮判断 |
| --- | --- |
| [V-COCO](https://github.com/s-gupta/v-coco) + [COCO](https://cocodataset.org/) | 采用。用 `ride=1` 的人物及其 `instr` 关系确认自行车，排除其他骑乘对象。 |
| [MPII Human Pose](https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/software-and-datasets/mpii-human-pose-dataset/download) | 有动作分类，可继续找骑行；本轮未下载其完整图像包。 |
| [Cyclist Actions](https://doi.org/10.5281/zenodo.4080940) | 公开内容为 motion-history 图等，原始 RGB 不在该发布中，未用于这轮重建。 |
| [Aschaffenburg Pose Dataset](https://zenodo.org/records/5724486) | 主要是姿态/轨迹数据；本轮未作为 RGB 图片来源。 |

筛选实现：[prepare_cycling_images.py](../../../tools/experiments/prepare_cycling_images.py)。

1. V-COCO `train` 中 `ride=1`，角色 `instr` 对应 COCO 类别 `bicycle`。
2. 人物非 crowd，框短边至少 100 px、高至少 180 px，距图像边界至少 2 px。
3. 得到 15 个人物—自行车候选对；按人物框面积排序，每张图最多取一人，下载前 12 张。
4. 本次 12 张均来自 COCO `train2014`。使用标注框，未运行人体检测器。

这里选相对完整、较大的源人物，是为了先获得可检查的姿态；最终训练所需的边缘截断、
多人遮挡和 DS 畸变应在重渲染阶段控制。框距边缘 2 px 也不等于人体必然完整或无遮挡。

`outputs/motion_inputs/cycling_images_20260929/manifest.json` 保存图像/人物/自行车 ID、
COCO 框、17 点标注、下载 URL、原 Flickr URL、每张图的 license。图片许可并不统一，
包含 NC、ND、SA 等不同条款；对外发布数据时需要逐图核对，不能统一称为任意用途可用。

本轮按框面积筛选，有明显偏差：多张是打电话或单手骑行。它能验证获取姿态这条路线，
还不能代表骑行动作分布。HMR2 下载包的训练配置明确包含 COCO train2014，
所以这些重投影数字只用于筛选姿态，不能作为独立测试集的模型泛化成绩。

## 重建方法与文件约定

入口：[reconstruct_cycling_hmr2.py](../../../tools/experiments/reconstruct_cycling_hmr2.py)。
复用 `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/4D-Humans`，未修改外部项目。
原用户模型缓存无读取权限，因此下载 [4D-Humans 作者的 HMR2 模型包](https://www.cs.utexas.edu/~pavlakos/4dhumans/hmr2_data.tar.gz)。
本地包名虽然以 `.tar.gz` 结尾，实际是未压缩 tar；以实际文件格式解包。

```text
COCO 原图 + person bbox xyxy
  → 官方 ViTDetDataset：框扩展至 192:256 比例，取最大边作为方形裁剪边长
  → RGB 标准化图像 [1,3,256,256]
  → HMR2 forward（骨干使用中间 192 列）
  → SMPL global/body rotation matrices、betas、估计相机
  → axis-angle 参数 + 6890 顶点网格
```

权重以 `strict=True` 加载。推理使用 RTX 3090、PyTorch 2.0.0+cu118、float32，
固定随机种子并关闭 TF32；这些设置不构成跨软硬件逐 bit 一致的保证。
原 SMPL pickle 包含 Chumpy 对象，本轮只在输出目录制作了 NumPy 版本，未更改原资产。

每人目录 `outputs/experiments/cycling_hmr2_20260929/<sample_id>/`：

| 文件 / 字段 | 内容 |
| --- | --- |
| `review.jpg` | 输入裁剪、SMPL 叠加、旋转 90° 的模型视图；第三张不是实际第二相机观测 |
| `body.obj` | SMPL 6890 顶点网格，包含预测 global orientation，未加估计相机平移 |
| `smpl_prediction.npz: global_orient` | `[1,3]`，axis-angle，弧度 |
| `body_pose` | `[1,69]`，23 个 SMPL 身体关节旋转 |
| `poses` | `[1,72]`，上述两项拼接，**不是 SMPL-X 的 165 维** |
| `betas` | `[1,10]`，该人物估计体型；不同人物不拼成共享 shape 的序列 |
| `vertices_model_m` | `[6890,3]`，SMPL 模型尺度下的米，绝对物理尺度没有图像外的测量保证 |
| `camera_*` | 官方 HMR2 估计的弱透视参数、透视平移与焦距；不是世界轨迹，不是 DS 标定 |
| `box_center / box_size` | 原图像素中心 / 完整方形裁剪边长 |
| `projected_openpose25_crop_normalized` | `[25,2]`，中心为 0、边长为 1 的完整方形裁剪坐标 |
| `coco_body_gt / coco_body_error_crop_px` | 12 个身体关节的原图 COCO 标注 / 裁剪像素重投影误差 |

12 个结果都通过了有限值、顶点形状、旋转矩阵↔axis-angle 检查。
另用独立 `smplx.SMPL` 实例从导出的 axis-angle 与 betas 重新生成网格，
最大坐标差为 **5.37×10⁻⁷ m**，小于检查容差 `2×10⁻⁵ m`。
见输出中的 `export_verification.json`；此检查也已加入后续重建脚本。
它证明导出可还原，不证明重建符合真实 3D 人体。

## 检查结果与候选选择

重投影只比较 COCO 肩、肘、腕、髋、膝、踝 12 点与 HMR2 OpenPose 对应关节。
目标裁剪坐标 `(original_xy - box_center) / box_size`，与预测比较后乘以 256。
COCO `v=2` 为可见，`v=1` 为有标注但不可见，`v=0` 不参与。
这些语义只属于本数据源；不是 DenseKP `keypoint_value` 的解释。

全批可见身体关节共 116 个，按关节汇总平均误差 **6.13 px**；单位是完整 256 方形
裁剪像素。每例均值的中位数为 5.17 px。关节定义和衣服遮挡也会影响误差。

以下为原图叠加与侧视图的人工目视初筛，没有自动验收阈值，也未验证真实深度或脚踏接触：

| image ID / person ID | 可见点数 | 平均误差 px | 初筛决策与原因 |
| --- | ---: | ---: | --- |
| 292226 / 1330527 | 12 | 5.13 | **优先候选**：前倾竞速、双手向前、双腿屈伸；适合先验证转换。 |
| 306800 / 225160 | 10 | 4.96 | **优先候选**：城市骑行，主体叠加较贴合；仍需检查遮挡侧手腿。 |
| 396853 / 209194 | 11 | 4.97 | 补充候选：夜间单手骑行/打电话；不代表双手握把姿态。 |
| 227186 / 483359 | 10 | 3.46 | 补充候选：单手骑行/打电话；正面深度仍有歧义。 |
| 3988 / 191596 | 10 | 3.53 | 补充候选：单手骑行；篮子遮挡处的腿部深度尚未核验。 |
| 394583 / 1702101 | 12 | 5.21 | 补充候选：单手抬起；后续需确认腕和腿的空间关系。 |
| 456545 / 1277729 | 10 | 5.45 | 补充候选：背面骑行；前臂/手深度仅由单视图估计。 |
| 530758 / 198525 | 10 | 4.12 | 补充候选：背面举手；适合作为动作多样性候选。 |
| 390315 / 254262 | 11 | 8.28 | 暂缓：强运动模糊，下肢细节和位置不稳定。 |
| 31813 / 197828 | 5 | 8.72 | 暂缓：冲浪板遮挡、可见身体关节少，手臂姿态不适合先验收。 |
| 387124 / 200738 | 6 | 6.86 | 暂缓：背面遮挡重，模型手臂下垂；全部有标注身体点误差 10.13 px。 |
| 193025 / 200903 | 9 | 15.74 | 暂缓：低清且手臂/上身叠加偏差明显，先排除出转换候选。 |

## 复现入口

下载的原始标注和权重保存在 `outputs/downloads/cycling_image_pilot_20260929/`。
图像获取脚本接收 V-COCO JSON 和官方 `annotations_trainval2014.zip`。
输出目录必须不存在；复跑请用新的目录，避免覆盖已审查结果。

```bash
python tools/experiments/prepare_cycling_images.py \
  --vcoco outputs/motion_inputs/cycling_images_20260929/vcoco_train.json \
  --coco-annotations outputs/downloads/cycling_image_pilot_20260929/annotations_trainval2014.zip \
  --output outputs/motion_inputs/cycling_images_repeat --limit 12

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/densekp-audit-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYOPENGL_PLATFORM=egl \
/home/yy/conda_envs/camerahmr/bin/python -B tools/experiments/reconstruct_cycling_hmr2.py \
  --hmr-repo /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/4D-Humans \
  --model-assets outputs/downloads/cycling_image_pilot_20260929/hmr2_assets \
  --smpl-model outputs/downloads/cycling_image_pilot_20260929/SMPL_NEUTRAL_numpy.pkl \
  --input outputs/motion_inputs/cycling_images_20260929 \
  --output outputs/experiments/cycling_hmr2_repeat --device cuda:0
```

## 下一步：先做两个姿态的 SMPL-X 转换验证

1. 从两个优先候选开始，使用经过验证的模型对应关系/拟合将 SMPL 转成 SMPL-X，
   检查几何残差和多方向视图。不能把 72 维补零到 165 维，也不能默认 betas 跨模型等价。
2. 当前 `generate_real_motion_geometry.py` 的 DenseKP regressor 是 `[138,10475]`，
   不能乘本轮 `[6890,3]` SMPL 顶点；转换验收前不产生声称有效的 DenseKP GT。
3. 每个姿态单独建资产，明确从图像相机朝向到场景朝向的转换，去掉估计相机平移。
   虚空骑行允许没有车，但人体姿态、摆放高度、图像和投影标签仍要一致。
4. 再接入 `cycling_forest`、`urban_street_assets`：2880² 鱼眼及实测遮罩、边缘截断、多人遮挡。
   渲染网格和标注必须读取同一个新几何；不能继续引用旧 Aaron 动作的 138 点。
5. 扩展图片时补充双手握把、不同踩踏相位和更多侧面；若需要连续 motion，另找视频并做时序处理。

SAM 3D Body 可用于候选对照，但它输出 MHR，需要单独的模型转换；
相关官方工具见 [MHR 模型转换](https://github.com/facebookresearch/MHR/tree/main/tools/mhr_smpl_conversion)。
本轮实际运行的是 4D-Humans，未把 SAM 3D Body 的可用性或结果当作已完成项。
