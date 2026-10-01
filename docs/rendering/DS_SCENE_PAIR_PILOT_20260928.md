# 双视角边缘截断人体：小批场景验证

后续已接入真实参考图提取的 DS 镜头遮罩，见 [遮罩更新](DS_LENS_MASK_20260928.md)。
本页原有指标对应 `ds_scene_pair_20260928_final/`；带遮罩的新产物位于
`ds_scene_pair_20260928_lens_mask/`，旧产物保留。

## 范围与相机约定

场景由用户指定为 `cycling_forest` 和 `urban_street_assets`；相机方向由用户明确为
**同一水平高度、180° 相反**。本批复用 20260320 cam2/cam4 的 Double Sphere 内参。
镜头中心间距使用显式试验参数 **0.18 m**，并未得到实际硬件间距确认。
因此这是共享世界几何的合成双视角试验，不是实测相机外参的完整复现。

两相机各自从不同光心渲染同一批固定人体。`rig.json` 同时记录：

- `physical_ds_camera_to_world`：真实成像方向意义上的合成 DS 相机位姿，两路光轴相反。
- `camera_to_world`：与生产 normal ERP 一致的虚拟全景坐标系，两路轴向对齐但光心不同。

cam4 的 180° 旋转在 DS 投影/重映射中应用一次。每路先独立渲染完整 ERP，再经过自己的
DS fisheye→normal ERP 变换；没有分别平移人体或对同一全景做不同 roll 冒充双视角。
沿用 `assets_external/real_erp_valid_masks.npz` 的有效区轮廓，并按试验分辨率缩放。
该本地资产未包含采集来源元数据，正式数据集仍需固定和核对其具体采集来源。

## 最终产物

根目录：`outputs/experiments/ds_scene_pair_20260928_final/`，每场景包含：

| 文件 | 含义 |
| --- | --- |
| `rig.json` / `people.json` | 内参、两套相机坐标约定、试验基线，以及两路共同使用的人体布局 |
| `source/` | 带原生场景光照的完整 ERP RGB、Depth、IndexOB，以及渲染 manifest |
| `{cam2,cam4}/normal_erp.png` | DS 往返后的 DenseKP 输入栅格，已应用有效区掩码 |
| `{cam2,cam4}/fisheye.png` | DS 投影图；有效 ERP 区域在后续 normal ERP 中实施 |
| `annotations.npz`（各相机目录） | 138 点 xy、DS xy、模型域、入视野、表面可见性、出视野/视野内遮挡等独立字段 |
| `boxes.json`（各相机目录） | 实例掩码得到的 oracle 框，xyxy、右下边界不包含；没有运行检测器 |
| `pair_annotations.npz` | 相同世界 3D 点、两路 xy、共同可见掩码 |
| `pair_review.jpg` | 两路完整 normal ERP 标注对照 |
| `diagnostic_loader.npz` / `crop_review.jpg` | 用真实 DatasetKeypoints 检查的诊断输入和逐人双路裁剪对照 |
| `report.json` / `validation.json` | 生成统计与数值/数据加载验收 |

旧目录 `ds_scene_pair_20260928/` 是初始布局试验，不作为最终样例；其中近处人物与后方人物
的三维包围盒重叠，存在穿插风险。最终布局增大了间距，要求各人体 AABB 至少沿一个轴有
5 cm 分离；该检查不等同于衣物或人体与全部环境物体的碰撞认证。

每个场景使用同一身份/动作序列的四个固定姿态：左右边缘各一个人、一个后方遮挡人、一个
完整单视角对照。没有逐相机改变同一个人的姿态或 shape；本批也不构造 train/val 划分。

裁剪对照图中绿色为可见监督点，灰色为其余投影点（包括视野外点）；所有点均来自合成
几何标注，**不是 DenseKP 网络预测**。`no box` 表示有效图像中没有该人的实例像素。

## 实现与验证边界

- 复用 `render_equirectangular_source.py`，新增可选 `--scene-blend` 和 `--device`。
  使用 `open_mainfile` 保留原生 World、灯光及场景材质；不改写 `.blend` 资产。
  原 HDRI-only 路径保持默认行为。
- 两份场景资产原先关闭了 `render.use_compositing`，加载后必须开启，才能输出深度和实例 EXR。
- RGB 与实例 ID 经过相同的两段几何映射，RGB 使用线性插值，实例 ID 使用最近邻。
- 表面可见性仍采用现有的同一实例 ID + 射线距离近似：源 ERP 上 5×5 邻域内误差小于 4 cm。
  这是近似标签，尤其衣物下的身体表面点和轮廓处仍需审查，不能称为精确逐点遮挡真值。
- 第三字段在诊断 loader 中显式定义为二值监督掩码；模型 `keypoint_value` 未当作置信度。
- 四项新增回归检查：水平/反向/间距关系，物理相机与 ERP 坐标投影一致，平移产生视差，非法间距拒绝。
  完整回归共 **35 项通过，0 跳过**。
- 额外检查：两路共享 3D 点逐项相等、138 点与仓库 regressor 一致、DS 投影逆投影、真实 loader
  `[3,256,256]` 输入有限、裁剪 xy 逆变换、第三字段保持不变、标注与实际渲染 manifest 一致。

RGB 为 1536×768 ERP、Cycles 8 samples 的低采样几何试验，存在明显噪声；服装沿用简化材质。
本生成验收阶段没有运行 DenseKP 网络，也没有修改检测、双视角人体重建或合图逻辑。
随后已完成 [现有模型预测诊断](DS_SCENE_PAIR_PREDICTION_20260928.md)，预测产物保存在独立目录，
不改变本页生成标注与几何验收结果。
后续效果评价仍以**重建前的原始 2D 点误差**为主，重建修正的收益另行记录。

## 最终验收结果

两个场景的 `validation.json` 均通过；每场景 4 人、2 路完整图像，有实例框的
6 个人体视图进入实际 loader。后方遮挡人和完整单路对照在 cam4 中无实例框。

| 检查 | cycling_forest | urban_street_assets |
| --- | --- | --- |
| 实际 loader 裁剪逆变换最大误差（原图像素） | 0.0000192 | 0.0000192 |
| 138 点与 `R @ vertices` 的最大误差（米） | 0.000000119 | 0.000000119 |
| 人物 1/2/3/4 的双路共同可见点数 | 6 / 25 / 0 / 0 | 6 / 25 / 0 / 0 |
| 两路共享世界 3D 点、同高反向光轴、人体 AABB 分离 | 通过 | 通过 |

共同可见点按上述近似可见性规则统计。两个场景复用相同人体与相机相对布局，因此这些
统计相同；这不构成独立姿态/身份样本，也不是跨场景泛化结果。人物 1 只有 6 个共同
可见点，后续数据应覆盖不同共同可见程度，不能把本批布局当作统一训练模板。

已人工查看最终全景标注和逐人裁剪图；噪声、简化服装、有限姿态和近似遮挡标注仍是
正式扩量前的限制。按用户要求不计算或保存 SHA256；`report.json` 记录映射矩阵路径，
其内容关系通过实际矩阵乘法与已有 3D 点直接比对。

## 复现

在项目根目录、可访问 Blender CUDA 的环境中：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. \
/home/yy/conda_envs/camerahmr/bin/python -B \
  tools/fisheye/run_ds_scene_pair_pilot.py \
  --output outputs/experiments/new_ds_pair --baseline-m 0.18 \
  --device CUDA --width 1536 --samples 8 \
  --fisheye-reference-root /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/Output/20260320_riding_dataset3/image/fisheye \
  --reference-frame 02214

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/densekp-audit-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
/home/yy/conda_envs/camerahmr/bin/python -B \
  tools/validate/validate_ds_scene_pair_pilot.py outputs/experiments/new_ds_pair
```

默认 Blender 路径是本机已有安装，可用 `--blender-bin` 指定。`--device CPU` 可做 CPU 渲染。
`--skip-render` 只重建诊断标注，读取已保存的相机与人体布局，并检查与渲染 manifest 一致。
已完成的渲染目录不会在普通运行中自动覆盖；换新输出目录进行新布局试验。

## 后续工作

先审阅两组 `pair_review.jpg` 和 `crop_review.jpg` 是否覆盖预期截断类型，再扩展距离、
边缘位置、遮挡程度、真实骑行姿态、身份、服装及光照。正式训练前应提高渲染质量、明确可见性
监督策略，并使用与训练独立的真实 `20260320-riding-dataset3` 固定框样本评估。
本批的单身份重复姿态和 oracle 框不满足正式训练语料/效果验证要求。
