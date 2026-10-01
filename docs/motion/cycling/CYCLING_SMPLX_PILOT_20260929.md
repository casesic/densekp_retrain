# 骑行图片 → SMPL-X → DS 合成 → 小规模训练（2026-09-29）

> 用户要求人物穿衣，后续已新增[独立骑行服接入](../../rendering/CYCLING_DRESSED_20260929.md)。
> 本页渲染和训练数字仍对应原裸身版本，不得复用为穿衣版指标。

已实际完成两个独立姿态的 SMPL-X 拟合、两个带原生灯光场景的渲染、138 点标签验收，
以及 8 步全模型训练和权重重载检查。**流程通过，定位效果验收未通过；原始权重保持不变。**

## 交接复核

- 读取了指定入口及其几何、相机、裁剪、监督调用链。旧文档含冻结审计状态，按当前源码判断。
- 输入 manifest 的 12 张图片均存在、尺寸一致；12 份 NPZ 均为 SMPL `[1,72]`、
  `[1,10]` betas、`[6890,3]` 顶点。重新从导出参数独立还原全部网格，最大坐标差
  `5.3644e-7 m`。没有重新下载图片/权重或重跑 HMR2。
- 从保存数组重算：116 个可见 COCO 身体点，平均误差 `6.133431` 完整 256-square px。
  保存的 HMR2 配置确实含 COCO train2014；这些结果仍是姿态获取，不能作为独立泛化评测。
- 检查两个候选的原图裁剪、SMPL 叠加和侧视图；初筛仍不构成真实 3D 验证。
- 当前仓库与外部 `densekp_delivery/camera` 的 20260320 两相机内参、cam4
  `diag(-1,1,-1)` 轴变换一致。`rig_poses` 使用同高度的反向光轴，光心间距 0.18 m
  是合成参数；normal ERP 的虚拟轴向一致。没有混用 0311 标定。
- GPU 0 在开始渲染和训练前均无计算任务。GPU 1 的既有 Blender GUI 未干预。

复核记录：[handoff_verification.json](../../../outputs/experiments/cycling_smplx_20260929/handoff_verification.json)。
本轮没有计算 SHA256，没有修改外部 pano-stitch，没有提交 commit，保留了原工作树和旧实验。

## SMPL-X 拟合方法和范围

入口：[convert_cycling_smplx.py](../../../tools/experiments/convert_cycling_smplx.py)。
复用本地 `pano-stitch/tools/smpl_models/transfer/smplx_to_smpl.pkl`，其矩阵与
CameraHMR `data/train-eval-utils/smplx2smpl.pkl` **逐项相同**。
检查了矩阵 `[6890,10475]`、重心权重行和、三角形顶点索引，并由 `bc/closest_faces`
独立重建矩阵，确认其三角形属于所加载 SMPL-X 拓扑。

拟合目标是通过这个正向算子映射 SMPL-X 网格，逼近保存的 SMPL 网格。
在文件提供的 5184 个 `valid_vertices` 上，LBFGS 优化 root、body pose、translation、
10 个 shape 参数，附加弱 shape/pose 正则。SMPL-X betas 从零初始化；共享身体旋转仅作
优化起点，不是转换结果。每个人单独拟合、单独保存，没有共享 shape 或连续序列。

这是本地实现的逆向表面拟合，**不是**直接运行官方 deformation-transfer 工具。
[SMPL-X 官方说明](https://github.com/vchoutas/smplx/tree/main/transfer_model)也明确指出模型间
shape 空间不同，不能直接复制参数。这里不把官方工具的能力等同于本实验验收结果。

预先写入脚本的试验门槛是对应子集 mean ≤15 mm、P95 ≤30 mm，参数/轴变换还原 ≤`2e-5 m`。
这只是本轮几何接入门槛，不是业务精度标准，也不是对真实人体的误差。

| 独立姿态 | 子集 mean / P95 / max mm | 全网格 mean / max mm | 轴变换还原 max m |
|---|---:|---:|---:|
| coco_292226_person_1330527 | 7.26 / 17.27 / 31.70 | 16.25 / 74.45 | 3.58e-7 |
| coco_306800_person_225160 | 6.05 / 12.84 / 21.08 | 9.90 / 43.48 | 2.38e-7 |

手指保持 flat neutral，jaw/eyes/expression 为零，没有恢复精确握把。全网格残差明显大于
拟合子集，不能用表中子集指标声称所有部位都精确。多方向预览支持躯干和屈腿的大体姿态
得以保留，尚无独立深度、接触或自相交验收。这些资产仅供受限合成试验。

- [转换报告](../../../outputs/experiments/cycling_smplx_20260929/report.json)
- [竞速姿态四方向对照](../../../outputs/experiments/cycling_smplx_20260929/coco_292226_person_1330527/conversion_review.jpg)
- [城市姿态四方向对照](../../../outputs/experiments/cycling_smplx_20260929/coco_306800_person_225160/conversion_review.jpg)

每个人目录有 `fit_parameters.npz`（图像相机轴下拟合结果）、`canonical_pose.npz`、
`geometry/{vertices,faces,densekp_3d}.npy`、`motion_params.npz` 和 `geometry_manifest.json`。
`fps=0, sequence=false` 明确表示单张图片姿态，不应送入要求正 fps 的时序工具。

## 同源几何和场景

HMR2 图像相机轴变为 y-up canonical 轴：`Q=diag(1,-1,-1)`。保留估计姿态的俯仰，
不使用 HMR2 相机平移，不声称恢复重力方向或真实世界朝向。SMPL-X root 绕 shaped pelvis
旋转，因此同时补偿 `transl' = Q(transl + pelvis) - pelvis`；直接检查变换后所有顶点。
随后由同一组顶点计算 `R[138,10475] @ V`，没有使用 BEDLAM 点索引或 Aaron 标签。

渲染和标注都依次使用 `(x,-z,y)`、身体最低 y 的地面抬升、Z 轴 yaw 和场景平移。
本试验场景平移的 z 是 0.25 m，最低身体顶点因而悬于地面 0.25 m；没有自行车或脚踏接触。
两个姿态各自作为独立资产，`people.json` 每个人显式保存 `geometry` 路径与 `frame=0`。
重复放置同一资产不增加独立人物或姿态数量。

入口：[render_cycling_ds.py](../../../tools/experiments/render_cycling_ds.py)。
使用两套原生 3D 场景、已有身体纹理，无旧动作专属 clothing mesh。
Blender 5.1.1 / Cycles CUDA / 32 samples / seed 29 / animated seed off / denoising off。
源 ERP 与最终 ERP 为 1536×768；鱼眼保持 2880×2880，LUT 按实际 ERP 尺寸计算。
外观仍有采样噪声，且只有一套身体纹理，不足以代表真实衣着分布。

遮罩继续使用 20260320 `02214.jpg` 提取的各镜头轮廓，与现有 ERP 有效区取交集。
没有用合成 RGB 是否黑色判断标签有效性；该轮廓仍是单帧估计，非精密光学测量。
镜头外、实际裁剪外的点不监督；有效裁剪内的遮挡点仍监督。预测第三值原样保留。

补齐的验证缺口：原 `validate_ds_scene_pair_pilot.py` 写死 Aaron 几何，现在按人物资产读取。
Blender 额外保存每个人实际的世界网格和物体矩阵；验证器直接检查 `R @ rendered_world_vertices`
与标注点，检查几何来源、相机投影、有效区及 RGB 遮罩，不能只验证两份旧标签互相一致。

| 检查 | 森林 | 街景 |
|---|---:|---:|
| 给定框 loader 行数 | 7 | 7 |
| 有效监督点（含遮挡） | 684 | 684 |
| 其中可见 / 遮挡 | 389 / 295 | 389 / 295 |
| 有效视野内但裁剪外点 | 3 | 3 |
| 实际渲染网格 → 138 点最大差 m | 8.03e-7 | 1.63e-7 |
| 裁剪坐标往返最大差 px | 1.25e-5 | 1.25e-5 |

两场景均包含两侧边缘截断、后方重遮挡和完整对照，共四个实例。cam2 后方人物仅露出
9×69 原图像素的实例框；其 15 个 in-view 点中 12 个进入监督，3 个在裁剪外。
完整对照在 cam4 无框，此视图不运行 DenseKP，也不计为检测失败。
人物 AABB 间距均超过 5 cm；未验证人体与全部场景物体的无穿插。

- [森林裁剪和 GT](../../../outputs/experiments/cycling_ds_20260929/cycling_forest/crop_review.jpg)
- [街景裁剪和 GT](../../../outputs/experiments/cycling_ds_20260929/urban_street_assets/crop_review.jpg)
- 两个场景目录各有 `validation.json`、`diagnostic_loader.npz`、`source/`、`rig.json` 和所有遮罩。

## 训练结果：流程通过，效果下降

入口：[train_cycling_smoke.py](../../../tools/experiments/train_cycling_smoke.py)。
原始 `assets_external/densekp.ckpt` 严格加载；森林 7 行训练、街景 7 行验证。
两个集合共享人物、姿态、布局和纹理，**不是独立 train/val split**，也未评测真实图片精度。

训练全 backbone/head，batch=1，AdamW，原学习率 `1e-5`、weight decay `1e-4`，
`with_var=True`，无数据增强，seed 29。训练 BF16 mixed，前后诊断 float32、TF32 off，
同一批原图、框、affine、GT，不按预测有效性缩小分母。

- 实际 8/8 次 AdamW 更新，所有步的 backbone 389 / head 107 个梯度张量均有限。
- backbone/head 参数探针最大变化 `7.218e-5 / 7.206e-5`，峰值分配显存 16.69 GiB。
- 街景验证覆盖 7/7 行，无零监督样本；可见/遮挡组均参与验证。
- 保存后严格重载，float32 预测最大差为 0。
- 仅保存权重，不含 optimizer 状态；本轮没有宣称完整训练恢复验收。

下表是按点等权、完整 256-square px 的平均误差；Lightning `val_loss` 则按人平均、
使用 BF16 验证且单位为归一化裁剪坐标，不能直接混比。

| 场景 | 分组（点数） | 训练前 px | 8 步后 px |
|---|---|---:|---:|
| 森林 | 全部监督点（684） | 8.26 | 18.08 |
| 森林 | 可见（389） | 5.84 | 15.65 |
| 森林 | 裁剪内遮挡（295） | 11.44 | 21.27 |
| 街景 | 全部监督点（684） | 7.52 | 16.96 |
| 街景 | 可见（389） | 5.63 | 15.20 |
| 街景 | 裁剪内遮挡（295） | 10.03 | 19.29 |

双视角同语义点的较差视角误差也上升，原始重遮挡窄框失败仍然存在。
不能将这次下降归因于某个单独因素：样本太少、框歧义、短训练、学习率、外观域和
异方差目标都尚未通过受控对照区分。BF16 的有限梯度只证明这 8 步数值可运行，不证明
长期稳定；本轮也没有证明遮挡监督策略优于其他策略。

[训练完整报告](../../../outputs/experiments/cycling_train_smoke_20260929/report.json)中
`status=passed` 指流程，`quality_status=regressed` 指效果。
`cycling_smoke_weights.ckpt` 仅供诊断，不替换或推荐部署。
`before/after/<scene>/` 保存每个框的全部 138×3 原始预测、误差、分组、配对指标和预览。

## 复现与后续

三个入口输出目录必须不存在。复现使用新目录，勿覆盖本次证据：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp-cycling-mpl
PY=/home/yy/conda_envs/camerahmr/bin/python
PANO=/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch
$PY -B tools/experiments/convert_cycling_smplx.py \
  --input outputs/experiments/cycling_hmr2_20260929 \
  --output outputs/experiments/cycling_smplx_repeat \
  --smpl-model outputs/downloads/cycling_image_pilot_20260929/SMPL_NEUTRAL_numpy.pkl \
  --model-dir "$PANO/tools/smpl_models/smplx" \
  --transfer "$PANO/tools/smpl_models/transfer/smplx_to_smpl.pkl"
CUDA_VISIBLE_DEVICES=0 $PY -B tools/experiments/render_cycling_ds.py \
  --converted outputs/experiments/cycling_smplx_repeat \
  --output outputs/experiments/cycling_ds_repeat \
  --fisheye-reference-root "$PANO/data_process/Output/20260320_riding_dataset3/image/fisheye"
$PY -B tools/experiments/train_cycling_smoke.py \
  --pilot-root outputs/experiments/cycling_ds_repeat \
  --output outputs/experiments/cycling_train_smoke_repeat --steps 8 --gpu 0
```

本轮相关测试 13 项通过（独立资产、轴/抬升矩阵、相机、镜头遮罩、遮挡监督）。
实际两场景 loader/渲染网格检查和 GPU 训练重载是额外的运行证据。

下一轮先扩充已初筛的独立骑行人物和姿态，并按源人物/图片划分训练与验证；建立真实
给定框评价和完整人体回归对照，再做学习率/更新范围的受控小试验。当前数据只能验证
接入，不能据此启动长时间正式重训。保留窄框失败，不能通过丢弃错误预测改善指标。
手部先验、衣着多样性、采样噪声、独立遮挡标签检查和真实镜头边界仍是数据质量缺口。
