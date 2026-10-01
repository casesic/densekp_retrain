# 骑行素材与 SMPL-X 拟合交接（2026-09-29）

> 后续决策：用户指定优先从公开 **2D 骑行图片重建人体姿态**。本页保留早先 BVH
> 输入整理记录，BVH 拟合暂停，不再作为当前下一步。实际图片与重建结果见
> [骑行图片重建试验](CYCLING_IMAGE_RECONSTRUCTION_20260929.md)。

## 本轮范围

用户确认可以“虚空骑车”：需要骑行人体姿态/动作，可从数据集或图像拟合，暂不要求
自行车资产、车体渲染或人车接触求解。新人体最终进入已有 3D 场景、DS 鱼眼遮罩和
138 点投影流程。当前已有本地骑行 mocap，可先用这两段完成拟合试验。

已完成片段整理与原始数据复核；**尚未完成 BVH→SMPL-X 拟合，不能将输入包当作训练标签。**
本轮没有下载新数据，没有把 BVH Euler 角直接写入 SMPL-X。

## 1. 已准备的输入包

`outputs/motion_inputs/cycling_fit_20260929/`：

| 片段 | 原始文件名 | 原始帧区间（0 起、右端不含） | 帧数 / 实际 fps | 动作证据 |
| --- | --- | --- | --- | --- |
| participant_1，主选 | `1_RB_M42_20230719_PK.bvh` | `[7560,7800)` | 240 / 23.9999808 | 约 72 RPM、双脚反相、双手较稳定 |
| participant_2，备选 | `2_RB_M27_20230809_PA.bvh` | `[5700,6000)` | 300 / 30.0000300 | 约 78.26 RPM、手臂较直的骑行姿态 |

原始文件位于 `/home/yy/datasets/cycling_mocap/candidates/` 的同名参与者目录。
来源与历史审计见 [专门骑行 mocap](DEDICATED_CYCLING_MOCAP_AUDIT.md)；输入包保留了
已有 Zenodo 元数据快照和作者/来源信息。无需自行车模型即可进行人体拟合。

每个片段目录包含：

- `source_clip.bvh`：原层级、原 Frame Time、精确截取的原始数值行；适合交给其他 BVH 工具。
- `clip_native.npz`：`channels_native[T,372]`、`joint_positions_native[T,75,3]`、
  `joint_names[75]`、`parents[75]`、`source_frame_ids[T]`、标量 `frame_time`，以及 22 个拟合关节的两侧索引。
- `manifest.json`：来源、帧区间、完整层级、单位边界和复核结果。
- `kinematic_diagnostic.png`、`clip_selection.json`、`kinematic_review.json`：此前的运动学诊断与选段证据。

75 个节点包括 62 个带通道关节和 13 个 End Site；它们不是 DenseKP 的 138 个语义点。
根目录 `BVH_TO_SMPLX_JOINT_MAPPING.csv` 是 22 个主要身体关节的拟合对应设计，
不是局部旋转可以直接复制的证明。

实际重新读取了两份原 BVH、计算所选帧 FK，与已审计位置按 `atol=1e-10`、`rtol=0`
比较通过；重新读取导出的 BVH 后，通道值逐项相等，层级和帧间隔相同。

复现入口：

```bash
python tools/experiments/prepare_cycling_fit_inputs.py \
  --output outputs/motion_inputs/cycling_fit_20260929
```

输出目录必须尚不存在。脚本复用已有 BVH parser/FK，没有重新实现旋转或计算 SHA256。

## 2. 拟合阶段需要解决的内容

1. **单位和轴向。** BVH 没有显式单位声明；骨长支持厘米推断，`0.01 m/unit` 仅是待校准
   候选。本包保持原值。先确认转换尺度、上方向和 rest frame，记录源坐标到模型坐标的变换。
2. **身体 pose 和固定 shape。** 用 22 个身体关节与时间连续性拟合，保持一段动作的 betas
   固定；检查膝肘方向、肢体交叉、手脚轨迹、踩踏相位和 root 漂移。
3. **关节映射。** BVH `LeftShoulder` 对应 collar，`LeftArm` 才对应 shoulder；Foot 对应
   ankle。局部 Euler 顺序为 ZXY、单位为度，不能当成 SMPL-X axis-angle 弧度直接复制。
4. **未可靠拟合的自由度。** 手指、jaw、eyes 可先用显式中性值，记录哪些参数来自拟合、
   哪些来自先验。不要把 4 节 BVH 手指直接逐项映射到 SMPL-X 的 3 节链。
5. **拟合验收。** 保存源骨架与模型骨架叠加、至少一个踩踏周期的多视角预览，以及按关节
   的残差和时间连续性结果。拟合残差阈值需要结合尺度、关节定义确认，不能只看文件能否加载。

## 3. 给现有 adapter 的输出约定

消费者：[load_real_smplx_motion.py](../../../tools/generate/load_real_smplx_motion.py)。
建议拟合器输出一个不含 pickle 对象的 `fitted_motion.npz`：

| 字段 | 要求 |
| --- | --- |
| `poses` | 浮点 `[T,165]`，SMPL-X 的完整 axis-angle，弧度 |
| `betas` | 浮点 `[B]`，整段固定；B 与所选模型支持的形状维度一致 |
| `trans` | 浮点 `[T,3]`，模型坐标系中的米；明确 root 是否已经去平移 |
| `gender` | 标量字符串，与实际拟合使用的模型一致 |
| `mocap_frame_rate` | 正的有限标量；显式填写，避免 adapter 缺省为 30 |

`poses` 的切片由当前 adapter 明确定义：

```text
0:3      global_orient
3:66     body_pose（21×3）
66:69    jaw_pose
69:72    leye_pose
72:75    reye_pose
75:120   left_hand_pose（15×3）
120:165  right_hand_pose（15×3）
```

所有参数必须有限。模型使用 `use_pca=False, flat_hand_mean=True`；输出手部 axis-angle
必须与这个约定一致。expression 当前不从此 adapter 传入，几何生成使用模型默认值。
若同时输出 `global_ori`，必须与 `poses[:,:3]` 一致。

另外提供拟合说明 JSON：源片段路径、源帧映射、单位转换、轴转换、模型版本/性别、shape
处理方式、优化设置、残差和先验参数。输入包帧 0 对应原始 BVH 7560 或 5700；后续 adapter
的 `source_frame_ids` 索引的是**拟合 NPZ 内的帧**，需要保留这层对应关系，不能误称原始 BVH 帧号。

拟合交付后可先抽一个周期的少量帧检查，例如对主选片段：

```bash
python tools/generate/load_real_smplx_motion.py fitted_motion.npz \
  --output outputs/motions/cycling_fit_preview/canonical.npz \
  --frame-ids 0,3,6,9,12,15,18,21

python tools/generate/generate_real_motion_geometry.py \
  --motion outputs/motions/cycling_fit_preview/canonical.npz \
  --model-dir /path/to/smplx_models \
  --output-dir outputs/geometry/cycling_fit_preview
```

这是未来交付后的命令示例，本轮没有生成 `fitted_motion.npz`。默认不加
`--use-source-translation` 时，canonical `transl` 为零，原始 translation 另存；
需要源平移时显式加该参数。抽帧没有改变存储的源 fps，预览播放应结合实际源帧索引间隔。

## 4. 几何接入和后续渲染边界

`generate_real_motion_geometry.py` 输出 `[T,10475,3]` SMPL-X vertices、faces 和
`R[138,10475] @ vertices → [T,138,3]` DenseKP 几何，并保存 canonical 参数和 manifest。
渲染器 `render_equirectangular_source.py` 已接受 `--geometry` 和可选 `--clothing-geometry`。

实际接入前还有三项需要处理：

- `run_ds_scene_pair_pilot.py` 和其 validator 目前固定读取 Aaron 1095 几何；选人 frame
  也是原动作的 0/8/16/24。必须在新几何验收后将相关入口统一参数化，不能只替换渲染网格而保留旧 GT。
- 当前服装序列绑定原动作；新动作必须配套服装几何或先省略服装做几何验收，不能直接套旧服装。
- `world_points` 和渲染器均按每帧身体最低 Y 点抬至地面。虚空骑行序列若逐帧这样处理，
  踩踏脚的高度变化会引入整体升降。连续骑行应选固定的整段抬升/摆放规则，并在图像与 GT
  两边一致应用；本轮没有修改现有基线的摆放行为。

若将来恢复 BVH 路线，具体交付是：**先把 participant_1 的一个周期拟合并验收为 SMPL-X，再接入两场景。**
participant_2 用于检查另一参与者和较直手臂姿态；完整训练集仍需更多独立身份、动作及划分。
