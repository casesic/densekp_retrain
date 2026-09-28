# DenseKP 正确性修复与边界重构（2026-09-28）

基线：`8553ccd2d3117755e595ae97c8522c303bd05ceb`。考古阶段完成后，经用户授权实施。
范围：给定人体框的 DenseKP 数据加载、裁剪、监督、训练与验证。目标场景仍为
`20260320_riding_dataset3`，20260320 标定；检测器、相机投影和 pano-stitch 源码未改动。

## 当前组件边界

| 组件 | 职责及输入输出 |
| --- | --- |
| `densekp/core/keypoint_contract.py` | 检查 NPZ 必需字段、样本数、138 点形状、center/scale、标签权重语义与数值 |
| `densekp/core/crop_geometry.py` | 像素坐标仿射、按 `(width,height)` 归一化、精确仿射逆变换；只处理 xy |
| `densekp/core/datasets/dataset_keypoints.py` | NPZ → 图像、裁剪标签、原标签、仿射；关闭 NPZ 文件，尊重增强开关 |
| `densekp/core/datasets/utils.py` | 图像增强与采样；沿用原几何函数导入名，以免破坏其他调用方 |
| `densekp/core/keypoint_objective.py` | float32 监督损失、按人平均的裁剪误差与有效/空监督人数 |
| `densekp/core/datasets/datamodule_keypoints.py` | split 与 loader；验证保留尾批，worker=0 可用 |
| `densekp/core/training_config.py` | 检查有效配置、初始化/恢复冲突、不支持的手动优化控制 |
| `densekp/core/checkpointing.py` | 当前权重验证结束后保存并排名，避免旧分数配新权重 |
| `densekp/train_keypoints.py` | seed、配置快照、Hydra Trainer 实例化、训练/测试调度 |

```text
NPZ + 显式标签语义
  → validate_annotations
  → DatasetKeypoints → get_example_projverts → crop_geometry
  → DataModule → DenseKP.forward_step
  → 原有 ViT（输入内部切为 256×192）→ 原有 138×3 head
  → keypoint_loss / keypoint_error_totals
  → 验证结束 → ValidationCheckpoint
```

模型参数名称和结构保持原有 ViT + head；旧 checkpoint 的 tensor state_dict 结构兼容。
旧 checkpoint 的 ModelCheckpoint 回调状态与新回调类型不同，继续训练时排名历史可能重建；
真实大型 checkpoint 的加载、GPU 前反向和导出仍须独立验证。

## 数据与坐标契约

- `imgname`: 非空 `[N]` 字符串；`center`: 有限 `[N,2]`，标签所在栅格的像素 xy。
- `scale`: 正有限 `[N]` 或 `[N,2]`，单位是像素 / 200。保持既有扩框和最终正方形裁剪规则。
- `proj_verts`: 浮点 `[N,138,3]`。形状检查不能证明 BEDLAM、SMPL-X 或 DenseKP 索引一致。
- **必须声明 `DATASETS.TARGET_VALUE_SEMANTICS: supervision_weight`**，且先确认生产器语义。
  标签第三通道须在 `[0,1]`；正权重点的 xy 必须有限；零权重 xy 可为未定义值。
  声明是调用方的责任，不是从数值推断语义。不同语义的数据须另写经过审查的适配器。
- 模型第三输出仅在 `with_var=True` 的既有损失中解释为 log sigma。
  外部 `keypoint_value` 不会因此被认定为置信度或推理权重。
- 图像 `[3,256,256]`；标签 xy 为完整裁剪空间 `(pixel_xy / [256,256]) - .5`。
  ViT 内部 `32:-32` 宽度切片不改变标签空间，恢复时不加 32 像素偏移。
- `_trans` 是标签所在栅格到裁剪的仿射。`restore_crop_xy` 返回该栅格的 xy。
  portrait 预旋转、ERP roll、ROI 偏移的外层逆变换仍由调用方处理。
- 监督掩码沿用生产器策略，不因随机裁剪自动重定义；visible-only 与 amodal 策略仍待实验明确。

损失保留原有 60 倍系数、可选方差分支的 1000 倍 xy 项以及
`max(sum(weights),1)` 分母。有限 float32 样本与旧方差公式数值一致。
零权重先屏蔽再计算，避免 `NaN * 0`；方差项改为 float32 的 `exp(-2*log_sigma)`
与 `2*log_sigma`，避免半精度中间方差下溢。此修复不保证任意异常输出的梯度都有限。
非有限预测、非法权重、非有限有效标签明确报错；不会静默忽略失败的预测。

## 已处理的审计项

| 基线项 | 本次处理 | 范围/限制 |
| --- | --- | --- |
| C01 水平翻转 | 启用时提前报错 | 未提供经过验证的 138 点左右置换，因此不声称已支持翻转 |
| C02 增强开关 | `use_augmentation=False` 生效 | 验证仍默认不增强 |
| C03 缺失默认数据配置 | 提交 `data/unselected.yaml`，默认完整配置可组合 | 未选数据集时禁止启动；本地旧 `data/train.yaml` 保留 |
| C04 Trainer 配置被忽略 | 实例化实际 Hydra Trainer 配置；CPU/GPU 配置、seed、train/test 生效 | 手动优化的累积/裁剪明确拒绝；DDP 尚未实机验证 |
| C05 验证尾批/空结果 | 保留尾批，累计误差和有效人数；逐数据集报告 | 全零监督的主验证集报错，禁用验证时不生成排名指标 |
| C06 掩码 NaN | 算术前屏蔽无效坐标 | 不掩盖有效点或模型输出中的 NaN |
| C08 反向抗锯齿判断 | DenseKP 活跃路径按源/目标缩小倍率判断 | 继承的通用 `get_example` 分支未整理 |
| C09 worker=0 | 仅多进程 loader 设置 prefetch | 训练仍 drop_last，样本数不足 batch 时给出明确错误 |
| C10 平移概率 | `TRANS_AUG_RATE` 生效 | 随机数消耗序列改变，旧训练轨迹不能逐位复现 |
| R01/R06 数据契约 | 显式第三字段声明、结构与数值校验 | 不证明数据来源或 138 语义映射正确 |
| R04/R05 评价/复现 | 分数据集人数、零监督计数、seed、有效配置快照 | 独立验证集、完整环境固定和恢复随机轨迹仍需建立 |

### 复查新增：checkpoint 排名与权重可能错配

基线 `train_keypoints.train` 将 `every_n_train_steps` 传入 `ModelCheckpoint`。
该回调在训练 batch 结束时读取 `trainer.callback_metrics['val_loss']`；此时当前 batch
后的验证尚未发生。`DenseKP.training_step` 已更新模型参数，旧验证分数可能被用于保存的新权重。
只要期间参数更新，这就不能保证分数与权重对应。若 checkpoint 不参与排名、参数未变或回调实际
安排在当前验证之后，则此具体触发条件不成立。

现改为 `ValidationCheckpoint`：每经过 `GENERAL.CHECKPOINT_STEPS` 个更新后，在第一次完成的
验证结束处保存。保存周期可能向下一次验证延后。`last` 也在这些验证保存点更新；这不是独立的
训练中途故障恢复定时器。仍用 **loader 0 的 `val_loss`** 排名；其他验证集分别报告，不混合加权。

验证指标：每人的有效点加权平均 L2，再对有效人平均，单位为归一化完整裁剪坐标。
乘 256 可得到当前固定输入尺寸的裁剪像素误差。零监督人单独计数，不把它们当零误差。
各进程累计量做 SUM 归约。Lightning DDP sampler 若补齐重复行，会影响唯一人数和平均值；
正式精确评测建议单设备，或先验证不补齐的分布式评估方案。
`test=True` 复用注册的验证 split，并非新增独立测试集；指标使用 `test/…` 名称。

## 验证证据

同步到当前目录后，完整回归 **31 项全部通过，0 跳过**（原有 18 项 + 新增 13 项）；
`git diff --check` 通过。未启动全量训练或真实 ViT 的 GPU 训练。

1. `tests/test_densekp_boundaries.py`：旋转/矩形裁剪与逆变换、增强后的图像与标签对齐、
   第三字段不被空间变换、翻转拒绝、平移概率、抗锯齿方向、标签契约、遮挡 NaN、
   半精度方差算术、旧方差公式对照、按人指标，以及实际训练入口保存/恢复/测试。
2. Lightning 集成测试使用真实 NPZ、DataModule、DenseKP 手动优化、AdamW、logger 和
   checkpoint 回调；仅 ViT/head 替换为小网络，在 CPU 上从 step 0→2 保存，再恢复至 step 4。
   batch=2 的三人验证集保留 `[2,1]` 两批；两个验证 loader 均记录三人。
   检查 checkpoint 保存分数与当前验证分数一致、优化器状态和参数更新。
3. 本机五套 tiny 数据共 33 行，对照基线和 pano-stitch 相同 bbox：
   - 原输入张量、裁剪标签和 `_trans` 最大差均为 **0**。
   - pano-stitch 归一化输入最大差 **2.3842e-7**。
   - 新逆仿射恢复原标签最大差 **3.5241e-5 像素**。
   - pano-stitch 恢复同一标签最大差 **1.5259e-5 像素**。
   - 本次未执行真实模型推理；这些是预处理与恢复一致性证据。
   - 逐样本记录：[crop parity JSON](densekp_crop_parity_20260928.json)。

运行回归（在项目根目录，使用本机现有环境）：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/densekp-audit-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
/home/yy/conda_envs/camerahmr/bin/python -B -m unittest discover -s tests -v
```

此测试环境是 Torch 2.0.0+cu118；其他版本兼容性没有据此得到保证。

## 后续训练会话的使用方式

无数据参数的配置现在可组合，但会明确拒绝训练，因为目前尚未选正式 NPZ。
先核对具体生成器、数据路径、标签语义与划分，再显式选择数据配置。
已提交的 `p0_dressed_tiny`、`ds_multiperson_tiny`、`omni_near_erp_tiny` 只用于 smoke。

例如仅检查组合结果（不会启动训练或加载权重）：

```bash
python densekp/train_keypoints.py --cfg job data=ds_multiperson_tiny trainer=cpu
```

正式运行必须显式选择合适的 batch、step、验证间隔和 checkpoint 来源。
`init_checkpoint` 是权重初始化，`ckpt_path` 是完整状态恢复，不能同时指定。
现有其他本地 data YAML 若缺少语义字段，应在确认生产器后加入字段；不要批量盲填。
诊断 evaluator 新增必需参数 `--target-value-semantics supervision_weight`，其
百分位/可视化报告只接受二值标签掩码，避免把任意连续值当作可见性。

### 全量重训仍需关闭的门槛

- [ ] 明确正式 NPZ、138 行语义对应、生产器与源输入 provenance。
- [ ] 建立按 motion/identity/scene 等独立单元隔离的验证集；tiny 集不能作为效果依据。
- [ ] 使用 20260320 dataset3 的固定人体框比较旧/新 DenseKP；多人、畸变、遮挡分组评分。
- [ ] 确定遮挡及输入外点的监督策略，核对裁剪后可见范围；不借此改动检测器召回。
- [ ] 跑真实 ViT/head 的目标精度前反向、保存/恢复及部署输出对照。
- [ ] 若启用 Omni 数据分支，解决相机 domain-validity 丢失（C07）；本批未改生成器。
- [ ] 如启用多设备，验证 sampler、指标归约和实际训练配置；固定环境和输入资产哈希。

本报告描述当前改动；仓库根目录四份考古报告保留基线事实，通过顶部说明指向这里。
