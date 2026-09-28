# 真实 DenseKP 模型验收（2026-09-28）

修复提交：`35ad01e`。运行工具：
`tools/validate/verify_densekp_model_lifecycle.py`。

## 结论

真实 ViT + 138×3 head 在 RTX 3090 上完成了权重加载、混合精度前反向、
backbone/head 参数更新、完整状态保存和独立进程恢复。固定同一验证样本在保存前与恢复后
的预测最大差为 **0**，恢复后的 AdamW 又完成一次更新。

**这是有限步骤的功能验收，不代表混合精度长期稳定或边缘截断人体效果达标。**
初始 16 个训练 batch 有 14 次 AMP 跳步，只实际更新了 2 次；这是本次最重要的新增发现。
`global_step` 仍增长，因此后续训练不能只用它判断模型是否在学习。

## 实际运行

- Torch `2.0.0+cu118`，Lightning `2.0.2`，GPU 0：RTX 3090 24 GiB。
- 输入 `assets_external/densekp.ckpt`；backbone 初始化使用
  `assets_external/vitpose_backbone.pth`。原始权重未改写。
- `p0-dressed-tiny`，12 train / 4 validation，batch=1，worker=0，不增强。
- `16-mixed`，`with_var=True`，默认 GradScaler 初始 scale=65536，原 AdamW 学习率。
- 每个训练 batch 后验证全部 4 人；P0 标签为全一监督，不能据此验收遮挡掩码策略。
- 初始 DenseKP checkpoint 来自 Lightning `2.0.7`；当前 `2.0.2` 发出了版本提示。
  本次实际加载/使用通过，但正式实验仍应固定并验证完整依赖版本。

| 阶段 | Lightning 步数 | 实际 AdamW 更新 | AMP 跳步 | 最终 scale | 结果 |
| --- | --- | --- | --- | --- | --- |
| 首次严格探针 | 第一批即停止 | 0 | 发现首个溢出 | — | 探针提前报错，随后改为观测 GradScaler 自动恢复 |
| 12-batch 探针 | 0→12 | 1 | 11 | 32 | 未达到至少两次更新的验收条件 |
| 正式有界检查 | 0→16 | 2 | 14 | 4 | 功能检查通过，保留数值风险 |
| 独立进程恢复 | 16→17 | 1 | 0 | 4 | 恢复并更新通过 |

16-batch 检查中：step-before-update 0–10 被跳过；11 有限并更新；12–14 再次跳过；
15 有限并更新。尚不能声称第一个有限更新后就持续稳定。
记录了每次 backward 后、AMP unscale 后的 backbone 389 个和 head 107 个梯度张量的有限性。
实际更新次数同时与 AdamW 内部 step 对照，backbone/head 的参数探针也均确认改变。
峰值 CUDA 已分配内存约 16.7 GiB；这是本机 batch=1 结果，不是其他配置的显存保证。

恢复检查覆盖模型预测、Lightning 步数、AdamW 非空状态及实际更新、保存的 scaler 的后续使用。
它没有证明恢复后数据顺序、随机轨迹和不中断训练逐位等价；该检查关闭了随机增强。
从预测 xy 通过保存的仿射恢复到标签栅格，结果有限；空间数值精度另见既有 crop parity 检查。

## 证据与复现

逐 batch 记录、输入权重 SHA256、参数探针变化和指标：
[JSON 报告](densekp_real_acceptance_20260928.json)。

本地详细产物位于：

- `outputs/acceptance/densekp_real_20260928/fresh/`：首次严格探针日志。
- `outputs/acceptance/densekp_real_20260928/fresh_amp/`：12-batch 失败条件记录与 checkpoint。
- `outputs/acceptance/densekp_real_20260928/fresh_verified/`：16-batch 报告、预测及完整 checkpoint。
- `outputs/acceptance/densekp_real_20260928/resume_verified/`：恢复报告、预测及完整 checkpoint。

这些大型本地产物被 git 忽略；报告和验收工具纳入版本控制。
独立的完整 checkpoint 约 7.5 GiB；没有覆盖原始权重。

以下命令需要可访问 CUDA 的环境，输出目录必须不存在（工具拒绝覆盖）：

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/densekp-audit-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
/home/yy/conda_envs/camerahmr/bin/python -B \
  tools/validate/verify_densekp_model_lifecycle.py \
  --phase fresh --checkpoint assets_external/densekp.ckpt \
  --output-dir outputs/acceptance/new_run/fresh --gpu 0

PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/densekp-audit-mpl \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src:. \
/home/yy/conda_envs/camerahmr/bin/python -B \
  tools/validate/verify_densekp_model_lifecycle.py \
  --phase resume --checkpoint outputs/acceptance/new_run/fresh/checkpoints/last.ckpt \
  --reference-dir outputs/acceptance/new_run/fresh \
  --output-dir outputs/acceptance/new_run/resume --gpu 0
```

## 后续门槛及目标场景

用户进一步明确：核心是**位于相机边缘、在单路图像中不完整的人体**，用 DenseKP 获得跨相机
同一人体的语义点对应。完整人仍可能落入检测器限定 FOV，保留作对照。检测召回和合图算法
不属于本次 DenseKP 实现范围。

下一份评测集应基于 `20260320-riding-dataset3` 的 20260320 标定，固定已检测的人体框，建立
同一时刻、同一人的双路样本。区分物理/有效 FOV 截断、检测 ROI 截断、DenseKP crop 排除和
ERP 接缝；明确每一步原图坐标与变换。

用户还说明，现有下游双视角人体重建通过共享 pose 与 shape 修复点对应。这是用户提供的
系统背景，本次未新增审计或修改重建实现。DenseKP 主指标应使用**重建前的原始 2D 预测**；
重建后的改善另行记录，不能把共享人体模型带来的修正算作 DenseKP 本身精度。

分组评测双路共同可见、仅一路可见和双路不可见的语义点，优先关注能用于对应的共同可见点。
跨相机像素坐标不能直接相减作为一致性误差；需要每路标注或经验证的几何约束。
出视野、物理遮挡和裁剪排除应保留独立标注；训练是否回归不可见点仍需明确策略。
模型第三值 `keypoint_value` 仍不自动作为置信度或对应权重。

在正式重训前还应：

- 追踪 scaler、AMP 跳步比例、实际参数更新、各损失分量及梯度量级，确认较长窗口稳定。
  FP32/BF16 或较低初始 scale 的对照属于后续受控诊断，不能未经验证直接改默认配置。
- 用以上边缘截断场景建立独立评测基线；当前 P0 smoke 不衡量跨相机对应效果。
- 完成真实部署后端与模型输出的对照。本次没有运行 pano-stitch 推理或修改其源码。
