# 遮挡点监督与评价（2026-09-29）

> 后续已完成两场景带遮罩图片的真实 GPU 推理，见
> [3D 场景＋鱼眼遮罩基线](../rendering/DS_MASKED_BASELINE_20260929.md)。
> 下文“没有生成新模型评分”描述本页原始监督改动阶段；旧预测复用尝试仍保留失败状态。

## 1. 本次决定

**可靠合成 GT 在镜头有效视野内、且落在实际完整裁剪内，就参与训练；遮挡不取消监督。**
网络仍预测全部 138 行。镜头外点没有本相机可用观测，不进入这套训练目标。
本次修改数据与评价规则，未重训模型，也没有把预测第三值当作置信度。

| 状态 | 含义 | 用途 |
| --- | --- | --- |
| `in_view` | GT 投影满足相机/图像有效域与镜头遮罩约束 | 排除镜头外 GT |
| `visible` | `in_view` 内通过深度和人物实例检查的点 | 分组评价，不直接决定监督 |
| `inside_full_crop` | 实际 affine 后，两轴归一化坐标都在 `[-0.5,0.5)` | 排除实际裁剪外 GT |
| 标注第三列 | `in_view & inside_full_crop`，明确的 0/1 | loss 监督 mask |
| 预测第三列 | 原始 `keypoint_value` | 原样保存，不用于本次误差加权或筛点 |

遮罩来自有效域文件，不能用 RGB 是否为黑色判断；衣服、阴影也可能是黑色。
也不能要求 GT 落在目标的可见实例 mask 上，否则被别人挡住的目标点会再次被排除。
训练策略包括完整 256×256 裁剪的左右各 32 列，模型实际只看中间 192 列；这些点是有意
保留的推断目标。诊断另报落在 192 列输入内的子集和完整裁剪外的子集。

当前表面可见性来自源 ERP 的 5×5 邻域、人物实例 ID 和 4 cm 深度容差，是近似标注。
暂时没有可靠字段细分自遮挡、其他人物遮挡和物体遮挡，因此统一报告 `occluded`，不能
把它全部叫作“被别人挡住”。`clear` 也可以有自遮挡。

## 2. 代码怎样接起来

```text
生成器：138 个原图 GT xy + 投影有效性 + 镜头遮罩 + 表面可见性
  → adapter：明确的 in_view / visible / supervision_policy + 原图 xy/mask
  → Dataset：图像和 xy 共用实际 affine；重新计算裁剪后的监督 mask
  → 原有 loss：使用标注 mask；目标公式和模型结构不变
  → 验证：总体误差 + 可见/遮挡组误差与样本数
  → 诊断：还原原图 xy；记录镜头区域拒绝率与双视角分组
```

- [监督规则](../../densekp/core/keypoint_supervision.py)：三个小函数集中定义裁剪范围、镜头像素检查和标签构造。
- [标注契约](../../densekp/core/keypoint_contract.py)：声明新策略时，必须有布尔 `in_view[N,138]` 和 `visible[N,138]`；可见点必须属于有效视野。
- [Dataset](../../densekp/core/datasets/dataset_keypoints.py)：平移/缩放/旋转后重新判断裁剪包含关系。旧 NPZ 没有新策略时保留原权重，返回 `visibility_known=false`，不把旧样本误归为遮挡。
- [训练与验证](../../densekp/core/densekp_trainer.py)：新增 `val/crop_l2_visible_0`、`val/crop_l2_occluded_0` 等日志。仍按监督点先算每人均值、再按有效人数平均，单位是归一化裁剪坐标；`val_loss` 仍取 loader 0 的总体误差。
- [tiny adapter](../../tools/annotation/build_omni_multiperson_tiny.py)：新 NPZ 同时存策略、有效性和可见性；旧 DS 数据缺少镜头遮罩时拒绝构造新监督。
- [诊断分组](../../tools/validate/occlusion_metrics.py)：共用 GT 分组和拒绝率计算，供两个场景评价脚本使用。

双视角评价先选择两边 GT 都在有效视野的同一人物、同一语义行，再分为两边可见、
仅一边可见、两边均遮挡。报告两视角平均误差和较差视角误差；这不是重建或拼接指标。

**错误预测不能缩小误差分母。** GT 有效但预测落在镜头外时，仍计入定位误差，另计运行时
拒绝率。非有限模型输出会使诊断失败；空 GT 子集的 JSON 误差为 `null`，不是 0。
诊断按点等权，裁剪误差单位是完整 256-square 像素；训练验证按人平均，二者不能直接混称。

新 DS 生成器保存完整 `erp_valid_mask.png`。兼容旧 DS 产物时使用报告引用的 ERP 遮罩；
若另有鱼眼镜头轮廓，要求旧报告证明该轮廓覆盖全部 ERP 有效像素，否则要求重新生成
完整遮罩。当前 tiny 来源的两相机覆盖率均为 1.0。遮罩与 RGB 分辨率必须一致。

## 3. 已生成的结果

| 产物（仓库相对路径） | 实际完成内容 |
| --- | --- |
| `datasets/ds_multiperson_amodal_tiny_20260929/` | 4 个样本，train 2 / val 2；393 个监督点，其中 160 个遮挡点；实际 loader 读取通过 |
| `outputs/experiments/ds_occlusion_amodal_20260929/` | 原有 15 份预测重新评分、新 NPZ、分组指标和对照图；没有重新推理 |
| `outputs/experiments/ds_scene_pair_amodal_20260929/` | cycling_forest、urban_street_assets 各 6 行新诊断 loader；各 622 个监督点，其中 330 个遮挡点 |

后两场景的坐标往返最大误差为 `1.92e-5` 原图像素。它们复用带镜头遮罩的场景资产，
相同人物布局使标签计数一致；这不增加动作多样性。**没有生成这两场景的新模型评分**：
尝试复用旧预测时发现原始输入与新镜头遮罩版本的裁剪像素不同，已停止复用。

遮挡控制实验使用原始来源目录、同一固定框、GT 和 affine，读取历史预测原值重新评分。
复用模式依赖原始图像内容未被改写；路径和坐标一致不能单独证明历史像素未变。
原始实验文件保留，新结果应理解为旧模型的更完整评价，而非训练效果改善。

人物 3，完整 256-square 裁剪平均误差：

| 相机/条件 | 可见点：数量 / 像素误差 | 遮挡点：数量 / 像素误差 | 全部有效视野：数量 / 像素误差 |
| --- | --- | --- | --- |
| cam2 clear | 52 / 4.42 | 71 / 5.04 | 123 / 4.78 |
| cam2 mild | 50 / 5.18 | 73 / 5.20 | 123 / 5.19 |
| cam2 heavy | 22 / 7.70 | 101 / 19.88 | 123 / 17.70 |
| cam4 clear | 28 / 7.87 | 13 / 6.84 | 41 / 7.54 |
| cam4 mild | 15 / 8.93 | 26 / 26.34 | 41 / 19.97 |
| cam4 heavy | 0 / — | 41 / 123.93 | 41 / 123.93 |

这里人物 3 的所有有效视野 GT 都在完整裁剪内。不同条件的可见/遮挡组成员会变化，
横向比较应优先看固定的全部有效视野集合；不能只比较不断变小的可见集合。
上述人物 3 的镜头拒绝率都为 0，仍可能有很大定位误差：落在镜头有效区不代表预测准确。
heavy/cam4 的目标实例完全不可见，使用 clear 条件的已知框仅作诊断，不代表现实检测器
能找到这个人，也不把恢复没有人体框的人作为 DenseKP 的要求。

先看 [人物 3 对照图](../../outputs/experiments/ds_occlusion_amodal_20260929/person_3_prediction_review.jpg)
和 [完整报告](../../outputs/experiments/ds_occlusion_amodal_20260929/report.json)。绿点是可见 GT，
橙点是遮挡 GT，红点是预测；每列为 clear / mild / heavy，每行为 cam2 / cam4。

## 4. 复现和训练入口

在仓库根目录、已有 CameraHMR 环境中运行。输出目录必须尚不存在：

```bash
python tools/annotation/build_omni_multiperson_tiny.py \
  --source outputs/experiments/ds_multiperson_boundary_lens \
  --output datasets/ds_multiperson_amodal_tiny_20260929 \
  --val-camera cam4

python tools/validate/evaluate_ds_occlusion_control.py \
  --root outputs/experiments/ds_occlusion_control_20260928 \
  --output outputs/experiments/ds_occlusion_amodal_20260929 \
  --reuse-predictions outputs/experiments/ds_occlusion_control_prediction_20260928
```

新增可选 Hydra 配置 `data=ds_multiperson_amodal_tiny`；注册名为
`ds-multiperson-amodal-tiny`。默认训练数据选择未变。该配置仅提供数据与已声明的标签语义，
正式运行仍需按现有入口设置 checkpoint、batch 和训练参数。4 行样本复用同一动作/场景，
**只用于链路检查，不是独立训练/验证集，也不足以启动正式重训**。

已通过 30 项相关测试：旧契约/坐标边界和 CPU 训练恢复 24 项，加新监督测试 6 项。
新测试覆盖遮挡点梯度、镜头外点排除、增强移入/移出裁剪、新旧样本混合、评价分母、
双视角分组、验证汇总和策略元数据约束。未运行完整 DenseKP 重训或部署后端比较。

下一步仍是准备独立动作/人物/场景的数据与真实给定框验证集，再比较训练前后可见和
遮挡误差。自遮挡与互遮挡细分、正式增强分布检查、真实镜头边缘标注质量仍需要验证。
