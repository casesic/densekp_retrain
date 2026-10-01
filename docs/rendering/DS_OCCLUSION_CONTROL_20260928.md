# 固定目标遮挡对照与输入框敏感性

## 本次问题及结论

此前街景后方人物 3 的可见点误差约 111 px，不能据此把失败全部归因于遮挡。
本次完成街景三档固定框实验，并补充相同图像、不同框中心的对照。

- 固定目标、相机、输入框和评分点时，cam2 的误差随遮挡从 **4.44→5.13→7.70 px** 增加。
- 同一重度遮挡图、同样 22 个可见点，仅切换输入框中心，误差从 **7.70→112.74 px**。
- 在旧的 8 采样图上也复现了 **8.49→111.29 px** 的框敏感性。因此本例的大幅失败可以由
  输入框变化触发，提高渲染采样并未消除它。不能进一步推断全部真实场景失败都由框导致。

单位均为 256×256 裁剪中的平均 L2 像素误差，不使用模型第三输出加权。
仍只研究已有输入框条件下的 DenseKP，未运行或修改检测器。

## 控制条件

使用 `urban_street_assets` 原生光照；1536×768 ERP，Cycles 64 samples，seed=0、关闭
animated seed，不启用降噪。先验证一个场景的因果对照，未重复生成森林三档。
相机沿用 20260320 DS 内参、同高且光轴相反；0.18 m 仍为合成试验间距。

人物 3 是固定后方目标，人物 2 是固定双路边缘对照，人物 4 是固定完整单路对照。
三者的身份、frame、姿态、位置、朝向均不变。只干预人物 1：

| 条件 | 人物 1 | 其他条件 |
| --- | --- | --- |
| clear | 移除 | 无此前景人物遮挡，仍有自遮挡和有效视野截断 |
| mild | 从原位置沿世界 Y 移动 -0.30 m | 姿态和朝向不变 |
| heavy | 保持原始位置 | 复现旧样例的重度遮挡布局 |

三档 light/World 配置不变，但前景人物变化也会改变投影阴影和间接光，所以它不是纯像素
遮挡率实验。轻/重是布局名称，实际严重度按各相机分别报告。人体 AABB 保持至少一个轴
5 cm 分离，未据此宣称衣物或环境物体的完整碰撞验证。

每个相机/目标的框取 **clear 条件下的可见实例框**，长边乘 1.2 形成方形裁剪，之后冻结。
这是诊断用框，不是身体完整投影框或部署检测器输出。三档 exact affine 逐项相等；目标
世界点、投影 xy、有效视野掩码也逐项相等。真实 `DatasetKeypoints` 加载，无增强；模型
仍内部保留 192×256 区域。重度下完全不可见的人物仍用 clear 的固定框推理，仅作诊断。

## 固定框结果

主要指标使用 **三档始终可见的同一组语义点**，避免可见点减少导致评分集合改变。

| 条件 | cam2 当前可见点 | cam2 同一组 22 点误差 px | cam2 剩余可见实例面积 | cam4 当前可见点 | cam4 剩余面积 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clear | 52 | 4.44 | 100% | 28 | 100% |
| mild | 50 | 5.13 | 74.8% | 15 | 42.6% |
| heavy | 22 | 7.70 | 23.2% | 0 | 0% |

面积以 clear 同相机目标的可见实例像素数为分母，不是完整人体面积。可见性仍为源 ERP
同实例 5×5 邻域、射线距离误差 <4 cm 的近似规则，不能当作精确轮廓/衣物遮挡真值。

cam4 没有跨三档始终可见点，因此主指标为 null，不记为 0 误差。报告额外提供：

- 每档当前可见点的误差；不同档的分母不同，不能直接作为固定点对照。
- clear 可见点全体的误差；后续包含新遮挡点，只表示隐蔽点外推诊断，不能称为可见点精度。
- 当档双路共同点，以及跨三档双路都可见的固定共同点。

人物 3 双路共同可见点为 14→14→0；重度不具备双路可见定位评分条件。
人物 2 保持 25 个固定双路共同语义点，双路平均误差 **4.640→4.648→4.645 px**，没有随
右侧前景干预明显变化。人物 4 的单路平均误差 **4.408→4.409→4.388 px**。
这支持干预影响集中在遮挡目标，但只针对当前固定布局。

## 为什么追加框对照

固定框重度误差与旧结果相差很大，而此前同时改变了框和渲染采样。为区分影响，再交叉
运行两份重度图和两种框，目标、相机、GT xy 与评分的 22 个可见语义点一致。

| 图像 | clear 冻结框 | heavy 可见区域框 |
| --- | ---: | ---: |
| 旧 8 samples 图 | 8.49 px | 111.29 px |
| 新 64 samples 图 | 7.70 px | 112.74 px |

两框分别是原 ERP 上 `[1081,325,1183,503]` 和 `[1085,325,1134,503]`（右下边界不包含）。
两者方形裁剪边长都是 **213.6 原图像素**；中心从 `(1132,414)` 移到 `(1109.5,414)`，
即横移 **22.5 原图像素，约 26.97 裁剪像素**。因此同图框对照具体检验的是中心变化。
同图同权重，改变中心即可复现大幅定位失败；这属于 DenseKP 对给定框/邻人上下文的敏感性，
不能将 oracle 框生成方式等同于检测器问题，也不证明 crop 逆变换有错误。

旧图未记录随机种子，跨图结果仅称为渲染版本对照，不能严格归因于采样数单一因素。
两种裁剪都含同一目标，但画面中前景人物所占位置不同；视觉上出现预测朝前景人物偏移的
迹象，具体内部机制尚未证明。

## 产物及图例

- 渲染及标注：`outputs/experiments/ds_occlusion_control_20260928/{clear,mild,heavy}/`。
- 固定框预测：`outputs/experiments/ds_occlusion_control_prediction_20260928/`。
  - `person_3_input_review.jpg`：主目标三档原始输入对照。
  - `person_3_prediction_review.jpg`：主目标预测；列依次 clear/mild/heavy，行依次 cam2/cam4。
  - `person_2_prediction_review.jpg`：双路边缘对照；人物 4 也单独保存。
  - `fixed_boxes.json`、分档 loader NPZ、全部 138×3 原始输出、还原 xy、affine、逐点误差和掩码。
- 框对照：`outputs/experiments/ds_occlusion_box_control_20260928/box_comparison.jpg`。
  - 左列 clear 冻结框，右列 heavy 可见区域框；上行旧图，下行新图。
- 可追踪数值副本：[三档报告](ds_occlusion_control_20260928.json)、[框对照报告](ds_occlusion_box_control_20260928.json)。

图中红点为全部 138 点预测，绿点为当前可见 GT，灰点为 clear 时可见但当前被遮挡的 GT；
黄色线只连接固定评分点，青色轮廓为目标可见实例，灰色竖线标出模型保留区域。
预览放大两倍，报告仍使用 256-square 像素单位。完全不可见视图中的预测不表示检测成功。

## 验证及限制

共 15 次固定框推理及 4 次框对照推理，使用同一原始 `assets_external/densekp.ckpt`，
严格加载、float32、无 autocast、关闭 TF32、种子 0；未训练。全部输出有限，模型第三值
原样保存，不作为置信度或误差权重。未计算 SHA256。

检查了三档固定几何/相机/affine、RGB 与标注来源、跨条件评分集合，以及保存的逐点预测
重新计算误差与 JSON 是否一致。GT crop 逆变换误差要求 ≤0.0002 原图像素。已查看主目标、
边缘对照和框对照图。采样数增加改善了预览噪声，但本例未测量图像质量指标。

这仍是一种身份、一组姿态、一个场景的诊断，不能据此选择训练数据比例或证明真实效果。
后续数据重点应包含 **边缘截断＋邻人遮挡＋给定框中心/尺度扰动**，明确目标身份，并同时
保留固定框评估和真实已有框评估；不能让大量无遮挡 GT 中心框样本替代部署困难情况。
扩展真实骑行姿态和独立真实 20260320 固定框标注仍是正式重训前的工作。

## 复现入口

使用项目 Python 环境，`PYTHONPATH=src:.`，GPU 命令需要 CUDA 访问；新输出目录：

```bash
python -B tools/fisheye/run_ds_occlusion_control.py \
  --output outputs/experiments/new_occlusion --samples 64 --device CUDA \
  --fisheye-reference-root /mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/Output/20260320_riding_dataset3/image/fisheye \
  --reference-frame 02214
python -B tools/validate/evaluate_ds_occlusion_control.py \
  --root outputs/experiments/new_occlusion --output outputs/experiments/new_occlusion_predictions
python -B tools/validate/evaluate_ds_box_control.py \
  --control-root outputs/experiments/new_occlusion \
  --old-root outputs/experiments/ds_scene_pair_20260928_final/urban_street_assets \
  --output outputs/experiments/new_box_control
```

当前生成命令已包含 [镜头遮罩更新](DS_LENS_MASK_20260928.md)；本页原有预测指标仍对应旧产物。

本机 Python：`/home/yy/conda_envs/camerahmr/bin/python`。渲染器提供可选 `--seed`，默认省略
时保留既有行为；本实验显式传 0。输入场景和源权重未被覆盖。
