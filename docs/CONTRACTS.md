# 技术契约

当前实现的边界，2026-10-03。精简不改变以下语义；不能依据历史字段同名就统一规则。具体例子见 [样本旅程](SAMPLE_WALKTHROUGH.md)。

## 138点与几何

DenseKP 固定预测138个身体语义点，`R[138,10475] @ vertices[10475,3]`，R作用于身体网格，不作用于衣物或头发。BEDLAM官方437点selection operator不是该138点算子的完整子集；仅有相同shape或部分顶点重合不足以证明语义一致。原R的微小负权重也不能擅自裁零/归一化。详细原证据保留在 [算子审计](data/bedlam/BEDLAM_OPERATOR_AUDIT.md)。

SMPL-X参数姿态为轴角弧度，身体/平移以米计。SMPL72维与SMPL-X165维不能直接拼接使用。`body.parameters.parameters_from_pose` 按七块原切片生成batch维float32，betas保留原对象，transl为负pelvis；源shape与目标角色shape不得混用。独立照片每人/姿态独立保存，不拼成共享shape的连续motion。

`body.transforms.world_points` 将模型Y-up转为世界Z-up：`(x,-z,y)`，再用同一身体最低点lift、yaw与location。人体、衣物、groom、138点必须共用这个变换，不能各自贴地。旧真实motion是否保留source translation以manifest为准；保存reference/rest数组不等于当前环境即时rest_body逐位复现。

## 相机、成像与有效域

目标是20260320-riding-dataset3，真实鱼眼2880²、真实ERP5760×2880；合成ERP常为1536×768，坐标/阈值必须带栅格单位。不混031/0311，0.18m只是现有合成rig间距，不是重新测量的外参。

CV camera-to-world矩阵的列为右/下/前轴。行向量世界点到相机：`q=(world-camera[:3,3]) @ camera[:3,:3]`。ERP经度`atan2(qx,qz)`、纬度`asin(-qy/||q||)`；映射到调用者给定W/H，接缝可能返回0或W，消费方各自执行周期处理。Blender相机局部看向-Z、+Y朝上，不得与CV轴混用。两物理DS相机朝向相反，normal ERP虚拟轴对齐，cam4映射不能再额外翻一次。

| 路径 | 深度来源/含义 | 可见性与mask |
| --- | --- | --- |
| DS | `imaging.ds.SourceERP.ray_range_m`：原始源ERP EXR射线距离，米；不是后处理depth.npy | 源ERP周期5×5深度/实例邻域，严格距离差<0.04m，再结合有效域；本流程不写新的depth.npy |
| Omni | `imaging.omni.OmniBuffers.ray_range_m`：nearest重映射并清理非有限值后的鱼眼距离，写depth.npy | 鱼眼内截断5×5邻域及原实例/inside策略；不拿此距离替代DS原ERP源距离 |
| 历史实例规则 | `camera.visibility.visible_at_instance_points` 不比较深度 | 四舍五入中心、非周期7×7实例邻域；不等价于ERP深度规则 |
| 透视P0 | 既有Eevee实测Depth为optical-Z，`-camera_blender.z` | 单人体Alpha mask、原3×3与容差规则；Alpha不是多人IndexOB，不能推广到任意场景遮挡 |

RGB插值、周期padding、实例取整、深度nearest等保持每条路径原行为，不为文档一致而替换。环境Omni预览的Lanczos4也不是所有后处理的默认规则。

`in_view` 表示相机模型/传感器/显式成像mask允许观测；`surface_visible`来自深度/实例；`visible=in_view & surface_visible`。**黑色RGB不等于无效区**。DS域mask采用floor/横向modulo；预测诊断 `keypoint_supervision.points_in_mask` 采用rint，边界策略未统一。Omni inside/domain的历史差异见 [已知行为](KNOWN_BEHAVIOR.md)。

## Dataset与裁剪

当前校验入口为 [keypoint_contract.validate_annotations](../densekp/core/keypoint_contract.py)。

| NPZ字段 | 契约 |
| --- | --- |
| imgname | 非空字符串[N]；一行是一个指定人物，可共享图片 |
| center | 有限[N,2]，标签所在图像栅格的像素xy |
| scale | 正有限[N]或[N,2]，像素边长/200；BBOX_SHAPE扩展后取最大边 |
| proj_verts | 浮点[N,138,3]，原图xy及明确声明的标注权重 |
| part / gtkps | 条件辅助关节，极端裁剪使用；辅助NUM_JOINTS=44不是138点数 |
| supervision_policy | 新标签为标量in_view_full_crop；必须同时有bool[N,138]的in_view/visible，visible不能超出in_view |

调用方核对生产者后声明 `DATASETS.TARGET_VALUE_SEMANTICS=supervision_weight`。权重有限且在[0,1]，正权重点xy有限；零权重xy可以未定义。数字恰为1不自动证明第三列是权重；BEDLAM原齐次常数需由适配器明确声明。

Dataset可显式接收annotation_path/image_root，未提供时仍解析旧ID。相对路径按cwd，相对imgname拼image_root，绝对imgname覆盖根；不按NPZ所在目录猜测。dataset名称仍控制日志/返回与portrait/closeup历史分支。

主流程：read_image → determine_crop → transform_crop → Dataset._supervision → 返回样本。
读取BGR，按实际增强顺序和OpenCV插值生成affine；颜色标准化产生RGB float32[3,256,256]，display保留未标准化BGR。
图像与138点共用affine；标签像素写入float32后再归一化，不能合并表达式而改变中间舍入。

```text
crop_xy = affine[2,3] @ [x,y,1]
normalized_xy = crop_xy / [256,256] - 0.5
original_xy = inverse_affine((prediction_xy + 0.5) * [256,256])
```

完整输入/标签空间为256-square；ViT只取中间192列，**输出不加32px偏移**。网络不读取bbox数字、人物ID、相机或GT，目标由裁剪内容表达。
中心平移、尺度、旋转改变图像和点，随机调用/增强顺序保持。box_size是原缩放前范围，不一定是增强后实际视野；可视化报告仍用plan.box宽高。未验证的138点水平翻转明确拒绝。

新策略在实际affine后计算 `in_view & inside_full_crop`，后者两轴为 `[-0.5,0.5)` 且有限。有可靠GT的遮挡点继续监督；域外或实际完整裁剪外不监督。旧NPZ常数/visible-only权重保持原值，不因删除旧Python接口而重解释历史数据。visible未知不等于全遮挡。

## 训练与评价

预测[138,3]中的第三值原样保存。仅当前with_var损失把它解释为log sigma，不能据此将外部keypoint_value当置信度、拒绝门槛或指标权重。

记 `e²` 为归一化xy平方距离、w为GT权重、s为预测log sigma：

- with_var=False：`60 * sum(w*e²) / max(sum(w),1)`。
- with_var=True：`60 * sum(w*(500*e²*exp(-2s)+2s)) / max(sum(w),1)`；有效s下限log(1e-6)。

当前计算为float32，零权重在算差前屏蔽，非有限模型预测不静默跳过。`keypoint_error_totals` 是每人加权L2再按有效人数平均，空监督人另计，不把空集当零误差；训练loss、验证val_loss和诊断点均值不是同一公式。

验证保留尾批，训练drop_last；主val_loss取loader0。ValidationCheckpoint在有效验证结束后保存/排名，不是独立故障恢复定时器。FixedValidation另保持步数触发、固定样本顺序、FP32、写预测/报告→保存checkpoint→恢复train的顺序，异常仍在原处停止。init_checkpoint仅初始化，ckpt_path恢复训练状态。多设备配置能组合不证明DDP sampler补齐、归约或唯一人数已经实测。

`densekp.core.evaluation.summarize` 严格拒绝非有限；`densekp_retrain.evaluation.candidates`保留失败/null与分母计数策略。点平均、等人/等框平均、crop像素、周期ERP像素必须分列。预测出有效域仍计其GT误差并另报拒绝率，不以丢错点提高指标。配对比较各视角自身GT误差，不能直接相减不同相机的像素位置。

## 来源和资格

人物家族、shape、脸部纹理风险组、姿态/动作人物、所有父版本和外观衍生需按组隔离。每场scene的全部人物和遮挡者、同图、双视角、邻帧及配对控制不能跨split。通用SMPL-X模型、衣型/groom模板共享不直接等于同身份，但未知皮肤供体无法认证绝对人物独立。

`data.source_split.validate_split`核查所有scene人物；`pack_index_rows`保持索引行序和来源，不能依据模型误差筛选输入。框构造与实验配额分离；当前合成change_box按1536×768裁边，不冒充任意分辨率算法。
最终资格以活动split、冻结index及frozen_candidate_assignment为准。旧integration_only和诊断渲染manifest不自动升级为正式训练集，也不能推翻明确的后续冻结划分。数据数量、排除和版本见 [数据与模型](DATA_AND_MODELS.md)。
