# 现存行为、限制与退役能力

本页区分现存行为、已修复问题和明确停止维护的能力。精简没有趁迁移改变数值语义，也不代表训练效果提升。活动实验与门槛见 [数据和模型](DATA_AND_MODELS.md)。

## 现存行为与待办

| 项目 | 当前边界 |
| --- | --- |
| portrait/closeup | 名称分支仍旋转图像；不自动修正未随生产者旋转的标签坐标。不能把该旧行为默认为所有输入正确 |
| 有效mask取样 | GT侧floor与部分预测侧rint等历史规则保留，不能借公共化统一取整 |
| 水平翻转 | 未验证138点左右置换，显式拒绝；辅助44关节点置换不可代用 |
| NPZ第三值 | 旧标签按其权重语义读取；新标签按有效域与实际完整裁剪重算。预测第三值不默认是置信度 |
| 有效域 | 当前监督使用显式mask；P0某些旧产物的全1 mask并不证明镜头有效区，黑色RGB也不是mask |
| 空/非有限集合 | 评价中的严格拒绝与返回null两类策略分别保留；某些旧空集合广播/max失败仍属历史行为 |
| NaN框拒绝记录 | 框已被拒绝，但strict JSON写拒绝记录仍可能因NaN失败；未修复为新回退 |
| groom全inactive | 原失败及先写部分文件的行为保留；迁移不增加自动筛毛/清理 |
| 配置与默认值 | YAML中存在的LOSS_TYPE/head等字段不等于已接入执行；未启用原来无效的配置。`dataset_config`、注册表全部key及重复覆盖顺序、`to_lower`暂缓清理 |
| 实验依赖 | DS框评价仍从DS遮挡入口取实现；SOURCES、mesh_panel及专用布局关系继续保留。未扩大到新一轮评价重构 |
| 设备与环境 | GPU0曾异常，当前健康未知；任何新GPU执行须即时检查。共享依赖冲突见安装说明，未在收尾时升级修复 |
| 可复现性 | 原canonical路径、默认outdoor_courtyard、真实BEDLAM tiny输入存在缺口；选定副本或受控输入检查不替代原资产可用性 |

真实目标是20260320-riding-dataset3；031/0311标定不混用。部署侧是否扩大检测框、目标居中和邻人干扰尚需独立判断，合成挑战不能推导真实YOLO误差分布或定位全部根因。旧报告中可靠性头、目标锚点等建议不是已实现功能。

## 已解决的历史问题

当前Dataset直接调用具名裁剪阶段，显式接收annotation_path/image_root；不再patch全局注册表。旧输入增强开关、下采样抗混叠、平移增强门控、断点等早期审计项不能照搬成当前缺陷。显式来源检查包含遮挡者，分组与打包不依赖实验脚本提供算法。

训练loss在FP32处理权重与数值边界；当前检查保留零监督/非有限值拒绝等规则。验证后保存与训练模式恢复已有独立验收，但不代表所有DDP/设备组合完成真实测试。test阶段复用验证划分也不产生独立测试集。

双包editable发现、被忽略公共data源码和Hydra配置交付已修复并提交。父配置现为loop；完整对象pickle的旧loss路径兼容已明确退役，当前state_dict严格加载能力保留。详见 [验证覆盖](VALIDATION.md)。

活动实验上下文/展示字段的历史修订应按版本读取，不能把当时报告状态当成当前错误；也不能将精简中的等价迁移称作额外算法修复。

## 明确停止支持的能力

| 范围 | 退役内容 | 保留能力 |
| --- | --- | --- |
| B1 | 旧loss、训练辅助、纯重导出包装；测试从迁移工具解耦 | 现行loss、归一化、来源/成像/指标契约及原数值fixture |
| B2 | 旧get_example/get_example_projverts、tuple与旧SMPL/HMR裁剪；旧Omni自动框/ERP居中/三相机评分CLI；注册ID评分CLI | 具名CropResult、当前Dataset、历史NPZ及权重语义、Omni生成与公共投影 |
| B2配置 | trainer=default_hmr；data=train/densekp_train/bedlam_v2_v1/eval/eval_smplx；experiment=camerahmr/bedlam2；旧JSON示例 | loop、六个当前data组、现行experiment/trainer、显式CLI覆盖；不表示BEDLAM数据能力删除 |
| B3 | 固定mixed制作CLI、旧crossings路径、两个旧回归调度CLI | 公共参数/衣物/groom/crossings、旧mixed资产消费、一个薄suite入口；不统一评价公式 |
| B4-D | BVH/BABEL骑行检索/拟合输入；失败衣物repair/union/settling/polo；指定画廊/诊断/迁移capture replay/smoke；独立riverfront、两套Omni sweep、P0 pyrender检查 | 当前生产者与数值契约、两套DS评价、真实模型与Blender按需检查 |
| B4测试 | 旧capture拒绝测试、整段迁移AST fixture、失败支线DQS/scan测试参考 | 原归一化/来源/裁剪/成像数值fixture，公共crossings四项，普通Python与真实Blender导入安全 |
| B4-P | verify_refactor_inference、verify_fixed_validation及旧类路径/AST capture | 默认只展示的新runtime入口，显式输入/独立expected/设备/新output后执行 |

这些是接口/能力退役，不声称新评分与旧分母、过滤或报告等价；不承诺当前代码重制失败配方。旧源码和文档用固定提交查阅，已有资产、保存预测、失败版本和历史结果原位保留。

## 暂缓而不是遗漏

注册表key、to_lower、专用透视时间序列/相机压力/BEDLAM适配、现行资产生产者继续保留。无需为“全部清理”再扩展删除。两份BEDLAM审计仍含独有operator/帧映射信息；活动报告及两份衣物来源报告仍有元数据消费者。本轮保留它们，原因和历史定位见 [最终交付](refactor/PRUNING_FINAL_20261003.md)。
