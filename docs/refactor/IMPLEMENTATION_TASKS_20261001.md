# 全仓库重构：执行任务与验收门槛

更新：2026-10-02。P1、P2 已通过用户整体评审；**P3.1–P3.4 已通过评审；P3.5 实现与验收完成，P3 整体待评审**。
P4–P5 仍待实施，整个重构尚未完成。
范围、代码证据、逐阶段风险和回退边界见 [完整方案](FULL_REFACTOR_PLAN_20261001.md)。
此文件为非忽略路径的执行交接清单；大清单和运行证据保留在 `outputs/`。
提交交付时已补公共数据包及数据文档的忽略例外，见 [交付记录](COMMIT_DELIVERY_20261002.md)；P4 安装与入口验收仍待执行。

## 已确定的取舍

- 保留两个包：`densekp` 管模型和直接数据处理；`densekp_retrain` 管生成、实验组织和应用评价。
  公共模块不能反向依赖实验脚本。活动调用方直接使用公共模块；旧导入只用于兼容。
- 先支持仓库开发与 editable 安装；携带全部资产的独立 wheel 不在目标内。
  两包安装配置在 P4 落实，本批不宣称已经解决完整安装问题。
- 旧入口保留薄包装。清理必须逐项证明无调用、无唯一复现用途；不移动或删除输出、失败版本和资产。
- 报告字段、图像旋转、mask 取整等新行为单列修复任务，不能借迁移决定新语义。
- 必须在旧实现仍独立存在时冻结 before。新旧包装共用实现的比较只证明兼容，不证明等价。
- 验证随计算路径安排：纯指标/配置用 CPU；真实模型、训练生命周期、Blender 验收留到对应迁移。

## 任务清单

“待执行”不是已通过。每批先保存独立证据，再迁移、改活动调用方、验收和同步阅读路线。

| 任务 | 状态 | 文件/产物及执行要求 | 验收门槛 |
| --- | --- | --- | --- |
| P0.1 七样本完整输入 | 已完成 | `verify_refactor_metrics_config.py`；迁移前源码/配置/七个 NPZ | 完整 `[3,256,256]` 输入、中心输入、GT、affine、mask 精确一致 |
| P0.2 真实模型基线 | 已完成 | 原/B/C、手动/Lightning 两加载路线、七输入独立 GPU before；GPU 1 按 UUID 在沙箱外可用 | 各 496 个 key，shape/dtype 严格匹配；原始预测、输入、模式与 CPU/CUDA RNG 已保存 |
| P1.1 纯评价 | 已完成 | `densekp/core/evaluation.py`、`densekp_retrain/evaluation/candidates.py`；旧导出兼容 | 保存预测重算精确；空组/非有限/按点与按人汇总、JSON 字段及顺序保持 |
| P1.2 配置工厂 | 已完成 | `densekp/core/datasets/configuration.py`；活动调用方直引 | YACS 两种配置 dump、默认语义、可扩展性、独立实例、七样本输入一致 |
| P1.3 公共推理 | 已完成 | `inference.py`；四评价入口直引；smoke predict 兼容导出，source trial 直引；保留原模式/异常策略 | 3 权重×2 加载路线×7 输入×5 预测路径最大绝对差均为 0；严格加载、输入和 RNG 一致 |
| P1.4 训练审计 | 已完成 | `training_audit.py` 的 CheckUpdates/probes；smoke 兼容导出，source trial 直接导入 | CPU 独立 before/after：2 步保存、恢复至 4 步，18 个事件/174 数组精确；6 个梯度失败用例；参数顺序、优化器、模式和 RNG 保持 |
| P1.5 应用报告与画图 | 已完成 | candidate main 按输入、逐行预测、配对/报告拆分；公共 visualization 与 io/json；恢复入口直引公共模块 | 独立旧预测回放：37 JSON、438 数组、78 图片精确；恢复入口另 17 JSON；仅列明时间/输出路径变化 |
| P1.6 有界验证回调 | 已完成 | `training/trial_callbacks.py`；FixedValidation 迁移及旧导出；主流程展开/重命名 | 真实 GPU 小网络 10 情形/742 数组精确；旧 checkpoint 恢复、state_key、参数/步数/RNG、FP32、保存/train 顺序和失败停止保持；74 项 CPU 回归通过 |
| P2.1 显式数据源 | 已通过评审 | Dataset + `datasets/loading.py`；保留 ID/name，15 处 patch.dict 和 1 处直接注册表赋值改为显式路径 | 45 用例×旧 ID/显式两路线、2814 数组精确；部分路径回退及无全局写入测试通过 |
| P2.2 裁剪与增强 | 已通过评审 | augmentation/image_processing/crop_pipeline；Dataset 直接按步骤调用；utils tuple 兼容 | 七真实样本关闭/启用增强完整输入精确；参数、Python/NumPy/Torch RNG 一致；11 旧接口用例/156 数组精确；80 项 CPU 回归通过 |
| P2.3 来源与打包 | 已通过评审 | `data/source_split.py`、`data/packing.py`；全场景人物/遮挡者核查 | 198/79 和 400/143 行精确；37 来源边界与缺字段/空行/空监督行为保持；历史邻人元数据差异单列 |
| P2.4 框构造 | 已通过评审 | `data/boxes.py`、`data/loading.py`；实验保留配额和 RNG 配方 | 框/中心/尺度/回退一致；400 父目标顺序等于既有日志、419 RNG 调用及状态精确；无训练 |
| P2.5 监督契约 | 已通过评审 | 显式旧权重/pre-crop in_view 适配；共用实际裁剪监督；contract 原样 | 旧/新 adapter 31 数组精确；543 行回读、七样本/RNG 回归和保存预测回放；87 CPU 测试通过 |
| P3.1 人体/服装 | 已通过评审 | body/parameters、measurements、transforms；clothing/shells、binding；io/geometry；配方保留实验 | 男/女×两骑行姿态独立 CPU 基线；顶点/面/138 点/衣物/参数保存恢复精确，历史选定资产误差 0 |
| P3.2 相机/ERP | 已通过评审 | camera/erp、rig、domain、visibility；旧名兼容，活动调用方直引 | 主回放 409 数组/12 JSON 精确；原始 EXR 另 20 数组及既有两相机标签精确；92 CPU 测试通过；未运行 Blender |
| P3.3 成像后处理 | 已通过评审 | imaging/source、ds、omni、remap；visualization/imaging；旧导出兼容 | 原始 EXR 独立完整回放：189 数组/44 图片/29 JSON 精确；另 55 数组/35 图片/7 JSON 边界对照；97 CPU 测试通过；未运行 Blender |
| P3.4 Blender 边界 | 已通过评审 | rendering/blender 的装配、材质/groom/bridge；clothing/contracts；两条旧 `--` CLI 保持 | 独立 native/linked 装配 78 数组精确；3 相机的 9 个 EXR 通道/3 预览精确；后处理 39 数组/23 图片/6 JSON 精确；97 CPU 测试通过；未验 CUDA/其它 Blender 版本 |
| P3.5 场景与配方 | 实现与验收完成，待评审 | jobs 具名命令/共用执行；4 配方显式 main；32 文件资产消费索引 | 独立 30 用例/20 argv 精确；4 执行体 AST、Blender 4 导入/3 CLI；100 CPU 测试通过；无场景制作/渲染 |
| P4 包与入口 | 待执行；依赖已迁公共职责 | 两包 setuptools、依赖/路径、CLI；检查整个新公共包的忽略和跟踪状态，修复 data/ 规则遗漏 P2 实现的交付问题；不迁资产 | 根目录脚本和 editable 下承诺的入口；Blender 不强制安装 Torch；先冻结旧命令 |
| P4 历史清理 | 待执行 | 逐项调用与复现用途签收表 | 查 import/符号/subprocess/shell/config/docs/产物/外部接口；证据不足保留 |
| P5 阅读路线 | 持续更新，整体待执行 | README、READING_GUIDE、SAMPLE_WALKTHROUGH、契约与产物索引 | 从一个实际样本追到来源、标签、输入、权重、输出和指标分母 |

## 每批共用的验收要求

1. before 保存来源、必要配置、环境、命令和旧实现；禁止覆盖旧 before 或为通过检查重新生成期望值。
2. 可精确比较的数组要求值、shape、dtype 一致；整数计数、监督集合和配置必须一致。
   已完成的指标与 CPU 审计浮点对照均要求精确，不使用容差。后续几何/模型沿用相应已有数值依据，不设全局宽松容差。
3. 保留 138 点/regressor 语义、完整 256-square/中间 192 列约定、不加 32 px、遮挡可靠 GT 监督、
   第三输出不默认加权，以及空集合和非有限值的原有处理。
4. 分开真实 2880 鱼眼/5760×2880 ERP 与合成 1536×768；真实目标用 `20260320-riding-dataset3`，不混入 031/0311 标定。
5. 验证记录注明实测与待测。真实推理、训练生命周期、Blender 不得以纯 CPU 数值通过替代；不靠重跑 400 步验收。
6. 回退以当前任务源码为单位；保留用户修改、before/after、历史输出、失败记录和所有资产；不 commit/reset/clean。

已完成批次见 [指标/配置](P1_METRICS_CONFIG_20261001.md)、
[公共推理与训练审计](P1_TRAINING_AUDIT_20261001.md)、
[评价流程与 FixedValidation](P1_REPORTING_CALLBACKS_20261001.md)、
[显式数据源与裁剪](P2_DATASET_CROP_20261002.md)、
[来源、打包、框与监督适配](P2_SOURCES_BOXES_20261002.md)、
[人体、衣物与坐标相机计算](P3_GEOMETRY_CAMERA_20261002.md)、
[成像后处理与标注](P3_IMAGING_20261002.md)、
[Blender 对象、材质与装配](P3_BLENDER_20261002.md)。
P3.5 见 [场景配方与执行入口](P3_SCENE_RECIPES_20261002.md)。
[P3 整体检查表](P3_COMPLETION_CHECKLIST_20261002.md) 提交评审；评审后再进入 P4。有限 CPU Blender 实测不代表所有场景、GPU 或版本均验收。
