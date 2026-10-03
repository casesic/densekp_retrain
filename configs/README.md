# 配置归属与历史兼容

P4 已整体通过评审；P4/P5 随本次提交交付，P5 待最终评审。当前 [命令表](../docs/COMMANDS.md) 与 [交付报告](../docs/refactor/P5_READING_DELIVERY_20261002.md) 是运行和状态入口。

本目录的同名字段不保证被同一个入口消费。保留配置的值与默认选择不变；本次停止支持的历史组名见文末。

| 配置 | 消费入口/用途 | 状态 |
| --- | --- | --- |
| `regression/suites.json` | `tools/validate/run_regression_suite.py` 的 source_trial / box_robustness | 仅固定输入路径角色和 matplotlib 目录；任务顺序在两个具名 builder，指标/门槛不复制到 JSON。默认只展示，显式 --execute 加设备选择才运行 |
| `hydra/train.yaml` | `densekp/train_keypoints.py`；组合 data/trainer/paths/experiment | 推荐训练配置入口；默认 data=unselected，先打印配置再核验数据 |
| `hydra/experiment/densekp.yaml`、`default.yaml` | 当前 DenseKP 训练；仍需查实际代码读取字段 | 保留活动配置，不因 YAML 存在就启用未实现字段 |
| `hydra/data/unselected.yaml`、本地 tiny 系列 | 当前 DataModule 选择已有 dataset ID | unselected 是安全默认；其它 ID 需要对应本地 NPZ/图片 |
| `hydra/trainer/`、`paths/`、`hydra/` | Lightning/Hydra 生命周期与运行目录 | 保留 cpu/gpu/ddp/mps/default 组、路径变量和 CLI override；共享循环父配置改名见下文；CPU 配置不等于资产已经准备好 |
| `fisheye*.json`、`perspective_camera_stress.json` | 对应 tools/fisheye、tools/render 具体配方 | 按调用者解释，不能与 Hydra 混用；参见场景索引 |

P4 安装批次曾补齐此前被忽略的 Hydra data YAML；本批仅退役下列已明确批准的预设，不改变保留配置或训练授权。
历史配置组是动态选择接口，本次按明确退役决定删除；[上轮保留表](../docs/refactor/LEGACY_RETENTION_20261002.md) 记录的是当时状态。

B2a 首批仅将训练循环父配置改名为 `hydra/trainer/loop.yaml`，四个 trainer 的 defaults 顺序与配置值保持不变。旧父组选择 `trainer=default_hmr` 已停止支持；对应选择为 `trainer=loop`，但父组不是完整 Trainer 配方，日常仍选择 cpu/gpu/ddp/mps。默认 data=unselected、trainer=gpu、experiment=densekp 不变。首批未删除其它预设；本次第二批的退役范围见下文，B2b 仍未实施。配置对照与既有解析异常见 [B2a 验收报告](../docs/refactor/PRUNING_B2A_20261003.md)。

B2a 第二批已停止以下七个 Hydra 组选择，均明确报 MissingConfigException，不提供别名或回退：

- data=train、data=densekp_train、data=bedlam_v2_v1、data=eval、data=eval_smplx。
- experiment=camerahmr、experiment=bedlam2。

旧 JSON 路径示例也已移除；这八份已跟踪文件的原内容可从 Git 历史找回。当前使用保留配置，显式核对 DATASETS.DATASETS_AND_RATIOS、VAL_DATASETS、TARGET_VALUE_SEMANTICS 及训练参数，先用 --cfg job 查看配置；Dataset 的 annotation_path/image_root 显式路径接口继续支持。tiny 配方不等价于旧全量组合，不能直接替换后声称训练或评价语义相同。

注册表全部 key、历史 NPZ 读取与监督语义、BEDLAM 数据来源及处理能力均保留。六个当前 data 文件（unselected 和五个 tiny）、experiment/default 与 densekp、所有现行 trainer/loop/paths/Hydra 配置未变。详见 [预设退役验收报告](../docs/refactor/PRUNING_B2A_PRESETS_20261003.md)。

回归 suite 内的历史数据/权重路径相对源码仓库 ROOT；CLI 的 trial/data-root/experiment 仍相对调用 cwd。
source 显式传入 model_config 和 cuda:0，box 保持原来不传的行为，由 evaluator 决定默认值。
这不是 Hydra 配置，不改变训练参数，也不将 tiny 配方或不同指标统一。旧两套 evaluate_* 调度 CLI
已退役，使用 [当前命令与执行边界](../docs/COMMANDS.md)。
