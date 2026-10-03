# 配置归属与历史兼容

当前使用入口见 [命令表](../docs/COMMANDS.md)，已验证范围见 [VALIDATION](../docs/VALIDATION.md)。B1–B4精简已包含在d8bd387；本页描述现存配置。

本目录的同名字段不保证被同一个入口消费。保留配置的值与默认选择不变；已停止支持的历史组名见文末。

| 配置 | 消费入口/用途 | 状态 |
| --- | --- | --- |
| `regression/suites.json` | `tools/validate/run_regression_suite.py` 的 source_trial / box_robustness | 仅固定输入路径角色和 matplotlib 目录；任务顺序在两个具名 builder，指标/门槛不复制到 JSON。默认只展示，显式 --execute 加设备选择才运行 |
| `hydra/train.yaml` | `densekp/train_keypoints.py`；组合 data/trainer/paths/experiment | 推荐训练配置入口；默认 data=unselected，先打印配置再核验数据 |
| `hydra/experiment/densekp.yaml`、`default.yaml` | 当前 DenseKP 训练；仍需查实际代码读取字段 | 保留活动配置，不因 YAML 存在就启用未实现字段 |
| `hydra/data/unselected.yaml`、本地 tiny 系列 | 当前 DataModule 选择已有 dataset ID | unselected 是安全默认；其它 ID 需要对应本地 NPZ/图片 |
| `hydra/trainer/`、`paths/`、`hydra/` | Lightning/Hydra 生命周期与运行目录 | 保留 cpu/gpu/ddp/mps/default 组、路径变量和 CLI override；共享循环父配置改名见下文；CPU 配置不等于资产已经准备好 |
| `fisheye*.json`、`perspective_camera_stress.json` | 对应 tools/fisheye、tools/render 具体配方 | 按调用者解释，不能与 Hydra 混用；参见场景索引 |

P4安装批次补齐此前被忽略的Hydra data YAML；B2随后退役下列预设，保留配置与默认选择不变。
历史配置组是动态选择接口，以下组名已明确退役，不提供别名或回退。

B2a 首批仅将训练循环父配置改名为 `hydra/trainer/loop.yaml`，四个 trainer 的 defaults 顺序与配置值保持不变。旧父组选择 `trainer=default_hmr` 已停止支持；对应选择为 `trainer=loop`，但父组不是完整 Trainer 配方，日常仍选择 cpu/gpu/ddp/mps。默认 data=unselected、trainer=gpu、experiment=densekp 不变。其余退役预设见下文。配置对照与既有解析异常见 [验证记录](../docs/VALIDATION.md)。

B2a 第二批已停止以下七个 Hydra 组选择，均明确报 MissingConfigException，不提供别名或回退：

- data=train、data=densekp_train、data=bedlam_v2_v1、data=eval、data=eval_smplx。
- experiment=camerahmr、experiment=bedlam2。

旧 JSON 路径示例也已移除；这八份已跟踪文件的原内容可从 Git 历史找回。当前使用保留配置，显式核对 DATASETS.DATASETS_AND_RATIOS、VAL_DATASETS、TARGET_VALUE_SEMANTICS 及训练参数，先用 --cfg job 查看配置；Dataset 的 annotation_path/image_root 显式路径接口继续支持。tiny 配方不等价于旧全量组合，不能直接替换后声称训练或评价语义相同。

注册表全部 key、历史 NPZ 读取与监督语义、BEDLAM 数据来源及处理能力均保留。六个当前 data 文件（unselected 和五个 tiny）、experiment/default 与 densekp、所有现行 trainer/loop/paths/Hydra 配置未变。详见 [验证记录](../docs/VALIDATION.md)。

回归 suite 内的历史数据/权重路径相对源码仓库 ROOT；CLI 的 trial/data-root/experiment 仍相对调用 cwd。
source 显式传入 model_config 和 cuda:0，box 保持原来不传的行为，由 evaluator 决定默认值。
这不是 Hydra 配置，不改变训练参数，也不将 tiny 配方或不同指标统一。旧两套 evaluate_* 调度 CLI
已退役，使用 [当前命令与执行边界](../docs/COMMANDS.md)。

六组准确名称：`data=unselected`、`p0_dressed_tiny`、`omni_near_erp_tiny`、`omni_multiperson_tiny`、`ds_multiperson_tiny`、`ds_multiperson_amodal_tiny`（后五项同属data组）。配置组合成功与插值解析成功是两件事；历史GENERAL缺项和Hydra运行上下文解析异常仍保留，不为打印补默认值。
