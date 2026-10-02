# 配置归属与历史兼容

P4 已整体通过评审；P4/P5 随本次提交交付，P5 待最终评审。当前 [命令表](../docs/COMMANDS.md) 与 [交付报告](../docs/refactor/P5_READING_DELIVERY_20261002.md) 是运行和状态入口。

本目录的同名字段不保证被同一个入口消费。配置值、默认选择与拒绝规则在本次清理中未改变。

| 配置 | 消费入口/用途 | 状态 |
| --- | --- | --- |
| `hydra/train.yaml` | `densekp/train_keypoints.py`；组合 data/trainer/paths/experiment | 推荐训练配置入口；默认 data=unselected，先打印配置再核验数据 |
| `hydra/experiment/densekp.yaml`、`default.yaml` | 当前 DenseKP 训练；仍需查实际代码读取字段 | 保留活动配置，不因 YAML 存在就启用未实现字段 |
| `hydra/data/unselected.yaml`、本地 tiny 系列 | 当前 DataModule 选择已有 dataset ID | unselected 是安全默认；其它 ID 需要对应本地 NPZ/图片 |
| `hydra/trainer/`、`paths/`、`hydra/` | Lightning/Hydra 生命周期与运行目录 | 保留既有组名、路径变量和 CLI override；CPU 配置不等于资产已经准备好 |
| `hydra/data/{train,densekp_train,bedlam_v2_v1,eval,eval_smplx}.yaml` | 上游 BEDLAM/HMR 数据选择与历史命令 | 历史配置，保留原键和文件路径；本地完整数据未承诺存在，未自动启用 |
| `hydra/experiment/{camerahmr,bedlam2}.yaml`、`hydra/trainer/default_hmr.yaml` | 继承的 HMR/历史超参组合 | 不是新的 DenseKP 推荐配方，不删除动态 Hydra 选择入口 |
| `datasets.example.json` | 早期数据路径格式示例 | 未发现活动加载器读取它；保留示例，不伪装成当前 Dataset 配置。显式路径在 Dataset 构造参数中传入 |
| `fisheye*.json`、`perspective_camera_stress.json` | 对应 tools/fisheye、tools/render 具体配方 | 按调用者解释，不能与 Hydra 混用；参见场景索引 |

六份此前被忽略的 Hydra data YAML 已在 P4 安装批次解除忽略，随本次提交交付。解除忽略只是补齐交付，不是允许使用未验证监督或重新训练。
历史配置是用户可通过 Hydra 组名动态选择的接口，静态没有字符串调用不足以证明可以删除；清理结论见 [保留表](../docs/refactor/LEGACY_RETENTION_20261002.md)。
