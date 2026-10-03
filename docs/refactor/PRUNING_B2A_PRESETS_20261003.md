# B2a 第二批：八份历史配置/示例退役

日期：2026-10-03。实施与纯配置验收完成，**未提交，停在本批验收点；未进入 B2b**。

[执行清单](../../outputs/acceptance/pruning_b2a_presets_20261003/execution_manifest.json) · [依赖复核](../../outputs/acceptance/pruning_b2a_presets_20261003/dependency_review.json) · [比较记录](../../outputs/acceptance/pruning_b2a_presets_20261003/comparison.json)

## 基线和实际范围

实际分支 master，起始 HEAD `8db7ca5ec7583d1749db2e5f33e96abf3eff5a1e`。起始工作区已有上一批父配置改名、四个继承者修改、configs/README.md 修改、新 loop.yaml 与上一批报告，另有用户 docs/TODO.md 修改。本轮全部保留，没有重置或重新实施父配置改名；before 中父组已经是 loop。

本轮新增删除严格为以下八项：

| 文件 | 明确停止支持的选择/用途 |
| --- | --- |
| configs/hydra/data/train.yaml | data=train |
| configs/hydra/data/densekp_train.yaml | data=densekp_train |
| configs/hydra/data/bedlam_v2_v1.yaml | data=bedlam_v2_v1 |
| configs/hydra/data/eval.yaml | data=eval |
| configs/hydra/data/eval_smplx.yaml | data=eval_smplx |
| configs/hydra/experiment/camerahmr.yaml | experiment=camerahmr |
| configs/hydra/experiment/bedlam2.yaml | experiment=bedlam2 |
| configs/datasets.example.json | 旧 train/val JSON 路径示例 |

上一批已有删除 `configs/hydra/trainer/default_hmr.yaml` 单独记录，不计入本轮八项。没有创建兼容包装、别名或 archive；八份文件的原字节已另存验收目录，已跟踪历史也可从 Git 找回。

文档只修改 configs/README.md，以及 docs/training/TRAINING_DATA_FORMAT.md 的简短退役提示。后者历史正文原样保留，提示旧 data=densekp_train 命令已退役，当前命令/配置看现行导航。上一批报告、训练审计等历史正文未改。

## 依赖复核结论

- 逐项读取 Hydra defaults：train.yaml 默认使用 unselected/gpu/densekp；四 trainer 继承 default/loop；两个旧 experiment 向 default 继承，没有保留配置反向继承待删除组。
- 七个组是确实可供用户显式选择的接口，删除依据是本次明确退役授权，不是“字符串搜索无命中”。smoke_densekp_lightning 的 --dataset 参数也可接收用户组名，其默认 p0_dressed_tiny 保留；传入退役组名将明确失败，不加回退。
- check_workspace 按现存 JSON/YAML 遍历并解析，没有硬编码这些文件或固定数量。verify_installation_runtime 使用 train + trainer=cpu，仍被保留。本轮只读其实现，没有运行 Dataset 或这些工具。
- 当前训练入口使用 train.yaml；相关配置测试选择默认或 ds_multiperson_tiny/cpu。安装包发现没有依赖这八份预设。未发现 datasets.example.json 的新增实际加载器消费者。
- README、COMMANDS、INSTALLATION 的当前推荐命令不依赖退役组；configs/README 已更新。旧数据接口说明增加退役提示；上游训练审计、已标历史的代码地图及历史阶段报告保留其语境。densekp_trainer 模块名等同名文本没有计为 data=densekp_train 依赖。

未发现需要暂停某项删除的保留流程依赖；未迁移业务实现。

## 独立 before 和执行方式

使用全新目录 `outputs/acceptance/pruning_b2a_presets_20261003/`，未覆盖前批证据。删除前保存 910 个文件字节到 protected/，包括八个原文件、其它配置、源码、文档、原 fixtures、TODO、上一批改名成果及上一批验收目录的文件。原配置快照在 protected/configs/；本轮 after 配置另存 after/configs/。

capture_config.py 复用上一批已经验证的纯配置采集方式；40 个保留组合的 ID/override 定义与上一批 accepted after 逐项相等，但**结果全部按本轮删除前状态重新采集**。本轮 before/after 均选择 loop，不做父名归一化；追加七个退役组选项，两阶段都采集。

两个实际 cwd 分别为仓库根和新验收目录内 temporary_cwd。配置路径为绝对路径，固定 now、PROJECT_ROOT、Hydra runtime cwd/output_dir 和 job 上下文，避免比较时剔除时间/目录字段。每轮先 initialize，再设置固定时间 resolver；compose 后按上一批确定的方法设置 runtime.cwd。未补任何业务字段或缺失值。

脚本只导入 Hydra/OmegaConf 等配置依赖，禁止 torch、pytorch_lightning、densekp、densekp_retrain、numpy、bpy、smplx 导入，禁用 instantiate/call，阻断子进程及验收目录外写入。解释器为现有 `/home/yy/conda_envs/camerahmr/bin/python`，Hydra 1.3.2；未新建环境或安装依赖。

已执行的命令形式如下；before 两个 cwd 和 before_gate 均完成后才删除文件，再执行 after 与比较：

```text
python -B <新验收目录>/capture_config.py --phase before --cwd root
python -B <新验收目录>/capture_config.py --phase before --cwd temporary
python -B <新验收目录>/capture_config.py --phase after --cwd root
python -B <新验收目录>/capture_config.py --phase after --cwd temporary
python -B <新验收目录>/compare_config.py
```

实际使用上述绝对解释器并设置 PYTHONDONTWRITEBYTECODE=1。结果文件用独占创建方式写入，不可复用命令覆盖旧证据。未运行 unittest 或生命周期测试。

## 验收结果

保留矩阵为：默认 1 + 四 trainer × 六 data 组 24 + 四 trainer × 两保留 experiment 8 + trainer=default × 两 experiment 2 + 四 trainer 显式覆盖 4 + loop 父组单独选择 1，共 40 种。

六个 data 组是 unselected、p0_dressed_tiny、omni_multiperson_tiny、ds_multiperson_tiny、ds_multiperson_amodal_tiny、omni_near_erp_tiny；保留 experiment 为 densekp/default。显式覆盖 TOTAL_STEPS/LOG_STEPS/VAL_STEPS、trainer.precision/devices、DATASETS_AND_RATIOS/VAL_DATASETS 和 paths.data_dir，与上一批定义完全相同。

| 检查 | 结果 |
| --- | --- |
| 保留组合 × 两 cwd | 80 组 before/after 精确一致；全部组合成功 |
| 保留任务配置解析 | 72 组成功，8 组既有失败原样保留 |
| 完整结构、顺序、插值、类型、异常 | 原容器与 13,400 个解析叶记录精确相等；不忽略 hydra 子树或任何字段 |
| 七个退役组删除前 × 两 cwd | 14 次实际组合/任务解析成功；原记录保存 |
| 七个退役组删除后 × 两 cwd | 14 次明确 MissingConfigException，消息指明缺失组，无组合结果或回退 |
| 两 cwd 一致性 | before 与 after 各自全部 47 条记录一致 |
| 本轮新增删除 | 精确八项，与上一批 default_hmr 删除分开 |

8 次既有任务解析失败来自 cpu/gpu/ddp/mps × experiment=default × 两 cwd：缺少 GENERAL.LOG_STEPS，异常为 InterpolationKeyError，位置 trainer.log_every_n_steps。逐叶记录中的其它缺失 GENERAL 引用也精确相等。没有为通过验收补默认值。

Compose-only 的完整 Hydra 容器仍在 hydra.hydra_help.hydra_help 处报 MissingMandatoryValue；这一既有占位符异常也保持。不能称完整 Hydra 容器解析全部通过，因此另逐叶比较后续所有配置和异常。**本批允许的保留组合字段差异为空**，不再沿用上批父名称变化的例外。

## 保护、限制与停止点

- 除八项删除和两份必要文档修改外，900 个保护文件逐字节一致。其中包括上一批验收目录的全部 424 个文件、上一批报告与 loop/四继承者、原 fixtures 和用户 TODO；未假定 ignored 文件有 Git 备份。
- 所有 DATASET_FOLDERS、DATASET_FILES 与注册表 key、Dataset/裁剪/监督/模型/评价源码字节不变。六个当前 data、两份保留 experiment、全部现行 trainer/loop/paths/Hydra 及 train.yaml 字节不变；默认选择和 override 行为未改。
- 本次只退役命名预设和 JSON 示例，未删除数据格式、来源、BEDLAM 处理或历史 NPZ 读取能力。tiny 配方不是旧全量混合配置的等价替代。
- 未调用训练入口、Trainer.fit、optimizer.step，未构造模型或 Dataset，未读取真实数据/权重，未训练、推理、渲染、生成资产、下载、安装、升级或计算 SHA256。未提交 commit、未修改外部项目。
- 四 trainer 的验收是现有环境中的纯配置组合，不是 CPU/GPU/DDP/MPS 设备运行验收；也不代表空环境安装或旧实验效果复现通过。

没有未解决的删除依赖或新增配置差异。完成后停在 B2a 第二批验收点，不进入 B2b 或其它清理。
