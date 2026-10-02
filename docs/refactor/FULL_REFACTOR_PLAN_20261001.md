# DenseKP 全仓库重构核查与执行方案

执行日期：2026-10-01（Asia/Shanghai）。**方案已通过用户评审，进入分批实现。**
本文主体保留迁移前核查基线，旧代码位置和“本轮”均指规划轮次。
当前进度及已确定取舍见 [执行任务与验收门槛](IMPLEMENTATION_TASKS_20261001.md)；
首批结果见 [P0.1 + P1.1 + P1.2 实施记录](P1_METRICS_CONFIG_20261001.md)。
后续批次见 [公共推理与训练审计](P1_TRAINING_AUDIT_20261001.md) 与
[评价展示及 FixedValidation](P1_REPORTING_CALLBACKS_20261001.md)：P0.2、P1.1–P1.6 已完成，
真实 GPU 前后对照、保存预测回放与 CPU 生命周期分别验收。
2026-10-02 [P2.1/P2.2 显式数据源与裁剪](P2_DATASET_CROP_20261002.md) 已通过评审；
[P2.3–P2.5 来源、框与监督适配](P2_SOURCES_BOXES_20261002.md) 已通过评审，P2 整体完成。
[P3.1/P3.2 人体、衣物与坐标相机计算](P3_GEOMETRY_CAMERA_20261002.md) 已通过评审。
[P3.3 成像后处理与标注](P3_IMAGING_20261002.md) 已通过评审。
[P3.4 Blender 对象、材质与装配](P3_BLENDER_20261002.md) 已通过评审，结论限于两条代表路径。
[P3.5 场景配方与执行入口](P3_SCENE_RECIPES_20261002.md) 实现与验收完成；[P3 整体检查表](P3_COMPLETION_CHECKLIST_20261002.md) 待评审。
P4–P5 继续按执行清单推进，不将有限场景验证扩大为全链覆盖。

建议保留 `densekp` 与 `densekp_retrain` 两个包的职责边界，从训练、推理和评价中的公共函数开始，随后整理 Dataset、生成链与安装入口。不要先搬整个目录，也不要统一当前实际不同的评分、采样和渲染规则。当前可开始离线公共逻辑整理；真实 checkpoint 和 Blender 的迁移验收仍有明确的证据门槛。

最新框增强权重仍是诊断产物，部署替换未获认可；重构不优化效果。

## 1. 实际基线与核查范围

### 1.1 Git 与操作边界

| 项目 | 本轮实际核查 |
| --- | --- |
| 仓库 | `/mnt/data1/yy/home/yy/Projects/densekp_retrain`；shell 的 `pwd` 显示 `/home/yy/Projects/densekp_retrain`，本机路径别名需保留 |
| 分支 | `master` |
| HEAD | `a3a5e7bb641bf1dee774d015127fee6b39c28bda` |
| 最近提交 | `2026-10-01 16:36:05 +0800`，`Add bounded box-robustness trial and fixed three-model evaluation` |
| 开始时工作树 | `.gitignore`、`docs/TODO.md` 已修改；没有 Git 可见的 `??` 文件 |
| 已有修改内容 | `.gitignore` 将忽略范围改为整个 `/assets_external/`；TODO 新增鱼眼成像问题和参考图片路径。两者完整保留 |
| AGENTS.md | 已检查仓库、适用祖先路径及本次涉及的源代码/配置/文档目录，未找到适用文件；未从第三方项目引入指令 |
| 本轮写入 | 仅本报告与 `outputs/plans/full_refactor_20261001/` 新规划目录；测试临时文件在临时目录 |
| 明确未执行 | 不计算 SHA256；不加载真实模型/大 checkpoint；不训练、渲染、下载、全量模型评价、提交、reset/clean、移动旧文件；不修改外部项目及第三方资产 |

完整初始状态与已有 diff 见 [repository_baseline.json](../../outputs/plans/full_refactor_20261001/repository_baseline.json)。这里的 Git HEAD 是读取已有提交标识，不是重新计算资产哈希。

`outputs/` 被 Git 忽略，故本地规划 JSON **不会随普通 commit 自动保存**。本报告保留关键结论；后续决定是否把必要小清单纳入版本控制，不能为了本轮核查擅自修改已有 `.gitignore` 或强行 add。

### 1.2 commit 没有包含全部运行依赖

本轮只扫描代码目录、顶层资产目录、代码中的明确绝对路径及选定产物引用，没有递归加载 outputs、纹理包或权重。

| 依赖 | 当前情况与迁移含义 |
| --- | --- |
| 原 DenseKP 权重 | `assets_external/densekp.ckpt` 存在，8,049,335,298 bytes；本轮仅 stat |
| ViTPose 初始化权重 | `assets_external/vitpose_backbone.pth` 存在，2,523,785,910 bytes；`DenseKP.__init__` 仍必读，本轮未加载 |
| 活动框增强权重 | `outputs/experiments/box_robustness_400_20261001/training/step_0400/checkpoint.ckpt` 存在，8,049,317,445 bytes；未反序列化 |
| regressor | `assets/densekp_regressor_R.npy` 存在但不在当前 tracked 文件清单；实际读取检查为 float32 `[138,10475]`、204 个非零值，行和约 1 |
| 镜头有效域 | `assets_external/real_erp_valid_masks.npz` 存在；DS 单测实际使用此小型文件 |
| 本地 Python 扩展 | `assets_external/python_deps/OpenEXR.cpython-310-x86_64-linux-gnu.so` 存在，相关脚本用 sys.path 接入；不是标准安装依赖 |
| 数据 | 六套 ignored tiny 数据目录存在；各自 images 直接目录层的 3–16 个条目未见符号链接。此结论不覆盖整个输出树 |
| 隐藏的本地配置 | `configs/hydra/data/{bedlam_v2_v1,densekp_train,eval,eval_smplx,omni_multiperson_tiny,train}.yaml` 共六份存在但不 tracked，普通 `rg --files` 默认可能漏掉 |
| 外部模型/纹理/Blender | 对代码中的 9 个完整绝对路径字面量做了存在性检查，均存在，包括外部 SMPL-X 目录、纹理 ZIP、他人 Blender。未认证目录内部完整性、版本或许可可迁移性 |
| 第三方源码 | 四个带 `.git` 的 BEDLAM 目录读取了 HEAD，另有 clothing download 脚本等；未检查其全部工作树，更未改动 |
| tracked 大资产 | Git 中仍有 `cycling_forest.blend1`（347,621,829 bytes）、`riverfront.blend1` 和 hidden_alley 分片。忽略规则并不取消已跟踪文件；本轮不删除 |

详见 [local_environment.json](../../outputs/plans/full_refactor_20261001/local_environment.json)、[supplemental_inventory.json](../../outputs/plans/full_refactor_20261001/supplemental_inventory.json)、[absolute_path_checks.json](../../outputs/plans/full_refactor_20261001/absolute_path_checks.json) 与 [local_links_and_third_party.json](../../outputs/plans/full_refactor_20261001/local_links_and_third_party.json)。`local_environment.json` 中的 Git ignored status 命令不能完整枚举忽略资产，以上结论以目录、`git check-ignore` 和 tracked 对照为依据。

默认 shell Python 不具备 NumPy/Torch 等项目依赖。实际 CPU 核查使用 `/home/yy/conda_envs/camerahmr/bin/python`：Python 3.10.20、Torch 2.0.0+cu118、Lightning 2.0.2、NumPy 1.26.4、OpenCV 4.13.0.92、Hydra 1.3.2。`requirements.upstream.txt` 的 NumPy 1.23.5 / OpenCV 4.10.0.84 与实际不同。该环境没有安装本仓库 distribution，靠 `PYTHONPATH=src:.` 工作。本轮没有安装或升级依赖。

### 1.3 实际活动产物

已读取 `outputs/experiments/box_robustness_400_20261001/status.json`：

- `status=completed_and_stopped`，400 次更新；`deployment_replacement_accepted=false`。
- 活动输入 `data_v2/`，活动上下文 `actual_crop_context_v2.json` / `data_context_indices.json`。
- 活动训练 `training/`，评价 `evaluation_v2/`，比较 `comparison_final/report.json`，流程审计 `final_audit_v2.json`。
- 原始 `data/`、`evaluation/`、旧 comparison、`actual_crop_context.json`、`final_audit.json` 及旧 `case.json` 均保留；活动 case 元数据优先按 status 指向的 `reviewed_case.json` 解释。
- `split_v2.json` 是来源隔离活动划分；`split.json` 是早期被否决方案，不因文件名短就优先使用。

本轮读 JSON 并 stat 引用确认 A/B/C 挑战各 143、C 原 val 79、C integration 51，共 559 个主预测路径存在；history B/C 报告各有 4 个 case，合计报告计数 567。history 报告使用不同文件字段，**本轮未逐个回读其预测数组**。559 与 567 是计数范围不同，不是缺了 8 个主要样本。

### 1.4 阅读与检查覆盖

先读 README、READING_GUIDE、SAMPLE_WALKTHROUGH，再读用户列出的数据要求、遮挡监督、遮挡控制、来源隔离数据、普通/框增强 400 步报告；复核 refactor 旧依赖、删除、遗留、正确性与真实模型验收报告。根目录考古报告按顶部基线声明解释，未当作当前实现。

扫描覆盖主目录 276 个 `.py/.yaml/.json/.sh` 文件，其中 Python 239 个；另读并 AST 解析资产旁的 `assets/scenes/photoreal/urban_streets_and_alleys/preview_render.py`，共 240 个项目 Python 文件语法通过。没有解析第三方源码树，也没有把 outputs 内入口快照当活动源码。

| 范围 | 核查深度 |
| --- | --- |
| `densekp/` | Dataset、裁剪、监督、objective、trainer、DataModule、checkpoint、配置入口逐条阅读；网络创建与head输出/参数命名核对；Transformer/ViT内部做结构和依赖检查，未重验网络数学 |
| `src/densekp_retrain/` | 22 个 Python 文件清单与依赖；几何、DS、runtime、EXR、render jobs主线阅读；现有数值回归实际执行 |
| annotation / generate | NPZ adapter、来源打包、SMPL-X几何/帧选择和serialization链核对；未重新生成资产 |
| fisheye / render / scene | DS主线、ERP来源、world变换、mask、visibility、两个renderer及材质/发型边界；其他场景配方与渲染脚本做结构和引用核查 |
| experiments | 全部70个文件结构/依赖检查，深入当前训练、框准备、评价、来源资产及公共几何被导入链；历史衣物迭代未逐条证明算法正确 |
| validate / visualize | 全部43/8个文件结构/依赖检查，主评价/来源验证/画图复用与模块顶层副作用深入核对 |
| configs / scripts / tests | 包括6个ignored本地YAML；执行56项选定CPU测试；下载脚本和全盘扫描脚本只读 |
| 外部 pano-stitch | 只读定位 delivery `ort_runner.py`、camera `fisheye_equirect.py`、pipeline；核对真实分辨率、new→20260320映射及框长边无1.2扩张。未导入或运行外部项目 |

扫描结果不是“每个文件的算法都已验证”。[source_inventory.json](../../outputs/plans/full_refactor_20261001/source_inventory.json) 保留 import、函数位置、长度、参数数和副作用候选；[migration_inventory.json](../../outputs/plans/full_refactor_20261001/migration_inventory.json) 给全部 277 项源/配置/脚本文件安排保留或迁移类别。

执行者可直接读取 [execution_tasks.json](../../outputs/plans/full_refactor_20261001/execution_tasks.json)：23个待执行任务均列出前置依赖、文件、目标、验证、风险与回退边界。[dependency_summary.json](../../outputs/plans/full_refactor_20261001/dependency_summary.json) 是关键依赖摘要，[dependencies.dot](../../outputs/plans/full_refactor_20261001/dependencies.dot) 可用于生成完整静态依赖图；原始import证据在 [dependency_edges.json](../../outputs/plans/full_refactor_20261001/dependency_edges.json)。

## 2. 当前架构与关键调用链

### 2.1 依赖现状

静态解析发现 **151 对不同的 tools→tools 文件依赖**、96 个含 sys.path 操作的文件。原始 `dependency_edges.json` 有按 import 名称产生的重复边，不能把其中386条记录当386对独立依赖。已解析静态 import 图没有强连通循环；动态导入、子进程和 Blender 的多种模块名别名不在此结论的完整证明范围。

值得先处理的反向依赖：

| 被依赖实现 | 不同源模块数量 | 说明 |
| --- | ---: | --- |
| `tools/experiments/prepare_cycling_characters.py` | 26 | 参数字段、人体计算、JSON输出和两张历史姿态配方共存 |
| `tools/validate/validate_bedlam_densekp_tiny.py` | 15 | 许多新实验仍从此处取 `audit_cfg` |
| `tools/fisheye/run_ds_scene_pair_pilot.py` | 8 | 主线依赖pilot中的rig/有效域/后处理 |
| `tools/validate/evaluate_candidate_baseline.py` | 6 | 训练、补汇总、比较复用此入口中的summary/aggregate |

当前 src 静态 import 没有回指 tools 或 densekp；但 `rendering/jobs.py` 通过子进程脚本路径调用 tools，属于已有编排依赖，需要单独迁移。

### 2.2 当前消费端

```text
configs/hydra + densekp/core/configs注册表
  → DatasetKeypoints → get_example_projverts → crop_geometry
  → 图像[3,256,256]、GT[138,3]、affine[2,3]
  → DenseKP.forward_step → 宽度32:-32 → ViT → head[ B,138,3 ]
  → keypoint_objective → 手动AdamW zero_grad/backward/step
  → 分loader验证 → ValidationCheckpoint

train_source_isolated_trial
  → 历史validate脚本audit_cfg + 临时注册表
  → IndexedRows + 固定shuffle或400行冻结框顺序
  → train_cycling_smoke.CheckUpdates/probes/predict
  → FixedValidation（独立于默认Lightning val loop）
  → evaluate_candidate_baseline.aggregate → checkpoint + 原始预测

evaluate_box_robustness
  → 按UUID隔离的多个Python子进程
  → evaluate_candidate_baseline / evaluate_ds_box_control
  → 输入身份/来源核对 → 严格加载 → 固定预测 → 分组/配对指标 → 报告/图
```

普通训练入口与有界实验的验证/checkpoint调度不同，不能为了“一套训练循环”把后者直接改成前者。当前 `predict` 会把模型切到 eval 且不恢复；FixedValidation 则主动检查 RNG/参数/步数并在结束时切回 train。公共函数迁移必须逐调用点保持这些事实。

### 2.3 当前生产端

```text
图片重建姿态 / BEDLAM动作 / 冻结角色shape
  → generate_* 或 prepare_*实验 → body.geometry → 顶点与R@V
  → 身体/衣物/groom派生制品 + 来源manifest
  → layout/people/rig配置
  → 普通Python subprocess → Blender render_equirectangular_source
       → 原生场景 + 同一物体变换 + RGB/Depth/IndexOB
  → process_scene → DS ERP↔fisheye重映射 + 显式有效域
       → normal ERP / world+2D标签 /实例mask /框
  → 来源验证 + candidate冻结 → diagnostic/merged loader NPZ
  → 来源隔离198/79行 → 框增强400行及143挑战行
```

生成时的整球 ERP 是中间成像缓冲，随后转鱼眼再生成实际网络 normal ERP；本轮保持此算法。TODO中直接渲染鱼眼的质量问题属于另一个成像设计/效果任务，不能借重构换采样链。

BEDLAM原图适配、透视P0、Omni-Radtan、DS normal ERP仍是不同路径，不能用“都是相机投影”替换其不同标定、坐标、可见性或历史标签策略。

## 3. 有代码证据的问题与历史复核

完整21项结构化清单见 [findings.json](../../outputs/plans/full_refactor_20261001/findings.json)。这里给出评审重点；位置以本轮HEAD工作树为准。

| ID / 分类 | 代码证据与具体问题 | 处理边界 |
| --- | --- | --- |
| F01 当前确认 | `train_source_isolated_trial.py:23–27` 从早期smoke取CheckUpdates/probes/predict；`evaluate_ds_box_control.py:11` 从另一个evaluator间接导入torch、DenseKP等 | 按定义归属迁库，旧路径保留转导出 |
| F02 当前确认 | `validate_bedlam_densekp_tiny.py:25` 定义通用audit_cfg；`DatasetKeypoints:38` 强制注册表寻址；`prepare_box_robustness.py:28` 等使用patch.dict | 配置工厂先移，显式数据源接口后移；不可只把mock再包一层作为最终设计 |
| F03 当前确认 | `evaluate_candidate_baseline.py:128–144,195`、`evaluate_ds_scene_pair.py:198`、`evaluate_densekp_npz.py:36` 重复加载、精度与推理；各自画裁剪点图 | 将数值策略作为显式现有参数，保留medium/highest等差异 |
| F04 当前确认 | `occlusion_metrics.py:7` 非有限值报错；candidate `summary:26` 返回失败数量和null；训练objective:50按人归约 | 不能以公共summary偷偷替换所有异常策略或分母 |
| F05 当前确认 | `experiment/densekp.yaml:24` 写l1；活跃objective:30实际平方距离；head:17读取constants，ViT:12固定结构，YAML结构参数未接入 | 先标明实际生效/历史兼容字段，不启用原无效配置 |
| F06 当前确认 | `densekp_trainer.py:25–26` 在构造时无条件读取ViTPose；默认常量为相对cwd路径 | 即使完整checkpoint推理也有第二份权重依赖。首阶段不移除该读入，不改变构造随机数消耗 |
| F07 当前确认 | `datasets/utils.py:430–537` 19参数/10返回值，混合读图、增强、颜色、坐标；Dataset:76变量名rgba与三通道活跃输入不符 | 分解时保留旧tuple接口、dtype与计算顺序；不是仅格式化长行 |
| F08 当前确认 | `run_ds_scene_pair_pilot.py:148–244` 同时做EXR/几何读取、投影、remap、mask、框、存储、拼图 | 先抽纯计算并接已有buffer；不要求每次重渲染 |
| F09 当前确认 | `prepare_cycling_characters.py:19–43` 的PARAMETERS/evaluate/zero_pose/write_json成为26模块依赖 | 几何/衣物稳定函数迁库，POSES/MATERIALS配方留实验层 |
| F10 当前确认 | 两个ERP renderer:21/25起重复compositor，但一个open原生场景、一个factory+link+SUN；历史衣物IndexOB=2而新多人按人物 | 合并前分别验收，不直接让旧CLI指向新renderer |
| F11 当前确认 | `audit_skin_texture.py:10–33` 顶层读模型/ZIP、写body_uv、计算SHA256；`prepare_procedural_eyes.py:9` 顶层造资产；`make_skin_texture_review.py:8` 顶层读输出 | 本轮只静态看，没有import。后续main保护与资源加载拆分需要保持旧执行顺序 |
| F12 当前确认 | pyproject只打包src且无dependencies；`runtime/paths.py:6` 按parents[3]找仓库；96文件sys.path | 当前根目录执行模式可以用，不代表wheel安装后任意cwd可用 |
| F13 当前确认 | `evaluate_box_robustness.py:23–26` 固定日期；box evaluator:41/62固定cuda:0；DS pilot:251默认他人Blender路径 | 新实验配置显式来源和设备；保留旧默认以重放 |
| F14 当前确认 | README末尾仍称训练未验证，旧报告说losses不存在；实际训练和新模块已出现 | 新现行入口指向活动版本，历史报告保持原意 |

**行为修复事项单列，不属于本轮纯重构的默认决策：**

1. B01，已确认报告语义冲突：`evaluate_densekp_npz.py:54–63,81` 将`weights>0`命名为visible，并声明GT-visible only。新amodal数据包含遮挡监督；不能继续解释为可见点指标。是否增加正确字段、保留旧字段别名或使旧CLI仅接受旧策略，需要独立schema兼容修复。
2. R01，可能风险：`datasets/utils.py:449–451` 根据`portrait/closeup`名字旋转图像，没有在该处同步旋转点。部分历史标签可能原本就在旋转后的坐标，未核实前不能删除旋转或宣称所有旧数据错位。
3. R02，可能风险：DS pilot 的GT有效域用floor，`points_in_mask`用rint；旧adapter也调用后者过滤GT，新评价调用它统计预测出镜头。先用边界fixture判断同一路径是否冲突；当前活动floor约定与预测nearest诊断各自保留。
4. 配置“修正”若让LOSS_TYPE=l1或YAML head参数真正生效，会改变模型目标或权重形状，必须成为独立行为任务。缩小checkpoint构造的初始化工作也需先核查RNG和完整恢复语义。

**已解决或已设防的历史问题：**

- losses文件已存在；活跃trainer直接使用`keypoint_objective`，不能再报“缺loss无法import”。
- 未验证的138点flip被明确拒绝；`use_augmentation=False`、平移概率、DenseKP路径的抗锯齿方向已有实现和本轮CPU测试。
- 默认data为unselected，显式语义校验；Trainer由Hydra实例化；worker0不传prefetch；验证不丢尾批；当前权重验证结束后保存排名。这些代码已确认，训练保存/恢复测试本轮未执行。
- 监督已集中为`in_view & inside_full_crop`，遮挡不取消监督，旧NPZ以`visibility_known=false`区分；本轮实际回读通过。
- 旧cleanup的warp/renderer遗留仍存在，但已有DS低分辨率mock buffer测试，不再是“完全无测试”；它们仍不足以证明完整Blender RGB/Depth/IndexOB一致。
- `evaluate_candidate_baseline.py:222` 的图像缓存键现在包含scene+camera；活动产物中修正过的图片引用已存在，不重复报告旧缓存问题为当前必现问题。衍生框上下文使用status指定v2，不使用父框旧计数冒充当前框。

## 4. 建议目标架构

### 4.1 保留两个包，依赖方向明确

```text
tools/明确CLI与历史配方
  ├── densekp.core：模型、实际Dataset、crop/监督/loss、checkpoint、推理原语
  └── densekp_retrain：身体/相机/成像/标注/来源组织、实验数据评价与编排
          └── 可依赖densekp的轻量数据契约、crop、监督和评价函数

densekp.core 不依赖 tools 或 densekp_retrain
body/camera/annotation基础数学 不依赖 Blender
rendering.blender 只由Blender入口导入
普通Python编排 通过显式subprocess边界调用Blender入口
```

不是把全部实现塞进一个包。`densekp` 保持 checkpoint 相关类路径和网络module层级；`src/densekp_retrain` 承接已经发生复用的生成/实验逻辑。包内 `__init__` 不导入全部下游模块，避免只用相机却加载Torch/Lightning/bpy。

拟新增模块只在迁入具体函数时建立，不预建空目录或基类：

| 职责 | 保留/目标位置 | 首批来源与接口 |
| --- | --- | --- |
| 138契约、完整crop、监督 | 原 `densekp/core/{keypoint_contract,crop_geometry,keypoint_supervision}.py` | 原位维护，函数签名/数组返回保持 |
| 网络、loss、optimizer/checkpoint | 原trainer/backbones/heads/components/objective/checkpointing | 原位；不重命名`backbone.*`、`head.*`参数 |
| Dataset配置与显式数据源 | `densekp/core/datasets/{configuration,loading}.py`，原Dataset保留 | 从audit_cfg起步；P2新增显式annotation_path/image_root，不依赖可变全局注册表 |
| 固定权重推理 | `densekp/core/inference.py` | 严格加载审计、输入/输出形状、单batch推理；保留现有load路径、精度、非有限策略和mode要求 |
| 纯评价规则 | `densekp/core/evaluation.py` | 从occlusion_metrics移point_errors/masks；strict summary与failure summary分名保留 |
| 候选数据汇总/展示 | `src/densekp_retrain/evaluation/{candidates,visualization}.py` | aggregate、pair aggregation、panel；action/layout/scale属应用元数据，不能塞进神经网络 |
| 训练审计 | `densekp/core/training_audit.py`；实验回调`src/densekp_retrain/training/trial_callbacks.py` | CheckUpdates/probes与有界FixedValidation分开；入口保留400步/seed/数据顺序 |
| 身体与坐标 | 现body.geometry加`parameters.py`、`transforms.py` | PARAMETERS/evaluate/zero_pose/world_points；单位m，body/Y-up→world/Z-up矩阵明确 |
| 衣物/发型几何 | `src/densekp_retrain/clothing/{shell,binding,contract}.py` | 只迁稳定运算；sports shell、commuter修复和版本配方不强行统一 |
| 相机与成像 | 现camera加`erp.py`、`rig.py`；`imaging/{erp_remap,ds_remap,lens_masks,omni_warp}.py` | camera为投影数学，imaging为采样/插值/有效域；DS与Omni接口分开 |
| 标注与样本组织 | 现annotation加`visibility.py`、`ds_scene.py`；`data/{source_split,packing,boxes}.py` | 保留GT/visible/in_view区别、原行序和来源闭包 |
| Blender实现 | `src/densekp_retrain/rendering/blender/{erp,materials,groom,scene}.py` | bpy依赖隔离，旧renderer脚本仍处理`--`参数并调用明确mode |
| IO/runtime | 现io/runtime，按需要加`io/json.py` | 保留路径相对基准与JSON空值格式；运行时显式选择root/Blender |

`get_example_projverts`将来用具名结果表示image、display image、original image size、crop center/size、transformed labels、affine；旧10元tuple通过兼容包装继续返回。不要新增每层都转换一次的大型对象体系。

### 4.2 数据与单位契约

| 量 | 固定契约 |
| --- | --- |
| `R` / vertices / DenseKP | `[138,10475]`、`[T,10475,3]`、`[T,138,3]`；138行语义不变；人体/衣物共用摆放矩阵 |
| 图像与相机 | 真实鱼眼2880×2880、真实ERP5760×2880、当前合成ERP1536×768；分别记录width/height，不能靠全局常量猜单位 |
| 标定 | `20260320-riding-dataset3` / DS_20260320；031/0311保持单独历史路径 |
| bbox | `xyxy_exclusive`像素；`center`像素xy；`scale*200`为源栅格像素尺寸。合成历史1.2扩张与部署1.0不合并 |
| 输入 | 完整RGB归一化`[3,256,256]`；forward内部取`[:,:,32:-32]`后`[3,256,192]` |
| 标签/输出坐标 | 完整crop归一化xy=`pixel/256-.5`；restore用实际affine；无额外32px偏移 |
| 监督 | `in_view & inside_full_crop`，后者有限且`[-.5,.5)`；可靠遮挡GT仍监督；原有旧NPZ权重不重写 |
| 第三值 | GT为生产器声明的supervision_weight；预测原值保留，非默认置信度；with_var分支含义不变 |
| ERP周期 | x按对应宽度W周期处理；ROI偏移/ERP roll/球面旋转是不同外层操作，不混为同一个affine |
| 指标 | crop px、ERP px、点平均、等框平均、训练按人归约分开命名；非有限/空组/失败数量保留 |

## 5. 可独立验收的迁移阶段与任务

本次“第一阶段”是规划；下文P1–P5指**后续实现阶段**。阶段交付后停止扩大范围，交规划侧评审。每个阶段仅修改本阶段源码与新验收目录，不移动旧输出；回退是撤回该阶段源码变更/切回旧入口，不是reset整个工作树。

### P0：锁定迁移用证据（本轮已完成一部分）

- 已有：Git状态、静态图、资产/环境清单、56项CPU检查、7样本回读、原始/衍生来源划分复核。
- 待执行者在动相关计算路径前采集：选定样本完整float32输入；两种当前模型加载路径的真实strict-load key/shape清单；固定权重FP32预测；CPU小网络训练/恢复与采样RNG快照；两渲染器对应的有限Blender几何/pass对照。
- 产物：新目录 `outputs/acceptance/refactor_<阶段>_<日期>/before/`，包含来源路径、size/mtime、配置、命令、环境、样本ID；不做SHA256。
- 风险/回退：如果before采集失败，保留日志；只阻止相关迁移，不重新训练400步、不修GPU0、不覆盖before。

### P1：训练、推理、评价公共逻辑（建议首先执行）

| 任务 | 涉及文件与迁移方式 | 产物/验收 |
| --- | --- | --- |
| P1.1 纯指标提取 | `tools/validate/occlusion_metrics.py` → `densekp/core/evaluation.py`；candidate的summary/aggregate → `src/.../evaluation/candidates.py`；旧函数名转导出 | saved raw预测重算与旧函数逐数组/计数/JSON语义精确比较；strict异常和failure-null两种行为都有fixture |
| P1.2 配置工厂归位 | `validate_bedlam_densekp_tiny.audit_cfg` → `densekp/core/datasets/configuration.py`；修改当前训练/评价导入 | 保持YACS结构、默认值、augmentation开关；旧import仍可用；7样本输入/GT/affine精确相同 |
| P1.3 公共推理 | 新`densekp/core/inference.py`；接入candidate、ds_scene_pair、ds_occlusion_control、ds_box_control，先不变CLI参数 | `strict=True`、map_location、构造顺序、数值精度和模式逐调用点保持；真实before/after七样本 ≤1e-5原始输出差门槛 |
| P1.4 审计回调脱离smoke | `train_cycling_smoke` 中CheckUpdates/probes迁`training_audit.py`；`train_source_isolated_trial`改导入；predict旧行为转包装 | 不改变参数顺序、AdamW分组、zero/backward/step；CPU小模型生命周期与梯度测试；本轮未运行这项训练验收 |
| P1.5 分离应用汇总/绘图 | candidate main拆输入核查、逐行预测、配对汇总、报告输出；panel迁visualization；DS专用轮廓覆盖图保留专用函数 | 保持case顺序、命名、JSON字段、空组、图片尺寸/颜色/点选mask；不要为共用画图丢失不同图层 |
| P1.6 有界回调拆分 | FixedValidation迁`src/.../training/trial_callbacks.py`，仅在前五项通过后做；训练入口保留400/100、seed930、shuffle开关 | eval→FP32→保存→train顺序、参数version、global_step、CPU/CUDA RNG检查保持；成功路径与异常停止日志兼容 |

第一批建议只交付P1.1–P1.2，便于纯CPU评审；P1阶段整体完成还需P1.3–P1.6的对应验证。P1不会提前改变Dataset构造签名、ViT结构、loss、两个大权重的初始化依赖或默认训练配置。

**P1主要风险**：新通用predict自动恢复mode导致旧调用语义变化；load helper少一次初始化导致RNG序列变化；聚合改用不同dtype或先mean再sum改变浮点结果；兼容包装漏导出原来被其他脚本间接引用的符号。每一项用旧/新函数同输入比较，失败就保留旧调用链，不带着“误差应该很小”继续迁移。

### P2：Dataset、裁剪、来源与监督路径

| 任务 | 涉及文件与迁移方式 | 产物/验收 |
| --- | --- | --- |
| P2.1 显式数据源 | DatasetKeypoints新增可选显式annotation_path/image_root；旧dataset ID由兼容resolver转换；DataModule和tools逐个接入 | 删除活动调用方patch.dict；旧ID、scene相对imgname与merged绝对imgname三路同样本精确一致 |
| P2.2 裁剪职责拆分 | utils中的读图、增强采样、图像warp/normalize、标签transform拆成小函数；crop_geometry原位 | 具名结果+旧tuple包装；full image/center input/affine/GT/监督和NumPy/Python RNG精确比较；保留不支持flip保护 |
| P2.3 来源与打包 | validate_source_isolated_split迁data/source_split；build_source_isolated_trial迁data/packing | 原198/79与衍生400/143逐ID/行序/来源/人物/遮挡者闭包相等；拒绝fixture仍拒绝 |
| P2.4 框构造 | prepare_box_robustness.change_box等纯运算迁data/boxes；固定采样配方仍在实验配置 | center/scale/单侧裁剪逐项一致；400次父目标曝光序列逐项等于既有日志，无二次shuffle |
| P2.5 统一契约入口 | 标注adapter消费既有contract/supervision，旧策略显式适配；不重写旧标签 | 旧NPZ与新NPZ混batch、遮挡梯度、裁剪外排除、有效区全黑仍有效fixture通过 |

依赖P1配置/评价边界。风险是随机采样顺序、图像读取标志、dtype、相对路径解析和监督重算时机变化。回退单位为单adapter/Dataset路径；不对NPZ做原地迁移。portrait、GT采样规则如需修复，单独提交评审，不混在P2等价迁移中。

### P3：几何、相机、成像、渲染和标注

| 任务 | 涉及文件与迁移方式 | 产物/验收 |
| --- | --- | --- |
| P3.1 人体与服装共用函数 | prepare_cycling_characters / prepare_commuter_pilot / shell与binding、contract → body/clothing/io | 字段、shape、vertex/face索引、单位和矩阵明确；纯数组fixture精确等价；真实参数回读沿已有容差 |
| P3.2 world/ERP/rig | multiperson pilot的world_points/project_erp/erp_visibility，DS pilot的rig/point_in_view迁各职责模块 | 同世界点、两个相机轴、DS域、周期坐标、可见/遮挡mask与旧函数对比；不统一不同历史visibility算法 |
| P3.3 后处理 | process_scene拆buffer读取、几何标签、DS重映射、实例/框、序列化/预览；warp_case先建立全输出fixture再迁 | 复用选定旧EXR或已保存buffer，在新目录CPU后处理；RGB/ID/mask/框/标签/manifest数值对比，不需要重渲染全部场景 |
| P3.4 Blender边界 | 两renderer和character_materials/groom/scene_bridge迁隔离子包；CLI仍支持原`--`参数与路径 | 分别冻结native多人与linked旧单人路径，先检查实际body/clothing世界网格与R@V，再少量pass/RGB对照 |
| P3.5 场景与历史recipe | build/validate场景入口保持独立；有复用的数值函数迁库，来源/失败版本留原位 | 原场景资产不保存覆盖；用户从配方到制品仍能追踪；不建立场景插件系统 |

依赖P2数据契约和来源元数据稳定。Blender验证未具备时可以交付P3.1–P3.3中已有CPU证据的独立项，**不能宣称P3.4完成**。旧Phase4“需完整warp/Blender回归”门槛保留；当前DS mock测试补充证据但不替代真实渲染。

风险包括IndexOB语义（衣物独立ID与人物共享ID）、EXR深度含义、插值顺序、地面抬升、cam4重复翻转、Blender版本API差异。回退以一种renderer/一种camera路径为单位，旧脚本调用和输出不删除。

### P4：包安装、配置、命令入口及历史清理

1. 在P1前期就冻结旧命令清单，P4再改安装：保留`densekp`包名和根目录源码位置，用显式setuptools配置把两个包纳入同一distribution；不要为了src统一而移动checkpoint类。Blender采用显式src/root路径启动，不能要求其安装整套Torch。
2. 以实际metadata写普通数值、训练、可选后处理/Blender环境清单。先固定当前环境，不在重构中升级NumPy/OpenCV/Torch；原`requirements.upstream.txt`标为历史来源并保留。
3. 先承诺“仓库根目录直接脚本 + 开发editable安装”的运行方式；wheel任意cwd运行属于额外验收，不默认承诺。root与资产路径由显式参数/环境/配置解析，保存resolved值；旧绝对路径先兼容，不迁移assets/datasets/outputs。
4. 主要用户入口控制在数据准备/渲染、训练、推理、评价几条文档化命令。可加小console entry，但旧脚本参数、默认值和exit行为保留包装；实验目录仍按配方管理，避免一个无限subcommand总入口。
5. 每个删除候选查import、符号、subprocess、shell、config、docs、产物snapshot及外部接口。删除/归档前形成逐项签收表，见第7节；有唯一复现用途的一律保留。
6. 下载脚本保留独立且显式运行；`scripts/audit_workspace.sh`的全盘`find .`改为后续显式可选大文件扫描，默认只查源代码及命名资产。本轮没有运行它。

验收包括未安装但根目录运行、editable安装后临时cwd运行所承诺入口、Blender入口参数解析、无副作用import、旧CLI帮助及冻结argv构造、配置实际消费者清单。环境安装/Blender实测本轮未执行。回退不动已有环境，使用新临时venv或独立检查目录，不覆盖本地ignored配置。

### P5：文档与完整阅读路线

- 重写README的“当前状态”，从少数入口指向实际数据→模型→评价；READING_GUIDE/SAMPLE_WALKTHROUGH跟随新代码位置。
- 当前架构、字段/坐标表、运行环境、来源与输出版本索引各有明确职责；旧考古文档保持基线说明，不伪装成新实现记录。
- 每阶段同步其入口链接；P5完成整体阅读路线和旧命令对照，不等到最后才修断链。
- 阅读验收：仅从README沿七样本中的一个，能找到来源/人物/rig/label/实际crop/权重/原始预测/分母；能解释两个checkpoint文件与活动版本；能区分真实5760 ERP和合成1536 ERP。
- 无自动训练、无下载、无资产迁移。回退仅文档/导航；历史报告不覆盖。

## 6. 最小充分行为一致性基线

### 6.1 已实际回读的七个样本

选择依据只用覆盖条件与冻结索引顺序，未按模型误差挑样。路径、loader行、字段shape/dtype及依赖存在性在 [selected_baselines.json](../../outputs/plans/full_refactor_20261001/selected_baselines.json)。前三个来自来源隔离原权重baseline；后四个来自活动C权重挑战报告。

| 样本ID | 覆盖 | 监督点 / 遮挡监督 / in-view裁剪外 |
| --- | --- | ---: |
| `val_baseline_002` | 完整人体对照，138点有效；完整不等于无自遮挡 | 138 / 41 / 0 |
| `val_baseline_001` | 后排遮挡、仍监督 | 138 / 83 / 0 |
| `val_baseline_004` | 有效区边缘并有裁剪截断 | 14 / 5 / 26 |
| `challenge_000` | 边缘目标原参考框 | 112 / 71 / 0 |
| `challenge_001` | 同目标left中心移动 | 112 / 71 / 0 |
| `challenge_005` | 同目标deployment_tight尺度 | 112 / 71 / 0 |
| `challenge_007` | 同目标单侧收缩 | 89 / 65 / 23 |

这七个均为1536×768合成ERP，不能作为真实5760×2880精度证明。ERP接缝另外实际检查了 `x=1535↔1`、W=1536得到2px的微型fixture，以及既有periodic remap单测；没有把不跨缝的真实样本冒称跨缝覆盖。

已比较：实际中心网络输入、完整square RGB、affine、GT crop xy、visible、supervision、保存预测逆变换、逐点crop/ERP误差、监督分母和均值，均**精确一致**。GT正逆变换最大绝对误差 `1.244142185896635e-5 px`，在已有 `2e-4 px, rtol=0` 门槛内。未重新推理模型；重算的是已保存prediction_raw。

第一次回读脚本错误地把scene-local NPZ的image_root设为`/`，在读第一张图片时失败；改为既有scene根后通过。该核查脚本错误记录在audit_attempts.json，不是仓库Dataset故障；没有修改原样本或掩盖失败。

### 6.2 逐项比较要求与已有容差来源

| 比较对象 | 要求 | 本轮证据 / 后续缺口 |
| --- | --- | --- |
| 样本/来源/行序/整数计数 | 精确相等；空集合与失败状态同样比较 | 原198/79及衍生400/143来源验证实际通过；后续比完整采样序列与每人身份/所有遮挡者 |
| 实际Dataset输出 | image、affine、GT、visible、mask、bbox、dtype精确一致 | 七样本center input与square RGB已通过；完整归一化`[3,256,256]`独立before快照待补 |
| 增强与随机性 | 在同seed/同状态比较参数、输出及Python/NumPy/Torch RNG前后状态；随机调用次序精确 | 本轮已有增强边界测试；多worker与恢复轨迹未证明，后续先worker0再命名支持配置 |
| crop逆变换 | `atol=2e-4 px, rtol=0`对原GT；旧/新相同计算结果要求精确 | 已有生成/训练验收代码与历史报告门槛；本轮最大1.244e-5，不放宽 |
| DS投影/逆投影与ERP | 同函数移位要精确；已有DS单位射线误差1e-8、测试像素1e-9等按原断言 | DS/Omni/周期单测通过；新增分辨率和cam4翻转fixture沿既有规则 |
| 已存透视/Omni几何 | 透视2D≤5e-5 px、camera xyz≤2e-7 m；Omni≤1.3e-4 px | `tests/test_numeric_regression.py`现有float32落盘门槛，本轮两个测试实际通过 |
| 真值138/人体/衣物/groom | 同算式/索引/数组搬迁精确；真实回读保留原门槛：body参数2e-5m、pelvis2e-6m、R@V1e-6m、世界网格5e-6m、世界标签2e-6m | 来自SOURCE_ISOLATED_TRIAL_DATA既有门槛；本轮没重新SMPL-X forward或Blender网格导出 |
| 固定权重原始预测 | 同硬件/库/FP32/TF32关闭/同batch次序，先追求exact；最大abs≤1e-5，rtol0是已有重载门槛 | 本轮只读已有138×3；实际before/after推理与strict加载待后续GPU门槛 |
| loss/梯度 | 两with_var分支、0权重NaN、空监督、第三值语义；纯移位应精确；旧代数公式测试保留原torch.testing.assert_close默认 | 56项含loss梯度测试通过；不要用预测1e-5门槛泛化为梯度容差 |
| optimizer/checkpoint | state_dict key/shape/dtype精确；优化器参数组**顺序**、真实step、LR/WD、scaler/回调state、RNG精确 | 静态构造与历史496 key报告核对；本轮未反序列化真实checkpoint，也未跑CPU训练恢复 |
| 指标 | 每点error、mask、count、case/group顺序、NaN/empty/failure规则精确；相同dtype/reduction顺序的均值精确 | 七样本重算通过；新增跨双视角/共同GT分母fixture；时间、耗时和新输出根是允许差异字段 |
| 旧命令/输出 | 参数、exit状态、报告字段与NPZ键、case命名/含义不变；图像固定解码库下逐像素 | 历史argv/schema清单待冻结；并非所有CLI已执行help |
| Blender RGB/Depth/IndexOB | mask/ID/框精确；固定buffer下后处理图像精确；真实渲染先同版本/seed/device收集重复性 | 本轮未跑。现无通用RGB容差依据，不写臆造阈值；若非确定性，先同设置重复渲染建立噪声范围再由规划侧评审，不能用宽容差吞行为变化 |

### 6.3 后续真实基线采集的具体步骤

**真实模型，仅在获得实现阶段授权并检查设备/进程后：**

1. 读本轮seven-sample清单与原/candidate的model_config、loader_config；原样冻结实际输入和环境，不重新选择样本。
2. 先 `nvidia-smi` 定向查询候选设备及compute进程，再看相关主机进程。GPU0历史故障不能视为恢复；任何计算前复查，禁止reset或结束别的任务。按核实的UUID隔离，记录物理与逻辑序号映射。
3. 依次检查原权重A、普通400步B、框增强C：每次只加载需要的一份；记录全部tensor key/shape/dtype、strict missing/unexpected=0。保留DenseKP现有ViTPose初始化路径；不用strict=False规避错误。
4. 使用同精度/批次顺序对七样本做FP32 prediction_raw及坐标回读；A的前三份、C的后四份有本轮已读参考；B补少量基线即可，不重做完整567份评价。RNG/模式前后也记录。
5. 将新before存独立目录，迁移后用同环境重放。无400步训练，无模型效果门槛替代数值一致性。

**训练生命周期：**先跑现有CPU TinyHead生命周期（2步→恢复4步），它只在后续授权的验证窗口执行；另加“不中断 vs 保存恢复”相同样本次序/RNG对照。真实模型只在P1触及优化/回调路径且CPU不能充分覆盖时做最少更新探针，由规划任务明确次数；不把旧生命周期报告当新实现已通过。

**Blender：**先从选定场景的rig/people/source_manifest/已有buffer冻结输入，在普通Python比较P3后处理。P3.4需在同一已有Blender版本，分别选native多人和linked单人各一个代表输入，保存body/clothing世界网格、相机、compositor设置、Depth/IndexOB/RGB。几何类更改先只导出数据；只有渲染装配变化才执行有限渲染。输出到新目录，不开保存覆盖源blend，不运行场景builder或下载器。

### 6.4 本轮已经通过与明确未做

- 主代码239个Python及资产旁1个Python AST通过；不是导入全部模块。
- 56项CPU测试通过，0失败、0错误、0跳过，耗时约3.92s；CUDA_VISIBLE_DEVICES为空。列表和结果见 [cpu_tests.json](../../outputs/plans/full_refactor_20261001/cpu_tests.json)，原始日志见 [cpu_tests.log](../../outputs/plans/full_refactor_20261001/cpu_tests.log)。timm旧导入有FutureWarning，不影响结果。
- 两套来源清单实际复核通过：parent 198/79、6/3人、48布局；box 400/143、6/3人、36个被引用布局。检查每图所有人物/遮挡者和来源风险组；不能认证未知纹理供体的绝对独立性。
- regressor实际shape/稀疏/行和检查通过；七样本和周期fixture实际通过。
- 未运行`LifecycleTests.test_loader_and_training_entrypoint_save_and_resume`（会训练小网络），未运行Blender专用测试；未运行真实模型、checkpoint反序列化、pip安装、渲染或GPU探针。
- 未运行全量`audit_workspace.sh`，因为末尾`find .`递归扫全仓；以限定源文件AST和单独regressor检查替代。

## 7. 兼容、归档与删除策略

### 7.1 必须兼容的接口

- `densekp.core.densekp_trainer.DenseKP`、backbone/head module层级、state_dict所有key/shape；完整checkpoint中的hparams、优化器和callback state不能仅靠tensor加载通过就视为恢复兼容。
- `ValidationCheckpoint`当前state_key及`last_validation_save_step`字段、loader0排名规则；完整恢复与weights-only初始化分别验证。
- 现有tools直接执行路径、import公共符号及Blender subprocess脚本路径；先薄包装，不在一次迁移中删除入口。
- NPZ的imgname/center/scale/proj_verts/in_view/visible/supervision_policy，以及camera/person/source等sidecar字段；旧版本缺失visibility的含义不补猜值。
- 图片、几何、manifest、checkpoint、报告的路径引用，包含绝对路径和旧失败版本。路径迁移如以后实施，应新建明确映射，不原地改历史记录。
- 138×3原始预测及第三值、完整crop坐标、ROI/ERP外层转换、JSON空值/分母/组名。外部pano-stitch只通过这些契约核对，本轮不修改它。

### 7.2 当前没有批准直接删除的业务文件

静态无import不等于无用CLI；历史失败配方也可能是唯一复现证据。以下是**候选与门槛**，不是已授权的删除清单：

| 候选 | 当前调用核查 | 推荐策略 / 删除前必须补的证据 |
| --- | --- | --- |
| `datasets/utils.py`的resize_image、generate_image_patch_skimage、get_example | 在densekp/src/tools/tests中搜索调用模式，仅见定义；活跃使用get_example_projverts | P4进一步查符号导入/动态调用/外部兼容与历史快照；优先移到legacy模块并留导出。不能连同共享增强helper一起删 |
| `core/utils/train_utils.py`的load_valid、convert_to_full_img_cam | 当前调用模式只见定义；denormalize_images仍被trainer使用 | 函数级处理，不能删整文件；旧weights加载不改用宽松load_valid |
| `densekp/core/losses.py`及非DenseKP常量 | 当前活跃trainer不用旧loss类，许多SMPL/HMR常量只属继承代码 | 先核pickle/hparams/外部import；默认暂存legacy兼容，不凭未调用删checkpoint可能引用的类 |
| `configs/datasets.example.json`及旧HMR配置 | 没有活动消费者证据；部分是用户说明入口或本地ignored文件 | 明确标为示例/legacy，补真实配置后再决定归档；六份ignored文件不删除 |
| 旧衣物修复/眼睛/BVH实验 | 多个有import或唯一产物生成用途 | 先迁被复用函数，剩余保留复现配方和输入/输出/状态索引；不统一删除失败路径 |
| riverfront/build/场景旁preview_render.py | 仍是对应资产可追溯制作或预览recipe | KEEP_PROVENANCE；可加导航，不移动或删资产/备份 |
| 两个ERP renderer / warp_case | 当前有主线、旧CLI或历史复现依赖，完整等价尚无证据 | 兼容包装必须保留到两种行为都验收；不列直接删除 |
| outputs/archive、checkpoint、blend1、纹理包/分片 | 存在性/来源不等于冗余证明 | 全部保留；本任务无删除或移动授权 |

后续若允许删除代码，每项交付`原位置→替代位置→代码调用→CLI/subprocess→文档→产物/外部依赖→验证结果→回退方式`记录。首选函数级整理；不以减少文件/行数作为验收指标。

## 8. 需要规划侧决定的少数取舍

1. **支持的运行方式**：建议第一轮承诺根目录脚本+editable开发，显式支持两个包；暂不承诺独立wheel携带全部资产或任意cwd运行所有历史recipe。如要求wheel，P4需增加资源定位与打包验收，不能仅改pyproject。
2. **历史代码清理深度**：建议先保留旧入口薄包装和独特实验recipe，隔离确认无调用的继承函数；是否在本次整体重构末尾删除已证明无用函数，由逐项签收表决定。历史输出与失败版本一直保留。
3. **行为修复是否跟进**：建议B01报告字段修复、R01方向元数据、R02mask取整差异独立任务；纯重构保留现有不同策略。不统一默认loss、bbox扩张或渲染链。

这些选择不阻塞P1.1–P1.2的离线实现。真正约束后续阶段的是缺少本轮真实strict-load/预测与Blender完整对照，以及GPU0健康未核实；不能因CPU测试通过而跳过对应门槛。当前没有发现阻止全部重构启动的代码/资产缺失。

## 9. 给规划对话的简短回报

最值得先改的三项：**评价/推理与训练审计脱离历史实验脚本；Dataset配置与数据源显式化；DS/人物几何公共逻辑脱离pilot和角色配方。**

整体采用“两包保留、单向依赖、旧入口包装、逐职责迁移”。首个实现交付建议只改occlusion_metrics、candidate汇总、audit_cfg及其调用方，新增evaluation/configuration公共模块，得到可用保存预测和实际Dataset精确验收的公共入口；随后完成P1推理与回调迁移，再推进P2–P5。

本轮已完成静态全目录核查、关键链人工阅读、环境/ignored资产核查、56项CPU测试、七样本回读及来源隔离检查；无训练、渲染、真实模型加载、下载、SHA256、commit或旧文件迁移。**在此停止，交规划侧评审。**
