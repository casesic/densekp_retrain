# DenseKP 进一步精简：只读核查与删除规划

执行日期：2026-10-02。**本文件是建议清单，不是删除授权或实施记录。** 本轮仅新增本文件和 [inventory.json](../../outputs/plans/pruning_20261002/inventory.json)，不修改运行代码、配置、测试、既有文档或产物。所有删除与迁移均待评审；不建立新的 archive 目录。


> B1 评审修订：本轮仅批准 B1 清单修正与实施前核查，不迁移、不删除、不运行测试；B2–B5 均须逐批评审，193 项不能批量执行。此前依赖扫描把同名字符串与标准库 json 误连到仓内模块，已修正 inventory，并人工核查 B1 删除候选的消费者。116 个保留入口只是清单数量，不是简洁性验收；后续须区分日常入口、按需资产工具、诊断入口，再决定透视/压力/BEDLAM整条能力是否退役。

> 最新状态：用户随后授权B1实施，指定10文件现已删除并完成验收，详见[B1实施报告](PRUNING_B1_20261002.md)。以下初版/准备记录保留当时状态；B2–B5仍须另行评审。

## 1. 基线、证据范围和结论

实际 HEAD 为 `e7905904fe84b7a1b107485fc681a62ced6fccda`，master，提交标题为
`Complete dual-package setup, legacy cleanup, and reading guides`。开始时工作区仅有用户的 `docs/TODO.md` 修改，完整保留。
未找到适用 AGENTS.md。此前报告里的“未提交”是历史状态，不能据此重复 P4 安装或忽略规则修复。

已读 README、COMMANDS、READING_GUIDE、SAMPLE_WALKTHROUGH、LEGACY_RETENTION、P5 交付报告、配置和场景索引。
本次范围是 Git 跟踪的源码、配置、测试、小型 fixture、文档和脚本：AST import（含相对导入；短名须结合实际脚本目录和搜索路径核查，不按全仓唯一文件名推断）、函数/类、动态模块字符串、subprocess/命令路径、Hydra defaults/_target_、安装发现规则、文档引用。对资产只核查指定元数据/路径，不读纹理包或递归遍历 outputs。

特别只读核查了三份指定 checkpoint 的 ZIP 目录和 data.pkl，**未反序列化、未读 tensor storage、未加载模型**；以及各自已有严格加载记录、活动 status.json。外部 pano-stitch 仅查 Git 跟踪的相关源码引用及其 runner，未改文件或全局 Git 配置。

建议：保留两个公共包和当前端到端能力；撤销旧 API/完整对象 pickle 的默认兼容要求；先迁长期测试辅助与剩余真实调用，再删旧包装和失败支线，最后合并文档。不会统一 DS/Omni、旧标签权重/新监督、不同汇总策略，也不会重新评价效果。

### 证据如何解释

- inventory 每个文件包含职责、分类、原行数、调用方行号/导入名、依赖、动态调用、文档引用、迁移位置、停止支持能力、各类影响和验收条件。
- import 解析、命令字符串和文档引用是不同证据。literal 引用可能只是注释/快照，不能当成实际执行；动态外部调用也不能仅靠静态分析完全排除。
- 删除依据是“保留能力及明确退役边界 + 调用闭包处置 + 配置/序列化检查 + 测试迁移”，不是单次 rg 无命中。未知外部导入不再无限阻止已声明退役接口。
- 表中的“直接删除”指无需迁出业务实现；仍须同步撤销**仅验证废弃接口**的断言、更新链接/停止支持清单，并在该批检查无保留消费者。存在真正消费者的条目列为“迁移后删除”。
- 统计是建议基线；旧实现物理行数是整文件毛减少量，包含注释/空行，不是净减少量。迁入 helper、新增配置/入口及保留文件中的符号删减不在净值内。
- 本轮没有运行测试、训练、推理、渲染、安装、下载或 GPU 检查。这里的验收方法都是后续执行条件，既有验证证据明确标为复用。

## 2. 精简后必须保留的系统

不强制将清晰的小模块拼成大文件。建议目录：

```text
densekp/core/                     Dataset、具名裁剪、138 点网络、loss、加载、训练审计
src/densekp_retrain/
  body/ clothing/ camera/        人体、必要衣物/groom、世界与相机计算
  imaging/ rendering/            DS/Omni 后处理、Blender 装配、透视任务
  annotation/ data/              标注、来源、打包、框派生
  evaluation/ training/          应用评分、有界训练回调
  io/ runtime/ visualization/    文件/路径/展示
tools/                           保留的明确 CLI 与具体资产/实验配方
configs/hydra/                   DenseKP、当前 tiny 数据、trainer 与路径组合
configs/regression/suites.json   拟新增：显式回归输入、权重、分组及原 argv
tests/                          长期数值契约、小型 fixture 与 CPU 生命周期
tests/integration/               拟整理：需真实权重、CUDA 或 Blender 的显式选测
docs/                           少量当前主题文档；已跟踪历史通过 Git 查阅
```

### 保留能力及入口选择

| 能力 | 建议主入口/公共实现 | 输入、产物与保留理由 |
| --- | --- | --- |
| 公开图片骑行姿态 | prepare_cycling_images → reconstruct_cycling_hmr2 → convert_cycling_smplx | 图像/人物框及逐图来源许可 → 独立 canonical_pose；仍只有独立照片姿态，不能称为连续动作序列 |
| 人物与必要衣物 | prepare_cycling_characters、prepare_adult_candidates、prepare_commuter_pilot；body.parameters/geometry、clothing.shells/binding | 保留原角色 betas/来源、4 mm fitted/clipped-shell 与实际接受候选。commuter_pilot 中形状与模型准备不因失败衣物同名而删除 |
| 行人姿态/来源组合 | audit_pedestrian_poses、prepare_source_isolated_assets | 完整 165 维姿态切片、target pelvis、显式来源分组；parameters/transfer_groom/衣物保存先迁公共模块 |
| groom 与外观准备 | export_groom_reference、bind_commuter_groom、bind_adult_grooms、prune/relax_commuter_groom、prepare_procedural_eyes；必要 UV/来源审计 | 接受资产仍有这些制作/绑定依赖；只删已完成且不被当前能力使用的诊断拼图，不删除来源/绑定元数据 |
| 原动作与衣物模拟 | load_real_smplx_motion、generate_real_motion_geometry、prepare_p0_dressed_geometry | 保留主例及既有动作/衣物帧映射 80（切片）/181（源时间轴），不是把照片姿态套到该历史来源 |
| 场景与渲染 | build/upgrade_urban_street、build_cycling_forest、对应 validate；两条 render_equirectangular_*；render_adult_candidates_ds | 保留 native 多人、linked 单人各自灯光/ID；城市/森林配方与实际资产消费链完整。河岸独立构图制作建议退役 |
| 成像与标签 | run_ds_scene_pair_pilot、run_ds_occlusion_control、Omni 各生成/ERP adapter；imaging.ds/omni；build_omni_multiperson_tiny | 源 EXR/几何/标定/mask → 图像、有效域、138 点、可见性、框、NPZ；不统一源 ERP 距离与重映射鱼眼距离 |
| 来源→打包→框 | prepare_source_isolated_split、validate_source_isolated_split、run_source_isolated_batch/occlusion_pairs、freeze_source_isolated_candidates、build_source_isolated_trial、prepare_box_robustness | 人物/姿态/外观分组、同图全部人含遮挡者；198/79 行及400/143派生序/字段/RNG继续可解释 |
| Dataset/训练/推理 | DatasetKeypoints、train_keypoints、train_source_isolated_trial、core.inference | annotation_path/image_root、当前 tiny ID；默认 unselected；两种加载路线/初始化顺序/模型模式和回调语义保持 |
| 评价与回归 | evaluate_candidate_baseline、evaluate_ds_occlusion_control、evaluate_ds_box_control、compare_box_robustness/source_trial、summarize_* | 保留严重遮挡、同图框偏移、paired、完整控制与旧 riding/mixed 51 行；新统一调度入口只整合 argv/配置，不整合指标公式 |
| 日常与交付核查 | audit_workspace.sh、check_workspace、verify_installation_runtime | 默认源码/配置/必要 CPU 合约；无资产 editable/双包导入/配置打印/小数据读取。保持不扫描整个 outputs |
| 历史透视、原 BEDLAM 适配 | run_temporal_case/render_p0_blender/jobs、build_bedlam_tiny_frame_manifest/generate_bedlam_densekp/validate_bedlam_densekp_tiny | 本建议暂保留：已有推荐命令和唯一深度/投影/帧映射能力，不能把候选 DS 入口视作等价替代 |

完整保留入口、配置、公共模块和测试列表见 inventory 中“保留”条目；本文末尾另列所有保留命令文件。没有把“无 import 的制作 CLI”当成死代码。

### 删除前后仍须满足的数值与业务边界

- DenseKP 固定 138 点及原 regressor 语义；不假定 BEDLAM 原点索引一致。
- 预测 keypoint_value 不默认解释为置信度或训练/评价权重；历史标签权重及现行 loss 语义保留。
- 有可靠 GT 的遮挡点，在有效观测区且完整裁剪内继续监督与评价；裁剪外/有效域外按当前策略排除。不能根据 RGB 黑色判断有效域。
- 完整裁剪 256-square，网络取中间 192 列；不加额外 32 px 输出偏移。图像插值、标签变换、增强/RNG 顺序不变。
- 人体、衣物和标签共用几何变换；DS/Omni、原 ERP 射线距离与重映射深度、周期边界及可见性历史规则分别保持。
- 实际目标为 20260320-riding-dataset3，不混用 031/0311 标定；真实 2880×2880 鱼眼与 5760×2880 ERP 不与合成 1536×768 ERP 混淆。
- 同源人物、姿态与外观衍生版本按组隔离；检查场景中所有人及遮挡者。
- 不改变优化器更新、模型模式、checkpoint key/shape/dtype、指标分母、空集和非有限值处理；不重评效果，框增强权重仍未获部署替换认可。

### 相似入口的具体选择

1. 候选图片预测以 evaluate_candidate_baseline 为主；旧 evaluate_densekp_npz 和 evaluate_omni_densekp 退出用户接口。前者 weights>0 的 visible 报告命名有历史问题，本轮**停止该接口**而不是修改旧结果。Omni 投影/后处理/标签仍保留，早期专用 box/roll 评分不声称与 candidate 完全同义。
2. 严重遮挡与历史四框保留专用 DS evaluator，不能为减少文件而换成候选 evaluator 的另一组分母。只把 draw_case 等真正共用展示迁走。
3. evaluate_source_trial/evaluate_box_robustness 两个调度器迁到一份显式 suite 配置和一个薄入口；每种 checkpoint、冻结索引、reference_report、旧历史 case、执行顺序和 gate 都保留。默认展示任务，显式执行才计算。
4. 人物新组合以 source-isolated asset 入口为主；mixed_pedestrians 的固定 JOBS 不再制作，但既有 mixed 回归索引/预测仍保留。角色初始准备、款式配方不因此消失。
5. 原生多人/链接单人、DS/Omni/透视均有算法或装配差异，暂不因名字相似淘汰。长期职责已经归位的公共模块原位保留。

## 3. 重新确认的关键依赖和阻塞点

| 原位置 | 真实依赖/风险 | 删除前的具体动作 |
| --- | --- | --- |
| datasets/utils.py | evaluate_omni_densekp、visualize_annotation、validate_bedlam_densekp_tiny；dataset_crop_fixture/test_densekp_boundaries | Omni 旧评分接口退役；visualize 用 crop_example + CropResult；BEDLAM validator 坐标直引 crop_geometry；测试改具名字段，旧输入/GT/RNG期望不重录 |
| core/losses.py | 当前 trainer 直接 keypoint_objective；旧类仅 legacy 测试；三权重 metadata 无旧类 GLOBAL | 可停止六旧 loss 类及完整对象 pickle 支持；保留 keypoint_loss/旧标签数值语义与当前 loss 梯度测试 |
| core/utils/train_utils.py、legacy/train_utils.py | denormalize 的当前消费者已直引 image_normalization；旧模块只剩兼容测试 | 将 denormalize 独立期望留 tests，删除旧 prefix/load_valid/透视 helper 导入承诺 |
| default_hmr.yaml | cpu/gpu/ddp/mps 均在 defaults 中继承；含 precision、val_check_interval、max_steps 等 | 原内容迁 trainer/loop.yaml，四组仅替换父名并保持覆盖顺序；不能直接删，也不能顺手换 precision |
| prepare_source_isolated_assets.py | 导入 audit_pedestrian_poses.parameters、prepare_adult_candidates.clothing、prepare_mixed_pedestrians.transfer_groom | 分别迁 body.parameters、clothing.preparation、clothing.groom_transfer；STYLES 仍在具体配方显式传入 |
| source packing / imaging 测试 | 分别直接 import verify_source_packing 和 verify_imaging_pipeline | 迁 tests/source_packing_fixture.py 与 tests/artifact_comparison.py；包括 write/outcome/normalize 闭包，不复制整只 CLI 到 tests |
| tests/geometry_camera_fixture.py | TARGETS 字符串 + importlib 动态选择旧脚本；直接 import 图找不到 | 删除 api(original=True) 兼容模式；保留公共 api/synthetic/独立期望与几何/距离/拒绝检查 |
| tests/test_commuter_intersections_blender.py | 真实调用 repair_commuter_top 的 DQS、commuter_union_volume.scan 及 crossings | DQS/scan 最小 oracle 闭包迁 tests/blender_geometry_fixture；crossings 迁 rendering/blender/intersections；保留七项几何断言，不把整文件归为失败方案删除 |
| tests/test_inference/test_training_audit/test_trial_callbacks | 有 smoke 重导出与旧 callback pickle 身份测试，同时有有效当前数值/生命周期测试 | 只撤销旧导出/pickle 身份断言；保持严格加载、模式、梯度、优化器/RNG、state_key、步数与恢复测试 |
| constants.py、core/configs/__init__.py | 大量继承常量和数据 ID，部分仍活跃 | 文件保留，按明确符号/key 清理；NUM_JOINTS=44 是辅助关节，不能改成138；保留现行 tiny ID/显式路径/名称用途 |
| t_cond_mlp.py | pose_transformer 真实导入归一化/频率嵌入 | 保留，不按“旧模型辅助”整文件删除 |
| setup.py/pyproject | find_packages 自动包含两个包/legacy，fixture 与 Hydra 在 checkout 内 | 删除 legacy 后重新验证可交付列表，不仅测本机 import；新 test helper/配置必须非忽略，不碰依赖升级 |

公开 API 的停止范围与数据兼容是两回事：**删除 legacy Python 接口不等于废弃旧 NPZ 权重或其监督策略**。
旧 legacy_weight_targets、fractional label weights、portrait/closeup 名称分支、mask floor/rint 的现有规则继续保留。
参数/几何/坐标/loss 等行为修复另立任务。

### 三份权重与外部接口

三份文件为原始 assets_external/densekp.ckpt、普通400步 source_isolated_trial_400_gpu1_20261001/step_0400/checkpoint.ckpt、
框增强 box_robustness_400_20261001/training/step_0400/checkpoint.ckpt。

- data.pkl 分别为 274260、256965、256969 字节；都有 state_dict、optimizer_states、callbacks、hyper_parameters 等容器标记。
- GLOBAL 仅 Torch storage/rebuild、OrderedDict、OmegaConf/typing 等；无 DenseKP/旧 loss/legacy 完整对象类引用，没有 STACK_GLOBAL。不能把 payload 中 callbacks 字段的存在等同于序列化了 callback 类对象。
- 既有严格加载记录均为496个 state_dict key、无缺失/额外项；当前 stat 的 bytes/mtime_ns 与记录吻合。**不是内容哈希，也不是本轮重新严格加载**。pickle 中匹配 backbone/head 的字符串还含模块 metadata，不能作为 state_dict key 数量。
- 当前手工路线先 torch.load payload 再构造模型并 strict=True；Lightning 路线显式提供 cfg、strict=True。保留 DenseKP、ViTPose 初始化、网络 key/shape/dtype、OmegaConf 依赖和 CheckUpdates/FixedValidation/ValidationCheckpoint 当前 state 语义。
- 外部 pano-stitch runner 把自己的 src/CameraHMR 加到 sys.path，导入 core.*；此处不是本仓库 densekp.*。该已知引用不构成本仓库保留全部旧 utils 的理由；不修改外部 runner，也不声称审计了未知外部用户。

## 4. 长期测试与一次性工具

长期保留 tests 下所有现行算法契约。最终候选中的 test_legacy_cleanup.py 在 B1 仅拆出当前归一化/导入安全部分，并保留裁剪检查；整文件删除须等裁剪接口退役批次评审。其余测试逐方法处置，**不能按文件名含 legacy/pilot 删除**。

| 保留能力 | 现有测试/fixture | 调整 |
| --- | --- | --- |
| 读取/裁剪/增强 | dataset_crop_fixture、dataset_crop_v1、test_dataset_crop_contract、test_densekp_boundaries | 保留旧 ID/显式路径、完整输入、形状/dtype、GT/affine/mask、增强参数与三套 RNG；低层具名结果适配旧独立期望 |
| 来源/打包/框 | test_source_isolation、test_source_packing_contract、test_box_robustness、source_packing_v1 | helper 进 tests；同图非目标/遮挡者、缺字段、空监督、NaN拒绝、400父序继续验 |
| loss/训练/推理 | test_densekp_visibility_loss、test_occluded_supervision、test_inference、test_training_audit、test_trial_callbacks | 不删当前目标与模式/优化器/恢复；只删退役脚本 import 或整对象 pickle 身份检查 |
| 指标/展示 | test_evaluation_contract、test_candidate_baseline_metrics、test_candidate_reporting 与对应fixture | 两种非有限策略、按点/按人、分母、空组、像素/JSON异常保留；只删旧导出包装签名兼容断言 |
| 几何/衣物/相机 | body/cycling/character/adult/commuter tests、geometry_camera、double_sphere/omni/multiperson | 公共函数直引；保持固定138语义、共同变换、距离/轴与ERP周期/显式mask |
| 成像/装配边界 | test_imaging_pipeline、test_ds_scene_pair、test_scene_recipes、test_commuter_intersections_blender | compare helper归tests；保留DS/Omni深度来源区别、拒绝时部分输出及Blender专属七项几何断言 |
| 保存数值回归 | test_numeric_regression | 缺外部产物继续明确 skip，不伪造 pass；透视/Omni冻结产物不因为旧制作入口减少而删 |
| 安装/日常检查 | verify_installation_runtime、check_workspace、audit_workspace.sh | 真正长期能力，名字含 verify 也保留；无资产交付必须可运行，不只做 --help |

一次性 capture/replay 工具逐文件列在附表。verify_source_packing/verify_imaging_pipeline 先迁 helper；其余已有 pure fixture 的 capture CLI可在证据索引归并后删。
verify_refactor_inference、verify_fixed_validation、verify_densekp_model_lifecycle、三个 Blender 对照工具迁成 tests/integration 中明确选择的验证驱动，去掉只为本次迁移存在的旧 AST/capture 分支。不把这几条真实路径的验收能力完全丢掉。

P4 已验证过无资产 editable/CPU 交付，本轮仅复用其证据；**实施删除后必须重验**。GPU/Blender 验收按实际改动安排，不为删除文档重渲染，也不以 CPU mock 代 CUDA RNG/autocast 检查。任何 GPU 计算前检查设备和进程，不假定 GPU 0 已恢复。

## 5. 文档归并与历史保留

保留 README、COMMANDS、READING_GUIDE、SAMPLE_WALKTHROUGH、配置索引、INSTALLATION/依赖版本、SCENE_RECIPE_INDEX、fixture说明与用户TODO。
新增少量稳定主题页（以下均为**后续计划，本轮不创建**）：

| 当前落点 | 必须从哪里归并 | 必须保留的信息 |
| --- | --- | --- |
| docs/CONTRACTS.md | data/camera/旧架构、OCCLUDED_SUPERVISION、DS/Omni与深度报告 | 138 regressor不等于BEDLAM索引；坐标轴/单位/矩阵方向；原ERP射线距离与重映射距离；256/192不加32；显式mask；in_view/visible/监督；来源/同图防泄漏；旧标签权重与预测第三值的不同含义 |
| docs/DATA_AND_MODELS.md | SOURCE_ISOLATED_TRIAL_DATA、普通/框400步、mixed/riding/遮挡评价及活动status | 三权重/数据/索引/报告的精确路径与active版本；198/79/400/143/51含义；source partition/人物与姿态衍生；框增强未获部署替换认可；历史邻人修订和失败版本；已知报告字段区别 |
| docs/ASSETS.md | appearance、cycling motion、scenes、BEDLAM来源/帧映射、P5缺口 | 模型/角色/图像/衣物/groom来源版本；SMPL-X/BEDLAM约束、Meshcapade来源许可文本、COCO逐图NC/ND/SA等差异、场景CC0原记录；不重新作法律授权结论；canonical旧引用缺失、遮罩版本和外观文件未完全冻结 |
| docs/VALIDATION.md | P1–P5/更早refactor、fixture README、各几何/渲染/训练验收 | 独立before的生成顺序、容差依据、状态/RNG/异常、实际验证环境与缺项；选定制品和输入集合；无GPU/Blender替代声明；历史工具的 Git 取回方式，不要求日常读全部报告 |
| docs/KNOWN_BEHAVIOR.md | BUG_AUDIT、POTENTIAL_ISSUES及最新阶段修订 | portrait/closeup、floor/rint、NaN拒绝JSON、无效YAML字段、历史visible命名、数据修订差异；逐条标现存/已解决/退役接口，不能照抄旧bug列表 |
| 现有 INSTALLATION / SCENE_RECIPE_INDEX | 环境/场景历史报告、requirements.upstream | 双包editable、不支持自带资产wheel、共享冲突未修复、Blender5.1.1 CPU代表路径范围；命令输入输出/运行边界、路径基准、必要资产完整消费链 |

所有历史报告的建议分类是“迁移后删除”。inventory 为每份报告保存了主题/来源/限制的行号摘录和具体目的文档；摘录不是“无遗漏”的自动证明。执行 B5 要逐条核对全文的独有结论，新文档完成并链接到实际产物后才删。历史全文可用 `git show e790590:<path>` 查阅，不新建 archive。

**Git 只保障已跟踪内容**：ignored outputs、权重、纹理、模型与场景不因此有备份。本计划不删除、移动或打包它们；既有资产目录下的 README/许可/manifest及全部跟踪的大资产也列为保护对象。
删除报告不等于删除 JSON/NPZ 原始证据。未来要清理 outputs 必须另列任务，不能夹在此轮代码精简中。

## 6. 批次及验收门槛

| 批次 | 实施范围/先后依赖 | 产物与删除条件 |
| --- | --- | --- |
| B1 测试脱钩与无业务包装 | 先迁 source packing / imaging helper，提取 legacy test中的有效部分；处理测试的旧身份导入；删纯材质/groom/scene/metrics包装、旧loss/train_utils；clothing wrapper换测试import | 全部当前CPU合约、导入安全、两种指标和source/成像fixture通过；没有tests反向依赖删掉的工具；未改模型构造无需重跑三权重推理 |
| B2 裁剪接口与配置 | visualize/BEDLAM validator/低层tests迁具名接口后删utils/legacy.crop；停旧Omni/NPZ评分CLI；迁default_hmr→loop；退役列明的5 data组/2 HMR实验与example；按符号清constants/registry/misc | 独立七样本+启用增强before，完整输入/GT/affine/mask/shape/dtype/RNG精确；4 trainer组合配置精确；旧标签行为保留；有效组/root与另一cwd配置读取不变 |
| B3 来源准备与回归组织 | parameters/衣物/groom/helpers迁出；source-isolated消费者直引；删除mixed固定配方；DQS/intersection测试先脱钩；两套回归调度迁1入口+显式suite | 身体/衣物/groom和来源链条选样精确；候选/遮挡/框/51旧样例的argv及保存预测分母不变。只在触及相关真实计算时做必要验证，不新400步 |
| B4 专用失败支线与捕获工具 | B1/B3后删除明确退役的BVH/失败衣物/一次性拼图与已替代smoke；保存有价值的DQS/scan/CUDA/Blender验证；旧sweep边界进入长期测试 | 保留入口闭包无删文件依赖；共享资产/失败版本原样；相交/groom/衣物实际修改才安排必要CPU几何/Blender数组检查，不为扩大覆盖渲染 |
| B5 文档与交付 | 归并独有信息、所有保留文档/配置/测试/安装引用更新；完成逐文件清单最终勾选 | 链接/符号有效；无资产editable交付及CPU流程；真实环境/资产限制准确；停止支持表可查；Git状态/忽略/fixture齐全；不自动提交 |

**推荐首批只做 B1**，不要把B2裁剪、B3人物准备或B4失败制作链混进第一次删除。
B1 源/成像 test helper 的期望必须来自既有独立fixture，不可先把两边转向同一新函数再宣称等价。
每批以删除文件和直接消费者为回退单元；Git用于恢复已跟踪源码，不用于恢复本地资产。发现依赖不符就暂停该条删除并更新清单，不放宽容差/更改算法来过验收。

## 7. 需评审的取舍

1. **建议接受明确停止旧导入、旧loss类及完整对象pickle兼容**；保留指定三份 state_dict checkpoint 的加载/恢复能力和当前 callback state_key。若存在必须继续开的其它完整对象pickle，应明确列出文件与用途，仅为该资产安排一次转换，不无限保留所有旧类。
2. **建议停止 BVH 骑行拟合支线、loose/polo/union/settling 失败衣物的再制作 CLI**。保留照片骑行/行人姿态、接受的简单衣物/groom链和全部既有产物；这不是衣物质量问题已解决。
3. **建议暂保留当前透视时间序列/相机压力与原始BEDLAM小型适配能力**。它们与DS候选流程不是等价实现；若产品范围明确只剩DS生成和已有资产消费，可另减这一整条能力及其专属配置/测试。当前计数按保留计算，不拿未知外部兼容作为保留理由。

以上是整份建议的评审点，不是本轮继续实施的条件；本轮到规划产物为止。

## 8. 逐文件清单与数量

JSON 包含全部被核查文本文件，不只列删除项；每行有原职责、调用证据、后续位置、停止支持能力、测试/配置/文档/安装/权重影响和验收方法。
下列统计与附表从同一分类清单生成，迁移条目仅在前置条件完成后计入实际删除。

| 计数口径 | 建议数量 |
| --- | ---: |
| 逐文件核查的已跟踪文本文件 | 494 |
| 拟删除文件（含迁移前置条件） | 193 |
| 其中 Python 文件 | 78 |
| Python 整文件物理行毛减少 | 8420 |
| 全部文本整文件物理行毛减少 | 27858 |
| 当前命令文件 | 180 |
| 保留现有命令文件 | 116 |
| 拟新增回归调度入口 | 1 |
| 精简后命令文件合计（不含 tests/integration） | 117 |

分类数量：保留 301；迁移后删除 174；直接删除 19。本建议没有悬而未列的自动删除项；第7节的替代选择如获采纳需重算清单。

### 8.1 拟删除文件：逐项原职责、前置条件与证据

每行 group 对应第9节完整迁移要求，JSON 内提供全部调用方行号及各影响字段。引用较多时本表只显示前三个，不能解释为只有这些引用。

| 文件 | 类别 / 批次 / group | 当前职责 | 调用/引用证据（节选） |
| --- | --- | --- | --- |
| `ARCHITECTURE.md` | 迁移后删除 / B5 / documents | DenseKP repository architecture | BUG_AUDIT.md:11；DATA_FLOW.md:16；docs/README.md:14 |
| `BUG_AUDIT.md` | 迁移后删除 / B5 / documents | DenseKP technical audit | ARCHITECTURE.md:11；DATA_FLOW.md:16；TRAINING_READINESS.md:132 |
| `DATA_FLOW.md` | 迁移后删除 / B5 / documents | DenseKP data and coordinate flow | ARCHITECTURE.md:11；BUG_AUDIT.md:11；TRAINING_READINESS.md:128 |
| `TRAINING_READINESS.md` | 迁移后删除 / B5 / documents | DenseKP training readiness | ARCHITECTURE.md:11；docs/rendering/ds_scene_pair_prediction_20260928.json:7 |
| `configs/datasets.example.json` | 直接删除 / B2 / direct | 早期 train/val JSON 路径示例，包含 image_root/annotation_npz；当前 Dataset 不读取此文件。 | BUG_AUDIT.md:200；configs/README.md:15；docs/CURRENT_CODE_MAP.md:34 |
| `configs/hydra/data/bedlam_v2_v1.yaml` | 直接删除 / B2 / direct | 上游 BEDLAM v2/v1 全量场景混合及 EMDB/3DPW/RICH 验证数据选择。 | docs/refactor/P4_INSTALLATION_20261002.md:52 |
| `configs/hydra/data/densekp_train.yaml` | 直接删除 / B2 / direct | 早期 AGORA/BEDLAM bbox44 全量混合与 EMDB 验证组。 | docs/refactor/P4_INSTALLATION_20261002.md:52；docs/refactor/REFACTOR_REPORT.md:10；docs/training/TRAINING_DATA_FORMAT.md:27 |
| `configs/hydra/data/eval.yaml` | 直接删除 / B2 / direct | 上游 EMDB/3DPW/SPEC/RICH/COCO 验证数据选择，训练选择为空。 | docs/refactor/P4_INSTALLATION_20261002.md:52 |
| `configs/hydra/data/eval_smplx.yaml` | 直接删除 / B2 / direct | 上游 EMDB/3DPW/RICH 验证数据选择，训练选择为空。 | docs/refactor/P4_INSTALLATION_20261002.md:52 |
| `configs/hydra/data/train.yaml` | 直接删除 / B2 / direct | 上游 AGORA/BEDLAM、Insta/COCO/MPII/AIC 混合训练组。 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `configs/hydra/experiment/bedlam2.yaml` | 直接删除 / B2 / direct | 上游 SMPL-X 参数与顶点目标训练配置；不是当前 138 点训练实验。 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `configs/hydra/experiment/camerahmr.yaml` | 直接删除 / B2 / direct | 上游 SMPL 参数与顶点目标训练配置；不是当前 138 点训练实验。 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `configs/hydra/trainer/default_hmr.yaml` | 迁移后删除 / B2 / hydra_parent | 四个当前 trainer 的共同循环配置：日志/验证间隔、精度、步数及验证批比例。 | Hydra configs/hydra/trainer/cpu.yaml；Hydra configs/hydra/trainer/ddp.yaml；Hydra configs/hydra/trainer/gpu.yaml |
| `densekp/core/datasets/utils.py` | 迁移后删除 / B2 / crop | Compatibility imports and tuple adapters; Dataset uses crop_pipeline directly. | tests/dataset_crop_fixture.py:268；tests/test_densekp_boundaries.py:20；tests/test_legacy_cleanup.py:20 |
| `densekp/core/losses.py` | 直接删除 / B1 / direct | Historical CameraHMR loss classes, kept at this path for pickle compatibility. | tests/test_legacy_cleanup.py:22；ARCHITECTURE.md:40；BUG_AUDIT.md:338 |
| `densekp/core/utils/train_utils.py` | 直接删除 / B1 / direct | Compatibility exports for historical training/camera helpers. | tests/test_legacy_cleanup.py:21；BUG_AUDIT.md:338；docs/READING_GUIDE.md:220 |
| `densekp/legacy/__init__.py` | 迁移后删除 / B2 / crop | Historical CameraHMR helpers, retained for old imports and reproduction. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `densekp/legacy/crop.py` | 迁移后删除 / B2 / crop | Historical SMPL/HMR crop variants; not the active 138-point Dataset pipeline. | densekp/core/datasets/utils.py:24；docs/READING_GUIDE.md:81；docs/refactor/LEGACY_RETENTION_20261002.md:10 |
| `densekp/legacy/train_utils.py` | 直接删除 / B1 / direct | Historical camera/checkpoint utilities; retain old quirks for reproduction. | densekp/core/utils/train_utils.py:4；BUG_AUDIT.md:338；docs/READING_GUIDE.md:220 |
| `docs/ARCHITECTURE.md` | 迁移后删除 / B5 / documents | Architecture（完整数据流，逐箭头标注实现） | BUG_AUDIT.md:11；DATA_FLOW.md:16；docs/README.md:14 |
| `docs/CURRENT_CODE_MAP.md` | 迁移后删除 / B5 / documents | 当前代码地图（CURRENT CODE MAP） | docs/ARCHITECTURE.md:4；docs/READING_GUIDE.md:4；docs/README.md:13 |
| `docs/POTENTIAL_ISSUES.md` | 迁移后删除 / B5 / documents | POTENTIAL ISSUES（当前状态与历史问题） | docs/CURRENT_CODE_MAP.md:83；docs/README.md:15；docs/refactor/REFACTOR_REPORT.md:54 |
| `docs/appearance/APPEARANCE_ASSET_AUDIT.md` | 迁移后删除 / B5 / documents | 外观资产审计 | docs/history/README_BEFORE_P4_CLEANUP_20261002.md:22；docs/rendering/SYNTHETIC_RENDER_P0_PLAN.md:5 |
| `docs/appearance/APPEARANCE_BASELINE.md` | 迁移后删除 / B5 / documents | 最小外观基线 | docs/README.md:65；docs/history/README_BEFORE_P4_CLEANUP_20261002.md:22；docs/rendering/SYNTHETIC_RENDER_P0_PLAN.md:5 |
| `docs/appearance/APPEARANCE_V1C_HAIR_SHOES.md` | 迁移后删除 / B5 / documents | Appearance V1-C — Hair + Shoes audit | docs/appearance/APPEARANCE_V1_PLAN.md:38 |
| `docs/appearance/APPEARANCE_V1D_HDRI.md` | 迁移后删除 / B5 / documents | Appearance V1-D — HDRI environment | docs/appearance/APPEARANCE_V1_PLAN.md:52 |
| `docs/appearance/APPEARANCE_V1_PLAN.md` | 迁移后删除 / B5 / documents | Appearance V1 计划 | docs/README.md:64；docs/history/README_BEFORE_P4_CLEANUP_20261002.md:32 |
| `docs/appearance/BODY_TEXTURE_INTEGRATION.md` | 迁移后删除 / B5 / documents | 最新执行状态（2026-09-06） | docs/README.md:66；docs/appearance/APPEARANCE_V1_PLAN.md:13；docs/history/README_BEFORE_P4_CLEANUP_20261002.md:34 |
| `docs/appearance/CHARACTER_LIBRARY_10PLUS1_PLAN_20260930.md` | 迁移后删除 / B5 / documents | 十名成人＋可选一名儿童：角色设计与公开资源接入方案 | docs/README.md:62；docs/data/DATASET_REQUIREMENTS_20260929.md:190 |
| `docs/appearance/CLOTHING_INTEGRATION.md` | 迁移后删除 / B5 / documents | 服装接入约束与方案 | docs/README.md:67；docs/history/README_BEFORE_P4_CLEANUP_20261002.md:24 |
| `docs/appearance/CLOTHING_TEXTURE_INTEGRATION.md` | 迁移后删除 / B5 / documents | BEDLAM 服装纹理接入 | docs/history/README_BEFORE_P4_CLEANUP_20261002.md:32 |
| `docs/appearance/COMMUTER_CLOTHING_REPAIR_20260930.md` | 迁移后删除 / B5 / documents | F02 / M02 通勤衣物修复与失败归因 | docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md:5；tools/experiments/finalize_adult_candidates.py:72 |
| `docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md` | 迁移后删除 / B5 / documents | F02 / M02 通勤角色试制记录 | docs/appearance/CHARACTER_LIBRARY_10PLUS1_PLAN_20260930.md:4；tools/experiments/finalize_adult_candidates.py:72 |
| `docs/appearance/CYCLING_MULTI_CHARACTER_FEASIBILITY_20260929.md` | 迁移后删除 / B5 / documents | 公开多角色资产可行性核查 | docs/README.md:63 |
| `docs/appearance/EYE_GEOMETRY_AUDIT.md` | 迁移后删除 / B5 / documents | Eye geometry audit | docs/README.md:68；docs/appearance/APPEARANCE_V1_PLAN.md:62 |
| `docs/appearance/TEN_ADULT_DENSEKP_CANDIDATES_20260930.md` | 迁移后删除 / B5 / documents | 十名成人：简化衣物与有限 DS 候选 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/camera/COORDINATE_SYSTEMS.md` | 迁移后删除 / B5 / documents | 坐标系文档（COORDINATE SYSTEMS） | docs/ARCHITECTURE.md:114；docs/README.md:40；docs/refactor/REFACTOR_REPORT.md:52 |
| `docs/camera/FISHEYE_CAMERA_P0.md` | 迁移后删除 / B5 / documents | FISHEYE CAMERA P0 — Omni-Radtan Validation Report (Phase 17) | docs/README.md:42 |
| `docs/camera/P0_COORDINATE_CONVENTION.md` | 迁移后删除 / B5 / documents | P0 坐标与透视投影约定 | docs/README.md:41；docs/camera/COORDINATE_SYSTEMS.md:4；docs/refactor/REFACTOR_REPORT.md:24 |
| `docs/camera/PERSPECTIVE_CAMERA_STRESS.md` | 迁移后删除 / B5 / documents | Phase 15A — Perspective Camera Stress P0 | docs/README.md:43 |
| `docs/data/ANNOTATION_AUDIT.md` | 迁移后删除 / B5 / documents | DenseKP 现有标注实证审计 | docs/README.md:33；docs/camera/COORDINATE_SYSTEMS.md:154；docs/data/DENSEKP_GT_PIPELINE.md:80 |
| `docs/data/DATASET_REQUIREMENTS_20260929.md` | 迁移后删除 / B5 / documents | DenseKP 重训数据集要求汇总 | README.md:62；datasets/README.md:3；docs/README.md:10 |
| `docs/data/DATA_FORMAT.md` | 迁移后删除 / B5 / documents | DenseKP 数据格式 | docs/README.md:34；docs/data/DENSEKP_GT_PIPELINE.md:4；docs/environment/MIGRATION.md:42 |
| `docs/data/DENSEKP_GT_PIPELINE.md` | 迁移后删除 / B5 / documents | DenseKP GT Pipeline（138 点从哪来、到哪去） | docs/README.md:35；docs/refactor/REFACTOR_REPORT.md:53 |
| `docs/data/bedlam/BEDLAM_OPERATOR_AUDIT.md` | 迁移后删除 / B5 / documents | BEDLAM 与 DenseKP Surface Operator 审计 | docs/data/ANNOTATION_AUDIT.md:17；docs/data/DENSEKP_GT_PIPELINE.md:4 |
| `docs/data/bedlam/BEDLAM_PLAN.md` | 迁移后删除 / B5 / documents | BEDLAM / BEDLAM2 可行性与转换方案 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/data/bedlam/BEDLAM_RERENDER_FEASIBILITY.md` | 迁移后删除 / B5 / documents | BEDLAM / BEDLAM2 定制重渲染可行性审计 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/data/bedlam/BEDLAM_TINY_DATA_AUDIT.md` | 迁移后删除 / B5 / documents | BEDLAM v1 Tiny 数据审计 | docs/data/bedlam/BEDLAM_TINY_FRAME_MAPPING.md:47 |
| `docs/data/bedlam/BEDLAM_TINY_FRAME_MAPPING.md` | 迁移后删除 / B5 / documents | BEDLAM Tiny 帧映射审计 | docs/POTENTIAL_ISSUES.md:25 |
| `docs/data/bedlam/BEDLAM_TINY_GENERATOR.md` | 迁移后删除 / B5 / documents | Tiny BEDLAM DenseKP 监督生成器 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/environment/HEADLESS_ENVIRONMENT.md` | 迁移后删除 / B5 / documents | Headless 渲染环境审计 | docs/README.md:88 |
| `docs/environment/HEADLESS_RENDERER_SELECTION.md` | 迁移后删除 / B5 / documents | Headless synthetic renderer 选型 | docs/README.md:89 |
| `docs/environment/MIGRATION.md` | 迁移后删除 / B5 / documents | 迁移至用户 yy | docs/README.md:90 |
| `docs/history/README_BEFORE_P4_CLEANUP_20261002.md` | 迁移后删除 / B5 / documents | 清理前 README 历史快照 | README.md:67 |
| `docs/motion/TEMPORAL_MULTI_SCENARIO_VALIDATION.md` | 迁移后删除 / B5 / documents | Phase 16 — Temporal Continuity + Multi-Scenario Validation | docs/README.md:74 |
| `docs/motion/cycling/BABEL_CYCLING_CANDIDATE_SEARCH.md` | 迁移后删除 / B5 / documents | Phase 12.1 — BABEL Cycling Candidate Search | docs/README.md:76 |
| `docs/motion/cycling/CYCLING_FIT_HANDOFF_20260929.md` | 迁移后删除 / B5 / documents | 骑行素材与 SMPL-X 拟合交接（2026-09-29） | docs/README.md:73；docs/rendering/DS_MASKED_BASELINE_20260929.md:81 |
| `docs/motion/cycling/CYCLING_IMAGE_RECONSTRUCTION_20260929.md` | 迁移后删除 / B5 / documents | 公开骑行图片 → SMPL 姿态试验（2026-09-29） | docs/README.md:72；docs/data/DATASET_REQUIREMENTS_20260929.md:209；docs/motion/cycling/CYCLING_FIT_HANDOFF_20260929.md:5 |
| `docs/motion/cycling/CYCLING_MOTION_AUDIT.md` | 迁移后删除 / B5 / documents | Phase 12 — Cycling Motion Asset Audit | docs/CURRENT_CODE_MAP.md:155；docs/README.md:75；docs/refactor/PHASE4_DELETE_CANDIDATES.md:12 |
| `docs/motion/cycling/CYCLING_SMPLX_PILOT_20260929.md` | 迁移后删除 / B5 / documents | 骑行图片 → SMPL-X → DS 合成 → 小规模训练（2026-09-29） | TRAINING_READINESS.md:19；docs/data/DATASET_REQUIREMENTS_20260929.md:210；docs/motion/cycling/CYCLING_IMAGE_RECONSTRUCTION_20260929.md:4 |
| `docs/motion/cycling/DEDICATED_CYCLING_MOCAP_AUDIT.md` | 迁移后删除 / B5 / documents | Phase 13 — Dedicated Cycling Mocap Audit | docs/README.md:77；docs/motion/cycling/CYCLING_FIT_HANDOFF_20260929.md:26 |
| `docs/refactor/COMMIT_DELIVERY_20261002.md` | 迁移后删除 / B5 / documents | P1–P3 阶段提交交付记录 | docs/refactor/IMPLEMENTATION_TASKS_20261001.md:7 |
| `docs/refactor/DENSEKP_CORRECTNESS_20260928.md` | 迁移后删除 / B5 / documents | DenseKP 正确性修复与边界重构（2026-09-28） | ARCHITECTURE.md:7；BUG_AUDIT.md:7；DATA_FLOW.md:12 |
| `docs/refactor/DENSEKP_REAL_MODEL_ACCEPTANCE_20260928.md` | 迁移后删除 / B5 / documents | 真实 DenseKP 模型验收（2026-09-28） | TRAINING_READINESS.md:119；docs/README.md:106；docs/refactor/DENSEKP_CORRECTNESS_20260928.md:95 |
| `docs/refactor/FULL_REFACTOR_PLAN_20261001.md` | 迁移后删除 / B5 / documents | DenseKP 全仓库重构核查与执行方案 | docs/refactor/IMPLEMENTATION_TASKS_20261001.md:5 |
| `docs/refactor/IMPLEMENTATION_TASKS_20261001.md` | 迁移后删除 / B5 / documents | 全仓库重构：执行任务与验收门槛 | README.md:66；docs/READING_GUIDE.md:11；docs/history/README_BEFORE_P4_CLEANUP_20261002.md:17 |
| `docs/refactor/LEGACY_RETENTION_20261002.md` | 迁移后删除 / B5 / documents | 兼容代码、专用配方与清理依据 | README.md:29；configs/README.md:19；docs/READING_GUIDE.md:223 |
| `docs/refactor/OUTPUTS_INVENTORY.md` | 迁移后删除 / B5 / documents | Outputs inventory | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/refactor/P1_METRICS_CONFIG_20261001.md` | 迁移后删除 / B5 / documents | P0.1 + P1.1 + P1.2 实施记录 | docs/refactor/FULL_REFACTOR_PLAN_20261001.md:6；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:59 |
| `docs/refactor/P1_REPORTING_CALLBACKS_20261001.md` | 迁移后删除 / B5 / documents | 第三批：评价流程、展示与有界验证回调 | docs/refactor/FULL_REFACTOR_PLAN_20261001.md:8；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:61 |
| `docs/refactor/P1_TRAINING_AUDIT_20261001.md` | 迁移后删除 / B5 / documents | 第二批：真实推理与训练审计迁移 | docs/refactor/FULL_REFACTOR_PLAN_20261001.md:7；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:60 |
| `docs/refactor/P2_DATASET_CROP_20261002.md` | 迁移后删除 / B5 / documents | P2.1/P2.2：显式数据源与裁剪流程 | docs/READING_GUIDE.md:82；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:10；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:62 |
| `docs/refactor/P2_SOURCES_BOXES_20261002.md` | 迁移后删除 / B5 / documents | P2.3–P2.5：来源、打包、框与监督适配 | docs/READING_GUIDE.md:135；docs/SAMPLE_WALKTHROUGH.md:242；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:11 |
| `docs/refactor/P3_BLENDER_20261002.md` | 迁移后删除 / B5 / documents | P3.4 Blender 对象、材质与渲染装配 | docs/READING_GUIDE.md:151；docs/SAMPLE_WALKTHROUGH.md:126；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:14 |
| `docs/refactor/P3_COMPLETION_CHECKLIST_20261002.md` | 迁移后删除 / B5 / documents | P3 几何与生成链完成检查表 | docs/history/README_BEFORE_P4_CLEANUP_20261002.md:15；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:68；docs/refactor/P3_SCENE_RECIPES_20261002.md:4 |
| `docs/refactor/P3_GEOMETRY_CAMERA_20261002.md` | 迁移后删除 / B5 / documents | P3.1/P3.2：人体、衣物与坐标相机计算 | docs/READING_GUIDE.md:149；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:12；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:64 |
| `docs/refactor/P3_IMAGING_20261002.md` | 迁移后删除 / B5 / documents | P3.3 成像后处理与标注流程 | docs/READING_GUIDE.md:150；docs/SAMPLE_WALKTHROUGH.md:154；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:13 |
| `docs/refactor/P3_SCENE_RECIPES_20261002.md` | 迁移后删除 / B5 / documents | P3.5 场景配方与执行入口 | docs/refactor/FULL_REFACTOR_PLAN_20261001.md:15；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:67；docs/refactor/P3_COMPLETION_CHECKLIST_20261002.md:50 |
| `docs/refactor/P4_CLEANUP_20261002.md` | 迁移后删除 / B5 / documents | P4 历史清理与 P5 导读整理 | docs/refactor/FULL_REFACTOR_PLAN_20261001.md:16；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:68 |
| `docs/refactor/P4_INSTALLATION_20261002.md` | 迁移后删除 / B5 / documents | P4：双包 editable 安装、路径与运行入口 | docs/environment/INSTALLATION.md:4；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:16；docs/refactor/IMPLEMENTATION_TASKS_20261001.md:7 |
| `docs/refactor/P5_READING_DELIVERY_20261002.md` | 迁移后删除 / B5 / documents | P5 最终阅读与交付核查 | README.md:5；configs/README.md:3；docs/READING_GUIDE.md:13 |
| `docs/refactor/PHASE3_DELETE_CANDIDATES.md` | 迁移后删除 / B5 / documents | Phase 3 delete candidates | docs/README.md:97 |
| `docs/refactor/PHASE3_DEPENDENCY_MAP.md` | 迁移后删除 / B5 / documents | Phase 3 dependency map | docs/README.md:96 |
| `docs/refactor/PHASE4_DEFERRED_CLEANUP.md` | 迁移后删除 / B5 / documents | Phase 4 deferred cleanup | docs/README.md:99 |
| `docs/refactor/PHASE4_DELETE_CANDIDATES.md` | 迁移后删除 / B5 / documents | Phase 4 deletion audit | docs/README.md:98 |
| `docs/refactor/REFACTOR_REPORT.md` | 迁移后删除 / B5 / documents | REFACTOR_REPORT（readability / 整理报告） | docs/POTENTIAL_ISSUES.md:9；docs/README.md:100 |
| `docs/refactor/densekp_crop_parity_20260928.json` | 迁移后删除 / B5 / documents | 早期独立裁剪样本逐行误差、基线位置和验收范围记录。 | docs/refactor/DENSEKP_CORRECTNESS_20260928.md:110 |
| `docs/refactor/densekp_real_acceptance_20260928.json` | 迁移后删除 / B5 / documents | 早期真实模型 fresh/resume 验收及当时资产标识记录。 | docs/refactor/DENSEKP_REAL_MODEL_ACCEPTANCE_20260928.md:47 |
| `docs/rendering/CYCLING_DRESSED_20260929.md` | 迁移后删除 / B5 / documents | 骑行合成的穿衣版本（2026-09-29） | docs/data/DATASET_REQUIREMENTS_20260929.md:211；docs/motion/cycling/CYCLING_SMPLX_PILOT_20260929.md:3 |
| `docs/rendering/CYCLING_FOUR_CHARACTERS_20260929.md` | 迁移后删除 / B5 / documents | 四角色 × 两骑行姿态接入验收 | docs/README.md:47；docs/appearance/CYCLING_MULTI_CHARACTER_FEASIBILITY_20260929.md:3；docs/data/DATASET_REQUIREMENTS_20260929.md:179 |
| `docs/rendering/DS_LENS_MASK_20260928.md` | 迁移后删除 / B5 / documents | DS 镜头成像区域遮罩 | docs/README.md:49；docs/SAMPLE_WALKTHROUGH.md:135；docs/data/DATASET_REQUIREMENTS_20260929.md:213 |
| `docs/rendering/DS_MASKED_BASELINE_20260929.md` | 迁移后删除 / B5 / documents | 3D 场景＋鱼眼遮罩：现有 DenseKP 基线 | docs/README.md:48；docs/rendering/DS_LENS_MASK_20260928.md:3；docs/training/OCCLUDED_SUPERVISION_20260929.md:4 |
| `docs/rendering/DS_OCCLUSION_CONTROL_20260928.md` | 迁移后删除 / B5 / documents | 固定目标遮挡对照与输入框敏感性 | TRAINING_READINESS.md:106；docs/README.md:52；docs/rendering/DS_SCENE_PAIR_PREDICTION_20260928.md:3 |
| `docs/rendering/DS_SCENE_PAIR_PILOT_20260928.md` | 迁移后删除 / B5 / documents | 双视角边缘截断人体：小批场景验证 | TRAINING_READINESS.md:96；docs/README.md:50；docs/rendering/DS_SCENE_PAIR_PREDICTION_20260928.md:9 |
| `docs/rendering/DS_SCENE_PAIR_PREDICTION_20260928.md` | 迁移后删除 / B5 / documents | 双视角场景：现有 DenseKP 预测诊断 | TRAINING_READINESS.md:99；docs/README.md:51；docs/rendering/DS_SCENE_PAIR_PILOT_20260928.md:68 |
| `docs/rendering/P0_DEPTH_VISIBILITY.md` | 迁移后删除 / B5 / documents | P0 depth 与 visibility | docs/POTENTIAL_ISSUES.md:55；docs/README.md:56；docs/camera/COORDINATE_SYSTEMS.md:4 |
| `docs/rendering/P0_DRESSED_DATASET.md` | 迁移后删除 / B5 / documents | P0 穿衣合成数据集状态 | docs/history/README_BEFORE_P4_CLEANUP_20261002.md:24 |
| `docs/rendering/P0_DRESSED_RENDER_INTEGRATION.md` | 迁移后删除 / B5 / documents | P0 穿衣渲染接入 | docs/README.md:58；docs/history/README_BEFORE_P4_CLEANUP_20261002.md:28 |
| `docs/rendering/P0_PERSPECTIVE_CLOSED_LOOP.md` | 迁移后删除 / B5 / documents | Perspective P0 closed loop | docs/CURRENT_CODE_MAP.md:237；docs/README.md:55；docs/rendering/SYNTHETIC_RENDER_P0_PLAN.md:118 |
| `docs/rendering/P0_REAL_MOTION_REGRESSION.md` | 迁移后删除 / B5 / documents | P0 real SMPL-X motion regression | docs/README.md:57 |
| `docs/rendering/SYNTHETIC_RENDER_BASELINE.md` | 迁移后删除 / B5 / documents | synthetic-render-baseline-before-cycling | docs/README.md:53；docs/motion/cycling/CYCLING_MOTION_AUDIT.md:12 |
| `docs/rendering/SYNTHETIC_RENDER_P0_PLAN.md` | 迁移后删除 / B5 / documents | Synthetic render P0 计划 | docs/README.md:54 |
| `docs/rendering/ds_occlusion_box_control_20260928.json` | 迁移后删除 / B5 / documents | DS 遮挡框控制评价：固定人物/相机、指标、第三值语义与适用限制。 | docs/rendering/DS_OCCLUSION_CONTROL_20260928.md:94 |
| `docs/rendering/ds_occlusion_control_20260928.json` | 迁移后删除 / B5 / documents | DS 遮挡对照：权重、环境、种子、配对、mask、指标策略与邻人提示。 | docs/rendering/DS_OCCLUSION_CONTROL_20260928.md:94 |
| `docs/rendering/ds_scene_pair_prediction_20260928.json` | 迁移后删除 / B5 / documents | DS 场景配对预测：权重与环境、评价器版本、指标/配对/邻人策略。 | docs/rendering/DS_SCENE_PAIR_PREDICTION_20260928.md:34 |
| `docs/scenes/PHOTOREAL_OUTDOOR_ASSETS.md` | 迁移后删除 / B5 / documents | 写实户外场景选型 | docs/README.md:82 |
| `docs/scenes/URBAN_STREET.md` | 迁移后删除 / B5 / documents | Urban Street environment | docs/README.md:81 |
| `docs/training/BOX_ROBUSTNESS_TRIAL_400_20261001.md` | 迁移后删除 / B5 / documents | 框鲁棒性 400 步小试：挑战集改善，完整对照仍退化 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/training/DS_REAL_ERP_PILOT.md` | 迁移后删除 / B5 / documents | 真实 ERP 坐标系下的 Double Sphere 多人小样本 | docs/training/OMNI_FISHEYE_TO_ERP.md:7；docs/training/OMNI_MULTIPERSON_PILOT.md:38 |
| `docs/training/MIXED_RIDERS_PEDESTRIANS_BASELINE_20260930.md` | 迁移后删除 / B5 / documents | 骑行者与行人混合样例、原始 DenseKP 基线 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/training/OCCLUDED_SUPERVISION_20260929.md` | 迁移后删除 / B5 / documents | 遮挡点监督与评价（2026-09-29） | BUG_AUDIT.md:220；DATA_FLOW.md:6；TRAINING_READINESS.md:26 |
| `docs/training/OMNI_FISHEYE_TO_ERP.md` | 迁移后删除 / B5 / documents | 公司 Omni-Radtan 鱼眼 → 经纬图小样本 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/training/OMNI_MULTIPERSON_PILOT.md` | 迁移后删除 / B5 / documents | 经纬图 / 鱼眼多人小样本闭环 | docs/README.md:27 |
| `docs/training/SOURCE_ISOLATED_TRIAL_400_20261001.md` | 迁移后删除 / B5 / documents | 固定 400 步小试：流程通过，预设效果未通过 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/training/SOURCE_ISOLATED_TRIAL_DATA_20260930.md` | 迁移后删除 / B5 / documents | 来源与身份隔离的小试数据准备 | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `docs/training/TINY_REPRODUCTION.md` | 迁移后删除 / B5 / documents | DenseKP 小样本训练链路复现 | docs/README.md:26 |
| `docs/training/TRAINING_AUDIT.md` | 迁移后删除 / B5 / documents | DenseKP 训练审计 | docs/POTENTIAL_ISSUES.md:34；docs/README.md:28；docs/data/DENSEKP_GT_PIPELINE.md:4 |
| `docs/training/TRAINING_DATA_FORMAT.md` | 迁移后删除 / B5 / documents | 训练数据接口（TRAINING_DATA_FORMAT） | BUG_AUDIT.md:330；docs/POTENTIAL_ISSUES.md:26；docs/README.md:29 |
| `requirements.upstream.txt` | 迁移后删除 / B5 / documents | 上游完整环境参考依赖（含 VCS 依赖），不是本仓库离线 editable 安装承诺。 | ARCHITECTURE.md:50；TRAINING_READINESS.md:213；docs/CURRENT_CODE_MAP.md:53 |
| `tests/fixtures/legacy_cleanup_v1.json` | 迁移后删除 / B2-or-later-review / legacy_tests | 冻结测试期望（不是本机大资产） | tests/fixtures/README.md:38；tests/test_legacy_cleanup.py:25 |
| `tests/fixtures/legacy_import_bodies_v1.json` | 迁移后删除 / B2-or-later-review / legacy_tests | 冻结测试期望（不是本机大资产） | tests/fixtures/README.md:38；tests/test_legacy_cleanup.py:33 |
| `tests/legacy_cleanup_fixture.py` | 迁移后删除 / B2-or-later-review / legacy_tests | Small independent legacy behavior capture; no model or optimizer updates. | tests/test_legacy_cleanup.py:13 |
| `tests/test_legacy_cleanup.py` | 迁移后删除 / B2-or-later-review / legacy_tests | Independent historical behavior and import boundaries, without large assets. | tools/validate/check_workspace.py:23 |
| `tools/experiments/audit_cycling_character_assets.py` | 迁移后删除 / B4 / once_audits | Inventory existing BEDLAM archives; copy only bundled review images/text. | docs/appearance/CYCLING_MULTI_CHARACTER_FEASIBILITY_20260929.md:44 |
| `tools/experiments/audit_cyclist_bvh.py` | 迁移后删除 / B4 / bvh | Read BVH channels and perform forward kinematics for audit, not retargeting. | tools/experiments/prepare_cycling_fit_inputs.py:14；tools/experiments/review_cyclist_audit.py:13；docs/CURRENT_CODE_MAP.md:155 |
| `tools/experiments/audit_eye_geometry.py` | 迁移后删除 / B4 / once_audits | Audit native SMPL-X eye components and create frame-0 close-up diagnostics. | docs/CURRENT_CODE_MAP.md:151；docs/refactor/PHASE3_DELETE_CANDIDATES.md:14；docs/refactor/PHASE4_DELETE_CANDIDATES.md:13 |
| `tools/experiments/audit_skin_texture.py` | 迁移后删除 / B4 / once_audits | Record official texture provenance and component/eye UV compatibility evidence. | docs/CURRENT_CODE_MAP.md:151；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:174；docs/refactor/LEGACY_RETENTION_20261002.md:31 |
| `tools/experiments/build_commuter_clothing.py` | 迁移后删除 / B4 / failed_clothing | Rest-shaped separates with straight trouser legs, fitted tops and folded polo collar. | tests/geometry_camera_fixture.py:22；docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md:221 |
| `tools/experiments/build_commuter_polo_collar.py` | 迁移后删除 / B4 / failed_clothing | Curved five-row polo collar anchored to the actual trimmed shirt neckline. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/experiments/build_continuous_commuter_pattern.py` | 迁移后删除 / B4 / failed_clothing | Independent continuous torso/sleeve surface; no inherited anatomical folds. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/experiments/commuter_union_volume.py` | 迁移后删除 / B4 / failed_clothing | Blender oriented scanline volume for an explicit garment/body-region union. | tests/test_commuter_intersections_blender.py:15 |
| `tools/experiments/diagnose_commuter_clothing.py` | 迁移后删除 / B4 / failed_clothing | Read-only geometry audit of preserved commuter prototypes (run with Blender). | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/experiments/evaluate_box_robustness.py` | 迁移后删除 / B3 / regression_launcher | Run missing fixed A/B/C predictions, leaving previous predictions untouched. | docs/refactor/FULL_REFACTOR_PLAN_20261001.md:176 |
| `tools/experiments/evaluate_source_trial.py` | 迁移后删除 / B3 / regression_launcher | Run only the frozen final-checkpoint evaluation batches, sequentially on one UUID. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/experiments/extract_commuter_union.py` | 迁移后删除 / B4 / failed_clothing | Extract a 3 mm outward envelope; no anatomical or pose edits. | docs/refactor/LEGACY_RETENTION_20261002.md:34；tests/fixtures/legacy_import_bodies_v1.json:3 |
| `tools/experiments/find_cyclist_clip.py` | 迁移后删除 / B4 / bvh | Rank 10-second native-BVH windows by measurable cycling stability. | docs/CURRENT_CODE_MAP.md:155；docs/motion/cycling/DEDICATED_CYCLING_MOCAP_AUDIT.md:144；docs/refactor/PHASE3_DELETE_CANDIDATES.md:19 |
| `tools/experiments/inspect_bedlam2_groom.py` | 迁移后删除 / B4 / once_audits | Blender Alembic import audit; preserves raw curves without claiming a head binding. | docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md:223 |
| `tools/experiments/inspect_smplx_rigs.py` | 迁移后删除 / B4 / once_audits | Blender: compare standard and locked-head rotation conventions, not their betas. | docs/refactor/LEGACY_RETENTION_20261002.md:37；tests/fixtures/legacy_import_bodies_v1.json:4 |
| `tools/experiments/prepare_commuter_cloth_path.py` | 迁移后删除 / B4 / failed_clothing | Numerical rest-to-pose initialization for static cloth, not a reconstructed motion. | docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md:225 |
| `tools/experiments/prepare_cycling_fit_inputs.py` | 迁移后删除 / B4 / bvh | Package the two audited cycling clips for SMPL-X fitting, in native BVH units. | docs/motion/cycling/CYCLING_FIT_HANDOFF_20260929.md:47 |
| `tools/experiments/prepare_mixed_pedestrians.py` | 迁移后删除 / B3 / source_helpers | Transfer reviewed local walking poses onto frozen adult identities. | tools/experiments/prepare_source_isolated_assets.py:17 |
| `tools/experiments/prepare_source_trial_config.py` | 迁移后删除 / B4 / once_audits | Prepare a bounded next-run recipe and audit immutable regression references. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/experiments/repair_commuter_top.py` | 迁移后删除 / B4 / failed_clothing | Component experiment: smooth the rest pattern, then elastic collision fitting. | tests/test_commuter_intersections_blender.py:14；tools/experiments/build_commuter_polo_collar.py:11；tools/experiments/settle_commuter_top.py:16 |
| `tools/experiments/review_cyclist_audit.py` | 迁移后删除 / B4 / bvh | Save mapping design, joint hierarchy, and native kinematic review plots. | docs/CURRENT_CODE_MAP.md:155；docs/motion/cycling/DEDICATED_CYCLING_MOCAP_AUDIT.md:145；docs/refactor/LEGACY_RETENTION_20261002.md:33 |
| `tools/experiments/search_babel_cycling.py` | 迁移后删除 / B4 / bvh | Audit all BABEL v1 annotation sets, preserving ambiguity and provenance. | docs/CURRENT_CODE_MAP.md:155；docs/motion/cycling/BABEL_CYCLING_CANDIDATE_SEARCH.md:3；docs/refactor/PHASE3_DELETE_CANDIDATES.md:21 |
| `tools/experiments/settle_commuter_cloth.py` | 迁移后删除 / B4 / failed_clothing | Blender Cloth pilot with moving body collision and boundary pinning; preserves failures. | tools/experiments/settle_commuter_top.py:15；docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md:225 |
| `tools/experiments/settle_commuter_top.py` | 迁移后删除 / B4 / failed_clothing | Isolated, initially valid top on the preserved P2 initialization path. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/experiments/smooth_commuter_weights.py` | 迁移后删除 / B4 / failed_clothing | Remove nearest-body assignment discontinuities on a garment's own graph. | docs/refactor/LEGACY_RETENTION_20261002.md:35；tests/fixtures/legacy_import_bodies_v1.json:7 |
| `tools/experiments/train_cycling_smoke.py` | 迁移后删除 / B4 / duplicate_smoke | Bounded full-model updates on the cycling pilot, with fixed-box before/after. | tests/test_inference.py:104；tests/test_training_audit.py:61；tools/validate/verify_training_audit.py:21 |
| `tools/experiments/trim_commuter_neck.py` | 迁移后删除 / B4 / failed_clothing | Trim the excess neck tube after checking actual faired shoulder height. | docs/refactor/LEGACY_RETENTION_20261002.md:36；tests/fixtures/legacy_import_bodies_v1.json:8 |
| `tools/experiments/validate_commuter_union.py` | 迁移后删除 / B4 / failed_clothing | Open cap surfaces, bind to the parent garment, test the untouched body. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/fisheye/run_fisheye_seam_sweep.py` | 迁移后删除 / B4 / old_sweeps | Phase 17.1: seam yaw sweep stress for the dual-fisheye overlap scenario. | docs/ARCHITECTURE.md:98；docs/CURRENT_CODE_MAP.md:131；docs/refactor/PHASE3_DEPENDENCY_MAP.md:32 |
| `tools/fisheye/run_human_scale_sweep.py` | 迁移后删除 / B4 / old_sweeps | Phase 17.2: human scale sweep for the frozen Phase 17 fisheye pipeline. | docs/ARCHITECTURE.md:99；docs/CURRENT_CODE_MAP.md:131；docs/refactor/PHASE3_DEPENDENCY_MAP.md:33 |
| `tools/render/character_groom.py` | 直接删除 / B1 / direct | Historical import compatibility; active callers use the public module. | docs/refactor/P3_BLENDER_20261002.md:42；docs/rendering/SCENE_RECIPE_INDEX.md:52 |
| `tools/render/character_materials.py` | 直接删除 / B1 / direct | Historical import compatibility; active callers use the public module. | docs/appearance/CHARACTER_LIBRARY_10PLUS1_PLAN_20260930.md:97；docs/refactor/P3_BLENDER_20261002.md:42；docs/rendering/CYCLING_FOUR_CHARACTERS_20260929.md:131 |
| `tools/render/commuter_clothing_contract.py` | 迁移后删除 / B1 / clothing_wrapper | Historical import compatibility; active callers use the public module. | tests/test_commuter_clothing_contract.py:7；docs/refactor/P3_BLENDER_20261002.md:42；docs/rendering/SCENE_RECIPE_INDEX.md:52 |
| `tools/render/review_commuter_component.py` | 迁移后删除 / B4 / failed_clothing | Small fixed-light component reviews, without regenerating body or hair. | docs/rendering/SCENE_RECIPE_INDEX.md:51 |
| `tools/scene/blender_scene_bridge.py` | 直接删除 / B1 / direct | Historical import compatibility; active callers use the public module. | TRAINING_READINESS.md:84；docs/ARCHITECTURE.md:88；docs/CURRENT_CODE_MAP.md:127 |
| `tools/scene/build_reference_riverfront.py` | 迁移后删除 / B4 / scene_unused | Blender background script: editable riverfront inspired by assets/example/00001.jpg. | docs/CURRENT_CODE_MAP.md:134；docs/refactor/P3_SCENE_RECIPES_20261002.md:20；docs/refactor/PHASE3_DEPENDENCY_MAP.md:41 |
| `tools/validate/audit_box_trial.py` | 迁移后删除 / B4 / once_audits | Audit completed frozen box experiment and correct image-copy metadata without re-inference. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/validate/audit_commuter_component_contacts.py` | 迁移后删除 / B4 / failed_clothing | Check proposed collar against the corresponding revised shirt, independently. | docs/refactor/LEGACY_RETENTION_20261002.md:38；tests/fixtures/legacy_import_bodies_v1.json:9 |
| `tools/validate/audit_commuter_pose_sources.py` | 迁移后删除 / B4 / failed_clothing | Compare existing source/retarget meshes only; never regenerate a pose. | docs/refactor/LEGACY_RETENTION_20261002.md:39；tests/fixtures/legacy_import_bodies_v1.json:10 |
| `tools/validate/audit_commuter_repair_bodies.py` | 迁移后删除 / B4 / failed_clothing | Recheck frozen body parameters/138 points and compare garment evidence bodies. | docs/refactor/LEGACY_RETENTION_20261002.md:40；tests/fixtures/legacy_import_bodies_v1.json:11 |
| `tools/validate/audit_interrupted_trial.py` | 迁移后删除 / B4 / once_audits | CPU-only preservation audit after an interrupted trial; never resumes training. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/validate/commuter_intersections.py` | 迁移后删除 / B3 / intersection | Blender BVH broad phase plus triangle-edge narrow phase, independent of kNN. | tests/test_commuter_intersections_blender.py:13；tools/experiments/build_commuter_polo_collar.py:12；tools/experiments/diagnose_commuter_clothing.py:11 |
| `tools/validate/compare_blender_acceptance.py` | 迁移后删除 / B4 / integration | Compare actual Blender assembly, EXR pass pixels and downstream artifacts exactly. | docs/refactor/P3_BLENDER_20261002.md:99 |
| `tools/validate/evaluate_densekp_npz.py` | 迁移后删除 / B2 / retired_eval | Score a registered DenseKP dataset with visibility-aware per-person metrics. | ARCHITECTURE.md:47；DATA_FLOW.md:201；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:166 |
| `tools/validate/evaluate_omni_densekp.py` | 迁移后删除 / B2 / crop | Evaluate the original DenseKP checkpoint on paired P0 ERP/fisheye renders. | tests/geometry_camera_fixture.py:33；ARCHITECTURE.md:47；DATA_FLOW.md:201 |
| `tools/validate/occlusion_metrics.py` | 直接删除 / B1 / direct | Historical import compatibility; active callers use densekp.core.evaluation. | tests/test_evaluation_contract.py:79；tools/validate/verify_refactor_metrics_config.py:95；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:167 |
| `tools/validate/p0_pyrender_crosscheck.py` | 迁移后删除 / B4 / once_audits | 用 pyrender EGL 对 P0 的 0/8/15 帧做 silhouette 交叉检查。 | docs/ARCHITECTURE.md:183；docs/CURRENT_CODE_MAP.md:109；docs/refactor/REFACTOR_REPORT.md:197 |
| `tools/validate/probe_commuter_overlap.py` | 迁移后删除 / B4 / failed_clothing | Multi-point solid-angle audit of saved F02/P2 geometry, no pose changes. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/validate/pyrender_headless_smoke_test.py` | 迁移后删除 / B4 / once_audits | 在无 DISPLAY 环境下验证 pyrender EGL 的最小离屏渲染。 | docs/CURRENT_CODE_MAP.md:110；docs/environment/HEADLESS_ENVIRONMENT.md:51；docs/refactor/REFACTOR_REPORT.md:198 |
| `tools/validate/smoke_densekp_lightning.py` | 迁移后删除 / B4 / duplicate_smoke | Exercise the upstream DenseKP Lightning training loop on the tiny dataset. | docs/training/OMNI_MULTIPERSON_PILOT.md:29；docs/training/TINY_REPRODUCTION.md:36 |
| `tools/validate/smoke_densekp_training.py` | 迁移后删除 / B4 / duplicate_smoke | Run a small training check against the real DenseKP loader and loss. | docs/training/TINY_REPRODUCTION.md:18 |
| `tools/validate/validate_commuter_studio.py` | 迁移后删除 / B4 / failed_clothing | Actual DenseKP preprocessing of studio diagnostics; no inference or DS claim. | docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md:227 |
| `tools/validate/verify_blender_assembly.py` | 迁移后删除 / B4 / integration | Run inside Blender: observe actual assembly immediately before rendering. | docs/refactor/P3_BLENDER_20261002.md:98 |
| `tools/validate/verify_blender_postprocess.py` | 迁移后删除 / B4 / integration | Connect finite Blender acceptance renders to existing DS and Omni label paths. | docs/refactor/P3_BLENDER_20261002.md:100 |
| `tools/validate/verify_candidate_reporting.py` | 迁移后删除 / B4 / capture_only | Replay saved predictions through old/new candidate CLI; never load a model. | docs/refactor/P1_REPORTING_CALLBACKS_20261001.md:146 |
| `tools/validate/verify_dataset_crop.py` | 迁移后删除 / B4 / capture_only | Independent CPU Dataset/crop baseline; no model, rendering or GPU computation. | docs/refactor/P2_DATASET_CROP_20261002.md:18 |
| `tools/validate/verify_densekp_model_lifecycle.py` | 迁移后删除 / B4 / integration | Bounded real-model GPU acceptance, in separate fresh/resume processes. | docs/refactor/DENSEKP_REAL_MODEL_ACCEPTANCE_20260928.md:4 |
| `tools/validate/verify_fixed_validation.py` | 迁移后删除 / B4 / integration | Real small-GPU callback baseline and migration acceptance; no DenseKP updates. | docs/refactor/P1_REPORTING_CALLBACKS_20261001.md:148；tests/test_trial_callbacks.py:1 |
| `tools/validate/verify_geometry_camera.py` | 迁移后删除 / B4 / capture_only | Freeze/replay selected CPU SMPL-X, clothing and camera arrays. Never render. | docs/refactor/P3_GEOMETRY_CAMERA_20261002.md:115 |
| `tools/validate/verify_imaging_pipeline.py` | 迁移后删除 / B1 / imaging_fixture | CPU replay of existing EXR assets into a fresh directory, never a renderer. | tests/test_imaging_pipeline.py:17；docs/refactor/P3_IMAGING_20261002.md:99 |
| `tools/validate/verify_refactor_inference.py` | 迁移后删除 / B4 / integration | Independent original/current inference comparison on seven frozen inputs. | docs/refactor/P1_TRAINING_AUDIT_20261001.md:27 |
| `tools/validate/verify_refactor_metrics_config.py` | 迁移后删除 / B4 / capture_only | Freeze pre-migration CPU evidence, or compare it with the public modules. | docs/refactor/IMPLEMENTATION_TASKS_20261001.md:26；docs/refactor/P1_METRICS_CONFIG_20261001.md:59；tests/fixtures/README.md:12 |
| `tools/validate/verify_scene_recipe_boundaries.py` | 迁移后删除 / B4 / capture_only | Blender-only import/CLI boundary checks, without building or rendering scenes. | docs/refactor/P3_SCENE_RECIPES_20261002.md:21 |
| `tools/validate/verify_source_packing.py` | 迁移后删除 / B1 / source_fixture | CPU-only independent source/packing/box acceptance; writes fresh output only. | tests/test_source_packing_contract.py:27；docs/refactor/P2_SOURCES_BOXES_20261002.md:120 |
| `tools/validate/verify_training_audit.py` | 迁移后删除 / B4 / capture_only | Capture independent historical CPU audit behavior, or verify its migration. | docs/refactor/P1_TRAINING_AUDIT_20261001.md:106；tests/fixtures/README.md:25 |
| `tools/visualize/commuter_failure_sections.py` | 直接删除 / B4 / direct | Measured mesh sections and component-level attribution, without rendering. | 无仓内消费者命中；仍须按明确退役能力/配置决定，非单独删除证明 |
| `tools/visualize/make_appearance_comparison.py` | 直接删除 / B4 / direct | 生成 P0 裸模调试基线与最小外观基线的并排对照图。 | docs/CURRENT_CODE_MAP.md:151；docs/refactor/REFACTOR_REPORT.md:199 |
| `tools/visualize/make_eye_fix_comparison.py` | 直接删除 / B4 / direct | Create a frame-0 eye-material comparison using verified eye component bounds. | docs/CURRENT_CODE_MAP.md:151；docs/refactor/REFACTOR_REPORT.md:200 |
| `tools/visualize/make_skin_texture_review.py` | 直接删除 / B4 / direct | Create native-render comparisons using projected SMPL-X joint-weight regions. | docs/CURRENT_CODE_MAP.md:151；docs/refactor/FULL_REFACTOR_PLAN_20261001.md:174；docs/refactor/LEGACY_RETENTION_20261002.md:42 |
| `tools/visualize/review_eye_background_fix.py` | 迁移后删除 / B4 / once_audits | Build honest native-pixel crops and before/after comparison artifacts. | docs/CURRENT_CODE_MAP.md:151；docs/refactor/REFACTOR_REPORT.md:200 |

### 8.2 保留的现有命令文件

数值/辅助支持模块不因没有 __main__ 而被删除；所有保留包模块、配置与测试另见 JSON。这里计 tools/scripts 的 __main__ 文件、shell 和 train_keypoints；新增的 run_regression_suite 另计1。

| 文件 | 当前职责 |
| --- | --- |
| `densekp/train_keypoints.py` | Part of the code has been taken from  |
| `scripts/audit_workspace.sh` | !/usr/bin/env bash |
| `scripts/download_bedlam_body_texture.sh` | !/usr/bin/env bash |
| `scripts/download_bedlam_clothing_single_shard.sh` | !/usr/bin/env bash |
| `scripts/download_bedlam_clothing_texture_single_shard.sh` | !/usr/bin/env bash |
| `tools/annotation/build_bedlam_tiny_frame_manifest.py` | 按 BEDLAM v1 官方状态机为已解压的 tiny PNG 建立 person-level 映射。 |
| `tools/annotation/build_omni_multiperson_tiny.py` | Convert a paired multi-person ERP pilot into per-person DenseKP NPZ samples. |
| `tools/annotation/build_rendered_tiny.py` | Convert a completed single-person P0 render into a tiny DenseKP training set. |
| `tools/annotation/build_source_isolated_trial.py` | Pack accepted index rows into actual loaders without copying them to datasets/. |
| `tools/annotation/generate_bedlam_densekp.py` | 从少量 BEDLAM 几何记录生成 DenseKP 138 点监督，不负责重建 SMPL-X。 |
| `tools/annotation/inspect_annotation.py` | 只读检查 CameraHMR DenseKP annotation NPZ 的结构与数值。 |
| `tools/annotation/inspect_surface_operator.py` | Inspect/export the sparse definition of all 138 SMPL-X surface points. |
| `tools/annotation/visualize_annotation.py` | 可视化真实 DenseKP NPZ 的原图坐标与无增强裁剪坐标。 |
| `tools/experiments/audit_body_uv.py` | 审计 SMPL-X body mesh 与 UV OBJ 的 face-corner 拓扑兼容性。 |
| `tools/experiments/audit_clothing_uv.py` | 全量核对 BEDLAM clothing NPZ 与参考 OBJ 的几何及 corner UV 索引。 |
| `tools/experiments/audit_identity_texture_sources.py` | Reproducible texture comparisons; similarity is evidence, never an identity score. |
| `tools/experiments/audit_new_pedestrian_source.py` | Review a complete new SMPL-X motion on unchanged target identities. |
| `tools/experiments/audit_pedestrian_poses.py` | Inspect existing full SMPL-X motion frames without inventing action labels. |
| `tools/experiments/bind_adult_grooms.py` | Bind locally available groom variants to new adult candidates, preserving bodies. |
| `tools/experiments/bind_commuter_groom.py` | Transfer official LH groom roots onto verified standard SMPL-X scalp semantics. |
| `tools/experiments/convert_cycling_smplx.py` | Fit independent SMPL-X bodies to saved HMR2 meshes via surface correspondences. |
| `tools/experiments/export_groom_reference.py` | Read the official locked-head neutral mesh as a groom reference, never a label body. |
| `tools/experiments/finalize_adult_candidates.py` | Record reviewed candidates separately from diagnostic rows and superseded versions. |
| `tools/experiments/finalize_cycling_characters.py` | Summarize completed integration evidence and preserve intermediate versions. |
| `tools/experiments/freeze_cycling_characters.py` | Freeze candidate metadata after a recorded visual review and geometry check. |
| `tools/experiments/generate_cyclic_validation_motion.py` | Generate a deterministic cyclic gait articulation for validation only. |
| `tools/experiments/prepare_adult_candidates.py` | Versioned simple-clothing candidates. Never edits a source body or pose. |
| `tools/experiments/prepare_adult_comparison.py` | Common outfit, illumination and no-groom comparison inputs; never new identities. |
| `tools/experiments/prepare_box_robustness.py` | Freeze box-only derivatives of accepted images. No rendering or model inference. |
| `tools/experiments/prepare_commuter_comparison.py` | Same sports outfit and studio setup for old/new identity shape comparison only. |
| `tools/experiments/prepare_commuter_pilot.py` | Two preserved identities: full-space regression, bounded shape fit, four poses. |
| `tools/experiments/prepare_commuter_scalp.py` | A versioned shader-only scalp mask; original face/skin PNG remains unchanged. |
| `tools/experiments/prepare_cycling_characters.py` | Instantiate fixed local characters and independent cycling derivatives. |
| `tools/experiments/prepare_cycling_clothing.py` | Build fitted jersey/shorts shells for each independent cycling body. |
| `tools/experiments/prepare_cycling_images.py` | Select explicit V-COCO ride+bicycle pairs and download a small COCO image pilot. |
| `tools/experiments/prepare_native_ground_repairs.py` | Bounded world-placement repair using native terrain probes, never pose edits. |
| `tools/experiments/prepare_procedural_eyes.py` | Generate material coordinates from native SMPL-X shaped rest vertices. |
| `tools/experiments/prepare_source_isolated_assets.py` | Reuse unchanged assets or regenerate body, simple clothes and bound hair per pose. |
| `tools/experiments/prepare_source_isolated_split.py` | Freeze conservative identity/source groups before producing new combinations. |
| `tools/experiments/probe_cuda_device.py` | Small isolated device check; never loads DenseKP or resets a GPU. |
| `tools/experiments/prune_commuter_groom.py` | Physically remove a bounded number of invalid strands, consistently across poses. |
| `tools/experiments/reconstruct_cycling_hmr2.py` | Recover independent SMPL poses from supplied cyclist boxes using local 4D-Humans. |
| `tools/experiments/refine_cycling_character_clothing.py` | Clip garment boundaries within body triangles instead of selecting whole faces. |
| `tools/experiments/relax_commuter_groom.py` | Bounded control-point/body collision correction; originals and bindings remain intact. |
| `tools/experiments/render_adult_candidates_ds.py` | Small integration-only DS layouts; numerical pass never implies appearance pass. |
| `tools/experiments/render_cycling_characters_ds.py` | Two bounded layouts covering eight independent character/pose derivatives. |
| `tools/experiments/render_cycling_ds.py` | Render two independent cycling bodies in masked, paired DS scene pilots. |
| `tools/experiments/render_source_ground_repairs.py` | Render only bounded, already-probed placement repairs into new directories. |
| `tools/experiments/review_source_layouts.py` | Contact sheets at full 256-square scale; does not approve any candidate. |
| `tools/experiments/revise_adult_skin.py` | Preserve similar-face failures; revise skin within the same synthetic identity family. |
| `tools/experiments/revise_cycling_eye_materials.py` | Version the four-character eye material correction; preserve all geometry. |
| `tools/experiments/run_source_isolated_batch.py` | Bounded layouts from a frozen split; pilots must be reviewed before expansion. |
| `tools/experiments/run_source_occlusion_pairs.py` | Four preselected controls: remove only foreground person 1, retain all world transforms. |
| `tools/experiments/train_source_isolated_trial.py` | Next-turn bounded trial. Defaults to data/config checks; never implicitly trains. |
| `tools/fisheye/build_ds_multiperson_pilot.py` | Pass a rendered whole-sphere ERP through production DS fisheye -> ERP geometry. |
| `tools/fisheye/build_omni_erp_pilot.py` | Convert an actual rendered Omni-Radtan fisheye sample back to ERP. |
| `tools/fisheye/render_equirectangular_scene_source.py` | Blender helper: equirectangular source buffers for Omni-Radtan warping with a |
| `tools/fisheye/render_equirectangular_source.py` | Blender helper: render equirectangular source buffers for Omni-Radtan warping. |
| `tools/fisheye/run_ds_occlusion_control.py` | Three-level same-target occlusion experiment using the existing street pilot. |
| `tools/fisheye/run_ds_scene_pair_pilot.py` | Small same-world, opposite-facing camera pilot in the two selected scenes. |
| `tools/fisheye/run_fisheye_camera_p0.py` | Run independent Omni-Radtan center and dual-camera overlap validation. |
| `tools/fisheye/run_fisheye_multiperson_pilot.py` | Render a paired ERP/fisheye multi-person pilot with per-person 138-point GT. |
| `tools/fisheye/run_fisheye_scene_p0.py` | Phase 18: place the SMPL-X subject into a static Blender scene and render the |
| `tools/generate/generate_p0_toy_motion.py` | 生成确定性的 16 帧 SMPL-X toy motion 和唯一几何制品。 |
| `tools/generate/generate_real_motion_geometry.py` | 从 canonical real motion 生成 P0 单一几何制品。 |
| `tools/generate/load_real_smplx_motion.py` | 把真实 SMPL-X motion NPZ 适配为 renderer 的 canonical 参数。 |
| `tools/generate/prepare_p0_dressed_geometry.py` | 从匹配的 BEDLAM 6 fps clothing NPZ 提取衣物几何。 |
| `tools/render/finalize_camera_stress.py` | Assemble renderer-native camera stress annotations and validate all samples. |
| `tools/render/finalize_p0_outputs.py` | 把 Blender EXR Z 转 NPY，计算 visibility，并生成 overlay/contact sheet。 |
| `tools/render/finalize_temporal_case.py` | Build videos and temporal statistics from a finalized DenseKP render. |
| `tools/render/render_p0_blender.py` | Blender 后台单进程渲染 P0 RGB/Z/IndexOB，并验证投影。 |
| `tools/render/review_cycling_characters.py` | Blender: identical studio lighting, orthographic scale and clothing palette. |
| `tools/render/run_camera_stress.py` | Render all eight deterministic configurations through the existing renderer. |
| `tools/render/run_eye_background_fix.py` | Run the fixed appearance with the previous baseline geometry/camera settings. |
| `tools/render/run_temporal_case.py` | Render one temporal validation case with the frozen appearance settings. |
| `tools/render/solve_camera_stress_placement.py` | Deterministic geometry-based camera translation solver with fixed intrinsics. |
| `tools/scene/build_cycling_forest.py` | Adapt Poly Haven Pine Forest into a smooth asphalt cycling greenway. |
| `tools/scene/build_urban_street.py` | Build a deterministic environment-only urban street, in Blender or via Python. |
| `tools/scene/download_urban_assets.py` | Download the CC0 Poly Haven apartment kit, with integrity checks. |
| `tools/scene/prepare_hidden_alley.py` | Adapt the CC0 Poly Haven scene without modifying the original .blend. |
| `tools/scene/render_scene_fisheye.py` | Render an environment-only Blender scene through the frozen Omni-Radtan model. |
| `tools/scene/upgrade_urban_street_assets.py` | Replace Urban Street facades with the CC0 Poly Haven apartment kit. |
| `tools/scene/validate_cycling_forest.py` | Reopen audit: dependencies, road geometry and sampled 2.5m riding clearance. |
| `tools/scene/validate_urban_street.py` | Reopen Urban Street and check scale, assets, and sampled standing clearance. |
| `tools/validate/assemble_source_visual_acceptance.py` | Assemble explicit manual reviews; never auto-accept a numerically valid image. |
| `tools/validate/audit_real_box_path.py` | Read a small saved dataset3 detector example; no detection, remapping or fitting. |
| `tools/validate/audit_source_pose_transfer.py` | Record the retained pose and explicit translation replacement per derivative. |
| `tools/validate/blender_headless_smoke_test.py` | 用法：blender -b --python tools/validate/blender_headless_smoke_test.py |
| `tools/validate/check_workspace.py` | Bounded daily checks: source/config syntax and asset-free CPU contracts. |
| `tools/validate/compare_box_robustness.py` | Compare accuracy and cross-box drift on fixed common GT in original ERP units. |
| `tools/validate/compare_source_trial.py` | Compare one predeclared final checkpoint against frozen baselines, without selection. |
| `tools/validate/diagnose_adult_contacts.py` | Independent BVH clothing crossings and sampled hair/cloth segments; diagnostic only. |
| `tools/validate/evaluate_candidate_baseline.py` | Strict checkpoint inference on frozen rows, with optional frozen-input comparison. |
| `tools/validate/evaluate_ds_box_control.py` | Cross two saved heavy-occlusion images and two fixed box definitions. |
| `tools/validate/evaluate_ds_occlusion_control.py` | Evaluate fixed oracle crops across a rendered three-level occlusion control. |
| `tools/validate/evaluate_ds_scene_pair.py` | Run an existing checkpoint on the diagnostic paired scenes, without training. |
| `tools/validate/freeze_mixed_candidates.py` | Audit actual buffers and freeze visually reviewed rows before any inference. |
| `tools/validate/freeze_source_isolated_candidates.py` | Freeze only reviewed, numerically valid new rows; never use model errors. |
| `tools/validate/freeze_source_occlusion_pairs.py` | Use each clear instance box unchanged in both images; recompute actual crop supervision. |
| `tools/validate/smplx_densekp_render_smoke_test.py` | 用同一组 SMPL-X 顶点渲染并生成 DenseKP 138 点 overlay。 |
| `tools/validate/summarize_saved_candidate_predictions.py` | Finish aggregation of a fully saved batch after a report-only failure; no inference. |
| `tools/validate/summarize_source_occlusion_pairs.py` | Compare preselected occluder controls using identical GT, boxes and semantics. |
| `tools/validate/summarize_source_trial_data.py` | Summarize frozen rows and measured failures; do not select data from predictions. |
| `tools/validate/validate_adult_candidates.py` | Independent parameter, pelvis and fixed 138-row readback of adult candidates. |
| `tools/validate/validate_adult_groom_bindings.py` | Read stored groom bindings independently; labels remain defined on the body. |
| `tools/validate/validate_bedlam_densekp_tiny.py` | 通过 DatasetKeypoints 验证 tiny BEDLAM DenseKP NPZ、裁剪与可视化。 |
| `tools/validate/validate_commuter_groom.py` | Independent saved binding/collision readback; does not replace curve or clothing visual review. |
| `tools/validate/validate_commuter_pilot.py` | Read back all candidate parameters and bindings without accepting failed components. |
| `tools/validate/validate_cycling_characters.py` | Independent saved-parameter, identity, garment-boundary and DS checks. |
| `tools/validate/validate_ds_scene_pair_pilot.py` | Validate a rendered DS scene-pair pilot and exercise the actual DenseKP loader. |
| `tools/validate/validate_p0_textured_regression.py` | 验证纹理集成没有改变 P0 的几何、相机和监督产物。 |
| `tools/validate/validate_source_isolated_split.py` | Fail closed on identity, source, image, occluder or paired-box leakage. |
| `tools/validate/verify_installation_runtime.py` | Exercise an installed checkout without weights, existing data, or path injection. |
| `tools/visualize/adult_candidate_sheets.py` | Contact sheets link actual render files; no generated or retouched imagery. |
| `tools/visualize/make_environment_comparison.py` | Generate native-render before/after reviews for an environment-only change. |
| `tools/visualize/summarize_cycling_character_review.py` | Arrange actual review renders, without changing their content or color. |

### 8.3 保留文件内的接口/调用精简

这些文件不整文件删除；动作不得改变保留流程数值。

| 文件 | 具体动作 |
| --- | --- |
| `densekp/core/configs/__init__.py` | 保留 dataset_config、六类本地 tiny/boundary ID 及 bool train/val 解析；逐 key 停用只由退役五份 Hydra data 组消费的上游 HMR/BEDLAM/COCO/EMDB 注册项。default_config/_C 无当前调用，可与旧配置一起删除；显式路径与 dataset 名称的 portrait/closeup 分支不改。 |
| `densekp/core/constants.py` | 保留 VITPOSE_BACKBONE、NUM_JOINTS=44（辅助关节）、TRANSFORMER_DECODER、NUM_DENSEKP_SMPL=138 及其依赖；其余逐 AST 定义/引用闭包删除。不能把 NUM_JOINTS=44 改成 138；不启用 YAML 中当前无效网络字段。 |
| `densekp/core/utils/misc.py` | 只删无当前消费者的 instantiate_callbacks/instantiate_loggers/get_metric_value 及只供它们的 imports；保留 task_wrapper/extras/save_file/log_hyperparameters/close_loggers 的异常和 finally 顺序。 |
| `densekp/core/datasets/crop_pipeline.py` | 保留具名阶段；在最后旧 tuple 消费者迁走后删除 CropResult.legacy_tuple。图像/GT 原地修改、舍入、RNG、异常顺序均保留。 |
| `densekp/core/components/t_cond_mlp.py` | 保留。pose_transformer 实际导入 AdaptiveLayerNorm1D/FrequencyEmbedder/normalization_layer，不因看似 HMR 残余而整文件删除。 |
| `tools/annotation/visualize_annotation.py` | 从 get_example_projverts 改为 crop_example + CropBox，按名取 image/display/dense_points/affine；expand_to_aspect_ratio 等直引 image_processing，坐标直引 crop_geometry。不得改其图像读取/插值及 is_bgr 参数。 |
| `tools/validate/validate_bedlam_densekp_tiny.py` | trans_points2d_parallel 直引 densekp.core.crop_geometry；保留独特 debug annotation 回读检查。删除仅供外部旧导入的 audit_cfg 别名承诺，不移除本函数实际使用的配置。 |
| `tools/experiments/audit_pedestrian_poses.py` | parameters(pose,beta,pelvis) → body/parameters.py 的参数切片适配；mesh_panel 若仍由 freeze_mixed_candidates 使用则移 visualization/geometry.py；CLI 保留用于来源姿态人工审核。 |
| `tools/experiments/prepare_adult_candidates.py` | clothing 的稳定保存/绑定部分 → clothing/preparation.py，显式传 profile/materials；STYLES 与形状 targets 继续留成人配方，不移成全局统一款式策略。 |
| `tools/experiments/prepare_source_isolated_assets.py` | 改直引 body.parameters 的参数适配、clothing.preparation 和 clothing.groom_transfer；依赖清单包括原 report 中的角色/geometry/groom 资产，不把删除旧 CLI 当成删除这些资产。 |
| `tools/experiments/audit_identity_texture_sources.py` | components(families,edges) → data/source_split.py；保留当前来源纹理审计 CLI。相似性仅是审计证据，不升级为身份真值。 |
| `tests/test_source_packing_contract.py` | 改从 tests/source_packing_fixture 导入 source_boundaries/contract_cases/pure_boxes，保留异常/空监督/旧标签权重策略；删除最后只比 old_split/old_box/old_official 身份的兼容测试。 |
| `tests/test_imaging_pipeline.py` | compare 改从 tests/artifact_comparison 导入；保留完整制品/partial failure/深度/周期测试；删 test_historical_exports_are_compatibility_only。 |
| `tests/test_evaluation_contract.py` | 保留独立值、空集、非有限与配置dump；只删 test_historical_imports_remain_available。 |
| `tests/test_candidate_reporting.py` | 保留成对分母、图片像素、JSON失败/复制顺序；test_panel_pixels 中旧 render_panel 关键字包装的身份/兼容部分移除，当前 render_candidate_panel 数值断言保留。 |
| `tests/test_geometry_camera_contract.py` | 保留冻结数组/拒绝、人体衣物共变换、两相机/周期/距离；删 test_old_imports_are_compatibility_exports。 |
| `tests/geometry_camera_fixture.py` | TARGETS 改成公共模块→符号清单；删 api(original=True) 旧脚本 dispatch，api() 及 synthetic 原算法与顺序保留。 |
| `tests/dataset_crop_fixture.py` | run_dataset 的旧实现选择分支停止支持，但旧 ID 路线仍验；run_lowlevel 改读取 CropResult 命名字段并保持独立 fixture 旧编码，删除仅为 tuple 顺序的期望。不是把期望重录成新实现输出。 |
| `tests/test_dataset_crop_contract.py` | 保留启用增强时完整输入、GT、affine、mask、参数与 RNG；低层 tuple 测试改对具名结果执行相同计算/异常，不整项删除行为覆盖。 |
| `tests/test_densekp_boundaries.py` | do_augmentation 改 sample_augmentation；get_example_projverts 改 crop_example 具名字段；保留loss空集/非有限/梯度、训练恢复、坐标与flip拒绝测试。 |
| `tests/test_source_isolation.py` | components 与 validate_split 直引 data.source_split；全场景遮挡者、纹理/姿态来源组及失败断言保留。 |
| `tests/test_box_robustness.py` | change_box 直引 data.boxes；补保留 verify_source_packing 中400父序/RNG核对方法，不删除相关固定结果。 |
| `tests/test_character_clothing.py` | clipped_shell 直引 clothing.shells，原测试输入/浮点期望不改。 |
| `tests/test_adult_candidate_garments.py` | clipped_shell 直引 clothing.shells，成人衣物覆盖和对应检查保留。 |
| `tests/test_cycling_geometry.py` | world_points/shell 等直引 body.transforms/clothing.shells；仍可测试保留 DS 入口调度，但不通过它重导出公共函数。 |
| `tests/test_ds_scene_pair.py` | rig_poses/point_in_view/project_erp 直引 camera 对应模块；保留显式 mask 和真实后处理契约。 |
| `tests/test_multiperson_pilot.py` | world_points/remap_periodic_erp 直引公共模块；roll_person 改直引 data/erp_crop.py；三项几何/周期行为断言保留。 |
| `tests/test_commuter_clothing_contract.py` | validate_binding 直引 clothing.contracts；不因为 commuter 旧制作 CLI 退役删除有效绑定/审阅 gate。 |
| `tests/test_commuter_intersections_blender.py` | crossings 直引 rendering/blender/intersections；DQS/scan 的既有数值 oracle 移 tests/blender_geometry_fixture；保留所有当前七项几何断言，不将 bpy 测试充当普通CPU验收。 |
| `tests/test_trial_callbacks.py` | 保留 state_key='FixedValidation'/empty state/触发及恢复检查；撤销 tools 旧类导出及 protocol-0 完整对象 pickle 导入兼容断言。 |
| `tests/test_scene_recipes.py` | 保留 jobs argv/路径/无执行与保留制作入口 import 安全；删被明确退役的河岸 AST 快照项，不整文件删除。 |
| `tests/fixtures/scene_recipe_bodies_v1.json` | 只随已退役 recipe 删除对应键；保留森林/街景等现行脚本独立期望，不全量重新捕获。 |
| `tools/validate/check_workspace.py` | MODEL_TESTS 改用提取后的长期 import/normalization 测试；默认扫描范围仍为源码/config/必要fixture，不扫 assets/outputs。 |
| `tools/validate/evaluate_candidate_baseline.py` | 保留 CLI；删除无现行消费者的 write/summary/aggregate/render_panel 重导出，调用者/测试改直引公共实现。 |
| `tools/experiments/train_source_isolated_trial.py` | 保留当前有界训练与实际使用的公共类 imports；不再承诺外部从本脚本 import 回调。类名/state_key和当前checkpoint恢复语义保留。 |
| `tools/validate/evaluate_ds_box_control.py` | 保留四历史框回归；np/torch/Image/Dataset/配置/restore 改直引定义模块；draw_case 迁 visualization/keypoints.py 的 DS 专用绘制函数。不得换成 candidate 同名但不同规则的分母。 |
| `tools/validate/evaluate_ds_occlusion_control.py` | 保留严重遮挡固定框/重放入口；draw_case 迁公共展示后本入口直引，保留评分/可见性历史策略。 |
| `README.md` | 改为少数推荐能力入口；停止把全部 tools 当公共 API；链接当前契约/资产/模型与测试文档。 |
| `docs/READING_GUIDE.md` | 核心→配方→显式测试三层；清除退役包装/报告链接，读者不需要历史报告完成主流程。 |
| `docs/SAMPLE_WALKTHROUGH.md` | 主例保持heavy/cam2/person3与新版amodal标签；保留canonical旧路径缺失、源/衣物帧80/181、遮罩版本限制；历史报告链接替换为当前契约/资产/验证文档。 |
| `docs/COMMANDS.md` | 保留配置打印/数据/模型/渲染计算边界，替换退役命令为保留入口或明确停止支持；不伪装不同指标语义完全等价。 |
| `docs/rendering/SCENE_RECIPE_INDEX.md` | 移除河岸/失败衣物/旧包装推荐链接；保留 native多人/linked单人、DS/Omni与透视差异和配方资产链。 |
| `configs/README.md` | 移除退役 Hydra 组和example推荐；loop父配置依赖与default data=unselected 明示；给出停止支持的组名。 |
| `docs/environment/INSTALLATION.md` | 保持双包 editable 与依赖/路径规则；删退役 CLI 列，保留既有环境冲突、外部资产要求和无资产交付限定。 |
| `pyproject.toml` | 保留双包元数据与依赖组，本轮不建议顺手升级/删除重型依赖；legacy删除后验editable discover不残留旧包。 |
| `setup.py` | 保留双根 find_packages；待删legacy后以独立离线环境验包清单，无需硬编码新包名。 |
| `tools/annotation/build_omni_multiperson_tiny.py` | roll_person → data/erp_crop.py，build 的标签装配 → annotation/omni_samples.py；此 CLI 留参数/输入输出组织，活动调用方与数值测试直引公共函数，保持ERP roll/图像/点/mask同变换。 |
| `tests/test_occluded_supervision.py` | build 改直引 annotation/omni_samples.py；遮挡/域外/裁剪外、loss和验证分组测试完整保留，不删 legacy_weight_targets 数据策略。 |
| `tests/fixtures/README.md` | 保留所有长期数值fixture来源/采集先于迁移的事实及原始制品链接；将已退役 capture 命令标为 git show e790590:<path> 历史来源，不要求继续安装旧工具。 |
| `tests/test_inference.py` | 保留两条加载顺序/严格key/模式/有限值的公共推理测试；只删从 train_cycling_smoke/train_source_isolated_trial 重导出的身份兼容断言。 |
| `tests/test_training_audit.py` | 保留 CPU 小网络保存恢复/174数组fixture、梯度失败和RNG/参数顺序；只删 smoke/历史脚本中 CheckUpdates/probes 的导出身份测试。 |
| `.gitignore` | 原位保留公共 data、docs/data、Hydra 配置例外；新长期fixture/helper必须非忽略，outputs/资产不假定有Git备份。 |

## 9. 迁移组的具体落点、停止能力与验收

### crop（B2）

旧 tuple/SMPL-HMR 裁剪接口退出支持；先迁仍有用的可视化与测试，再删包装和其唯一 legacy 实现。

迁到：visualize_annotation.py 改用 crop_pipeline.crop_example 的 CropResult；validate_bedlam_densekp_tiny.py 改直引 crop_geometry；tests/dataset_crop_fixture.py 改采集具名结果。evaluate_omni_densekp.py 的独立早期评分 CLI 停用，Omni 生成及公共相机回归仍保留。

停止支持：get_example/get_example_projverts/do_augmentation 的历史 tuple 导入；早期 Omni 专用推理 CLI。旧 NPZ 权重与 Dataset ID 子集并未一并删除。

删除门槛：旧独立 dataset_crop_v1 fixture 保留；逐字段比较完整输入/GT/affine/mask、增强参数/RNG/异常；旧 tuple 位置断言改名不重生成期望。

### clothing_wrapper（B1）

纯 re-export 仅被测试直接引用；删除旧导入，不删实际衣物契约。

迁到：tests/test_commuter_clothing_contract.py 改为 densekp_retrain.clothing.contracts.validate_binding

停止支持：tools.render.commuter_clothing_contract 旧模块导入

删除门槛：原重构/旋转与拒绝 gate 测试原样通过

### source_helpers（B3）

固定混合人物的旧组合主线由来源隔离入口替代，但 transfer_groom 仍被真实资产准备使用。

迁到：transfer_groom → src/densekp_retrain/clothing/groom_transfer.py；prepare_source_isolated_assets 直引。parameters → body/parameters.py，clothing 数组保存 → clothing/preparation.py，STYLES 仍属于 adult 配方，显式传入 profile/materials。

停止支持：固定 JOBS 的 mixed_pedestrians 再制作 CLI；既有 mixed 回归样本继续使用

删除门槛：选定既有 walking/cycling 身体、groom active_mask/offset、衣物对应和来源字段独立精确对照；不新增姿态资产

### source_fixture（B1）

长期来源/打包契约测试引用本工具，不能连带删测试；大数据 before 捕获 CLI 属于一次性迁移。

迁到：source_boundaries/contract_cases/pure_boxes 及 write/outcome/normalize 的必需闭包 → tests/source_packing_fixture.py；TracedGenerator/RNG 断言接 tests/test_box_robustness.py；真实资产 replay 配方/路径清单 → docs/VALIDATION.md。

停止支持：tools/validate/verify_source_packing.py --output / --before 一次性迁移基线与全数据包装命令；数值 fixture 和长期契约测试保留。

删除门槛：source_packing_v1.json 不重录；正常/泄漏/缺字段/空监督/NaN 拒绝及行序/RNG 回归通过

### imaging_fixture（B1）

compare/normalize 被成像长期测试调用；独立旧 before 捕获入口已完成使命。

迁到：compare/normalize → tests/artifact_comparison.py，test_imaging_pipeline 直引；选定 EXR replay 参数进入 docs/VALIDATION.md

停止支持：verify_imaging_pipeline 迁移捕获 CLI；DS/Omni 后处理入口继续存在

删除门槛：imaging_pipeline_v1.json 原图片/数组/JSON/失败部分输出精确；保持允许路径/耗时字段白名单

### legacy_tests（B1 拆分；整文件删除待 B2 及后续评审）

旧 loss/pickle/legacy 专用一致性断言可撤销；其中有效的导入安全和归一化验证必须先提取。

迁到：导入无资产/无写出/无进程 → tests/test_import_boundaries.py，仅列精简后保留脚本；denormalize 的独立期望 → tests/fixtures/image_normalization_v1.json 与相关测试；其余退役接口记录由 Git 保存。

停止支持：为已退役完整对象 pickle、SMPL 裁剪/相机 helper 的兼容性承诺

删除门槛：保留的导入副作用 gate、当前 denormalize 数值测试继续；删除只验证退役符号身份/旧执行体的断言

### capture_only（B4）

实现是 refactor 旧/新捕获或 AST 对照专用；长期纯数值 fixture 已在 tests，捕获入口不能继续当主路径。

迁到：按各自现有 tests/*_fixture.py 保留数值/RNG/拒绝检查；Blender 导入/CLI 安全必要部分 → tests/integration/scene_boundaries.py；命令/环境与证据索引 → docs/VALIDATION.md

停止支持：重新从旧 tools 路径冻结 before 的命令与一轮迁移专用参数

删除门槛：在无 outputs 的交付副本运行长期 CPU 契约；确认 helper 不反向 import 删除工具；Blender 检查在对应环境显式执行

### integration（B4）

应删除迁移专用 CLI，但真实加载/生命周期/CUDA/Blender 证据能力仍有长期价值，须先落到显式集成测试。

迁到：按文件职责分别迁 tests/integration/inference.py、fixed_validation.py、model_lifecycle.py、blender_assembly.py、blender_postprocess.py、blender_artifacts.py；仅保留当前实现+冻结基线检查，剔除旧 AST 抽取/兼容 capture 分支；数据路径显式输入。

停止支持：原 verify/compare 迁移工具 CLI 和硬编码 GPU 1/历史目录；不放弃真实 GPU/Blender 验收

删除门槛：数值/shape/key/dtype/state_key/RNG/异常继续；真实路径先设备进程检查，未运行时标为未验；CPU mock 不替代；Blender 5.1.1 CPU 既有范围不扩张

### regression_launcher（B3）

两套定日期调度器重复组装同一 evaluator；需要保留其不同 batch、权重与 gate，不能只删掉旧骑行回归。

迁到：新增 configs/regression/suites.json + tools/validate/run_regression_suite.py；原命令 argv、顺序、A/B/C/51 行 riding+mixed、paired/严重遮挡/历史四框 case 全部显式化。配置只打印为默认，执行要显式选择。

停止支持：按历史目录/UUID 硬编码的两条 orchestrator CLI；现有评分入口/算法不合并

删除门槛：独立旧 argv/manifest 对照；重放保存预测的分母和字段精确；时间/新输出根单列；不训练

### bvh（B4）

原生 BVH 骑行搜集/拟合输入支线不用于当前公开照片重建姿态；明确停止该支线而不是假称已被同算法替代。

迁到：两片段来源、单位/轴/未完成拟合结论 → docs/ASSETS.md；若有保留脚本用短名导入 audit_cyclist_bvh，随支线一并移除调用

停止支持：BVH 搜索/窗口排行/fit-inputs/kinematics 图的再制作命令；原 BVH 与失败输出不删

删除门槛：照片 → HMR2 → SMPL-X → 角色姿态链无此依赖；脚本命令/文档全引用移除；原输出只 stat

### failed_clothing（B4）

当前接受候选用 simple clipped-shell；该条 loose/polo/union/settling 修复主线仍属历史失败/诊断方案，建议停止再制作。稳定契约/绑定并不连带删除。

迁到：共用 binding/contracts/shells 原位；repair_commuter_top.qmultiply/dual_quaternion_binding 与 commuter_union_volume.scan 若仅作为几何回归 oracle，迁 tests/blender_geometry_fixture.py，保留原数值测试；crossings 独立公共化到 rendering/blender/intersections.py，diagnose_adult_contacts 与测试直引；失败原因和制品路径 → docs/ASSETS.md。

停止支持：历史领口、直筒裤、voxel union、cloth settling 各修复 CLI；不宣称解决衣物质量

删除门槛：保留相交/共面未知、四元数符号/刚体、scan overlap 的数值测试；资产准备调用闭包无旧 CLI；必要 Blender 数组测试，不因删 CLI 重渲染

### once_audits（B4）

固定一次性资产核查/已完成补救报告，不承担当前主流程；先把来源、失败与环境结论落到当前文档。

迁到：来源许可/版本与缺失 → docs/ASSETS.md；设备/环境 → docs/environment/INSTALLATION.md；A/B/C 修订、失败及未部署 gate → docs/DATA_AND_MODELS.md；不自动复写其历史 JSON。

停止支持：固定旧目录的诊断/修复记录再次生成；已有制品及失败记录保留

删除门槛：逐文件确认没有保留调用方；报告独有结论已摘录并指向实际产物；不把此前状态写成当前状态

### duplicate_smoke（B4）

训练入口已有 train_keypoints 和有界 source trial；通用生命周期/RNG测试归长期 integration，旧 tiny smoke 不再作为用户入口。

迁到：当前 CheckUpdates/probes/FixedValidation 保持公共路径；生命周期测试 → tests/integration/model_lifecycle.py，旧 smoke 配额/范围说明 → docs/VALIDATION.md

停止支持：三条早期 smoke/小试训练 CLI 与旧导入名；不删除当前训练审计

删除门槛：CPU 小网络更新/恢复与损失契约通过；保留真实模型生命周期选测，不能用训练效果代验收

### old_sweeps（B4）

固定目录全 sweep 调度并非唯一接缝/大人体数学实现；保留被这些试验发现的长期边界，而停止整套历史 sweep 制作。

迁到：接缝/有效域/尺度代表检查 → tests/test_omni_radtan.py、test_multiperson_pilot.py、test_imaging_pipeline.py；取样参数/已存在制品 → docs/VALIDATION.md

停止支持：完整历史 yaw/scale sweep CLI；不取消 ERP 周期/边缘测试

删除门槛：先核对现有测试覆盖每个独有失败类别，再删调度器；无新渲染

### retired_eval（B2）

注册数据 ID 的旧 evaluator 把监督 weights>0 命名为 visible；不用合并该语义来删接口。

迁到：推荐 candidate evaluator；必要显式 NPZ → candidate index 适配说明进入 COMMANDS；旧字段含义和旧结果保留在 DATA_AND_MODELS

停止支持：evaluate_densekp_npz CLI 及其误导性 visible 字段格式；不是修复或重算旧报告

删除门槛：现存需要的 candidate/DS/旧 riding 回归完整；不拿两个不同策略统计结果宣称完全相同

### scene_unused（B4）

河岸示意场景不被当前保留的森林/街景生成与回归引用；只是独立构图配方。

迁到：场景来源/用途记录 → docs/ASSETS.md；源 .blend/.blend1 和 outputs 留原处

停止支持：河岸程序式再制作 CLI

删除门槛：SCENE_RECIPE_INDEX 与测试旧 recipe body 清单更新；城市/森林加载路径不变

### hydra_parent（B2）

不是无用 HMR 配置：cpu/gpu/ddp/mps 全部继承，必须先保持组合结果再改名。

迁到：原字段原值 → configs/hydra/trainer/loop.yaml；四个 trainer defaults 仅替换父配置名，保持覆盖顺序

停止支持：trainer=default_hmr 旧组名；其日志/验证/precision/max_steps 行为继续保留

删除门槛：四组 compose 的 resolve=False 容器值及打印顺序独立 before/after；gpu/cpu precision、max_steps、val_check_interval 不变

### intersection（B3）

实际 BVH 相交计算被成人诊断与长期 Blender 测试使用，先迁计算再删 tools 公共库文件。

迁到：crossings → src/densekp_retrain/rendering/blender/intersections.py；tests/test_commuter_intersections_blender.py、diagnose_adult_contacts.py 直引；待退役修复脚本同批撤销。

停止支持：tools.validate.commuter_intersections 导入路径；保留计算能力

删除门槛：相交/不相交/共边/共面未知四类结果原样，Blender最小数组检查，不重渲染

直接删除组无需搬迁业务实现；测试、配置、文档、安装及权重影响逐文件写在 inventory。文档组按第5节逐份归并后删除。

## 10. 本轮规划产物核查

已核对494条记录的必填字段、源文件存在性、分类与物理行计数，以及本规划的相对链接。HEAD未变，TODO字节内容与开始时一致，暂存区为空；仅新增本规划和被现有规则忽略的inventory。未执行运行测试或删除迁移；所有后续验收条件均未因本次静态核查而标为通过。

## 11. B1 评审修正与实施前置条件

- 已去除 203 处普通 `import json` 指向项目 `io/json.py` 的误映射；`losses`、`crop`、`erp` 等非模块查找字符串不再进入调用图。完整点分模块字符串也仅先列候选，需结合 importlib/命令/配置上下文人工确认。
- `smoke_densekp_training.py:87` 的 `losses` 只是报告字段；不为此修改该脚本。B1 拟删文件的真实调用方已逐项阅读，详见 inventory 的 `dependency_review`。
- `verify_refactor_metrics_config.py` 的 capture 目前已因定义模块检查而不能用于当前包装。B1 明确退役 capture，入口应在旧模块导入和任何文件操作前给出退役说明；删除旧模块路径/capture代码，保留 verify。整工具删除仍留待 B4。验收既检查capture不落盘，也检查verify仍能读取独立基线，不重新预测。
- 修正原计划过宽的 legacy 测试删除时点：B1不删除 `legacy_cleanup_v1.json` / `legacy_import_bodies_v1.json`，也不取消尚未退役的裁剪测试。归一化与12脚本导入安全/执行体检查独立出来，原期望直接使用；旧loss和训练helper断言随接口退役。裁剪接口及其剩余测试到B2再评审。
- 来源辅助仅迁 `source_boundaries`、`contract_cases`、`pure_boxes` 及 `write/outcome/normalize` 必需闭包；成像辅助仅迁 `compare/normalize`。不把整个验收CLI搬入tests，不重录任何期望。
- B1验收按具体契约列出来源泄漏/遮挡者、缺字段/空监督、框边界、成像完整制品/异常部分输出、归一化、导入副作用、两个指标汇总策略以及保留入口可达性。测试数量只作记录，不作正确性依据。

此节是前述初版清单的收窄修订；B2–B5候选仍停留在规划。

## 12. B1 可执行文件清单（实施准备，未执行）

| 文件 | B1 动作 | 前置条件与验收 |
| --- | --- | --- |
| `tests/source_packing_fixture.py`（新） | 迁出6个必要函数：source_boundaries/contract_cases/pure_boxes/write/outcome/normalize | 原函数体与依赖闭包逐项核对；source_packing_v1.json字节不变；来源/遮挡者/拒绝和NPZ数组逐项比较 |
| `tests/artifact_comparison.py`（新） | 迁出compare/normalize | 保留图片文件字节、JSON字段/顺序、数组dtype/shape、异常部分输出；仍只豁免输出根和elapsed_seconds |
| `tests/test_source_packing_contract.py`、`tests/test_imaging_pipeline.py` | 只改上述helper导入 | 全部当前数值断言保留；不扩大本批到其它未退役CLI导出 |
| `tests/test_image_normalization.py`（新） | 独立调用公共denormalize_images | 直接读取原legacy_cleanup_v1.json中的denormalize期望；不生成新期望 |
| `tests/test_import_boundaries.py`（新） | 迁入12脚本导入安全和直接执行体AST检查 | 沿用原legacy_import_bodies_v1.json；禁止读资产/写文件/进程/扫描/加载重依赖，random和sys.path不变 |
| `tests/test_legacy_cleanup.py`、`tests/legacy_cleanup_fixture.py` | 保留裁剪及RNG旧回归，去掉已退役loss/train_utils的执行和断言 | 只比较原fixture中明确列出的裁剪键；不更改旧裁剪实现 |
| `tests/fixtures/legacy_cleanup_v1.json`、`tests/fixtures/legacy_import_bodies_v1.json`及source/imaging fixtures | 原字节保留 | B1前后直接bytes比较，不计算SHA256 |
| `tests/test_evaluation_contract.py` | 只移除occlusion_metrics身份兼容断言 | 19组独立数值/非有限/空集/分母期望及配置dump检查保留，其它未退役导出断言不扩大清理 |
| `tests/test_commuter_clothing_contract.py` | 直引densekp_retrain.clothing.contracts.validate_binding | 旋转、绑定误差、studio gate及拒绝断言原样 |
| `tools/validate/verify_refactor_metrics_config.py` | B1退役capture，保留verify；不留旧动态模块引用 | capture明确失败且不创建baseline/report；verify回读原before，比较配置、保存预测及七样本；无模型构造 |
| `tools/validate/check_workspace.py` | 加入新归一化/导入/成像检查，保留裁剪相关检查 | 显式测试列表，默认仍只扫源码和配置；两cwd核查，不扫outputs |
| `densekp/core/losses.py` | 删除旧六类及其完整对象pickle接口 | 当前目标函数未引用；本计划已记录三指定checkpoint元数据证据；不重新推理 |
| `densekp/core/utils/train_utils.py`、`densekp/legacy/train_utils.py` | 删除旧导出与实现 | 当前denormalize公共函数不改；仅退役旧训练/相机helper接口 |
| `tools/render/character_groom.py`、`character_materials.py`、`commuter_clothing_contract.py` | 删除纯导出文件 | 活动renderer已直引公共模块；衣物测试先迁 |
| `tools/scene/blender_scene_bridge.py`、`tools/validate/occlusion_metrics.py` | 删除纯导出文件 | 旧metrics capture先显式退役，身份测试先调整；不改实际Blender装配/指标 |
| `tools/validate/verify_source_packing.py`、`verify_imaging_pipeline.py` | helper迁完后删一次性验收CLI | 保存原输出和独立fixture；真实全数据回放的方法/路径在Git和原阶段证据中可追溯 |
| `docs/READING_GUIDE.md`、`docs/rendering/SCENE_RECIPE_INDEX.md`、`tests/fixtures/README.md` | 只修正已删除接口、当前测试入口与fixture用途 | 历史阶段报告主体不变，当前导航不再声称这些旧接口可用 |

B1 整文件删除目标收窄为10个；4个legacy测试/fixture文件不随本批整文件删除。这个数字只说明边界，不作为验收标准。

实施时先保存小型源码/fixture before与CPU前测，再迁helper、拆测试、退役capture并删除10文件；最后做有界CPU契约、保留入口依赖及无资产交付核查。当前仅完成源代码阅读与准备，未运行这些测试。fixture和TODO用字节比较保全，所有新验收输出必须使用新目录。

本轮范围已由用户明确为“仅修正清单与实施前核查”。本次没有执行本节计划中的代码迁移、删除或CPU测试；B1实施也未在本轮自动开始。

## 13. B1 实施状态

B1已按第11–12节完成，实际删除10文件、保留原fixture与裁剪回归，capture明确退役而verify通过。完整变更、验收、前测误纳入小网络更新的范围偏差见[B1实施报告](PRUNING_B1_20261002.md)。未执行B2–B5，未提交commit。
