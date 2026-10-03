# B3 实施前核查与分批计划

日期：2026-10-03。状态：**仅静态核查及规划，待评审；未迁移、删除或执行验收。**

## 1. 实际基线与核查边界

实际分支为 `master`，HEAD 为 `48b074901140a4ae9571ea3ff6828f2e851c5e83`（`Complete B2 configuration and legacy interface pruning`）。开始时工作区只有用户原有 `docs/TODO.md` 修改；B2 已提交。B2 完成报告中的“尚未提交”是当时状态，不重写历史。

本轮只新增本文件和 [机器清单](../../outputs/plans/pruning_b3_20261003/inventory.json)。TODO 原字节单独保全核对。没有运行项目代码、测试、Dataset、评价、模型、几何生成、Blender 或 GPU；没有安装、下载、SHA256 或提交。

写出后复核：已跟踪文件的差异仍只有 TODO，其字节与开始时精确一致；新增报告未被忽略，inventory 位于被忽略的 outputs。机器清单覆盖54项文件记录；当前相对链接和所有标为现有的文件路径存在。`git diff --check` 仅报告原有 TODO 第13行行尾空白，未修改。两份规划尚未提交，关键执行要求均在本报告，不以 ignored JSON 作为唯一落点。

核查方法：对相关 Python 源码做 AST import/函数定位，人工追踪调用体、subprocess 参数、动态路由、当前配置和导航；读取明确相关的 JSON、选定 NPY/NPZ 的头部及文件存在性。没有解压纹理、读取模型权重内容、加载整个 outputs 或访问外部项目代码进行修改。模型文件只检查存在性。普通报告字段和同名文本不计作依赖。

已阅读原精简规划及 inventory、B2 完成报告、COMMANDS、SCENE_RECIPE_INDEX，并核对当前公共实现。这里的“存在”“保存记录通过”不等于本轮重新验证数值通过。详细输入路径、形状、dtype、import 行号保存在机器清单。

## 2. 当前调用链与保留系统

| 当前入口 / 环节 | 当前真实依赖 | B3 后建议归属 |
| --- | --- | --- |
| prepare_source_isolated_assets.main | audit_pedestrian_poses.parameters；prepare_adult_candidates.clothing；prepare_mixed_pedestrians.transfer_groom | 直接使用 body.parameters、clothing.preparation、clothing.groom_transfer |
| audit_pedestrian_poses、audit_new_pedestrian_source、freeze_mixed_candidates | 共享 parameters；后两者另消费 SOURCES / mesh_panel | 参数转换公共化；固定来源、画廊和冻结配方继续有明确入口 |
| prepare_adult_candidates | STYLES、fitted_shape、角色/照片配方、衣物写出 | 拟合和角色制作保留；STYLES 放独立、无副作用的配方数据文件；写出公共化 |
| prepare_mixed_pedestrians | 固定 JOBS 六个行人衍生，加四个骑行复用；提供 groom 函数 | groom 迁出后退役固定制作 CLI；旧 report、几何、索引全部保留 |
| 来源准备 → 场景 | split_v2 → assets/report → run_source_isolated_batch → render/postprocess → pack | 全链不变，不移动场景布局、灯光或随机配方 |
| 成人/mixed 历史回归 | candidate_index 24＋27 行 → evaluate_candidate_baseline → original_baseline 51 行参考 | 数据和当前 evaluator 保留，不需要旧 mixed 制作脚本来读取 |
| evaluate_source_trial | 3 个 candidate 任务＋配对汇总＋普通试验比较 | 一个薄调度入口中的 source_trial suite |
| evaluate_box_robustness | 5 个 candidate 任务＋2 个 DS 历史框任务；比较入口单独执行 | 同一薄入口中的 box_robustness suite，保留独立评价/比较公式 |
| diagnose_adult_contacts 等 8 个工具＋1 个测试 | commuter_intersections.crossings | Blender 边界内的公共相交函数；所有调用方先迁 |
| test_commuter_intersections_blender | crossings、repair_commuter_top 的 DQS、commuter_union_volume.scan | 公共 crossings＋最小测试参考辅助，保留 7 个断言 |

保留人物/骑行/行人/衣物准备、场景渲染与成像、来源检查与打包、框增强、Dataset、训练推理和当前回归能力。当前 public shell、binding、body geometry、camera、imaging、evaluation 模块不另造一套。138 点、regressor、共同几何变换、256-square/内部192列、监督有效区和可靠遮挡 GT 不变。独立照片不拼成共享 shape motion；来源检查覆盖所有遮挡者。DS/Omni、ERP 像素与裁剪像素、不同汇总分母继续分开；预测第三值不转成置信度。

## 3. A：人物、衣物与来源准备

### 3.1 逐函数闭包及接口

| 原位置 | 输入输出 / 闭包 | 推荐接口与迁移约束 |
| --- | --- | --- |
| tools/experiments/audit_pedestrian_poses.py:22 parameters | pose 一行165维；beta 原对象；pelvis ndarray。返回按插入顺序的参数 dict；无 IO/RNG | 在现有 body/parameters.py 增加 `parameters_from_pose(pose, beta, pelvis)`。不经 zero_pose，不增加校验/强制 reshape，不转换 beta |
| tools/experiments/prepare_adult_candidates.py:67 clothing | destination/geometry 为 Path；rest/joints 为数组；short/cid；读取 vertices/faces，查 STYLES，调用 clipped_shell，创建目录并写数组/3份JSON，返回 str(destination) | 新 clothing/preparation.py 的 `prepare_simple_clothing(destination, geometry, rest, joints, style_id, character_id, *, style_table)`；style_table 是显式配方映射，原 short/cid 对应 style_id/character_id |
| tools/experiments/prepare_mixed_pedestrians.py:25 transfer_groom | manifest 的 reference 几何/groom/binding；geometry/out Path；读数组、转绑定并写6个数组及 manifest，返回 str(out) | 新 clothing/groom_transfer.py 的 `transfer_groom(manifest, geometry, out)`；保持原参数和返回，不留旧导出包装 |

parameters 的切片顺序必须原样：global_orient[0:3]、body_pose[3:66]、jaw[66:69]、leye[69:72]、reye[72:75]、left_hand[75:120]、right_hand[120:165]；每段 `np.asarray(..., np.float32)[None]`；最后 betas=beta、transl=-pelvis[None]。保留 beta 身份/输入不变及异常顺序。短 pose 目前可产生短切片，不借迁移新增165维拒绝；zero_pose 对 beta/pelvis 的转换不同，不能直接复用替代。

clothing 的顺序是先读 body/faces、再查 style key、再计算 shell、再 mkdir（没有 exist_ok）、再写文件。接口仍接收显式 style_table，是为了不在调用方提前查 STYLES 而改变缺文件/坏 key 的异常优先级。STYLES 的10套角色数据单独放 `tools/experiments/adult_candidate_recipe.py`，仅存原数据；公共模块不得反向导入 tools。不把衣服颜色、角色名单、路径、拟合目标放入公共包。

衣物契约：身体首帧及 rest/joints 同一坐标；沿用 clipped_shell 的 barycentric/外壳数组、dtype 和插值；数组写出顺序不变。face_material_ids 现为全零 int32，即使 materials 有 top/pants/panel 三项，也不能顺手修复。RGBA 追加1、profile.collar=.045、fabric、method、limitation、split=integration_only、diagnostic_pending、body_geometry/cid 和历史 clearance/offset 字段均保持；目录已存在时的失败和部分写出状态也比较。这里公共化的是已复用的制品保存规则，不重新实现 shell。

groom 闭包：binding 的 body_faces 是每根曲线的三个顶点索引；weights 为重心权重，curve_offsets 决定每点所属曲线。使用 CONV 变换到绑定轴系，以 posed/reference triangle_frame 的矩阵乘积旋转 target_rest_frames；根部沿第三轴偏移0.0004米；local_points 和冻结 collision_correction 随头皮三角形旋转；最后 points 转 float32。保留 source radius、active_curve_mask、offsets、文件名和 parent_groom/binding 引用；不重新筛毛、不更改头型或身份。

后续每50个点取样，经 active mask 过滤，closest_surface 分块8192（基于近邻面中心候选，不是新的精确全表面距离算法）。保留 sample_min_signed_m、低于-1mm计数和诊断状态。当前全 inactive 时会在保存数组后因空 concatenate 失败；保留该异常和已写出文件集合，不能当作本批修复。target_rest_frames 在既有 binding 中有 float64 和 float32 两种；不得统一 dtype。三函数没有显式 RNG 调用；外层 SMPL-X 构造和 rest_body 的调用次数/顺序仍有独立边界，纯函数回放不能证明整个 main 的模型/RNG 生命周期。

### 3.2 文件及消费者处理清单

| 文件 | 决定与真实消费者 / 前置条件 | 保留或停止支持 |
| --- | --- | --- |
| src/densekp_retrain/body/parameters.py | 修改：加上述转换，保留 PARAMETERS/evaluate/rest_body/zero_pose | 原有公共接口、模型行为不改 |
| tools/experiments/audit_pedestrian_poses.py | 修改：删除 parameters 定义，自己的 main 直接公共调用；audit_new、freeze、mixed、source 全部迁移 | 保留 SOURCES、mesh_panel 和姿态检查 CLI |
| tools/experiments/audit_new_pedestrian_source.py | 修改 parameters 导入/调用；mesh_panel 仍是专用画廊依赖 | 不扩展为通用可视化框架 |
| tools/validate/freeze_mixed_candidates.py | 修改 parameters 导入/调用；保留 SOURCES 固定来源和既有冻结审查 | fixed mixed 制作 CLI 退役不等于冻结/读取旧样本退役 |
| tools/experiments/prepare_adult_candidates.py | 迁出 clothing 和 STYLES；自身直接用公共 writer＋配方数据 | 保留 fitted_shape、角色名单、照片准备、材质/眼睛/许可记录 |
| tools/experiments/prepare_source_isolated_assets.py | 三调用逐批直引公共；其余 main 顺序不动 | 保留完成行优先、历史资产复用、27衍生、来源与足底检查 |
| tools/experiments/prepare_mixed_pedestrians.py | 第一批暂改参数/衣物调用；groom 迁出并核清后删除整文件 | 停止固定 JOBS 制作 CLI 和三个旧 helper 导入；不删除已制成资产 |
| src/densekp_retrain/clothing/shells.py、binding.py | 保留；writer/groom 复用当前实现 | 不合并不同历史壳层或绑定方案 |
| tools/experiments/prepare_cycling_characters.py | 保留照片 POSES 配方，adult 仍有真实消费 | 不改照片来源、许可证和各自 shape |
| tools/experiments/prepare_source_isolated_split.py、run_source_isolated_batch.py | 保留来源划分、布局、随机采样配方 | 不移动 default_rng(930)、roll/permutation 或角色配额 |
| tools/validate/audit_source_pose_transfer.py、summarize_source_trial_data.py | 保留来源回读和统计 | 不重定义来源字段或监督集合 |
| tools/experiments/render_adult_candidates_ds.py | 保留；main 的 --assets 读取 report.derivatives，支持 --pose-source-ids 和 --pedestrian-layout 消费 mixed | 不因制作入口退役删除渲染/旧回归能力 |

已核到 mixed 制品的场景消费者是 render_adult_candidates_ds，而不是另一个名为 mixed 的渲染脚本；不能按目录或入口名称推导整链可删。两个 audit recipe 间的 SOURCES/mesh_panel 共享仍明确保留为专用配方关系，不宣称所有 tools 之间已零依赖。

来源准备 main 仍会读取 adult 和 mixed 的历史 report，并按已完成 → 可复用 → 制作优先级处理。固定 mixed CLI 删除后这些 report/manifest/geometry 仍是本机运行资产，不能把 ignored outputs 当作 Git 备份。未来若要从空资产库完整重建 mixed 固定配方，可从 Git 历史恢复旧版本；当前不承诺该旧 CLI 在新树可用。

## 4. 有界实物核查和独立基线设计

以下是从当前 source-isolated report 选出的代表，**本轮只核对关联文件及头部，不执行转换**：

| 角色 / 分区 | 姿态 | 选择意义 |
| --- | --- | --- |
| cycling_F01_v2 / train | coco_292226_person_1330527 | 第一照片、女性、sport衣物、早期float64 binding frames |
| cycling_M01_v2 / train | KIT314_bend_left01_f060 | 男性行人、sport、头皮绑定转移 |
| cycling_F02_v3 / val | coco_306800_person_225160 | 第二照片、女性、daily长衣、float32 binding frames |
| cycling_M02_v3 / val | KIT425_walking_medium05_f040 | 隔离来源男性行人、新来源准备产物、daily |

四者的 manifest、几何、138点、canonical/motion参数、衣物数组/材料、groom/reference/binding、rest_joints、原姿态输入和模型路径均已核到；完整路径和头部在 inventory。rest_joints 均为 (127,3) float32，不能解释成138语义点。角色 beta 为30维；manifest 中头型策略为 shape 0:30、full-file 300:310 zero，不能与旧10beta几何证据混用。

source report 有27衍生；历史 body/clothing 与 groom readback 各36条、source transform27条，记录可用但本轮未复跑。mixed/adult 索引27/24行，所列 loader/image/square/network_center 共204个引用（含重复）存在；原始回归 report51行，riding_A_000/mixed_B_000 的预测文件存在。只抽查两份保存预测的存在性，没有宣称全部预测数值验收。

原几何证据正确路径为 `outputs/acceptance/refactor_geometry_camera_20261002/before/data/selection.json`。它覆盖原10beta角色几何，不能替代本次30beta准备、衣物写出或 groom 基线。新的独立 before 必须在迁移前调用原函数冻结；不得旧新都转发公共实现后比较。

### 计划验收矩阵（全部尚未运行）

| ID | 输入 / BEFORE来源 | 比较项 / 计算边界 |
| --- | --- | --- |
| A-PARAM | 原 parameters；合成165维顺序输入＋四角色已存参数；float32/64 beta、非连续数组、短pose、非法pelvis | keys顺序、切片、shape/dtype、beta别名、输入变异、异常和 Python/NumPy/Torch RNG；精确。仅CPU数组，无模型 |
| A-CLOTH | 原 clothing；四角色现存 geometry 与明确选定 reference/rest/joints 输入，输出全写新验收目录 | arrays、faces、barycentric、material IDs、全部JSON字段/写序；相同路径输入，输出根仅在报告字段明确列映射。未知style/缺vertices/faces/已存在目录/无面按原行为；不改变期望 |
| A-GROOM | 原 transfer_groom；至少 M01 与 M02 完整绑定，兼顾 f64/f32；F01/F02用选定相同数组补齐男女与两照片关联 | points/roots/corrections/radius/offsets/active mask、头型/来源引用、dtype/shape/顺序/诊断值精确；缺binding、坏offset/weights、全inactive、已存在out及部分写出顺序 |
| A-SOURCE | split、27衍生 report、原canonical输入、9角色manifest；不运行制作 main | 来源组/shape/pose父子、所有场景人物与遮挡者；引用存在和字段相等，合成泄漏用例继续拒绝；源main未改段AST与调用次序审查 |
| B-PLAN | 原两调度器在小型受控report下的截获轨迹，不启动真正子进程 | argv、顺序、cwd、env差量、日志/失败/输出规则；新默认展示stdout零写出/零Popen/零模型import；根目录及临时cwd |
| B-REPLAY | 明确现存预测/参考report/索引；先冻结旧summarizer输出至新目录 | 指标/门槛/分母/非有限值/units/报告精确；不能验证推理精度、设备、权重或训练 |
| C-TRI | 原 crossings 对四组三角形；小型原输入 | pairs/locations/None、次序/dtype/shape精确；Blender mathutils，非资产制作 |
| C-DQS-SCAN | 原 qmultiply/DQS/scan 对现有三项测试输入；独立解析期望保留 | 刚体矩阵期望和±179°符号、重叠体正绕数；原容差不放宽；同版本Blender CPU，只有测试临时cube，无渲染 |
| C-IMPORT | 原 import bodies fixture，所有9个消费者＋新公共模块 | 静态闭包、直接 -- 入口路径、导入无IO/RNG/可选依赖；精确的2处main导入/路径映射例外，见§6 |

A-CLOTH 的 saved reference vertices/rest_joints 可作为明确传给旧新函数的同一输入，但**不能谎称它就是每次 rest_body 的位级原输出**。如果要验收真实制作 main，另列 CPU SMPL-X 30beta 的 rest_body 重算与外层 RNG/order 基线，涉及模型实例化，须在对应实施批明确授权；本轮及建议首批不需要。不得新造角色或覆盖几何。四角色头部存在性不证明完整数组内容正常；实施采集发现坏输入应暂停，不下载/补生成。

groom 每份 binding 约28–36MB，曲线/点规模大；串行读取选中样本，不递归整个资产库。完整头发转移是写新验收制品的 CPU 工作，不是单纯metadata检查，实施时应明确授权。不为验证搬函数重新制作全部27衍生。

## 5. B：回归调度方案

建议合并**任务组织**为 `tools/validate/run_regression_suite.py`，配一份小型 `configs/regression/suites.json`，只有 source_trial 与 box_robustness 两个固定 schema 的 suite。不建公共执行引擎、不支持任意动态类/模板语言，不把指标计算迁入调度器。两套任务构造保留具名分支，共享最少的日志/顺序执行逻辑即可。公共模块没有第二个真实消费者时，不另开一层库。

建议新 CLI：`--suite source_trial|box_robustness`，沿用各套所需 trial/data-root 或 experiment、output-name、only；默认只展示解析后的计划；显式 `--execute` 才执行，GPU UUID/physical device 在执行模式必填并即时核查设备和进程。纯展示只用 stdlib、读取小型配置/报告，不调用 nvidia-smi、项目 evaluator、模型、mkdir 或任何子进程；不以 train=False 保证安全。不要自动选“最新” checkpoint/status，不覆盖活动目录。

这是明确的新运行界面：停止支持两个旧调度 CLI，不留包装；默认展示及 source 套件新增 GPU 核查是明确变化，不伪装成旧 CLI 完全等价。评价命令/任务顺序/权重/门槛保持。若不同失败流程使共用 runner 更复杂，应只共用 argv 数据构造，保留两个短执行分支。

### 5.1 行为对照表

| 项 | source_trial 当前 | box_robustness 当前 | 实施要求 |
| --- | --- | --- | --- |
| 路径基准 | trial/data-root 依调用cwd resolve；其余ROOT | experiment依cwd resolve；data/plain/mixed/adult固定ROOT | 两cwd明确比较，不把所有相对路径改ROOT |
| gate | actual_updates==400 且 reload_max_abs_error≤1e-5；创建out之前检查 | 同gate，但先mkdir再读report | 保留执行异常和副作用顺序；展示不模拟写出 |
| 输出 | trial/evaluation，exist_ok=False，复制两份入口快照 | experiment/output-name，默认evaluation_v2；exist_ok=bool(only) | 不覆盖历史目录；执行记录schema各自保留 |
| 任务 | main → paired → regression | challenge_A/B/C → val_C → regression_C → history_B/C | source后接2个汇总；box不悄悄加汇总 |
| 模型配置 | 每项显式 trial/model_config.yaml、device=cuda:0 | candidate未传model-config/device，使用现行默认 | 保留“未传”这一行为，不统一argv补齐 |
| 权重 | trial/step_0400 | A assets_external/densekp.ckpt；B普通400；C experiment/training/step_0400 | 固定角色和来源，不替换权重 |
| reference | main原277/paired原paired/regression51 | challengeB/C以A；val79以原报告；regression51；history独立 | reference不是复用预测选项；不得混指标 |
| split | main/paired带split，regression不带 | challenge/val带split，regression/history不带 | 保留来源审计集合 |
| 环境 | 父env＋PCI_BUS_ID、UUID、OMP/OPENBLAS=4、/tmp/densekp-trial-mpl | 相同但 /tmp/densekp-box-mpl | 比较env delta，勿保存敏感父env值 |
| GPU preflight | 原无内置核查 | 每个任务两次nvidia-smi，核返回码/UUID，不保证设备空闲 | 新执行统一先查设备/进程，拒绝不匹配；不沿用旧GPU状态 |
| only | 无 | 读现有execution list，按固定任务顺序筛选；未知名当前零任务 | 首轮不顺带改变筛选/错误规则；若要改未知名拒绝单列行为变更 |
| 失败 | 每项写execution后raise，无retry；postprocess check=True | 同样stop，无retry；only不自动跳过成功项 | stub覆盖非零、缺报告、已有out、缺execution、GPU失败 |
| 汇总 | paired汇总，再compare_source_trial；两项未写入batches | compare_box_robustness原来需独立调用 | 保留；新计划可展示“独立后续命令”，未经显式选择不自动执行 |

保留当前 summary CLI，不再新增第二个汇总入口。suite JSON 只存任务顺序/输入角色/显式固定路径或有限根参数，不存评价公式和门槛副本；Python 两个 builder 负责有界 argv 规则，避免同一数据定义两处维护。新入口的默认计划也要展示显式/默认model_config区别、将产生的路径和独立汇总步骤。

活动记录 `box_robustness_400_20261001/status.json` 指向 data_v2、evaluation_v2、reviewed_case.json、comparison_final/report.json、final_audit_v2.json。198/79父行、400exposures、18目标143框、567新预测、51历史回归是已有记录。original_four_gates 为3/4，deployment_replacement_accepted=false；本轮没有重评、更没有部署认可。

旧 execution 中 val_C 留有一次非零退出。当前公共 candidate inputs 已按 labels 筛 reference；保存恢复工具说明过宽参考集合问题，不能据旧失败记录认定当前仍有相同bug。保留 execution、原case和reviewed_case；不把恢复报告自动化为“失败后静默继续”。

### 5.2 保留与暂缓的评价文件

- 保留 evaluate_candidate_baseline、evaluate_ds_box_control、evaluate_ds_occlusion_control、summarize_source_occlusion_pairs、compare_source_trial、compare_box_robustness。普通试验的4门槛仍为18.449216、11.746464、4.156706、5.355821，对应后排/遮挡/完整/可见；来源和公式以当前 compare 实现为准，不在suite中复写。
- paired 汇总、box common-mask（至少10点）、大于32严重误差规则、等框/等目标/按点加权、22可见点历史控制分别保留；ERP分辨率映射不改。
- summarize_saved_candidate_predictions 保留为有明确前置条件的历史恢复工具，不是通用推理替代器；不会被默认suite调用。
- **尚存真实耦合**：evaluate_ds_box_control 从 evaluate_ds_occlusion_control 导入 numpy/torch/Image/DatasetKeypoints/DenseKP/Hydra/restore/draw 等名字。本轮只记录，调度迁移不改评价实现；不得宣称所有 evaluator 已完全脱离脚本依赖。另批需要授权。
- checkpoint、model_config、source group、active index、reference路径仅静态核查；不读取权重。本计划不改变 state_dict 或训练参数。

独立调度基线：迁移前在全新验收目录用受控小JSON驱动原 main，替换 subprocess.run 为只记录参数/返回值的桩，并阻断 Popen/os.system/模型 import；允许原入口只在该新目录创建自己的日志/快照。先保存原源码和轨迹，再改代码。失败/部分重跑/两cwd各有明确用例；记录时间可用固定clock，路径仅映射验收根。新默认展示不许有日志/快照写出。这是以后实施步骤，本轮没有运行桩。

保存预测回放可验证指标/报告/门槛，不能验证命令真实启动、CUDA精度或原权重加载。调度-only改动不需要三权重推理或400步训练；未来真实GPU工作另行检查设备、进程并授权。

## 6. C：相交公共化与失败工具测试依赖

### 6.1 最小分离方案

crossings 位于 `tools/validate/commuter_intersections.py:11–47`，本身只有函数、numpy/mathutils BVH依赖；建议迁到 `src/densekp_retrain/rendering/blender/intersections.py`。BVH结果顺序、自相交共享顶点过滤、float64坐标、int32 pairs、双向边测试、det/边界阈值、最后命中位置覆盖、空集合和 coplanar unresolved=None 全部保留。保持这是 Blender/mathutils 边界，不让轻量导入链顺带 import 它。

必须同时直引新模块的9个文件：

1. tools/validate/diagnose_adult_contacts.py
2. tools/experiments/diagnose_commuter_clothing.py
3. tools/experiments/repair_commuter_top.py
4. tools/experiments/settle_commuter_top.py
5. tools/experiments/build_commuter_polo_collar.py
6. tools/experiments/validate_commuter_union.py
7. tools/validate/audit_commuter_pose_sources.py
8. tools/validate/audit_commuter_component_contacts.py
9. tests/test_commuter_intersections_blender.py

这里8个工具中包含B4待退役工具，B3只能改导入及必要的 source-root 路径，不能提前删失败支线。它们当前有仅插ROOT的入口；Blender Python不一定装了editable包，需要在原执行位置加入 ROOT/src，不能依赖本机安装来掩盖。导入必须仍放原main内或原顶部位置，不提前加载资产/bpy。

第7、8项 main 的 AST 在 `tests/fixtures/legacy_import_bodies_v1.json` 中冻结，`ImportBoundaryTests.test_recipe_direct_execution_body_unchanged` 会比较。**原fixture字节必须不变**；建议在 `tests/test_import_boundaries.py` 中仅对这两文件，在实际 AST 校验确切的新ImportFrom与新增src路径语句后，将这两个已核准节点反映射到旧AST进行原比较。映射必须具体到文件、位置、模块和参数，不允许泛化忽略imports/sys.path；其余AST一字不松，inert导入测试继续保留。路径新增本身也有明确断言。如果不接受这种有限契约更新，暂缓两个审计调用方及旧crossings文件删除至B4，不能留断裂入口。旧规划遗漏此处前置条件，本计划已修正。

DQS/scan 只被自己的失败制作 main 和测试使用，当前生产的 simple shell/groom 不走它们。建议新增一个 `tests/blender_geometry_fixture.py`，仅保存 qmultiply、dual_quaternion_binding 和 scan 三个必要函数；不复制 capped、repair main、投影/体素制作/CLI，也不让运行代码导入 tests。原失败工具内实现暂留到B4，测试不再导入整套失败工具。

独立性：DQS预期仍为解析刚体矩阵和[-1,0,0]符号对齐结果，不由新helper生成期望；scan预期仍为重叠体内外的已知布尔值/未闭合计数。迁移前另存原实现的小输入完整输出作独立对照。helper 是被冻结的历史算法参考，不证明当前生产衣物采用DQS或质量通过；不能把公共被测实现与同源计算期望互相比当成独立验收。B4可再评审纯历史DQS/scan测试是否保留，不在本批悄悄删断言。

### 6.2 仍保留的 B4 依赖与停止点

repair_commuter_top 的 edges_of/average/project 仍被 validate_commuter_union、settle_commuter_top、build_commuter_polo_collar 消费；commuter_union_volume 的 capped/main 仍是失败方案CLI。B3不把这些全部搬到公共包或tests，也不删除它们。仅在所有9条import和有限冻结AST检查完成后，删除 commuter_intersections 原文件。退役它的旧Python导入，不新增兼容包装。

7个长期数值测试全部保留，Blender下逐项运行（未来）：
`TriangleCrossings.test_edge_pierces_face_while_vertices_are_outside`、
`test_disjoint`、`test_shared_edge_is_not_self_crossing`、
`test_coplanar_overlap_remains_unresolved`、
`test_oriented_scan_keeps_overlap_instead_of_xor`、
`test_dqs_exact_rigid_transform`、`test_dqs_quaternion_sign_alignment`。
原DQS atol=2e-7和默认rtol不放宽；其余原断言保持。

未来命令为 `<已核实Blender> -b --factory-startup --python-exit-code 1 --python tests/test_commuter_intersections_blender.py`；先核对同一Blender/mathutils版本。只用合成三角形和临时cube，不开源blend、不读人物、不渲染。本轮没有查设备/运行Blender；缺Blender或skip应记未覆盖，不能以普通CPU import成功替代。

## 7. 推荐实施批次与准确首批范围

### B3.1：参数组装与衣物保存，推荐先做

运行代码现有文件仅7项：
- src/densekp_retrain/body/parameters.py
- tools/experiments/audit_pedestrian_poses.py
- tools/experiments/audit_new_pedestrian_source.py
- tools/validate/freeze_mixed_candidates.py
- tools/experiments/prepare_adult_candidates.py
- tools/experiments/prepare_source_isolated_assets.py
- tools/experiments/prepare_mixed_pedestrians.py

新增运行文件仅2项：clothing/preparation.py（位于src/densekp_retrain）、tools/experiments/adult_candidate_recipe.py。
新增测试 `tests/test_source_asset_preparation.py`；小型长期期望如确需序列化，只加一个 `tests/fixtures/source_asset_preparation_v1.json`，从旧实现先冻结且不能期望生成自新实现；真实角色大数组只放新验收目录。必要导航仅 docs/SAMPLE_WALKTHROUGH.md、docs/rendering/SCENE_RECIPE_INDEX.md；实施报告和一次性证据放新的对应目录。

**本批不删除任何整脚本、不迁groom、不跑制作main。** 先 A-PARAM/A-CLOTH 独立before，再迁消费者，再精确对照、相关来源契约、导入闭包/无资产导入检查。所有图像/几何资产只读；衣物writer只写验收目录。SMPL-X实例化不在首批默认范围。暂停条件：额外动态消费者、缺输入、数值/异常差异或所选参考输入不足；不修算法/旧期望。

### B3.2：groom 与固定 mixed 入口退役

新增 clothing/groom_transfer.py；source入口直引；新增测试沿用同一个 source_asset_preparation 测试文件的有界groom用例，避免每函数一套框架。采集 A-GROOM，核验旧mixed report/27行索引读取完整及静态来源关系。完成后删 prepare_mixed_pedestrians.py，更新同一导读/索引；不删任何mixed输出、渲染入口、freeze工具。旧groom import与固定JOB CLI停止支持。此批无需Blender/模型，但完整绑定CPU回放需单独列实际资源规模。

### B3.3：相交公共化与测试依赖

准确文件：新增 rendering/blender/intersections.py 和 tests/blender_geometry_fixture.py；修改§6列9个消费者及 tests/test_import_boundaries.py；删除 tools/validate/commuter_intersections.py；只更新相关测试用法和诊断导航。先小型真实Blender独立before，后7测试/冻结AST/导入边界与所有CLI路径闭包。失败制作工具main的计算不改，资产制作不运行。未具备Blender时可做依赖准备，但不能将数值验收记完成；也不能先删桥接实现留断引用。

### B3.4：薄回归调度器

新增 tools/validate/run_regression_suite.py、configs/regression/suites.json、tests/test_regression_suite.py；删除 evaluate_source_trial.py、evaluate_box_robustness.py（均在tools/experiments）；必要更新 docs/COMMANDS.md、configs/README.md、docs/READING_GUIDE.md 和现行场景索引。按§5两suite执行规则冻结B-PLAN，再实现默认展示与显式执行；不改各evaluator/summary。B-REPLAY按真实改动选择，若只是argv组织，不为数量重跑全部回放。完整保留活动index/status/checkpoint和门槛。

四批不是一次性删除授权。B3.3、B3.4与A链可独立验收；优先B3.1，因为真实消费者和CPU接口最清楚。拟整文件删除候选仅原规划4个，均有前置条件：mixed、两个调度器、commuter_intersections；**本轮没有删除任何一个**。audit_pedestrian_poses、prepare_adult_candidates等有独有制作用途，不能因helper迁出就删整文件。

## 8. 后续有界检查与交付条件

以下仅列实施时的测试ID/方式，本轮一项未运行：

- 来源：`tests.test_source_packing_contract.SourcePackingContractTests.test_declared_sources_all_scene_people_and_pair_failures_match_before`。输入原合成fixture，仅CPU，无更新；保持人物/遮挡者泄漏拒绝。
- 几何：`tests.test_geometry_camera_contract.GeometryCameraContractTests.test_frozen_array_and_rejection_results`、`test_clothing_lift_uses_body_and_does_not_modify_either_mesh`。只在body/clothes依赖受影响时选用，不能当真实SMPL-X重建。
- 导入：`tests.test_import_boundaries.ImportBoundaryTests.test_recipe_direct_execution_body_unchanged`、`test_imports_are_inert_without_optional_dependencies`，尤其C批2处精确契约映射。
- 新参数/衣物/groom测试放同一文件，计划方法 `test_pose_blocks_preserve_dtype_alias_and_errors`、`test_clothing_artifacts_and_failure_order`、`test_groom_binding_masks_and_failure_order`，定义后先审调用闭包再执行。它们目前不存在，不能写为通过。
- 调度新方法计划 `test_suite_argv_order_and_context`、`test_plan_has_no_writes_imports_or_processes`、`test_execute_failure_and_partial_resume`；模型构造/权重读取/真正subprocess/GPU接口应被阻断，执行模式仅以桩验证组织逻辑。
- C批7个Blender方法见§6；不要运行unittest discover或整库生命周期测试。未来普通CPU命令以 `PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=.:src python -m unittest <上列具体方法ID>` 为界；根/临时cwd需要绝对repo/src路径，不能误靠outputs或本机editable掩盖导入问题。

每批记录源码/小输入独立before、数值/异常/写出轨迹、实际方法ID及skip；快照不用SHA256，原fixture字节比较。只复制当批必要小输入，不逐批复制整历史验收库。数组和整数精确；原有浮点断言保留原容差；任何新误差先定位，不能靠新宽限阈值通过。路径、时间等允许变化只能列具体JSON字段，不能整体忽略manifest。

无资产源码副本只验证轻量模块、默认计划、合成CPU契约；groom/SMPL-X/Blender依赖明确列出，缺资产不伪造通过。采用既有环境不安装升级；源码副本、editable导入和空环境安装是三个不同结论，没有实施对应步骤就不声称通过。新增公共包/配方/测试/fixture/suite要查git ignore与安装交付引用，ignored证据不是备份。

所有实施批保留TODO、B1/B2、原fixtures、历史失败制品和许可/来源记录。衣物原有全零材质ID、groom空mask异常、当前评价恢复记录都不是偷偷修复项。B3不动constants/注册表/Hydra/裁剪/loss/训练，不进入B4失败支线批量删除和B5历史文档清理。

## 9. 待评审的少数取舍与真实限制

1. 推荐独立的无副作用 STYLES 配方数据文件；比保留公共writer反向读adult脚本更清楚，也避免为两消费者建资产配置框架。接受这一点后B3.1即可实施。
2. 推荐一个薄入口＋两套显式suite，但保留两种执行/记录策略；默认展示/--execute是有意的新CLI规则，旧两个入口退役。不要以统一runner为由统一指标或默认model_config。
3. C批推荐把B4工具的相交import一起改完，并对两个冻结main作严格有限映射；若希望B4文件一律不动，则保留旧相交模块到B4，B3不能宣称它已删除。

目前选定输入没有发现缺失，B3.1没有已知阻断依赖。真正的实施前置条件是独立before、原fixture保护和明确的CPU写出范围；C批另需真实Blender小型验证。已存大绑定尚未载入数值，真实制作main/SMPL-X/groom/Blender均未在本轮运行，因此没有完整链路“已通过”的结论。**停在规划评审点。**
