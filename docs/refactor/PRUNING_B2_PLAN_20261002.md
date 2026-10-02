# B2 实施前核查与任务清单

日期：2026-10-02。状态：**只读核查完成，待评审；未实施 B2，未运行任何测试或项目计算。**

本报告收窄并具体化 [原精简计划](PRUNING_PLAN_20261002.md) 的 B2；第 11–12 节的 B1 修订继续有效。[逐文件、逐符号清单](../../outputs/plans/pruning_b2_20261002/inventory.json) 包含定位、原配置内容、注册表逐 key 记录及后续测试 ID。该 JSON 被 outputs 规则忽略，不视作 Git 备份；可执行要求同时保存在本文。

## 1. 实际基线与核查边界

- 分支 `master`，HEAD `e7905904fe84b7a1b107485fc681a62ced6fccda`。B1 已由用户评审通过，仍未提交；工作树中有其 10 个文件删除、消费者/测试/导航修改及新增文件。并非只有 TODO 修改。
- 用户 `docs/TODO.md`、B1 所有现存修改、原 fixtures、原计划及验收清单均保留。本轮只新增本文和 B2 inventory；保护记录包含起始 status、文件大小/mtime、TODO 文本，结束再核对。不计算 SHA256。
- 阅读了用户指定六项材料、B1 manifest、相关实现与测试。范围限定于 densekp、src、tools、tests、scripts 的源码，configs、安装定义及直接相关文档；outputs 仅读取具名清单/小 JSON、检查明确引用的文件存在性。没有遍历整个 outputs、权重或纹理，没有读取外部项目。
- AST import（含相对 import）、动态 dispatch、Hydra defaults/选择、CLI 调用和文档引用分别记录。普通 `json` import 不映射到项目 JSON 模块；`losses` 等字段不作为动态依赖。静态未发现调用，只是证据之一；退役接口必须同时明确停止支持的用途。
- B1 前测误运行小网络 4 次 CPU 更新的记录保留。**本轮没有执行测试、Hydra 组合、Dataset、真实模型、参数更新或数值验收。以下命令全部是实施获批后才执行的计划。**

## 2. 精确文件范围

建议分两个独立验收部分：**B2a 配置整理；B2b 裁剪调用、旧评分入口与限定符号清理。** 可以先仅批准 B2a 的父配置改名（旧父配置、四个继承者、新父配置及对应文档/验收），其余继续按下列清单评审。

共 **16 个整文件删除候选，原文件合计 1292 行**：其中 1 个是改名后的原文件，8 个配置/示例明确退役，7 个裁剪/评分/专用测试文件完成前置处理后删除。不是净减行数或本轮删除数量；不含 B1 已删的 10 个，也不包含任何 fixture。新增运行配置仅 `configs/hydra/trainer/loop.yaml`。建议新增长期测试 `tests/test_pruning_b2_contract.py`；一次性采集器放新验收目录。

| 文件 | 分类 | 真实消费者、处理及停止支持内容 | 验收 |
| --- | --- | --- | --- |
| `configs/hydra/trainer/default_hmr.yaml` | 迁移后删除 | cpu/gpu/ddp/mps 的 defaults；原字节迁 loop.yaml，停止旧父组名 | §4 四组合、默认、override 组合 |
| `configs/hydra/data/train.yaml` | 直接删除候选 | Hydra `data=train` 动态入口；AGORA/BEDLAM、Insta/COCO/MPII/AIC 混合预设 | 留存原配置快照；其它组合不变；旧组明确拒绝 |
| `configs/hydra/data/densekp_train.yaml` | 直接删除候选 | `data=densekp_train`，AGORA/BEDLAM bbox44 + EMDB；旧格式文档有引用 | 同上；不删除其 dataset ID/NPZ 支持 |
| `configs/hydra/data/bedlam_v2_v1.yaml` | 直接删除候选 | `data=bedlam_v2_v1`，BEDLAM v1/v2 + EMDB/3DPW/RICH 预设 | 同上；不废弃 BEDLAM 标注生成/验证 |
| `configs/hydra/data/eval.yaml` | 直接删除候选 | `data=eval`，训练选择为 null，EMDB/3DPW/SPEC/RICH/COCO 验证组合 | 同上；不是删除所有评价能力 |
| `configs/hydra/data/eval_smplx.yaml` | 直接删除候选 | `data=eval_smplx`，EMDB/3DPW/RICH 验证组合 | 同上 |
| `configs/hydra/experiment/camerahmr.yaml` | 直接删除候选 | Hydra 可动态选择，继承 default，含 SMPL 参数目标/超参 | 停止该组名；不据名称推断无法组合 |
| `configs/hydra/experiment/bedlam2.yaml` | 直接删除候选 | Hydra 可动态选择，继承 default，含 SMPL-X/手部目标/超参 | 停止该组名；densekp/default 配置不变 |
| `configs/datasets.example.json` | 直接删除候选 | 早期 train/val 路径格式文档示例；当前解析器未读取，键名也不是 Dataset 构造接口 | 更新当前配置导航，保留 annotation_path/image_root 说明 |
| `densekp/core/datasets/utils.py` | 迁移后删除 | visualize、BEDLAM validator、低层/边界测试和旧 capture 依赖；Omni 评分入口一起退役 | §3 所有消费者先迁；停全部旧 re-export/tuple 导入 |
| `densekp/legacy/crop.py` | 迁移后删除 | utils re-export 和 legacy 专用测试；不同算法不能伪装成 current crop 的别名 | 明确停止 §3.3 旧能力；现行数值测试保留 |
| `densekp/legacy/__init__.py` | 迁移后删除 | B1 train_utils 已删，crop 是最后实现 | 包发现不再含 legacy；不修改双包根或清理用户缓存 |
| `tools/validate/evaluate_omni_densekp.py` | 迁移后删除 | CLI 与 geometry fixture 的动态 erp_points 路由 | 先迁动态引用；停止旧三相机评分，保留 Omni 数值回归 |
| `tools/validate/evaluate_densekp_npz.py` | 迁移后删除 | 旧注册 ID 评分 CLI；历史文档命令 | 明确旧评分接口退役；不宣称 candidate 是相同分母替代 |
| `tests/test_legacy_cleanup.py` | 迁移后删除 | B1 后仅测 legacy crop，日常 MODEL_TESTS 引用 | 删除旧能力断言和日常列表项；归一化/导入安全仍运行 |
| `tests/legacy_cleanup_fixture.py` | 迁移后删除 | 上述旧 crop 专用 capture 辅助 | 最后消费者消失后删除；原冻结 JSON 不变 |

“直接删除候选”表示无需搬迁实现，不表示已获本轮执行授权；导航退役说明和原内容快照仍是前置要求。

**需修改的现存代码/测试（不额外增加整文件删除）：**

| 文件 | 修改内容与依赖依据 |
| --- | --- |
| `configs/hydra/trainer/{cpu,gpu,ddp,mps}.yaml` | 各只替换第二个 defaults 父名，内容覆盖顺序不动 |
| `densekp/core/datasets/crop_pipeline.py` | 仅在最后消费者迁走后删 `CropResult.legacy_tuple`；各处理阶段不改 |
| `tools/annotation/visualize_annotation.py` | `main` 直调 crop_example/CropBox；另两函数直引定义模块，见 §3 |
| `tools/validate/validate_bedlam_densekp_tiny.py` | trans_points2d_parallel 改直引 crop_geometry；实际 audit_cfg 调用保留，不承诺旧外部 re-export |
| `tests/dataset_crop_fixture.py` | run_lowlevel 读取具名结果，以测试内编码适配旧 fixture；run_dataset 旧实现 capture 分支明确拒绝 |
| `tests/test_dataset_crop_contract.py` | 低层 tuple 测试改名为具名结果测试，原断言与 11 组期望完整保留；其余 5 项不减 |
| `tests/test_densekp_boundaries.py` | crop_example / sample_augmentation 直引；不动 objective/lifecycle 测试实现，也不执行生命周期更新项 |
| `tests/geometry_camera_fixture.py` | 按原顺序将 TARGETS 改成公共模块→符号；api() 仍含 erp_points；停止 original=True 旧动态 dispatch |
| `tests/test_geometry_camera_contract.py` | 只移除旧导出 identity 测试；冻结值/拒绝、共变换、相机轴/距离、周期规则 4 项不变 |
| `tools/validate/verify_dataset_crop.py` | capture 在任何旧接口导入、读基线或 mkdir/write 前明确拒绝；verify 保留 |
| `tools/validate/verify_geometry_camera.py` | 无 --before 的历史 capture 提前拒绝；run(original=True) 也在 api/计算前拒绝；--before 验证算法保留，B2 不执行重型几何回放 |
| `tools/validate/check_workspace.py` | 仅移除 legacy_cleanup 项，加入新的有界 B2 契约；保留归一化、导入安全及 B1 有效覆盖，不扩大扫描范围 |
| `densekp/core/constants.py` | §6 的限定常量清理，四个活动常量原值原表达式保留 |
| `densekp/core/configs/__init__.py` | 仅 `_C` 构造树与 default_config 删除；注册表所有 key 暂不删，dataset_config 保留 |
| `densekp/core/utils/misc.py` | 仅三项旧工厂/取指标函数及它们独占 import；活动任务/日志/finally 顺序不改 |

当前文档修改候选逐项为 `README.md`（B2 状态）、`docs/READING_GUIDE.md`（裁剪及兼容区链接）、`docs/SAMPLE_WALKTHROUGH.md`（具名返回与框坐标）、`docs/COMMANDS.md`（退役接口及检查边界）、`configs/README.md`（组名/预设退役）、`tests/fixtures/README.md`（保留期望的实际消费者）、`docs/environment/INSTALLATION.md`（legacy 包退出但双包安装边界不变）。只作与 B2 对应的最小更新。

历史报告主体不重写。`docs/CURRENT_CODE_MAP.md` 已有历史提示，内部旧实现描述不是运行依赖；原报告与 B1 清单保留当时状态，现行导航通过 B2 报告解释退役。TODO 不改。

## 3. 裁剪迁移表与行为边界

### 3.1 参数和返回值

旧 `utils.py:65 get_example_projverts` → `crop_pipeline.crop_example`：

| 旧参数 | 公共接口 | 要求 |
| --- | --- | --- |
| img_path | 第一个参数原样传入 | str 或 ndarray；不擅自把 Path 转成 str；不追加旋转 |
| center_x/center_y/width/height | CropBox 四字段 | 源图像像素；不改变 float 转换位置 |
| keypoints_2d | keypoints | 同一个数组，保留原地写入 |
| proj_verts | dense_points | 同一个数组或 None；调用方原来的 copy 位置不动 |
| patch_width/patch_height | patch_size=(width,height) | 显式传入；不能误交换 H/W，也不加 32 px 偏移 |
| mean/std/do_augment/augm_config/is_bgr/use_skimage_antialias/border_mode/dataset | 同名关键字 | 原值和默认行为保留 |
| flip_kp_permutation | 不传 | 该包装本来未消费，不构造新语义排列 |
| return_trans | 不传 | 包装本来忽略，始终返回 affine；测试保留旧用例编码，不保留运行参数 |

| 原十项 tuple 下标 | 具名字段 |
| --- | --- |
| 0 | result.image：float32 CHW 归一化输入 |
| 1 | result.display：未归一化 HWC 裁剪图；当前 BGR 输入时为 BGR |
| 2 | result.image_size：源图 H/W |
| 3、4 | result.plan.box.center_x、center_y |
| 5、6 | result.plan.box.width、height：extreme/translation 后，尚未乘 augmentation.scale |
| 7 | result.keypoints |
| 8 | result.dense_points |
| 9 | result.affine：[2,3] |

`do_augmentation(cfg)` → `augmentation.sample_augmentation(cfg)`，原八项顺序为 scale、rotation、flip、extreme_crop、extreme_crop_level、color_scale、translation_x、translation_y。新返回为已有 Augmentation named tuple；不改随机调用或将传入 cfg 复制/归一化。

旧几何 re-export（rotate_2d、gen_trans_from_patch_cv、trans_point2d、trans_points2d_parallel、normalize_crop_xy）直引 `densekp.core.crop_geometry`；图像 re-export（expand_to_aspect_ratio、generate_image_patch_cv2、convert_cvimg_to_tensor）直引 `datasets.image_processing`；extreme_cropping、get_bbox 及各身体部分 crop helper 的原定义继续在 `datasets.augmentation`。不因旧包装消失而删除活动定义。

### 3.2 真实调用方

- `visualize_annotation.main:137` 使用下标 **1、5、6、8、9**，不仅是网络输入。保留先读图/按 dataset_name 旋转、绘制并保存原图，再调用 crop 的顺序。仍传已读 ndarray、dataset=None、44×3 float32 零辅助点、proj_verts.copy()、mean/std=None、BGR=True、CONSTANT、关闭增强/抗混叠；否则可能二次旋转或修改用于原图对照的点。继续用 plan.box 宽高写报告；不能以乘过 scale 的范围替代。
- `validate_bedlam_densekp_tiny:18,72` 只需要几何函数导入迁移，Dataset 已直用现行流程。保留 debug joints 画图、np.rint、报告字段和空集错误，不因名字 BEDLAM 淘汰此能力。
- `tests.dataset_crop_fixture.run_lowlevel:267` 保留 11 个用例的所有图像、数组、变异后输入、类型、异常与 RNG 比较。在 **tests 内**把具名字段依原下标展平到 `result_N`，匹配既存独立 JSON；不在公共包恢复 tuple wrapper，不重录期望。return_trans_flag 作为旧参数无效性的迁移记录保留同一期望；新调用不传该参数。
- `tests.test_densekp_boundaries.GeometryContractTests` 改具名取 display/dense_points/affine；既有误差断言原值不动。
- 旧 Omni evaluator 退役；不为删它另造等价评分入口。其 erp_points 动态消费者先改公共相机模块。

关键顺序：验证 flip → 读图 → 随机增强采样 → 有限/正尺度检查 → extreme crop / translation → 预滤波/Albumentations → cv2 warp → 归一化 → 辅助点和 138 点原地变换。标签先写仿射像素结果、再写归一化结果的两次 dtype 舍入不能合并。4 通道 alpha 的 border 处理、LINEAR 插值、抗混叠阈值、portrait/closeup 分支和 Python/NumPy/Torch/增强库状态不变。

低层错误也保留：Path 输入拒绝；缺图 IOError；无效宽度在已有采样之后拒绝；启用未验证 flip 在读图/采样前拒绝。旧接口退役不是修复这些行为的机会。

### 3.3 明确停止的旧能力

`legacy/crop.py` 的 resize_image、get_transform、transform、crop_img、generate_image_patch_skimage、fliplr_params、fliplr_keypoints、keypoint_3d_processing、rot_aa、get_example 和 do_augmentation 的旧导入接口停止支持。do_augmentation 的有效采样功能由现行 sample_augmentation 保留；其余旧实现没有当前业务消费者，专用 legacy 测试随之退役。

旧 get_example 的 9/10 项返回布局、RGB display、按 patch_width 同时归一化 XY、旧抗混叠比例和 flip/异常顺序，与 get_example_projverts/current crop 不同；**不声称可用同名包装等价替代**。停止其 SMPL 参数翻转、3D 关节处理、skimage/PARE 裁剪及旧 tuple 复现接口。已跟踪历史可查 Git；历史 NPZ、已有图片和监督权重的读取继续支持。

## 4. Hydra 继承与退役

`train.yaml` 顺序保持 `_self_ → data:unselected → trainer:gpu → paths:default → hydra:default → experiment:densekp → texture_exp:null → debug:null`。不补新的 `_self_`，不改变默认 data、安全拒绝、未实际使用字段或配置解释。

| trainer | 现 defaults → 建议 defaults | 自身覆盖（原样保留） |
| --- | --- | --- |
| cpu | [default.yaml, default_hmr.yaml] → [default.yaml, loop.yaml] | accelerator=cpu, devices=1, precision=32-true |
| gpu | 同上 | accelerator=gpu, devices=1；继承 precision=16-mixed |
| ddp | 同上 | strategy=ddp, accelerator=gpu, devices=3, num_nodes=1, sync_batchnorm=true |
| mps | 同上 | accelerator=mps, devices=1；继承 precision=16-mixed |

default.yaml 的 `_target_=pytorch_lightning.Trainer`、default_root_dir、deterministic、默认设备内容不动。loop 原样包含 num_sanity_val_steps=0、log_every_n_steps=${GENERAL.LOG_STEPS}、val_check_interval=${GENERAL.VAL_STEPS}、precision=16-mixed、max_steps=${GENERAL.TOTAL_STEPS}、limit_val_batches=1.0。

五个退役 data 组和两个 experiment 组当前确实能动态选择。没有当前默认选择它们，不等于它们不可运行；删除是明确撤销这些命名预设。保留六个 data 文件：unselected 与五个 tiny 配方；**六个本地 registry ID 不等于六个 tiny YAML**，boundary ID 没有对应独立 data YAML。保留 experiment/default、densekp 及所有当前覆盖参数。

后续组合矩阵：默认 + 四 trainer × 六保留 data 组；另测现有 densekp/default 实验选择、default trainer，以及显式覆盖 GENERAL.TOTAL_STEPS/LOG_STEPS/VAL_STEPS、trainer.precision/devices、DATASETS 路径选择字段。旧父单独选择与新 loop 单独选择也比较，但不推荐直接选择无完整 Trainer 父配置。每种均从根目录和临时 cwd 组合，不实例化 `_target_`。

先保存原 YAML、defaults 序列、未解析/解析 task config；新旧运行值和类型逐项相等。仅允许父配置**来源标识**从 default_hmr 变为 loop；固定输出目录/cwd/运行时间等 Hydra 上下文后比较，不能笼统忽略所有 hydra 字段。显式退役组的 MissingConfigException/CLI 非零退出属于宣布的接口变化。

组合成功与解析成功分开记录。例如单独选 experiment=default 可能缺少 GENERAL 等被引用字段：以独立 before 的实际解析异常为准，比较类型与原因，不为让验收通过补默认值。保留组合既有失败也属于需要保持的行为。

## 5. 两个旧评分入口

| 入口 | 独有行为与依赖 | 保留能力与明确退出 |
| --- | --- | --- |
| evaluate_omni_densekp | 三相机 P0；鱼眼框来自 >=100 像素 mask、1.2 倍 tight square；ERP 可见点最大周期间隙居中、凸包膨胀；鱼眼 visible 子集共享到 ERP。评分是 visible & in_crop & finite(error)，非有限误差退出分母 | 停此自动造框/居中和配对模型评分 CLI/报告格式；Omni 生成、warp_case、camera.erp.erp_points、周期投影和旧保存数值回归继续保留 |
| evaluate_densekp_npz | 从注册 ID 构造 Dataset，显式目标语义；二值标签检查、任意非有限预测拒绝、零正权重点拒绝；按 weights>0 逐人 mean/median/p90，报告称 visible | 停旧 ID 评分 CLI；不改历史 NPZ 权重读取，不将历史 visible 字段改写为当前监督定义，不重评旧结果 |

当前 candidate/index 入口覆盖固定样本推理、成对汇总；DS 控制入口支持保存预测重放。它们**不是上述两入口的逐参数/分母等价替身**。若用户仍要求旧 Omni GT 自动居中或注册 NPZ 全集评分，应暂停对应删除，另评能力需求，不偷偷迁入当前指标。

现有三权重元数据和严格加载证据引用原计划 §5 的已核查记录；本轮未重新读权重或加载模型。此次只删除函数/配置入口，不改 DenseKP、state_dict、优化器、checkpoint callback 类。旧完整对象 Python 接口不承诺恢复；仍需使用的三份 state_dict 权重及其配置字典不受配置文件改名直接影响。使用退役 Hydra 组名重新构造启动命令则会明确失败，需采用保留配置并显式给原参数。

## 6. 限定符号、注册表及暂缓

- `constants.py` 保留 **VITPOSE_BACKBONE**（densekp_trainer:13）、**TRANSFORMER_DECODER、NUM_DENSEKP_SMPL=138**（heads/smpl_head_keypoints:7）、**NUM_JOINTS=44**（dataset_keypoints:11）。NUM_JOINTS 是辅助关节数，不能改成 138。inventory 逐一列出其余定义与内部引用闭包：旧 checkpoint/SMPL/检测器路径、IMAGE_*、NUM_* 参数常量、H36M/SMPL 映射、permutation、JOINT_* 和 MANO 均为删除候选；同名 YAML 字段不是 Python 符号消费者。保留的四个定义 AST/值必须原样。
- `core/configs`：default_config 只 clone `_C`；当前训练/审计导入的是 dataset_config。删除 `_C` 初始化树和 default_config；保留 CN、dataset_config 与所需 import。**to_lower 暂缓**，不在本批原明确符号集合内。
- DATASET_FOLDERS / DATASET_FILES：当前 loading.resolve_source 仍动态索引，DataModule._names 接受配置字符串/列表；去掉五个预设后，显式 override 仍能合法选旧 ID。**本批暂缓全部非 tiny key 的删除**，逐 key 值表达式、train/val 所属、重复覆盖顺序记录在 inventory；六个 tiny ID 明确保留。不能把“只有旧预设静态出现”作为排除动态用途证据。也不改已有路径基准和重复 key 的最后赋值语义。
- `misc.py` 只删 instantiate_callbacks、instantiate_loggers、get_metric_value，及仅被其使用的 hydra、List、Callback、Logger imports。当前 train_keypoints 自行构造 callback/logger，导入 task_wrapper/log_hyperparameters；保留 extras、save_file、close_loggers 及调用关系，不执行装饰器去验证，以免写日志或启动生命周期。对保留函数作 AST 相等检查。
- 旧 DataModule 的其它 HMR 分支、人物制作工具、B3/B4 包装、透视/相机压力整条能力均不纳入。本批不根据附带发现扩大清理。

## 7. 跨批次关系与 fixtures

1. **Dataset capture**：verify_dataset_crop 当前先 mkdir 再因当前 Dataset 签名拒绝 capture；run_dataset(original=True) 还导入旧 utils。B2 删除 utils 前把 capture 的拒绝移到读数据/写文件/旧导入之前，辅助函数也拒绝旧模式。verify 使用 current Dataset + 独立 before，继续有效。不要使用旧 capture 当作本批 before 采集器。
2. **Geometry capture**：api(original=True) 真正通过 importlib 导入旧 Omni evaluator；完整 capture 已因 evaluate 来自公共模块而不能作为迁移前采集，却仍可能先导入/建目录。按原计划将 fixture 路由改成公共符号表，停止 original=True，移除纯兼容 identity 测试，保留四项行为测试。工具 CLI 和 run 的旧 capture 先拒绝；--before 的实际验证计算顺序保持，重型几何验收不在本批运行。
3. **Metrics/config capture**：B1 已明确拒绝，verify 只用公共模块，本批不再改；保留原 B1 拒绝测试。B4 才退役一次性工具整文件，B2 只清其必要跨批分支。
   `tests/test_evaluation_contract.py:test_historical_imports_remain_available` 还检查 BEDLAM validator 的 audit_cfg 身份；该入口仍实际调用并导入公共 audit_cfg，B2 不移除这项实际导入，因此测试没有断裂。不能为了撤销外部兼容承诺误删该入口正在使用的配置函数；其它包装身份清理留后续批次。
4. 原 `dataset_crop_v1.json` 的 31 Dataset + 11 低层用例仍全部有意义。`legacy_cleanup_v1.json` 中 legacy get_example/SMPL 专用部分停止断言，但 denormalize 是 B1 长期测试输入，**整个文件保留原字节**。`legacy_import_bodies_v1.json` 继续被 B1 导入安全/执行体测试使用。其它所有原 fixtures 一律不改不删。
5. setup.py 自动 find_packages，未硬编码 legacy，pyproject 也未指定该子包；预计不需修改构建定义。删除最后实现和 __init__ 后检查包发现不含 legacy、有效 imports 不依赖它。不要把工作树残留 __pycache__ 导致的 namespace 现象误认为仍需保留实现；也不要为清理缓存修改额外文件。

## 8. 独立证据与后续验收矩阵

### 已只读核查的证据

- `outputs/acceptance/refactor_dataset_crop_20261001/selection.json`：七例为 val_baseline_002、val_baseline_001、val_baseline_004、challenge_000、challenge_001、challenge_005、challenge_007；对应 loader 文件、scene 目录及七例原/增强 before NPZ 都存在。
- 同目录 specifications.json 共 45 例（31 合成 + 七例各关闭/开启增强）；增强七例 seed=930，scale=.2、translation=.15/rate=1、rotation=25/rate=1、color=.2。before/data 为独立迁移前实现结果；after_final/result.json 记载旧 ID/显式路径 90 路、2814 数组通过；lowlevel_after/result.json 记载 11 路/156 数组通过。**这些是历史结果，不是本轮复验。**
- 已检查 frozen JSON 的结构、用例和消费者；没有解码/重算全部数组。Omni 旧数值回归所指 camera_config 与 4.9 MB vertices 文件存在；未来必须检查该测试没有 skip，不能把 skip 当通过。
- `datasets/densekp_bedlam_tiny`（文档路径）与 `datasets/bedlam_densekp_tiny` 均不存在。不能宣称真实 BEDLAM 回读已可执行；不搜索全盘或补生成资产。后续以受控小输入检验 validator 导入迁移和图像/报告，真实数据验收保留此限制。

### 未来执行约定

定义 `R` 为仓库绝对路径，`PY=/home/yy/conda_envs/camerahmr/bin/python`，`A=$R/outputs/acceptance/pruning_b2_<新执行标识>`，`P=$R/outputs/acceptance/refactor_dataset_crop_20261001`。A 必须不存在；before/after 子目录仅写新 A，不覆盖 P、fixtures、B1 或历史输出。设置 PYTHONDONTWRITEBYTECODE=1、CUDA_VISIBLE_DEVICES=''。配置命令只 compose/print，**train=False 不能代替 --cfg job，也不调用 train()**。

| ID | 具体未来命令/输入 | 读写与断言 | 参数更新 |
| --- | --- | --- | --- |
| V01 | `$PY -B tools/validate/verify_dataset_crop.py --phase verify --acceptance "$P" --output "$A/before/dataset"`；迁移后同命令写 after | 读45规格/P before及指定图片NPZ；写90组小结果；完整 img、GT、affine、mask、dtype/shape、采样和各 RNG 精确；先运行before再编辑 | 无 |
| V02 | `run_lowlevel` 当前版的11例先由新 A 中 `capture_current.py --part lowlevel --output "$A/before/lowlevel"` 采集；迁移后该驱动走新具名接口；两次都对原 frozen JSON | 保存旧wrapper源码/输入/结果，再迁；测试内映射不得重录fixture或放宽断言 | 无 |
| V03 | `$PY -B tools/annotation/visualize_annotation.py <selection.loader_input> <selection.scene> --index <loader_row> --count 1 --output "$A/before/visual/<id>"`，七例逐一；迁移后写 after | 参数须按当前 argparse 命名（npz、image_root 为位置参数）；RGB/BGR裁剪与标注 PNG 字节、报告字段精确；补 portrait/closeup、相对/绝对路径、错误/部分输出用例 | 无 |
| V04 | `$PY -B tools/validate/validate_bedlam_densekp_tiny.py "$A/inputs/bedlam" --output "$A/before/bedlam"`，after重复 | 执行阶段仅在 A 制作2行小图/NPZ/debug（joints2d、densekp_in_image_count、densekp_in_bbox_count、surface_operator_max_abs_error）；同一输入先冻结 before。不是新增 BEDLAM 资产或真实数据验收 | 无 |
| V05 | 新采集器 `$PY -B "$A/capture_current.py" --part hydra --output "$A/before/hydra"`，after同矩阵；长期测试见下 | Hydra initialize_config_dir+compose，不 instantiate。根/tmp cwd、§4矩阵、值/类型/defaults顺序精确；固定Hydra上下文，来源名变化单列 | 无 |
| V06 | `python -B -m unittest <inventory.validation.safe_test_ids 逐项ID>`，必须展开为具体ID | 见下方范围；临时小数据及原 fixtures；拒绝运行整个 boundaries 模块或 discover | 无；Objective 部分仅求梯度 |
| V07 | 新 tests.test_pruning_b2_contract 的两个 capture 拒绝测试 | 子进程 audit hook：旧 utils/evaluator 导入、mkdir、write、模型构造/生成均阻断；无合法 before 时先清晰失败；verify 分支仍可达 | 无 |
| V08 | Omni `tests.test_numeric_regression.FrozenNumericRegressionTest.test_fisheye_densekp`；geometry 的4项有效契约 | 仅已有小数组，原 1.3e-4 投影容差不改；不是新旧模型预测对照。若缺文件 skip，验收记未覆盖 | 无 |
| V09 | `$PY -B tools/validate/verify_installation_runtime.py --mode data --output "$A/runtime/root"`；临时 cwd 以脚本绝对路径再运行写 tmp 子目录 | 仅既有依赖下双包import/配置/小CPU数据。另将保留源码/config/fixtures复制至A无资产checkout，显式源码路径导入；绝不从原项目偷偷找资产 | 无 |
| V10 | AST/配置/命令依赖核对 + `git check-ignore` 新测试/loop.yaml；setup 包发现；当前导航链接核查 | 验证旧模块不存在且无活动导入/动态 dispatch；原fixture逐字节比较 before快照；B1/TODO前后字节/差异保持。此项不依赖模型 | 无 |

V02/V05 的 capture_current.py **是实施阶段待编写的一次性验收驱动，不是现存命令**；必须先完成并审查其只读输入/新目录输出，再对当前实现采集 before。不得调用旧历史 capture，不得在旧新函数都转发新实现后才采集。

V06 明确白名单（inventory 列出每个完整方法 ID）：DatasetCropContractTests 六项；GeometryContractTests 六项；ObjectiveTests 五项；LifecycleTests **仅** test_configuration_requires_data_and_rejects_unsupported_controls；GeometryCameraContractTests 四项有效数值测试；ImageNormalizationTests 一项；ImportBoundaryTests 三项；保留现有 occluded supervision 和 evaluation 有效测试。重命名低层方法时仅改 ID/具名字段读取，不改期望。

建议新增 `tests/test_pruning_b2_contract.py` 的方法：

- `PruningB2ContractTests.test_hydra_parent_and_retained_combinations`：原父配置快照来自实施前 A；长期断言既有六字段与默认/覆盖语义，不以 after 生成 expected。
- `test_dataset_capture_rejects_before_legacy_imports_or_writes`、`test_geometry_capture_rejects_before_legacy_imports_or_writes`。
- `test_retained_modules_do_not_import_retired_paths`、`test_package_discovery_excludes_retired_legacy`。
- `test_visualization_named_result_matches_frozen_before`、`test_bedlam_validator_coordinate_import_preserves_results`：新验收 A 中独立 before 用于一次性对照；长期小测试选输入/固定契约，原有 fixture 不改。不把 A 的本机文件设为日常测试必需。

**明确排除**：LifecycleTests.test_loader_and_training_entrypoint_save_and_resume；test_training_audit/test_trial_callbacks 等含更新模块；verify_densekp_model_lifecycle、smoke_densekp_lightning、三权重 inference、verify_geometry_camera 的完整 SMPL-X 回放；所有 Blender/生成命令。不能先整模块跑了再扣除不合范围项。实施前可用 guard 禁止 Trainer.fit、optimizer.step、DenseKP 构造，并检查方法体/辅助调用闭包。

没有新的空环境安装验收承诺：源码 checkout、现有离线 editable 环境分别记结果；B2不改依赖/构建机制，已有P4安装证据继续适用。若以后要求重建独立editable环境，另授权且离线/no-deps；本批不默认安装，更不能据现有环境通过声称从空环境安装全部依赖通过。

## 9. 推荐实施顺序与评审条件

1. **保护与冻结**：再次核实 HEAD/B1/TODO；原 fixture 用字节快照保存到新 A，不计算 SHA256；记录所有拟改/删文件，确保 B1 新文件也在保护集合。建立精确白名单，采集当前独立 before。缺原基线或多出真实消费者，暂停对应项。
2. **B2a**：先 default_hmr→loop 与四继承者；V05/V10通过才退役八个列明配置/示例。所有保留组合值一致，退役组选项明确失败，当前文档同步。首个可批准子范围仅父改名，不需要触碰 Dataset。
3. **B2b 前置迁移**：visualize、BEDLAM validator、低层/边界测试迁具名接口；冻结数字期望保留；Dataset/geometry capture 显式退役、公共 fixture 路由完成；V01–V08按需通过。
4. **B2b 删除与限定清理**：删三裁剪文件、两旧评分入口、两旧测试辅助；删 legacy_tuple 及 §6 限定符号，注册表 key 暂缓。调整日常列表/导航。对保留函数 AST 和关键常量相等检查，V09/V10验证无资产交付、路径与闭包。
5. **停止验收**：记录实际删除/修改、配置退役名、测试逐项结果/skip、限制与未提交状态。B3–B5 不自动获准，也不提交 commit。

具备进入逐项实施评审的条件；没有需要新 GPU/模型实验的阻碍。实际限制是 BEDLAM 真实小数据缺失、旧评分并非被等价替换、注册表不能随旧预设一起安全删 key。建议评审确认旧评分/预设能力退出，并先批准 B2a；如需保留其中任何独有能力，仅暂停其对应删除，不扩大本批。

所有数值与业务边界继续保持：固定138点和regressor语义、辅助44点、完整256-square/内部192列无额外偏移、旧NPZ权重、可靠遮挡GT监督、有效mask、portrait/closeup和取整差异、keypoint_value非默认置信度、来源防泄漏以及DS/Omni各自规则。真实目标仍为20260320-riding-dataset3；不混031/0311标定，不改变模型/训练/评价公式，不将框增强权重视作已获部署认可。

## 10. 本轮规划文件检查

结束检查：HEAD 未变；原 Git status 条目全部保留，仅新增本报告的未跟踪条目，另新增被忽略的 B2 inventory。TODO 文本与开始时精确一致，保护文件大小/修改时间在采集精度内未变。起始纳秒时间戳经过工具 JSON 数值传递存在亚微秒舍入，不把该检查称为字节哈希；已跟踪原 fixture 另直接与 HEAD 的 Git 文件内容逐字节比对一致。被忽略的 `training_audit_v1.npz` 保留原位，未假定它有 Git 备份。报告本地链接有效，清单计数与16项候选一致。以上均为只读文件核查，未执行计划中的测试或项目计算。
