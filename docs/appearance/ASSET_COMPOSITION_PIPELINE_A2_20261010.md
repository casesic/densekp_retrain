# 标准 SMPL-X 人体、动作、贴身衣物和静态头发组合流程（A2）

日期：2026-10-10。范围是已有标准 SMPL-X 参数动作、M01/M02/F02 固定体型、simple 衣物表面绑定和可转移静态 groom。所有新组合为 `integration_only`；没有修改 A1 道路、ROI、Dataset 或训练接口，也没有把新组合加入活动采样池。

## 可选组件与真实边界

[组件清单](../../configs/asset_composition_a2_20261010/components.json)保存 body、motion、clothing、hair 四类独立 ID、版本、来源路径与转移策略；[组合配置](../../configs/asset_composition_a2_20261010/recipes.json)只选择四个 ID、源时间窗口和 root 策略。[组件 schema](../../configs/asset_composition_a2_20261010/components.schema.json)与[配方 schema](../../configs/asset_composition_a2_20261010/recipes.schema.json)给出机器可读字段；运行时仍以代码的引用/时间/拓扑检查为准。[实施冻结范围](../../configs/asset_composition_a2_20261010/scope.json)包含本批五项及受保护文件。配置与模块均在非忽略路径，新资产大数组和预览在 `outputs/experiments/asset_composition_pipeline_a2_20261010/`。增加同类标准组件时编辑清单/配方，不需要复制人物专属制作器。

| 维度 | 当前支持 | 明确拒绝或限制 |
| --- | --- | --- |
| body | M01、M02、F02 各自标准 SMPL-X male/female 模型、固定 30 beta、原皮肤/脸部 appearance、家族与父引用 | 未注册模型、缺模型、非标准模板/顶点语义不通过；10475 顶点数量本身不足以证明兼容 |
| motion | APD6526、bikefit、p2_straight、p1_comfortable、p1_alternate 五段已适配的参数；保留原整数 timeline，按窗口切片 | 不转换新的 BVH/SMPL/MHR；行走源 root、未知 pose 基或需重采样的输入不悄悄丢弃/猜测 |
| clothing | M01/M02/F02 的 `simple_clothing_v1`；保持衣型面集、barycentric 身体顶点语义、材质和 4 mm 法线偏移，在目标 rest 及全动作重建 | BEDLAM 原配仿真缓存锁定原身体＋原动作＋原时间；不可把旧逐帧缓存改配新 motion；无布料仿真 |
| hair | 原角色固定 active 曲线＋刚性 head；跨头型复用已有三角框架、局部点与 correction，生成新的目标 rest 参考 | 无动态摆发；头皮局部接触仍需人眼检查；不逐帧删毛，不复制 120 份曲线 |

内部坐标为标准 SMPL-X 模型 Y-up、米。body 描述指向权威角色 manifest 中的 gender、30 个实际 beta、模型文件与固定外观；motion 描述指向参数和逐条整数时间；clothing/hair 描述指向权威 reference 中的绑定、材质和固定有效集合。加载模型固定 `use_pca=False, flat_hand_mean=True`，双手为各 45 维 full axis-angle，body 为 21×3，jaw/eye 及 10 维 expression 与现有参数契约一致。`plan` 不构造模型，但验证参数维数、固定源 root、原时间、素材文件、源/目标注册体型的面索引顺序和衣物/groom 对应的 reference。真正制作时还核对目标 shaped-rest 与原角色 reference 顶点。衣物来源的面集/绑定拓扑固定，材质中的 `GarmentRest` 改为目标 rest 衣面坐标；原调色与衣型配置仍来自所选 clothing 组件，不把换色冒充换衣型。当前 studio 的头发着色由已有 `add_groom` 固定为深棕色，清单只独立选择 groom 几何/绑定；独立发色或新着色模型尚不属于首版自由组合能力。

motion 是基于清单所记 `fitted_body` 的估计动作，不是与人物 shape 无关的三维真值。跨 body 时只继承姿态旋转和时间，在目标 gender、固定 beta 上重新求值，并使用目标 shaped-rest pelvis 的负值作全段常量 `transl`。本批 motion 的 pelvis 固定，road yaw、行进与固定骑行放置高度均由 A1 场景拥有；源 root、不均匀原时间和场景变换没有被重复施加。body/cloth/R138 在同一模型时刻，groom 使用目标 rest 参考点及目标 body 求得的 head 矩阵；以后分别由 `scene @ model` 和 `scene @ head @ rest_hair` 消费。R138 始终为项目原 `R[138,10475] @ body`。

## 一个入口与缓存契约

[组合 CLI](../../tools/experiments/compose_asset_a2.py)调用[公共组合模块](../../src/densekp_retrain/assets/composition.py)：解析清单 → 兼容/依赖/时间检查 → 固定目标 shape 求值 → rest 衣物/头发适配 → 全段 cloth/head/R138 → 数值回读 → 统一 delivery manifest、A1 proposal 与只读 adapter 检查。`--plan` 只打印解析配方及依赖，不初始化 SMPL-X、不渲染也不写输出；`--build` 才制作；`--prepare-preview` 显式冻结短预览所需的源帧、组件、采样和设备。导入模块没有计算副作用。

```bash
OPENBLAS_NUM_THREADS=1 python tools/experiments/compose_asset_a2.py --name A_f02_p2 --plan
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 python tools/experiments/compose_asset_a2.py --name A_f02_p2 --build
OPENBLAS_NUM_THREADS=1 python tools/experiments/compose_asset_a2.py --name A_f02_p2 --prepare-preview --gpu-uuid GPU-6f046d56-acbe-d49b-774a-9d463e768cac
# 预览是显式可选步骤；实际 Blender 设备须先重新检查，不能照抄历史 UUID。
CUDA_VISIBLE_DEVICES=<空闲设备UUID> /mnt/data1/khb/app/blender-5.1.1-linux-x64/blender -b -t 4 --python tools/render/preview_apd_6526_dressed.py -- --stage probe --asset-root outputs/experiments/asset_composition_pipeline_a2_20261010/builds/A_f02_p2 --recipe-file preview_recipe.json
```

缓存拆为 `cache/body/<body,motion,window>`、`cache/cloth_rest/<body,clothing>`、`cache/cloth_motion/<body,motion,window,clothing>`、`cache/hair_rest/<body,hair>`；head 矩阵随 body motion 保存，groom 参考曲线只保存一份或引用来源。命中时比对解析组件版本及依赖绝对路径、大小、纳秒修改时间，包括 groom 的绑定文件和 Alembic 拓扑来源，并回读必要文件、姿态参数与 rest 衣面；不同配置占用同名目录或依赖改变时拒绝，不覆盖。此机制能识别正常文件更新；若有人刻意同时恢复文件大小和修改时间，仍须显式提升组件版本，不能把元数据标识当密码学内容证明。本轮未计算 SHA256。中断目录即使已有缓存键，也会因必需文件不完整而被拒绝。源模型、原参数、原纹理和既有接受视频保持引用。

后续接受的 [M03＋KIT11 BEDLAM2 转移衣物](BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md)作为 `retargeted_native_cache_locked_body_motion_time` 登记。它是**已生成的整段缓存引用**，只允许清单中精确的 M03 身体、KIT11_walk 动作、M03 groom 与 `[0,120)` 时间窗；组合入口核查权威文件后回用成品，换任一项即拒绝。这不把 BEDLAM2 原配服装升级成可自由换动作的简单衣型，也不表示 A1 公共材质消费者已支持它。

## 本轮有界实证

两个已接受组合用新入口从旧权威来源单独重放，之后以旧缓存作独立比较，详见[交叉检查](../../outputs/experiments/asset_composition_pipeline_a2_20261010/cross_validation.json)。M02＋p1_comfortable 的 body/R138 最大差 2.384×10⁻⁷ m，cloth/head/面集逐值一致；F02＋bikefit 的 body/cloth/head/R138/面集逐值一致。两者姿态参数和原 source frame/tick/time_base 一致，因而仅这两个同配置重放组合继承原用户接受记录；新换件组合不继承观感结论。

新 A/B/C 全部是 F02 固定女性 shape＋同一 participant_2 原帧 `[5700,5820)` 的 120 帧；BVH 发布 `Frame Time=0.0333333 s`，整数时基 `1/10000000`、步长 `333333` ticks，不是实际曝光 PTS。未重拟合、平滑、插帧、循环或修改 p2 原权威资产。三者共用同一身体/138点/head 动作缓存：

| 组合 | 独立选择 | 全段技术状态 | 动态观感 |
| --- | --- | --- | --- |
| A `A_f02_p2` | F02 原 simple 衣物＋F02 短波波头 | 120 帧有限、标准拓扑、R138/时间/A1 只读适配通过 | 前 60 原始连续帧获用户接受；后 60 帧未做人眼动态检查 |
| B `B_f02_p2_m02_cloth` | 只把衣物换为 M02 的 simple 衣型；衣面 7058→6032，非仅换色 | 共用 A 身体；重新生成目标 F02 rest 衣面及逐帧几何，通过 | 同一前 60 帧获用户接受；后 60 帧未做人眼动态检查 |
| C `C_f02_p2_m02_cloth_hair` | 在 B 上只换 M02 groom；目标 F02 rest 头皮适配，固定 active 80397 曲线 | 共用 B 衣物、A 身体；有限值/头矩阵与 A1 只读适配通过 | 同一前 60 帧获用户接受；后 60 帧未做人眼动态检查 |

头发适配的原始局部 collision correction 和 active 集合保留；约 5.1 万个抽样点中，转移 rest 表面有 2 个样本的近似法向距离低于 −1 mm，最小 −1.27 mm。这是定位指标，不是全发束碰撞证明。轻微局部穿模允许。短视频只覆盖前 60 帧；全 120 帧完成的是几何/时间/R138 数值检查，不能据短视频观感接受推断后 60 帧也已接受。

2026-10-10，用户观看三项新组合后表示“我觉得效果可以”。[人工接受记录](../../outputs/experiments/asset_composition_pipeline_a2_20261010/visual_acceptance_20261010.json)据此将 A/B/C 的**前 60 帧连续 studio 穿衣与头发观感**标为接受；全 120 帧数值状态、后半段未观看状态和公共场景 `not_consumed` 分别保留。此次仅同步权威交付 manifest、A1 proposal 与资产目录中的人工状态；原 `validation.json` 保留制作当时的待评审快照，几何、时间和预览媒体未改。

[统一 A/B/C 观看入口](../../outputs/experiments/asset_composition_pipeline_a2_20261010/index.html)的两固定方向、每向 512×512、Cycles 64 samples、无降噪；各 MP4 按 `10000000/333333` 的发布标称时钟编码，逐帧核对 60 条展示 PTS，最大整数 tick 差均为 0。64 samples 只是 studio 评审参数，不自动决定正式道路数据采样。

[机器资产目录](../../outputs/experiments/asset_composition_pipeline_a2_20261010/asset_catalog.json)只列五个实际组合，不宣称任意笛卡尔积已通过。[执行记录](../../outputs/experiments/asset_composition_pipeline_a2_20261010/execution_record.json)：两个旧组合回放与三个新组合制作、探针/补帧渲染、编码的成功实测计算合计约 **1181.5 秒（19.7 分钟）**；一次沙箱 CUDA 探针及两次预览封装失败额外约 **50.5 秒**，总计约 **20.5 分钟**，低于 45 分钟冻结预算。阅读、开发和等待不计入这些实测计算。使用当时空闲的 GPU1 UUID `GPU-6f046d56-acbe-d49b-774a-9d463e768cac`，任务结束显存回到 407 MiB 空闲状态；未连续采样显存峰值。目录约 **258 MiB**，其中 A 的失败封装目录及沙箱失败现场按要求保留；模型、原视频、旧资产和每帧全套 groom 未复制。

## A1 读取与来源划分

每项输出 `delivery_manifest.json`、`timeline.json`、`a1_proposal.json`、`a1_adapter_check.json` 和 `validation.json`。实调 A1 当前 `dynamic_scene.adapt_proposal/geometry_paths/select_timeline` 对全 120 条原时间及必要持久路径只读通过；这是资产字段可读取，不表示道路矩阵、相机、ROI、观测、打包行已生成。A1 后续只需按当前 proposal 引用新资产，选择固定骑行高度和场景轨迹，明确场景时间采样策略，并在独立批次做实际 ROI、标签和 Dataset 回读；本轮不改它的代码。跨衣型材质目前沿用 `GarmentRest_continuous_shell`，BEDLAM native cache 不在这个消费者/换件路径内。

来源同时保留：F02 身份家族、p2 motion 的 `zenodo_10668611_participant_2` 及发布录制、M02 衣物和发型原角色引用。A/B/C 只是一个动作源的三个外观衍生，不是三个独立人物或独立验证来源；F02 也与已有 F02＋APD/bikefit 组合有关联。普通标准 SMPL-X 模型与通用衣型本身不代表所有目标人物是同一身份。CC BY 4.0 源署名 Kyriakou 等；各目标角色/模型许可分别沿原资产，bikefit 的未知许可没有因回放提升。

本轮没有下载、安装、人体重建、衣物物理、场景渲染、DenseKP/YOLO 推理或训练，也没有改 TODO、旧接受资产和外部项目。三项短预览观感已接受；公共道路场景接入仍需另行实施与验收。
