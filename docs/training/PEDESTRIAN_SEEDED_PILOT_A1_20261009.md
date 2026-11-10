# A1：行人种子场景小试，停在探针后的预算门槛

日期：2026-10-09。范围：`integration_only`，不发布到 datasets，不替换权重。

**实际完成了新行走资产适配、种子配方、几何检查、3 个连续时刻的 C 分辨率双路生成、compact 打包和真实 Dataset 回读。完整两段、原权重基线、50 步训练及最终评价没有运行。** 探针后按当前串行生成流程估计完整 62 个时刻需 **92.35 分钟**，超过本批 90 分钟门槛，因此按执行指令停止后续重计算。这不是完整训练流程通过报告。

产物根目录：[pedestrian_seeded_pilot_a1_20261009](../../outputs/experiments/pedestrian_seeded_pilot_a1_20261009/)（仓库根相对路径实际为 `outputs/experiments/pedestrian_seeded_pilot_a1_20261009/`）。以下以该目录记为 `OUT`。

## 1. 实际阶段与计数

| 阶段 | 结果与边界 |
| --- | --- |
| 本地来源及 shape 核查 | KIT/425 walking_medium05；源 127 帧、名义 30 fps、16 维源 betas；目标各自固定 30 维 betas，未使用或补零源 betas |
| 行走与动态绑定 | 两个既有男角色的源帧 9–102 共 94 帧/角色完成 CPU 求值；固定衣物绑定、固定 groom 拓扑、逐帧头矩阵；不是 188 个独立动作 |
| 纯配方 | 10 个 episode 配方结果（5 次双 episode 调用，其中两次同 seed）；同 seed 完全一致，不同 seed 有变化；不加载模型、不渲染 |
| 几何布局 | 第一次 native-ground 检查拒绝；第二次沿道路纵向布局通过首 episode 的原生地面检查；多人段仅完成投影/间距预检，未做 Blender 实景检查 |
| C 分辨率探针 | 首 episode 的 0、1、2 时刻，源帧 12、15、18；6 张最终 normal ERP，均 5760×2880；DS 2880×2880；64 samples |
| compact/索引/回读 | 6 个原生 mask 紧框、6 条 NPZ 行、6 次真实 Dataset 回读；没有重复生成旧 24 个扰动框 |
| 数据目标 | 3 个连续时刻双侧有效、域边界截断、非空共同 GT 与单人自遮挡成立；多人遮挡变化和完整 3 秒成片未验收 |
| 原权重、短训练、重载 | 权重文件存在性核查；未加载网络/权重，基线/最终/重载前向均 0，实际 optimizer 更新 **0/50** |

核心机器可读文件：`budget_decision.json`、`training_not_run.json`、`targeted_tests_final.log`、`layout_attempt_2/frozen_recipe.json`、`layout_attempt_2/road_pass_00/probe_pack/`。

## 2. 代码及实际调用

全部新增文件；未重写 A2 入口、历史 400 步入口、Dataset、loss、138 点定义或 A1 已评审的框/旁表实现。

| 文件 | 职责 |
| --- | --- |
| `src/densekp_retrain/body/sequence_binding.py` | 复用既有 vertex normals，固定 barycentric 衣物表面 + 4 mm 法线偏移；刚性 SMPL-X head15 矩阵；不选择角色或来源 |
| `src/densekp_retrain/data/road_pass.py` | 无模型、无渲染的确定性配方；人数接口 1–5；连续支持长度检查 |
| `src/densekp_retrain/data/sequence_packing.py` | 单 episode、单 NPZ 的 compact 消费与完整旁表；保留未渲染/空观测/拒绝原因 |
| `tools/experiments/prepare_pedestrian_seeded_assets.py` | 本次本地行走适配器；固定 shape，保留源 root；输出通用动态资产 manifest |
| `tools/experiments/plan_road_pass.py` | 独立 dry-run；可选只读取几何缓存的投影预检，不加载 SMPL-X 或 DenseKP |
| `tools/render/render_road_pass_sources.py` | Blender 逐帧消费者，消费资产 manifest 和显式矩阵，不硬编码 KIT/APD；共用既有材质、groom、ERP 相机与渲染基本函数 |
| `tools/experiments/consume_road_pass_probe.py` | 共用既有 DS 投影、visibility、remap、存储函数；compact 打包/回读/轻量视频 |
| `tests/test_road_pass.py` | 5 项新定向 CPU 测试；无模型或参数更新 |

独立纯配方用法（目标必须使用新路径）：

```bash
python tools/experiments/plan_road_pass.py \
  --config outputs/experiments/pedestrian_seeded_pilot_a1_20261009/layout_attempt_2/config.json \
  --output /tmp/road_pass_review_recipe.json
```

`--geometry-preflight` 会在输出父目录创建 episode 子目录，必须使用全新目录。实际已执行的 Blender/消费任务由 `layout_attempt_2/probe_job.json` 指定，帧号是 `[0,1,2]`。成功临时源已清理，不应直接复跑该 job 覆盖旧产物。消费者输入的是相机/人物配方、缓存 frame selector 和公共 DS 函数，没有复制一套投影算法。

本批没有实现自动推进剩余 59 个时刻的调度器，也没有交付已验收的 50 步训练入口。`training_not_run.json` 中的 AdamW、lr=1e-6、weight_decay=1e-4、batch=1、accumulation=1、with_var=False 等仅为后续明确配置，不能当作已执行训练证据；精度策略尚未初始化验证。

## 3. 来源、时间和动态绑定

动作来自 `outputs/experiments/source_isolated_pilot_20260930/sources/us_2495_L_2200.npz`。核实其 127 帧/165 维姿态、名义 30 fps。KIT/314 bend_left01 未使用。源末段速度明显下降，因此使用起始帧 9 或 12，最后分别为 99 或 102，`source_frame = start + 3 * frame_index`；31 个时刻首末跨度 3 秒，**10 Hz 是显式 stride=3，不是源原生逐帧**。

每个选定输出间隔的水平 root 速度均大于 0.1 m/s（实测最小约 0.45 m/s），相对骨盆的踝/足关节最大步间移动均大于 5 mm。逐间隔速度、足部变化保存在两个资产 manifest 的 `dynamics` 字段。未循环、复制静态姿势或补帧。当前实际成像仅对应单人段最初 0.2 秒；上述 3 秒动作证据是几何/参数检查，不是完整动态视频观感验收。

固定组合为 `cycling_M01_v2`、`cycling_M04_v1` 的既有角色、衣物、皮肤与 groom。名称中的 cycling 是角色历史名称，本批姿态来自行走源，不套用骑行静态姿态。随机选择结果：实际首段 M04、起始帧 12；冻结的多人段两个人均 M04、起始帧分别 9/12。其 `person_id` 不同、角色家族相同；不把重复家族称为外观多样性验收。

权威参数在 `assets_attempt_2/<character>/parameters.npz`，包括 94 个真实源帧号、完整姿态分块、root 平移、固定 betas。body/densekp/cloth 缓存对应同一 frame；groom 仅引用一份既有参考及 94 个小型头矩阵，不保存逐帧百万级头发点副本。绑定参考的 rest body 和 4 mm rest garment 与历史参考误差均在 2e-6 m 内。

刚性 groom 延用 A2 的 `H=[Q, J15(t)-Q·J15(rest)]` 关系；Q 已含 global orientation，H 已含 root。行走 pose corrective 下沿用 A2 最大 1 cm 检查失败，失败记录和初始日志保留。未修改 A2 绑定或偷偷宣称通过其门槛。当前独立**行走诊断**门槛为最大 2 cm、P95 5 mm，M01 实测 max 18.24 mm/P95 3.86 mm，M04 max 17.29 mm/P95 4.20 mm；最差帧分别有 104/321 个锚点超过 1 cm。保持固定拓扑和原刚性方案，**该放宽不代表与 A2 骑行验收等价，头皮贴合与头发动态仍需人工审核**。

源原始 100 fps 关系只保存名义 `3×index` 映射及 `uncertain/not_verified_original_timestamps` 状态，不认证原始曝光时间。序列时间为整数 `time_tick`、`time_base=[1,10]`，源为整数 `source_time_tick`、`source_time_base=[1,30]`，不以文件名或浮点近似配对。

该动作是旧实验验证来源。本批原拟进行诊断训练，不能据此建立新的独立 train/val。角色/seed 不同也共享动作来源。命名固定为多人 `training_inputs`、单人 `shared_source_diagnostic`；不改旧 split、旧指标。未来正式训练必须从原始 DenseKP 权重重新开始，另立来源隔离；任何本批诊断权重不得作为正式初始化（本次实际没有生成权重）。

## 4. 种子、世界变换和约束

`master_seed=20261009`；每 episode 保存独立 child seed。模式 `road_pass`；场景使用既有 urban_street，未新增场景资产。整套 rig 的原点和 yaw 有小范围随机扰动，双相机内部关系固定。标定目标始终 `20260320-riding-dataset3`，0.18 m 仅为**合成基线，不是实测基线**。

配方记录角色/动作/固定衣着、相位、入场时刻、起点、方向、root 与高度策略、相机和成像参数。本次入场时刻均为 0；源速度不独立随机缩放，变化由选中片段自然产生。没有为尚未实现的速度/步频/循环或其他场景模式作支持承诺。

root 所有权只有资产参数：`source.trans[f] - source.trans[9] - target_pelvis`。episode 的固定矩阵扣除所选起始帧相对 root，再一次性施加坐标换轴、行走方向、rig 整体旋转、场景起点及常量地面抬升。body/cloth/138 点共用 M，hair 使用 `M @ H`；没有另加场景速度，也没有每帧 min-Y 贴地。初始 root 的绝对世界位置不重复叠加。

第一版沿原生街道横向，末端到达 `Sidewalk_RIGHT_3.5m`，native ray cast 得到约 -13.3 cm 支撑间隙，被拒绝；没有进入 Cycles 图像渲染。保留原版配方、临时目录和 `probe_render.log`。第二次仅将整套 rig/路径旋转 90°，沿道路纵向；参考 `layout_attempt_2/revision_reason.json`。这是同一冻结资产/相位选择的第 2 次几何尝试，低于每 episode 最多 8 次限制，没有预测误差参与选择。该版本首 episode 的 0/1/2/15/30 时刻 native road 检查通过，支撑采样间隙约 0.45–34.91 mm；完整 31 帧理论最低脚部高度变化约 27.7 mm。没有针对行人结果推广骑行贴地规则。

几何预检记录每视角 138 点相对于显式 valid mask 的有符号边界距离，单位是**最终 ERP 像素**；边缘带 64 px 不是角度门槛。还记录身体域内比例、投影身体框尺度、域外点比例、完整 crop 截断及共同点。使用实际 DS 几何和显式 mask，不通过黑 RGB 或 ERP 左右接缝定位目标。预检框是投影估计；实际训练框使用渲染后的可见实例 mask 紧框，两者不混为同一类型。

冻结的实际覆盖要求：目标至少连续 3 个输出时刻两侧各有 ≥8 个有效点；至少一个域外语义点且实际完整 crop 内共同 GT ≥1；多人段至少一个视角的目标 `visible/in_view` 比率随时间变化范围 ≥0.10；保留单人对照。多人 root 水平间距下限 0.65 m，实际几何最小约 0.754 m。多人预期遮挡窗口仅记为 ticks 0–30 的视线候选，**尚无渲染证明，也未声称完成遮挡/恢复事件**。

## 5. 成像、标签、compact 和实际消费者

固定 C 配方：源/final ERP 5760×2880、DS 2880×2880、Cycles CUDA 64 samples、seed=1009、denoising=False、adaptive_sampling=False，场景原生灯光和 bounce 设置写入各时刻 source_recipe。64 samples 来源于已有 C 实验配置，不由 A2 studio 头发结果推导。共用已有 DS 标定、显式 lens footprint、ERP valid mask、原始源 range 的 visibility 规则与图像 remap。

probe 数据位于 `layout_attempt_2/road_pass_00/frame_000..002/`。每个时刻保留两路 `normal_erp.png`、annotations、valid/person mask、boxes、report、source_recipe 和 shared_world_densekp。首时刻额外保存两张轻量 overlay；不是每帧全套调试图。

主框为 `visible_instance_tight`，`crop_policy=deployment`，扩张 1.0；调用 A1 显式 `image_size_wh` 接口，2 个源栅格像素最短边拒绝规则、ERP 跨缝拒绝保持具名。检查身份、mask/manifest/image 尺寸、有限坐标、紧框相等及目标支持。没有检测器、没有新增随机框扰动，也未改历史 trial 的 1.2 扩张。

| time_tick / 源帧 | cam2 有效/可见 | cam4 有效/可见 | 实际共同 crop GT |
| --- | --- | --- | --- |
| 0 / 12 | 123 / 68 | 113 / 62 | 98 |
| 1 / 15 | 123 / 66 | 115 / 62 | 100 |
| 2 / 18 | 122 / 66 | 122 / 67 | 106 |

6 行共 718 个单视角监督点，其中 327 个 `~visible` 的可靠遮挡点仍参与监督。每路 15–25 个语义点位于有效域外；实际紧框 crop 均仍有非空共同 GT。单人可见数没有标成“全部可见”。本批无框/空观测/单侧有效/无共同点的实际已渲染计数均为 0；未渲染时刻另列，不静默删行。

`probe_pack/packed.npz` 完全沿用原有字段和 Dataset 返回接口。6 次真实回读独立重算 `cv2.warpAffine` 图像、138 点齐次坐标 affine、监督 mask；均精确一致。额外只读检查原始 annotation 与 NPZ 坐标/in_view/visible 一致；从权威 body 缓存用 R138 和 M 重算 world GT，最大误差 0。验证见 `readback.json`、`final_compact_checks.json`。

清理发生在真实 Dataset 和旁表检查通过以后，只删除本任务带所有权 marker 的 3 个成功时刻临时源。之后再次验证 compact 路径、图像头和 NPZ 元数据，仍通过。Dataset 不依赖 EXR、source 或 fisheye 图；初次失败布局保留。

序列旁表复用 `densekp_sequence_index_v1`，每 episode 独立映射：

- 单人 `probe_pack/sequence_index.json`：计划 31 时刻中 3 available、28 missing；6 observed、56 camera_missing；6 个 box 与 6 条 packed row 一一对应。
- 多人 `not_rendered_index/`：31 时刻全部 missing、124 camera_missing，packed path=null、row_count=0。这个旁表不是训练数据，也不能传给 Dataset 冒充空图像样本。
- 每个 observation 明确保留 person/camera/asset/asset_frame/source_frame、整数源时间、原始标注索引；原生框 parent=null，未创建衍生版本。多人相同角色家族不代表同一个 synthetic person_id。

合成两路共享同一状态只声明 `synthetic_same_state`，`exposure_sync_validated=False`；不认证真实 dataset3 曝光同步或检测 track_id 身份。

探针视频：[dual_view_probe.mp4](../../outputs/experiments/pedestrian_seeded_pilot_a1_20261009/layout_attempt_2/road_pass_00/probe_pack/dual_view_probe.mp4)。实际文件同样位于仓库 `outputs/`。视频解码校验仅 3 帧，PTS 为 0/0.1/0.2 秒，最后一帧保持 0.1 秒，容器约 0.3 秒；**不是 3 秒完整视频**，没有拉伸或复制补帧。完整两段视频未生成。视频和初始 crop 需要用户检查衣着/头发/步行观感，0.2 秒不足以批准整段动态质量。

## 6. 预算与资源账

运行前核查主机 GPU/进程/内存/磁盘。GPU0 UUID `GPU-d481acc7-106f-29e5-1298-8c0d398261b2`，RTX3090，启动前约 18 MiB、空闲约 24 GiB；CUDA_VISIBLE_DEVICES 显式设为该 UUID，Blender 仅启用这一个 CUDA device。GPU1 存在 Blender PID 43491，未终止、未同卡运行。主机内存可用约 140–142 GiB，仓库盘可用约 12 TiB，/tmp 约 333 GiB；预留临时空间 10 GiB。探针临时文件阶段根目录约 2.1 GiB，成功清理后约 106 MiB。

| 项目 | 实测 |
| --- | --- |
| 成功动态 CPU 适配 | 9.01 s，2 threads，峰值 RSS 786192 KiB；2×94 帧。另有初次失败的 1 帧和定位的 4 帧，以及 4 次 rest 求值；无优化器 |
| Blender setup | 17.77 s |
| 双路源渲染每时刻 | 65.657 / 72.043 / 77.603 s，均值 71.768 s |
| 成功 Blender 总时长 | 233.074 s；峰值进程 RSS 10540056 KiB（约 10.05 GiB，不是 VRAM） |
| DS 后处理每时刻 | 14.829 / 10.829 / 10.848 s（首次含 LUT 准备） |
| CPU 消费总时长 | 44.106 s，含打包、回读、清理与视频；峰值 RSS 1473152 KiB |
| 首次地面失败 | 没有完成渲染；精确 wall 未记录，预算保守计入 10 s，不伪造实测秒数 |
| GPU 结束状态 | GPU0 恢复约 18 MiB / 0% utilization；GPU 峰值显存未采样，不报告虚构值 |

预算门槛采用当前**串行源渲染 + 必需 DS 转换**的生成阶段 wall compute，不仅 GPU kernel 时间；开发、等待和读文档时间不计入。估计式：

`62 × (max(三次渲染)=77.603 + max(两次 warm DS)=10.848) + 两次 setup + 一次已测打包/视频开销 + 首次 LUT 额外开销 + 失败预留10s = 5541.074s = 92.35min`。

探针已经包含在 62 内，不重复渲染。该估计暂按多人和单人同成本，**没有偷偷乘上多人倍率**；多人实际额外开销未知。也保留乐观均值估计 5157.13s（85.95min）和只计 GPU 源渲染的同单人成本上界估计 4846.93s（80.78min），不能将 92.35min 称为已测总时长或纯 GPU 渲染必然超时。小样本下保守串行流程估计已超 90min，故不继续赌剩余预算，不改分辨率/采样，不追加探针或借重试扩大计算。

按当前时刻文件体积外推完整最终数据约 1,604,023,229 bytes（1.49 GiB），低于 5 GiB；不是此次停止原因。当前实际成功计算约 233.07+44.11 秒，另计 CPU 适配、定向检查与失败尝试。没有执行基线、网络构造、训练、重载前向；也没有用旧数据训练来代替本批数据。

## 7. 测试与剩余限制

运行：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MPLCONFIGDIR=/tmp/densekp_a1_mpl \
PYTHONPATH=src:. python -m unittest \
  tests.test_road_pass tests.test_high_res_boxes tests.test_sequence_index
```

**24 项通过**（包括新 5 项）；没有全库 discover，没有夹带小网络参数更新。覆盖确定性/种子差异、1–5 人及非法人数/模式、固定绑定组合、时间/root/crop 策略、rig 整体朝向、头矩阵只叠加一次、连续支持不能用离散累计冒充；复跑直接依赖的高分辨率框与旁表定向测试。实际消费者额外验证 compact 清理后路径/尺寸/原始 GT、6 条 loader 行和 3 帧视频时间。新增渲染器的附加 R138/body-matrix assertion 为静态及 CPU 等式核查，未为此额外重渲染；本次 Blender 运行已做逐帧 body 坐标精确读回、hair 点位矩阵和固定拓扑检查。

尚未通过：完整连续双 episode 数据、多人人体遮挡变化、完整 3 秒视频人工评审、原权重预测、50 次有限 loss/梯度/optimizer 更新、checkpoint 严格重载、最终评价。冻结了至多 24 个人物时刻点对的时间采样选择，但实际没有模型输出；因此 crop 1/2/4/8px 同时达标率、周期 ERP 误差、等点/等人物时刻/等 episode 聚合、退化与残差时序指标均 **未计算**，不能填 0 冒充结果。旧指标保持原样。

新公共消费者已经支持可追溯的动态资产 frame 与显式世界矩阵；当前配方采样器仍是本批 30fps→10Hz 的固定 stride 方案，不宣称任意来源帧率已经接通。A2 后续只需提供通用资产 manifest：权威参数、固定目标 betas/角色家族、衣物/groom 参考与绑定版本、body/cloth/138 缓存对应关系、head transforms、可用帧范围和准确源时间表、来源/许可/划分关系、动态人工审核状态。生成器负责 person 实例、种子选择、单 episode 场景/rig/地面/root 所有权、同步合成状态、显式 masks/框、compact、旁表与每 episode 行映射；A2 不需要复制一套骑行场景系统。

原文件 size/mtime 快照检查：本次新增代码前已有 Python 文件、TODO、README、共享计划没有变化；之后仅授权更新共享计划当前状态。A2 专用脚本、产物、报告及外部 pano-stitch 未修改；未下载/安装资产或依赖，未算 SHA256，未提交 commit，未替换部署权重。旧失败版本、用户修改与历史 fixture 保留。

本批新增 Python 文件语法/尾随空格检查与共享计划定向 `git diff --check` 通过。全工作区 `git diff --check` 提示用户原有 `docs/TODO.md:13` 尾随空格，未借本批清理或修改。

**当前停在预算与探针评审点。** 下一批需要先评审时间预算/串行消费吞吐和完整动态观感，再决定是否完成原定两段及 50 步诊断；本报告不能作为泛化、骑行提升、曝光同步或部署验收依据。
