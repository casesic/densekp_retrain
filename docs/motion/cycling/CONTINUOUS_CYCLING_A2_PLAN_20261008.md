# 连续骑行动作与穿衣资产接入准备（A2）

日期：2026-10-08。状态：**资产核查与方案交付完成，停在 A2 评审；未实施接入。** 用户本轮回复：**尚未制作**。因此没有待接收的新标准资产，也没有已验收连续骑行角色。机器清单：[interface_and_candidates.json](../../../outputs/plans/continuous_cycling_a2_20261008/interface_and_candidates.json)；只读核查脚本：[audit_continuous_cycling_a2.py](../../../tools/validate/audit_continuous_cycling_a2.py)。

最小建议是：一个既有成人角色、固定 shape、一段实际蹬踏的连续动作，交付参数与绑定；接入侧按同一时间索引生成身体、简化衣物、随头部运动的 groom 和 138 点，再统一施加场景变换。逐帧缓存是等价替代，不要求制作方重复提交参数链和全量缓存。第一阶段无需毛发/布料物理仿真、Unreal→Blender 完整管线、自行车模型或不可见手指/面部重建。

## 1. 范围、保护与证据等级

已阅读总计划、双视角审计、ASSETS、CONTRACTS、interface_draft，以及下文列出的消费者代码。接口草案是参考，不把其中全部数组、评价阈值、真实同步及训练要求转成制作方的必交项。

初始 Git 状态：`docs/README.md`、`docs/TODO.md` 已修改；两份 DUAL_VIEW_TEMPORAL 文档及 `tools/validate/audit_dual_view_temporal.py` 未跟踪。读取并保留用户 TODO。未发现适用 AGENTS.md。过程中另见 A1 范围的 `data/boxes.py`、三个历史调用脚本及框测试制品发生变化；没有编辑、恢复或接管它们。本轮只新增本报告、机器清单、独立只读核查脚本，不修改共享计划、README、导航、生产代码或历史测试。

本次范围限定在 manifest 指向的目录：`outputs/motion_inputs/{cycling_images_20260929,cycling_fit_20260929}`、`outputs/experiments/cycling_smplx_20260929`、十成人及其父版本目录、`source_isolated_pilot_20260930/sources`、两个已知 BEDLAM motion 文件、两份既有骑行 audit。外部仅阅读 `pano-stitch/data_process/tmp/production_regression_20260320_dataset3_multiperson` 与对应 human_fitting 源码；没有扫描整块磁盘。

证据分四级：本轮路径/数组头与小数组检查；历史 manifest/质量报告；本轮查看既有静态预览；尚待下一批连续播放验收。前三项不能代替第四项。未下载视频或模型、抽帧/批量解码、构造人体模型、重建/生成几何、运行 Blender/YOLO/DenseKP、训练、安装、SHA256、commit、删除或写外部仓库。媒体参数复用前批头信息，没有再次解码媒体。

## 2. 本地动作与候选

| 类别 / 具体输入 | 本轮与历史证据 | 能复用与尚缺什么 |
| --- | --- | --- |
| 两个独立照片姿态 `cycling_smplx_20260929/{coco_292226_person_1330527,coco_306800_person_225160}/canonical_pose.npz` | 实读各 `body_pose[1,63]`、`betas[10]`、`sequence=false`、`fps=0`、source frame 0 | 静态回归及外观对照；不同照片人物不能拼成一个连续 motion。来源逐图许可继续跟随照片 manifest，不继承统一授权 |
| 连续行人 `source_isolated_pilot_20260930/sources/us_2495_L_2200.npz` | 实读 `poses[127,165]`、`trans[127,3]`、neutral、16 betas、30 fps；126 对相邻 pose 均变化且有限 | 历史 `pose_selection.json` 认可 KIT/425 walking_medium05 的行走/站立片段；是时序适配参考，不是骑行。原100 fps的0–381与本地30 fps的0–126只有推断映射，不能称逐帧原时间已证实；不能用源16维 beta 覆盖目标30维 shape |
| 其它已知连续非骑行动作 | `/home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz` 为192帧；`third_party/bedlam_render/blender/smplx_anim_to_alembic/smplx_animation.npz` 为126帧，均30 fps、165维，实读相邻pose变化 | 历史 motion_info 分别关联 KIT/883 wipe_arm_dabbing01 与 KIT/314 bend_left01；不能凭旧“行人”配方名称把全段称为走路或骑行。程序化 cyclic gait 也不是采集的骑行动作 |
| **连续骑行候选 L1：participant_1** | `motion_inputs/cycling_fit_20260929/participant_1/clip_native.npz`：240帧、372通道、75关节、frame_time=0.0416667；原帧 `[7560,7800)`，约10秒；239对相邻通道变化，索引连续、有限 | 来源 [Zenodo 10668611](https://zenodo.org/records/10668611)，本地原记录作者 Panayiotis Kyriakou、Yiorgos Chrysanthou、Marios Kyriakou；历史许可 CC BY 4.0。保存的运动学报告约72 rpm；本轮看到既有踝轨迹图有左右交替周期。**不是视频内容验收，也不是 SMPL-X** |
| **连续骑行候选 L2：participant_2** | 同目录 participant_2：300帧，frame_time=0.0333333，原帧 `[5700,6000)`，约10秒；299对相邻通道变化，索引连续、有限；历史报告约78 rpm | 同一数据集另一参与者，候选保留；没有重做运动学或观看动态预览。L1/L2都只拟定过22关节映射，未 retarget，原生单位“cm”、+Y和前向仍属推断，无车座参考，手指等覆盖不完整 |
| **外部视频/拟合候选 L3：dataset3** | 既有4325–4575共251帧；头信息名义30000/1001 fps，区间占用约8.375秒；实读序列JSON，775次观测、54个算法track | 可定位真实候选时段及遮挡条件；不代表同一骑行者贯穿251帧。详见下一节；不是已交付motion，更不是真实GT |

L1/L2 的 `manifest.json` 明确 `prepared for fitting; not SMPL-X motion`、`smplx_fitted=false`。这补充了前批“未找到已验收连续骑行”的结论，并不推翻它。旧 BABEL/BEDLAM 审计没有确定可用骑行 motion；不重复历史全量扫描，不据此断言整库不存在。BVH 制作 CLI 已退役，复用原始候选需要用户标准适配流程，不能假定现仓库已有一键 BVH→SMPL-X 制作能力。

推荐优先级：先让制作方评审 L1 能否通过现有标准流程交付；若坚持视频重建来源，则从用户自有可授权视频或下节公开页面重新筛选，当前不定重建模型。L1 是动捕备选，**不能用它宣称完成“公开视频连续重建”这一来源要求**。

## 3. 外部真实拟合为何暂不能直接接入

外部根目录记为 `X=/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch`，运行目录记为 `P=X/data_process/tmp/production_regression_20260320_dataset3_multiperson`。

- 原视频由 `P/prepare_selected.py` 定位到 `X/Circular_Video/01_send/20260320_riding/dataset3/VID_20260317_112543_00_013.{0,1}.mp4`。本轮只复用此前媒体头与帧映射证据：0→cam2、1→cam4。缺实际逐帧 PTS、曝光偏移/漂移验证及用途授权记录；两个容器同起点不证明曝光同步。
- 实读 `P/standalone_sequence/sequence_result.json` 全部 person 字段：`person_id/track_id/initialization/point_ids/raw_pairs/optimized_pairs/valid/fitting/cpp_preparation`。**没有 pose、beta 或 root 数组**。`densekp/erp_rotate/004370.json` 也是二维对应导出。不能从成功次数推定完整三维资产已落盘。
- 本轮 track 统计：track16 有49次观测，4427–4477间有缺口，最长连续34帧；track4 是4370–4414共45连续帧。这里只是算法关联，既未证明是骑行者，也未人工确认同人。不能把所有775条记录混成一个人物序列。
- `src/lib_pano/human_fitting/matched_person_tracks.py` 的内存状态包含75维 `frame_parameters` 和10维 beta，序列输出却不保存它们；`per_person_cpp.py` 会更新pose/beta并保留部分hand/scale初始化状态。实际运行记录为 persistent_worker，不能把 subprocess 的潜在磁盘输出当作该次运行已保存的资产。未扫描其它未知运行目录，也不宣称外部绝无三维缓存。
- 所需补证：明确75维参数语义/坐标/尺度/模型版本及可导出位置；同一人的逐帧关联、固定shape策略；global orient、root translation与相机运动的分离；丢帧/恢复track规则；拟合残差及异常帧、脚踏变化与躯干连续性。若参数未保存，后续重算/导出属于新批授权任务。
- 本轮查看已有 `outputs/audits/dual_view_temporal_20261008/real_roi_04450.png`：有自行车、多人重叠与镜头有效域截断；一张静态预览不能证明蹬踏、连续同人、镜头运动或可用片长。原车体可能遮住腿/手，重建不能默认恢复这些细节。合成首段可不建车，但应标记 `source_bicycle_present=true`、`render_bicycle_present=false` 与差异。

优化点、平滑结果和模型拟合不是独立真实GT。尤其该结果依赖 DenseKP，本项目不能拿它认证 DenseKP 的真实精度或独立泛化；只能作为带 provenance 的候选或接入诊断。

## 4. 补充公开视频页面候选（仅2个）

本地有未适配动捕及自有视频线索，但没有已确认可用的同人连续视频交付。以下是2026-10-08只读网页核查，未获取媒体，也**没有实际播放视频**。页面描述是发布者描述，不能代替动作质量证据。

| 候选 | 原始页面 / 发布方 | 无需付费与许可范围 | 内容证据、适配理由与可能失败原因 |
| --- | --- | --- | --- |
| V1 | [Man Using Stationary Bike at the Gym，4367759](https://www.pexels.com/video/man-using-stationary-bike-at-the-gym-4367759/)，Pavel Danilyuk / Pexels | 页面标Free；许可有官方说明，**本项目重建/派生/训练用途未确认** | 页面列1920×1080、14秒、29.97 fps。发布者描述健身车；若全身/脚踏连续可见，适合原地动作分离。是否同人连续、可见蹬踏、遮挡、镜头运动、切镜与可用区间全部未知；近景裁腿、器材遮挡都可能失败 |
| V2 | [A Man Exercising on a Stationary Bike，6388401](https://www.pexels.com/video/a-man-exercising-on-a-stationary-bike-6388401/)，Tima Miroshnichenko / Pexels | 同上；作者个人简介不能当项目授权 | 页面列3840×2160、45秒、25 fps。描述为暗光健身车；若内容合适，可选择短连续蹬踏段。暗光是页面描述而非本轮视觉结论；遮挡、全身入镜、同人连续、镜头运动均未验证，可能存在暗部/运动模糊或切镜 |

[Pexels许可页](https://www.pexels.com/license/)说明免费使用与修改，并有用途限制；[服务条款§8](https://www.pexels.com/terms-of-service/)另有限制未经授权的数据采集/提取及机器学习用途相关行为。因而“免费”不等于本项目已取得下载、重建、派生数据与训练授权，两候选状态均为 `page_only_not_selected`。没有选定下载方法或重建模型；下一阶段先核实具体用途及观看内容，再冻结片段。

## 5. 十成人资产与现有消费边界

`cycling_ten_adults_candidates_20260930/character_manifest.json` 的10个角色均30维固定beta，5女5男；manifest所指模型、reference body、衣物重心绑定、groom绑定/ABC和主要外观文件均存在。下表是**reference数组头**，没有重新评估人体模型或读完整groom数据。身体均 `[1,10475,3]`，faces `[20908,3]`；静态认可数来自 combination_status。

| 角色 | 简化衣物顶点数 | groom控制点数 / 模板 | 已认可照片姿态数 |
| --- | ---: | --- | ---: |
| cycling_F01_v2 | 2618 | 1,591,400 / 03-1 short-casual | 1 |
| cycling_F02_v3 | 3616 | 1,890,570 / 18-1 short-bob | 2 |
| cycling_F03_v1 | 2701 | 2,076,256 / 07-1 layered-bob | 1 |
| cycling_F04_v3 | 3611 | 1,890,570 / 18-1 short-bob | 1 |
| cycling_F05_v2 | 3391 | 2,560,079 / 01-1 curly | 1 |
| cycling_M01_v2 | 2612 | 1,591,400 / 03-1 short-casual | 1 |
| cycling_M02_v3 | 3116 | 2,560,079 / 01-1 curly | 2 |
| cycling_M03_v1 | 3202 | 2,076,256 / 07-1 layered-bob | 1 |
| cycling_M04_v1 | 3427 | 2,560,079 / 01-1 curly | 1 |
| cycling_M05_v1 | 3586 | 1,591,400 / 03-1 short-casual | 1 |

合计12 accepted、8 not_attempted、3 excluded DS rows、3 failed appearance versions。静态资格未扩大；F04家族来源隔离限制继续生效。源照片人物、目标shape家族、皮肤供体、groom模板不是同一身份概念；皮肤仍有Meshcapade CC BY-NC 4.0记录及供体不确定性，模型/groom各自许可仍需随交付保留。

首角色建议 `cycling_M01_v2`：已有简单短袖/短裤式shell，短发控制点较少，可减少首段缓存和长发遮挡；不是因为它已通过动态验收。制作方若有现成更合适角色可换，但只验一个，重新固定其身份/shape/拓扑。M01历史shape来源关系也必须沿manifest保留，不因重新命名成为新身份。

| 组件 | 当前消费者实际能接什么 | 本轮确认的限制 |
| --- | --- | --- |
| 身体 | `body/parameters.py` 七块轴角；`generate_real_motion_geometry.py` 可将canonical参数变为 `[T,10475,3]`；已有 renderer 按person.frame取body | 旧 `parameters_from_pose` 单帧、transl=-pelvis；旧 adapter 默认稀疏选16帧、fps可fallback30、无完整PTS映射，不能直接充当新序列协议 |
| 138点 | `assets/densekp_regressor_R.npy` 作用于**身体**顶点，`R[138,10475] @ V` | 不使用衣物/groom顶点或BEDLAM437点索引；不裁零或归一化R的小负权重 |
| 简单衣物 | `clothing/preparation.py`、`shells.py` 保存 clothing_vertices、faces、body_barycentric_indices/weights、reference_body_vertices、rest及materials | preparation只读第0帧；clipped_shell可能按当前pose过滤退化面，逐帧重建不能默认保证固定拓扑。应冻结绑定/面集，再逐帧变形。历史局部0.4mm/1cm门槛不是新要求的“零相交” |
| 渲染衣物 | `render_equirectangular_source.py`可按相同frame取衣物，但验证body路径、同T、精确reference和绑定，普通shell距离≤0.010002m；旧commuter契约调用require_studio | 任意骨骼蒙皮/松衣缓存不能直接塞入这个旧shell分支；新增具名契约适配即可评审，不要求先改成复杂仿真，也不能静默放松历史验证 |
| groom | `groom_transfer.py` 使用头皮三角形重心根、局部frame/local_points、固定collision_correction/active mask；`rendering/blender/groom.py`读取points/radius/offsets并导入source_abc曲线结构 | 转移固定 `[0]`；renderer没有时间索引，只接受一个静态folder。ABC仍是渲染依赖，不能只交points。第一阶段逐帧随头皮/头部变换即可；没有毛发惯性或动力学 |
| 世界/成像 | renderer身体、衣物、groom共用body matrix；DS读取同frame的138点并调用world_points；双相机process_scene可作为帧单元 | 两处独立实现逐帧min-Y lift；普通scene spec仍为静态location/yaw。不能只循环照片或平移静态body冒充motion |

## 6. 最小两层交付契约

### A. 制作方提供（角色包与动作包可引用既有文件）

| 必交信息 | 最小表达与责任 |
| --- | --- |
| 来源关系 | asset_id、target_character_id、identity_family_id、appearance/source family；源视频或motion URI/本地文件、发布方、source_actor/person_id、片段半开区间、license_evidence、用途记录、处理/适配版本；unknown显式保存。目标shape不假称源视频本人 |
| 时间表 | asset_frame=0..T-1、source_frame_index、sample时间与source时间、fps或有理time_base、时间真实性（measured/nominal/inferred）。视频有PTS则保存；没有则null+原因，不能编造。BVH用frame_time及source帧号建立名义时间，PTS不适用。保留采样/变速/循环/缺帧映射 |
| 固定角色 | 模型类型/版本/gender/num_betas/固定betas引用，body topology及顶点顺序；绑定reference与target shape一致。源beta可作来源信息，不能覆盖已选角色。缓存分支至少能追溯固定shape定义/参考体，无需再拟合一套beta |
| 参数分支（推荐） | SMPL-X body_pose `[T,63]`；root rotation/translation `[T,3,3]`/`[T,3]`或等价 `[T,4,4]`，轴角rad、平移m，注明旋转约定。手 `[T,45]`或一个固定45维值及broadcast声明；jaw/eyes同理。expression可固定零并标明，不要求恢复未观察细节 |
| 缓存分支（替代） | 身体 `[T,10475,3]`和固定faces；每件衣物 `[T,Vc,3]`和固定faces，头发逐帧points或头部刚性变换＋静态点。缓存本身承载逐帧pose；不强制补一套等价完整参数。必须声明root/global orient是否已烘焙、所属空间及同一时间映射，保留root/场景分解信息 |
| 衣物表示 | 任选简单贴身表面绑定、rest mesh＋骨架/权重/inverse bind、或逐帧缓存；提供绑定版本、reference body、面集、UV/材质及许可来源。既有简单shell用重心索引/权重，不需要重新制作皮肤纹理 |
| 头发表示 | 任选当前头皮绑定包、rigid head attachment或逐帧cache；注明 `head_follow/scalp_bound` 与 `physical_simulation` 区别。复用现renderer时还需source_abc、source_objects的点/曲线顺序、radius、curve_offsets、active mask及冻结修正 |
| 坐标及质量标记 | 源坐标轴/手性、up/forward、原点、单位与转换矩阵；rest/bind/pose参考一致。身体/手/面各参数标observed/inferred/fixed/missing及固定值，坏帧、遮挡、推断尺度与已知限制单列 |

不要求制作方交相机、DenseKP预测或两套重复全量缓存；138点由接入侧按仓库R计算。若制作方交了138点，接入侧只作回读核对。许可证据可以引用原文件，不需要本轮下载或重新制作附件。

### B. 接入侧生成与保留

1. 验证交付时间/身份/shape/拓扑与绑定；执行一次明确的源坐标转换、root分解及目标shape适配记录。生成场景轨迹、序列固定放置、world matrix；制作方不负责把每帧脚底贴到道路。
2. 用同一asset_frame生成body/clothing/groom；在身体网格回归138点，再以同一world transform输出所有组件。参数与静态绑定长期保留；渲染需要的逐帧几何可以按帧生成，不要求持久化整段groom缓存。
3. 同一时刻双路渲染与相机/标签输出：source ERP 5760×2880、DS 2880×2880、normal ERP 5760×2880，默认compact。rig/calibration版本显式；0.18m既有合成基线不能当实测外参。A1提供的尺寸感知框接口由后续driver使用，本轮不更改框逻辑。
4. 保存 `episode_id/sequence_id/frame_index/time_tick/time_base/person_id/asset_id/asset_frame/camera_id`；来源表、seed实际配方、root/scene变换及文件状态旁表。两相机共享整数时刻及geometry frame；`sample_time_s`作显示值，不用浮点近似匹配身份。来源组贯穿全部邻帧、双路、邻段、变速、循环及外观派生。
5. compact长期保留最终PNG、138标签、world/pair小数组、valid/person mask、框、camera/配方/来源/版本、序列索引及双路连续检查视频；几何生成权威输入必须在独立资产目录。EXR/临时RGB仅在本次任务回读成功后按原生命周期清理，失败保留；不会引用已清理文件作为必需依赖。检查视频不替代训练PNG。

推荐参数＋绑定的依据：既有30维shape和七块SMPL-X路径可复用；简单衣物已有重心对应；groom已有冻结绑定。本轮数组头显示短发points单帧float32约18.21 MiB，72帧全存约1.28 GiB（未计其它数组），绑定加按帧消费更小。制作方若本来就有合格缓存，直接适配缓存，不能要求其为了此建议重新做参数重建。

## 7. 连续变换、采样与接缝的明确方案

推荐首段采用 **in_place_plus_scene（原地动作＋场景轨迹）**。列向量约定：

`p_world(t) = T_scene(t) · C_yup_to_zup · T_root_residual(t) · p_local(t)`，其中 `C(x,y,z)=(x,-z,y)`。

- `p_local` 是目标shape的pelvis中心局部pose几何，根朝向/平移没有重复烘焙；制作方给出的global_orient由接入适配器转成 `T_root_residual` 一次。现SMPL-X求值若已包含global_orient/transl，则必须显式去除/分解，或标记baked并不再次应用同一root矩阵。不能直接再套上述公式。
- 首段平直道路可去掉源平面行进位移及固定初始heading，保留root的垂直/姿态残差；去除方法、源root和残差均记录。若源存在真实转弯，必须另定rotation分解，不默默将源yaw与场景切线yaw同时叠加。
- `T_scene(t)`拥有场景平面行进和world-Z yaw；height采用序列固定放置或显式路面高度轨迹。源origin到pelvis、Y-up→Z-up各只转换一次。脚在踏板上的最低点不是地面接触点：不能为了贴地逐帧改变root高度，也不能将骑行者误按行人脚底放置。
- 可接受另一模式 **source_root_plus_static_placement**：保留全部源root运动，`T_scene`整段固定。两模式互斥，不悄悄混用。首段建议前者，因为目的包含可控场景轨迹；若来源root分解不可靠则先用后者做诊断。
- body、cloth、groom与138点采用同一时刻和同一最终矩阵。groom先随当帧头皮/头骨运动再施加world，不把头部局部变换重复加到body上；active曲线集合固定，避免按帧删毛闪烁。head-follow已满足首段，不要求头发动力学。
- 首段默认原速、原采样、不循环、不插补；截取一个完整连续区间。视频fps无法代表VFR逐帧时间时使用PTS表。缺帧先缩短/拆分有效区间，不能无标记复制上一帧。
- 若后续确需重采样或变速，逐输出帧保存源相邻帧、插值权重及source_time；关节旋转用SO(3)/四元数插值，不能轴角分量直接线性混合。速度改变后身体/衣物/头发统一时间映射。循环要另验pose/root/cloth/hair位置及速度接缝，记录周期与相位；接缝未通过禁止循环。

## 8. 下一批逐文件最小改动建议（全部未实施）

下列新文件名是建议，不是现有入口；共享模块待A1结束并统一评审后由单一实施方修改。历史调用方与测试不归本轮A2。

| 文件 / 接口 | 建议及最小范围 |
| --- | --- |
| 新 `src/densekp_retrain/motion/contract.py` | 参数或缓存二选一；时间表、固定身份/shape/拓扑、root模式/烘焙标记、缺帧与参数observability验证。无自动推断fps、形体或许可通过 |
| 新 `tools/generate/prepare_continuous_cycling.py` | 仅消费用户完成的标准资产；与旧照片/旧16稀疏帧adapter分开。选一个角色、一个区间，按帧生成并记录R/绑定回读；不新增重建模型选择或BVH自动制作支线 |
| `body/parameters.py`、`body/geometry.py` | 先复用现计算，新adapter显式处理batch/目标固定beta和pelvis/root所有权；非必要不改旧函数。禁止把源beta零填充成目标beta |
| 新 `clothing/sequence.py`，参考 `preparation.py/shells.py` | 冻结一次rest重心对应和faces，再消费每帧body；固定退化面策略、局部offset上限/异常标记，避免逐帧裁面。骨骼蒙皮或缓存走另一个具名契约；允许轻微局部穿模，明显翻折/脱体/持续闪烁拒收 |
| `clothing/groom_transfer.py` / 新逐帧适配器 | 提取或包装只读单帧绑定计算，显式frame，复用reference/active mask/修正；不反复做绑定/剔除。首段可给renderer逐帧folder，避免先重写毛发系统 |
| `body/transforms.py` | 增加具名sequence-explicit matrix入口，保留legacy min-Y语义供历史结果复核；新序列禁止隐式逐帧lift |
| `tools/fisheye/render_equirectangular_source.py` | 读取driver给定matrix，与标注共用；衣物按contract验证，不把骨骼/任意cache套旧1cm shell门槛。现ground gate若按脚底接地，需要新模式显式骑行放置规则；旧模式不改 |
| `rendering/blender/groom.py` | 最小使用driver选定的逐帧folder；若采用rigid head mesh需另有明确head矩阵入口。复用ABC/source_objects，不能复制同一points目录声称头发动态 |
| `imaging/ds.py` 的 world_ground_truth | 与renderer共享显式world matrix和同一asset_frame，DS投影/可见性语义不变 |
| 新 `tools/render/run_continuous_cycling.py` | 一个episode、按时刻顺序的帧单元；两个相机同一几何、显式尺寸/compact、逐帧状态/断点与固定source group。使用A1接口，不编辑A1历史调用修复 |
| 新序列旁表/打包wrapper，参考 `data/packing.py` | pack_index_rows目前仅固定8字段；wrapper按同一row顺序写packed_row_id→时间/来源旁表，避免强制修改Dataset和网络。后续source_split检查新增源视频/邻段关系 |
| 复用 `imaging/output_modes.py`，新增有界验收/视频工具 | 原清理失败保护已存在，不重复修复。各帧任务独占临时目录，检查视频时间/帧数可追溯，检查视频失败单记。下一批定向测试新增独立文件，不修改A1的框测试 |

## 9. 一个角色、一段动作的有界任务及验收

**来源和制作未完成，下面是任务建议，不是自动执行授权。** 不扩十人、不扩多场景、不训练、不把首段打包成正式train/val。

1. 制作方先选来源和格式，给一个角色包＋一个motion包。若接受L1动捕路线，建议只截原帧 `[7560,7632)` 共72帧，保留 `frame_time=0.0416667`（约3.0000024秒占用时长、首末跨度约2.9583357秒），历史72rpm意味着约3.6周期，足够检查实际交替蹬踏；**这是原生候选对应的预算建议，尚未完成适配或认可该子段**。若改用视频，待观看后选择包含至少两个完整可见蹬踏周期的最短连续段，帧率/帧数按来源冻结，不强套72帧或30fps。
2. 只做CPU回读验收：来源/许可/time字段可追溯，所有数组同T、finite、索引单调、target identity/beta/拓扑不变；旋转正交与det检查，固定值broadcast明确；允许unknown证据但不能把阻塞项标passed。BVH单位未确认或视频身份不清时，不进入正式接入生成。
3. 获下一批授权后生成该短段几何；body、cloth、hair使用同一asset_frame。保存 `R@body` 回读差异、世界矩阵及已知点闭环；检查root/heading/轴变换只应用一次。模型空间回读建议沿用历史1e-6m级数值门槛，world浮点容差随dtype/尺度预先冻结；R行和若非严格1，需把数值差异写明，不能改R掩盖问题。
4. 至少检查连续变化的相邻帧；对拟用短段统计**全部**关节/root/衣物/头发位移变化并人工连续播放。重点看蹬踏交替、躯干/头部连续性、root高度不被min-Y重置、衣物不明显翻折/脱体、头发随头不冻结。轻微局部穿模允许；不能只看首尾或一张静帧。若源躯干较稳，不强制人为添加摆动。
5. 先用同一个角色的侧面及前斜角简单诊断预览，时间轴一致；几何/服装问题定位后才启动同一场景双路ERP的短段。双相机同tick/同frame，至少连续一组3帧作投影与compact回读，然后推进**已冻结的同一短段**；不自动延长。视频按真实timebase播放，缺帧显示/拒收策略明确，不能静默跳帧。
6. 双路成品读回PNG/标签/masks/旁表，并在临时源清理后再回读依赖；高分辨率框接口使用A1交付版本作后续联合验收。单视角监督仍按in_view与full crop，pair集合不覆盖它；没有真实GT不做真实精度结论。
7. 首段统一 `integration_only`、`dynamic_acceptance=pending/pass/fail` 与失败原因。源视频/源演员/动作来源/角色家族/外观关系成组；同视频邻段、变速、重采样、循环、换seed和双视角都不构成独立验证。独立验证来源另行准备，不能用该首段自动证明泛化。

验收交付建议：asset/episode manifest、时间映射表、参数与绑定或权威缓存引用、逐帧状态/坏帧表、几何/138点/transform回读报告、侧面/斜角连续预览、同步双路检查视频及compact最终目录。素材不足时交明确的失败报告，不能通过平滑或复制帧伪造连续通过。

## 10. 资源与工作量建议（未实测）

| 阶段 | 模型/CPU/GPU/存储 | 工作量与停止条件 |
| --- | --- | --- |
| 交付检查 | CPU＋NumPy即可；读manifest/小数组，无GPU，无新模型 | 1个角色包、1段动作；格式/来源/许可/root未明确先出缺口，不开始重建 |
| 参数→几何/绑定 | 已有获许可的对应SMPL-X版本、R；CPU可按帧评估，视实际时间决定是否GPU。骨骼/缓存适配不依赖新视频重建模型 | 最多已冻结T帧；首次先连续3帧验证，再整段。人工适配工时未估定，取决于用户标准流程；不声称半天/某GPU必能完成 |
| 双路渲染/视频 | 已有Blender/Cycles与资产，1个健康GPU串行使用即可作为起点；启动前即时检查设备、显存和其它任务。CPU负责后处理/视频编码；不需要DenseKP/YOLO/训练权重 | 若选72帧是144个相机帧，另有前述小预览；先量连续3帧的耗时/显存/临时盘峰值，再审核余下预算。实际设备配置与耗时均未测 |
| 长期与临时存储 | 绑定＋参数优先。body+138点float32每帧127,356 B；M01衣物reference规模约31,344 B/帧，faces/材质一次保存；groom每帧约19,096,800 B，不要求全序列持久化 | 历史compact约25.80 MiB/帧对仅作规划代理；72帧约1.81 GiB，加视频/索引/资产与失败保留。这个数字不是新场景测量，更不是临时峰值；临时EXR/原ERP及内存按3帧探针另测，流式处理而非一次载入整段 |
| 新视频重建（若选） | 本轮没有选择或安装模型；下阶段按单人持续跟踪、固定shape、root导出、遮挡能力和许可评审工具 | 先冻结最短可用源段及模型/硬件预算，再计算；不沿用旧HMR2照片流程假称视频时序重建 |

本轮独立只读脚本成功运行，约1.96秒（该次shell记录，不含人工阅读/网页/写报告），没有运行生产测试或动态质量验收。两次辅助查询遇到字段/路径与语法问题后改正，未产生模型或渲染制品。机器清单内包含本轮证据快照与明确unknown项；outputs被Git忽略不是备份，应连同报告交接。

交付自检通过：JSON解析、脚本AST、10角色/12静态组合计数、所有已列依赖存在、body/faces数组头、两照片单帧标记、两BVH帧号连续及通道有限、参数/缓存互斥交付语义、公开视频未观看/未选定标记，以及报告本地链接。此自检只证明方案及清单内部一致，未验收动态资产。结束时重新检查Git状态；新增的A1报告、测试、验证脚本和调用方修改均保留。

## 11. 评审只需决定的事项

1. 制作方是否接受推荐的“参数＋既有绑定”，或选择已经熟悉的逐帧缓存流程。用户已说明尚未制作，本方案不要求先做齐所有高级资产。
2. 首来源选：L1现有动捕经标准适配，还是坚持同人连续视频来源。后者尚未选片，V1/V2仅页面线索；具体用途许可与连续内容核查完成后再定模型/片段。
3. 是否以M01_v2、原地动作＋场景轨迹作为首样。角色可替换，但保持一个角色/一段动作/integration_only；L1路线72帧仅作有依据的有界预算建议。

**停止点：A2方案评审。** 下一阶段的下载、制作/适配、重建、几何生成、修改公共模块或渲染，均未在本轮执行。
