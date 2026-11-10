# A1：组件种子采样进入 ROI-only 道路场景

日期：2026-10-10。**已交付两套由组件规则采样的组合、两段单骑行者 31 时刻双路 ROI episode。** 两段均为 `integration_only/shared_source_diagnostic`；没有来源独立验证、模型推理、训练或正式数据集发布。直接观看：[正面组合视频](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_00/run/pack/dual_roi_components.mp4)、[背面组合视频](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1/run_gpu1/pack/dual_roi_components.mp4)及[统一页面](../../outputs/experiments/seeded_asset_composition_a1_20261010/index.html)。完整机器验收见[acceptance.json](../../outputs/experiments/seeded_asset_composition_a1_20261010/acceptance.json)。

## 采样、接口与兼容范围

[A1 采样配置](../../configs/seeded_asset_composition_a1_20261010.json)引用 A2 的[权威组件清单](../../configs/asset_composition_a2_20261010/components.json)，分别限制 body=M01/M02/F02、motion=APD6526/bikefit/p2_straight/p1_comfortable/p1_alternate、三件 simple 衣型和三种 groom。身份/固定 30 beta/皮肤随 body；motion 的原录制与参与者、许可先过滤。衣物只支持标准 SMPL-X 语义拓扑的固定 barycentric simple 衣型；头发只支持已有 rest 头皮三角框架转移及刚性头矩阵；骑行只接受 `fixed_pelvis_scene_owns_progression`。BEDLAM 原配仿真缓存、未知 pose/root 表示和不足 3 秒的原时间窗口被明确排除。该兼容条件不是已制作成品的四件套白名单；最终仍调用 A2 `composition.plan` 做实际源文件、参数维度、拓扑/面序、绑定、时间与依赖检查。所有新组合的人眼接受状态保持待观看，数值通过不自动继承原角色视频的接受结论。

[编排入口](../../tools/experiments/seed_asset_composition_a1.py)使用稳定 ID 顺序；body、motion、clothing、hair、time、layout、camera 各有带版本/seed/episode 的局部随机流。先对可用组件分别确定随机顺序，再按条件排名选择兼容组合，并对第二份施加事前冻结的“不同 body、不同 motion、至少一次跨 body 衣发”配额。这是条件采样，**不承诺全部兼容笛卡尔积均匀分布**。[8份冻结计划](../../outputs/experiments/seeded_asset_composition_a1_20261010/frozen_plans.json)由 master seed `20261020` 生成；重复计划结果逐项一致，不构造 SMPL-X 模型、不查询/占用 GPU、不启动 Blender。6份仅做 A2 `plan` 与原整数时间可用性检查；没有为它们制作几何。第二个 seed 的定向测试证明组件选择会改变，布局变化不是唯一随机项。

实际时间窗口在组合层保留全部已适配短动作缓存：F02＋p2 的 120 条，M02＋APD 的 76 条。场景分别选择合法源起点记录 12 和 0，使用 `nearest_original_record_ties_earlier` 映射到 31 个 10 Hz 合成时刻；源 tick/time_base、原 source frame、组合 asset frame 和量化误差均保留。p2 源时基为 `1/10000000` 的发布标称时钟，APD 为原整数微秒时钟及其既有不确定性；两者没有被硬写成 30 Hz，也没有插值、循环、倒播或重复裁切。每段的 body/衣物/头发/动作 ID 固定，姿态随原时刻推进。

| 已制作组合 | body / motion / clothing / hair | 场景视角与来源 | A2 缓存 |
| --- | --- | --- | --- |
| episode 00 | F02 / p2_straight / M02_simple / M01_groom | 正面；`zenodo_10668611_participant_2`，source 原点记录12 | 新制作；body、cloth-rest、cloth-motion、hair-rest 均为新键 |
| episode 01 | M02 / APD6526 / M01_simple / M02_groom | 背面；`APD_5724486_v1_identity_session_unknown`，source 原点记录0 | 新制作；定向修正时**核对完整交付后命中**，没有重复制作 |

两项均来自种子条件选择，非手写最终四件套。A1 调用 A2 公共 `plan/build`，组合缓存属于本任务目录；A2 清单、算法、旧产物只读。复用交付时核对解析组件版本、依赖路径/大小/纳秒修改时间、必要持久文件、全段 body/R138/head/cloth 有限值、时间和数值验证；同名不同依赖及半成品拒绝，不当作命中。[缓存记录](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1/build_record.json)中的 `validation.compute_seconds` 是最初制作的历史值，**不计成修正时的新制作**。元数据戳和显式版本不是密码学内容证明；本轮未计算 SHA256。

场景模板只提供道路、rig、光照和受约束参数范围。新 A1 adapter 指向本次 A2 交付的真实 body/cloth/groom、材质、参数和时间；参考 recipe 中人物的旧资产字段被替换，不会用旧衣发偷偷出片。实际 body、cloth、R138 共用 `scene_from_model[t]`；groom 用 `scene_from_model[t] @ head[asset_frame]`。源固定骨盆 root 只施加一次，整段一个固定道路高度与固定朝向；不逐帧贴地、缩放人体或修改原姿态。两段保留 5760×2880 逻辑 ERP、ROI-only 64 samples CUDA、物理有效域与渲染覆盖区分、原生可见实例框和 deployment 1.0 完整 crop。背景没有单独渲染，也没有完整 ERP RGB 读写。

旧整套 `seed_scene_modes.py` 的配置/seed/RNG 行为保持，独立 before 在[before目录](../../outputs/experiments/seeded_asset_composition_a1_20261010/before/)。新组件路线用新入口；旧单人/混合默认、M04 行人来源及两种 root 没有迁移。[混合 CPU 计划检查](../../outputs/experiments/seeded_asset_composition_a1_20261010/mixed_mode_cpu_plan_check.json)证实新骑行组合＋原 M04 可共同映射到合成 10 Hz 时钟，骑行源 `1/10000000`、M04 发布 `1/30` 分别保留；没有渲染混合段，也未扩大行人自由组合。

## 布局修正与真实验收

正面段 12 个冻结布局候选只检查前两个：attempt0 双边缘不足，attempt1 完整31时刻几何/精确 groom/ROI 检查通过。两机同时正面 **31/31**，双路物理边缘 **31/31**；每个时刻两机各至少8个域内、5个域外 R138 点且共同域内至少8，超过既定16/31与连续0.5秒门槛。最小共同crop支持仍有43点。首末同步20→5或5→20的完整端点过界未通过，按 `edge_coverage` 目的如实记录为诊断失败，不把边缘覆盖改称完整穿越。

背面段初次使用平均朝向求固定 yaw，12/12 个**同一冻结布局候选**均因既有身体前向与速度≤25°门槛拒绝。逐帧前向角范围约49.87°，说明存在满足±25°的单一固定朝向；这是放置求法错误，而非 motion 不可用。仅在新的[修正目录](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1/)用 `heading_minimax_fixed_yaw_v2` 将原角区间居中，重放完全相同的12个候选 seed/布局/rig 参数；attempt0 通过，未放宽门槛或改动作。原12次拒绝与[候选一致性记录](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1/candidate_replay_comparison.json)并存。背面两机实际视角 **31/31**、双边缘 **28/31**、连续2.7秒；端点过界同样只是诊断。首段原放置规则 `heading_mean_v1` 和后段 `v2` 各在冻结产物标记；当前可复用配置已显式选 `v2`，原[冻结策略](../../outputs/experiments/seeded_asset_composition_a1_20261010/frozen_policy.json)保留当时版本，不能把两段说成同一 yaw 实现的重渲染对照。

两段各一个 Blender 进程加载一次场景和人物、一个常驻同步 CPU 消费者，合计 **62时刻/124次 ROI 相机调用/124个观测槽位/124条有效原生框行**。没有无框槽位，但仍使用 `record_unavailable_v1`，未来自然无框不伪造训练行。[逐帧真实消费者审计](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_00/actual_consumer_audit.json)与[背面审计](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1/actual_consumer_audit.json)比较了全部62时刻的预检/实际整数 ROI、两台相机矩阵、R138 世界点、源帧与源 tick、body/cloth/head 同步和本次组件路径，全部通过；两份 compact 均无剩余临时 EXR。ROI 面积正面每路约0.802–5.392M像素，背面0.554–1.573M，均未超过7M上限。源时间严格递增，视频各31帧 PTS 0..3.0秒，末帧展示0.1秒另列，未复制帧。

每段在首次 Dataset item 读取前写好首/中/末双路6行计划，实际各读 **6行、合计12行**。旧同一 Dataset 的局部 ROI RGB→256-square、原 R138 仿射、可靠GT的 `in_view & inside_full_crop` 监督、`_trans` 与完整ERP `img_size` 均通过；遮挡点仍参与，第三值不作置信度。正面第0/15/30帧两路可靠监督点分别为129/132、112/118、86/95，包含遮挡点。packed episode 和序列旁表含组件ID/版本、组合交付、身体家族、motion 原来源、源起点与两时钟映射。正面段因谱系传播代码晚于预检冻结，保留[原渲染 recipe](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_00/run/recipe_render_frozen_before_provenance.json)，仅增补元数据后另打[带组件谱系的 pack](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_00/run/pack_components/)；[逐数组比较](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_00/run/pack_components/provenance_repack_comparison.json)为完全一致，没有改图像/标签/监督。原一次性脚本在成功打包后因返回值解包数写错退出1，原/新 pack 的独立校验均通过；失败没有藏成成功计算。背面段直接从生产 recipe 带谱系打包。

两段首/中/末 ROI 已目视核查，没有发现明显 body/衣物/头发错绑；这只代表本次有限道路探针，**不提升新组合的人眼接受状态**。黑色区域是物理有效域外，灰色视频排版区是未渲染；视频标题列组件、源整数时间和边缘目的。F02/p2 和 M02/APD 的不同角色/动作不是来源独立性的证明；两段与旧 body/motion 衍生版本仍按[来源关系](../../outputs/experiments/seeded_asset_composition_a1_20261010/source_lineage.json)整体追溯。participant_1 两段同源，KIT425 是历史验证来源，bikefit 许可仍 unknown，预训练来源重叠未知。当前不划正式 train/val、不移入 datasets。

## 成本与使用限制

| 实测项目 | 正面 F02/p2 | 背面 M02/APD |
| --- | ---: | ---: |
| 场景/人物装配回调 | 20.43 s | 21.46 s |
| 首时刻 | 14.93 s | 21.64 s |
| 后30时刻均值/范围 | 15.55 s / 10.06–31.82 s | 13.87 s / 12.43–16.07 s |
| Blender 含同步消费者 | 503.48 s | 460.74 s |
| 打包＋6行回读＋原视频后的整段交付 | **519.27 s** | **474.39 s** |
| GPU采样峰值 / Blender RSS峰值 | 7985 / 8653 MiB | 8525 / 9060 MiB |
| 临时源采样峰值 | 383.6 MiB | 165.1 MiB |

两段生产与必需收尾合计 **993.66秒（16.56分钟）**。A2新几何制作约7.49秒，三次CPU预检约52.62秒，另有谱系再打包约7秒、两段字幕视频重编码约2秒；这些均计入本批45分钟上限，显著低于上限。GPU0 正面运行期间实际空闲；背面准备启动时 GPU0 新出现他人计算，runner 在 Blender 前拒绝并保留资源现场，改用当时空闲的 GPU1；两次沙箱内 `nvidia-smi` 错误9也都发生在 Blender 启动前。实际成功 Blender 启动各一次，成功帧均未重渲染。最终输出目录约 **301 MB**，其中正面 run 含双 pack 约141 MB、背面 run 约77 MB、组合缓存约80 MB；这些是不同组合/ROI大小和额外谱系副本的实测存储，非等负载性能比较。函数分项及实际设备 UUID 见机器验收，不能把相机调用秒数当纯 GPU kernel 时间。

定向 CPU unittest **24项通过**，覆盖冻结计划重放、不同seed组件变化、来源前置过滤、短 motion/锁定衣型拒绝、缓存命中/失效与半成品拒绝、非首帧原时钟、固定yaw修正及旧整套入口兼容；未跑全库、DenseKP/YOLO 或参数更新。计划入口只读调用 A2 公共 plan；制作入口显式调用其 build 后复用公共 `plan_mixed/preflight_mixed`，渲染仍走 `run_continuous_roi.py`，没有第二套人体生成器、Dataset或监督实现。

在**新的空目录**中复用命令：

```bash
PYTHONPATH=src:. python tools/experiments/seed_asset_composition_a1.py plan --config configs/seeded_asset_composition_a1_20261010.json --output <new_output_dir>
PYTHONPATH=src:. OPENBLAS_NUM_THREADS=1 python tools/experiments/seed_asset_composition_a1.py prepare --config configs/seeded_asset_composition_a1_20261010.json --output <new_output_dir>
# 只对 geometry_preflight.json 为 passed 的 prepared episode 显式调用：
PYTHONPATH=src:. python tools/experiments/run_continuous_roi.py --output <new_run_dir> --prepared-episode <new_output_dir>/episode_00/preflight/<episode_id> --storage-mode roi_only --consumer-mode persistent --observation-policy record_unavailable_v1 --gpu-uuid <checked_idle_GPU_UUID> --budget-seconds 1800
```

实际本批第2段采用[修正后的 prepared episode](../../outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1/preflight/)，不能对旧失败 preflight 执行渲染。新 config 中 `heading_minimax_fixed_yaw_v2` 是明确版本，两个 episode 的精确历史重放应分别用产物内冻结配置。对以后其它 motion/身体的固定高度、朝向、衣物和头发仍须实测；缓存依赖无 SHA 只用显式版本与文件元数据，不能宣称对恶意同大小同时间替换有密码学保证。ROI只保障原生框及现有完整crop，不保障尚未冻结的框增强；来源划分、正式配额、多人/路口和训练仍属后续评审。
