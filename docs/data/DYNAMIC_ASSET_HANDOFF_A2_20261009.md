# A2 → A1：具体连续资产组合交付

日期：2026-10-09。四个明确组合、三个目标人物家族、两个动作来源组。全部integration_only；**资产侧检查通过不等于A1公共场景已消费**。不把人物、动作、衣服列成可任意组合的笛卡尔积。

机器入口：[index.json](../../outputs/plans/dynamic_asset_handoff_a2_20261009/index.json)、[只读检查结果](../../outputs/plans/dynamic_asset_handoff_a2_20261009/validation.json)、[最小契约样例](../../outputs/plans/dynamic_asset_handoff_a2_20261009/contract_sample.json)。schema显式标为proposal，`A1_schema_effective=false`、`public_consumer_tested=false`；样例没有observed/packed行或假图像。

## 组合与接受状态

| 组合 | 几何/衣物/头发 | R138 | 人工状态 |
| --- | --- | --- | --- |
| [M01 APD6526](../../outputs/plans/dynamic_asset_handoff_a2_20261009/cycling_m01.json) | 76帧、male固定shape；原simple绑定＋原groom | 标准模型原证据，compatible | 用户已接受身体/衣物/头发，保留左臂快摆与亮度限制 |
| [F02 APD6526](../../outputs/plans/dynamic_asset_handoff_a2_20261009/cycling_f02.json) | 76帧、female独立固定shape；F02 simple绑定＋short-bob | 当前female模板/索引回归＋原证据，compatible | 新动态候选，待用户看视频 |
| [BEDLAM原配基础版](../../outputs/plans/dynamic_asset_handoff_a2_20261009/bedlam_base.json) | 91帧、原neutral locked-head、16beta；原nt_belle_007缓存，无groom | 本轮限定模型compatible | 原身体/动作/衣物已接受 |
| [BEDLAM眼睛/头发版](../../outputs/plans/dynamic_asset_handoff_a2_20261009/bedlam_eyes_hair.json) | 同一身体/衣物；原生眼球材质＋本地选定短波波头 | 与基础版共用身体及结论 | 新眼睛/头发未收到接受反馈，仍pending |

播放入口：[女性76帧](../../outputs/experiments/cycling_female_asset_a2_20261009/preview/index.html)、[行人眼睛/头发](../../outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009/preview.html)、[行人原配基础版](../../outputs/experiments/bedlam_native_walk_a2_20261009/native_attempt_2/preview.html)。M01现有64 samples视频直接由其manifest引用，没有重复渲染。

BEDLAM兼容证据见 [R138核查](bedlam/BEDLAM_R138_COMPATIBILITY_A2_20261009.md)：原算子可精确复现，逐面角UV/蒙皮权重相同，5个受头部雕刻影响的点仍位于相同表面位置。仅新manifest标ready；原pending文件和历史报告保留。鞋位移、去趾、其它模板不继承结论。

## 必须消费的时间、root和材质

APD男女共享同一 `timeline.json`：76条严格递增的原整数时间，源帧30…105、interval39949…40010 ticks，time_base=1/1,000,000为推断微秒。不能套30Hz/stride3、假装10Hz、分别nearest或只改文件名。两目标人物各自固定beta和 `transl=-自身rest pelvis`；没有源行进残差。场景拥有行进轨迹、固定放置高度和一次轴转换，骑行脚不能套“脚底贴路”门槛。

BEDLAM使用已发布处理后的30Hz时间：asset/body/cache 0…90；衣物仿真帧101…191。原100Hz来源映射仍不宣称精确。身体和衣物已经烘焙原global orient/trans，scene只统一施加一次固定放置（若另加轨迹须另立显式所有权，不能重复源root）。若A1另选0,3,…,90成为10Hz，必须记录这是显式采样，本交付不偷偷降采样。

共同列向量约定：

```text
body_world(t)  = scene_from_model(t) @ body[asset_frame(t)]
cloth_world(t) = scene_from_model(t) @ cloth[asset_frame(t)]
kp_world(t)    = scene_from_model(t) @ (R @ body[asset_frame(t)])
hair_world(t)  = scene_from_model(t) @ head[asset_frame(t)] @ reference_points
```

manifest的模型坐标是米、Y-up；`scene_from_model`含一次Y-up→Z-up变换 `(x,-z,y)`。APD的head由各目标模型关节求出，参考点在各自shaped-rest、pelvis-centered空间。BEDLAM新groom参考点在其身体asset0模型空间，含一次固定适配；head为frame0→当前头部且已含原root。不能将两种reference空间混用或共享M01的H。

衣物两类明确区分：

- APD是固定barycentric/法线offset派生，缓存用于消费，权威绑定/拓扑与GarmentRest程序化材质一并引用。
- BEDLAM是原配仿真缓存，独立faces、face-corner UV、diffuse/normal及明确roughness/normal设置；不能运行simple shell重新生成，也不能借body UV代替衣物UV。

身体标签永远来自身体R138，不来自衣物、groom或原APD14点。轻微穿插不自动否决；没有“零相交即通过”的契约。

## A1当前进度与尚缺能力

本轮结束前只读核对 [ROI-only报告](../rendering/ROI_ONLY_RENDER_A1_20261009.md)、`acceptance.json`、`pack/same_buffer_comparison.json` 及现有视频文件：A1已交付11时刻、22份ROI、22行打包数据，6组Dataset同缓冲对照exact，生产全图RGB为0。一次JSON序列化故障后复用首帧，实际2次Blender启动。这里是引用A1现有验收，不是A2重跑或A2资产已接入。

后续接入沿用**同一公共生成器及ROI-only路径**，不恢复旧DS往返或另造骑行场景。ROI标签/框继续为完整ERP像素坐标，局部图offset只在既有图像读取契约处理；`in_view & inside_full_crop`监督与单视角语义不改。

| A1调用点 | 最小资产适配能力（proposal，尚未由A2实现） |
| --- | --- |
| `data/road_pass.py`、`plan_road_pass.py` | 从原timeline消费可变帧数/整数时刻；明确root_mode；不再强制 `start+3*k` / `source_root_relative` |
| 放置/地面支持逻辑 | APD显式固定放置高度及骑行support policy，不能以每帧最低脚点反推；行人保留原root及相应支撑检查 |
| `render_road_pass_sources.py`及既有ROI帧循环 | 同一asset帧消费body/cloth/R/head；支持显式逐时刻scene matrix、可变帧数；保留现有ROI-only输出能力 |
| 材质装配调用点 | APD沿用现有character/simple材质；BEDLAM读取独立UV/纹理、无groom基础版以及新增眼球属性材质；head矩阵路径可显式引用，不能强制只放geometry/head_transforms.npy |
| `sequence_packing.py` | 从timeline取source/episode tick及time_base；不要强制30Hz或伪造original_100fps_mapping；ROI布局字段及Dataset契约沿用A1已有实现 |

当前代码仍有31帧索引范围、30Hz/stride3、固定model_to_world、必需character appearance/groom等来源特定假设；proposal不代表已生效。本轮不改上述文件、公共材质、Dataset、ROI规则或训练。下一批可优先接**已接受M01**验证APD timeline/scene/support接口；F02待用户观看后可复用同一接口。BEDLAM基础版R障碍已解除，但需native材质和可选groom读取，眼睛/头发版还需用户确认外观。

## 可运行的只读资产侧检查

```bash
OPENBLAS_NUM_THREADS=1 python tools/validate/check_dynamic_asset_catalog_a2.py \
  --index outputs/plans/dynamic_asset_handoff_a2_20261009/index.json \
  --output /tmp/dynamic_asset_check_new.json
```

输出必须不存在。脚本只读参数/缓存/时间/绑定及持久依赖，不加载SMPL-X、Blender、DenseKP、道路或Dataset；验证全帧有限值/拓扑/时间/R回读、shape与参数、材料引用、固定毛发集合、小型scene/head代数。groom复合抽样stride1024明确记录；各资产生成阶段另有全部毛发点有限值检查。任意诊断矩阵不是道路放置或公共投影成功的证据。实际4项检查0.362秒。

小型契约样例只有资产帧、时间和支持策略；scene matrix由A1实际配方给出，没有伪造相机/图像/观测。源模型、纹理、groom、旧参数/缓存均引用，不复制到新目录；长期路径清单见各manifest。四个组合不是独立train/val：M01/F02共享APD来源组，BEDLAM两外观共享KIT425动作与同一body family，旧KIT425项目划分不改，换seed/邻段/衣服不能制造独立验证来源。

## 2026-10-09 增量：F02＋Bike Fit Adviser 山地车动作

新增第五个具体组合 `cycling_F02_v3_bikefit_XBLMidr4dps_8662_8782_simple_v1`，旧四条manifest/接受状态/来源引用保留。[观看入口](../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/index.html)、[报告](../motion/cycling/MTB_F02_DRESSED_ASSET_A2_20261009.md)、[manifest](../../outputs/experiments/mtb_f02_dressed_asset_a2_20261009/delivery_manifest.json)。源身体120帧已获用户接受，新女性比例/穿衣/groom仍pending。

原源帧8662–8781，PTS8670662–8789781，time_base1/30000、步长1001；不改成APD25Hz、30Hz整数或10Hz。固定F02自身30beta/female模型，transl=-女性rest pelvis；body/cloth/R已含root/global，groom用女性自身head，scene拥有道路行进与固定放置。衣物仍F02原simple绑定、原程序化材质，groom固定reference/active集合；不是BEDLAM换衣。

只读核对当前实际A1 `dynamic_scene.py` 并调用adapt_proposal/select_timeline/geometry_paths，120条原时间及真实路径通过，提供 `a1_asset_adapter.json`（densekp_scene_asset_v2）。这更新了上文历史proposal的能力判断，但没有改写旧场景证据或声称新组合已实际渲染。当前已有native timeline、scene-owned progression、fixed cycling height和simple/groom材料能力；`person_time.py`提供显式多人物nearest原观察及有理数误差记录，本轮未调用它重采样。

仍由A1选择公共场景yaw/道路轨迹/固定高度并实际完成ROI、旁表、Dataset回读；不套M01的1.2m为已验收高度，几何最低Y范围[-0.951465,-0.767322]m。详见 `contract_example.json`、`a1_interface_check.json`；没有伪造scene matrix或packed rows。

新动作来源族bikefit_XBLMidr4dps与APD分开，但角色家族cycling_F02共享；两者共同约束划分，不因换动作/外观成为独立验证。视频许可unknown，仅内部integration_only，不发布或移入datasets。全目录五组合只读检查通过，统计中的3个motion group仅指不同素材来源组，不是统计独立性认证。
