# DenseKP B4 历史工具退役规划

2026-10-03。**仅核查与规划，全部实施建议待评审。本轮未执行测试或计算，未删除或修改运行文件。**

## 1. 实际基线与范围

HEAD：`ada22953bb0696e5a3870d14fb60ccbd715925dc`，master，`Complete B3 asset helper migration and regression suite consolidation`。
开始时仅用户 `docs/TODO.md` 修改；原样保留。未发现适用AGENTS.md。历史B3报告“未提交”是当时记录，不重复实施。
本轮仅新增本文及 [逐文件 inventory](../../outputs/plans/pruning_b4_20261003/inventory.json)。

旧B4清单 **57个工具均仍存在**；建议48个随完整依赖组直接退役，2个先提取必要长期检查，5个保留、2个暂缓。
集中批次还需明确退役3个测试/辅助Python和2份迁移AST fixture：共 **53文件**；前置两旧驱动完成后累计 **55文件**。
“直接删除”指不用迁业务算法，仍须先同步测试/导航和核查闭包；不是本轮授权。

纠正旧清单：audit_skin_texture生成当前body_uv；inspect_bedlam2_groom生成绑定入口消费的groom数组/report；prepare_source_trial_config生成训练/比较必需记录。这三者不能按audit/inspect命名或无import判死。

## 2. 证据与保留系统

核查相关源码AST、直接/相对import、动态字符串、runpy/subprocess/CLI、Hydra、安装/日常检查、当前文档与制品生产关系。
BVH短名按脚本目录确认；标准库json不映射到项目io/json，报告losses不算模块。字面命中在inventory单列，不自动视为执行依赖。
旧AST中的模块名/报告文字与真实capture测试runpy调用分别判断。未知外部私有调用不无限阻止明确接口退役。

只读指定小记录/元数据：body texture审计JSON和body_uv文件大小（未载数组）；
`outputs/experiments/cycling_commuter_pilot_20260930/groom_import_audit_v2/report.json` 的Blender版本、cases/export；
source-isolated的training_recommendation/regression_reference_audit；B2/B3报告与已有Blender验收manifest。
不遍历整个outputs/资产库，不读checkpoint内容。历史运行证据不作为本轮新通过结果。

| 保留能力 | 当前入口与实现 |
| --- | --- |
| 骑行/行人、来源及角色 | prepare_cycling_images → reconstruct_cycling_hmr2 → convert_cycling_smplx；audit_pedestrian_poses、prepare_source_isolated_assets；body.parameters/geometry，SOURCES/STYLES配方 |
| 简单衣物、皮肤、groom | prepare_adult_candidates/prepare_commuter_pilot/prepare_cycling_characters；clothing.preparation/shells/binding/groom_transfer；audit_skin_texture → body_uv；inspect_bedlam2_groom → bind_commuter_groom/bind_adult_grooms；export/prune/relax和眼部准备 |
| 场景/成像/标注 | 街景/森林/hidden alley、原生多人/链接单人、当前DS/Omni/透视；rendering.blender、imaging.ds/omni、annotation。独立河岸另提退役 |
| 来源→样本→框→Dataset | split防泄漏、所有人物及遮挡者、打包/框派生、具名裁剪；mixed27/adult24既有资产和索引仍消费 |
| 模型/训练/评价 | train_keypoints、train_source_isolated_trial、core.inference/training_audit；candidate、DS遮挡/框控制、两suite及summary/compare；旧骑行/行人回归 |
| 诊断/安装 | diagnose_adult_contacts+公共crossings；安装检查、限定源码/CPU日常检查；长期集成见§4 |

`evaluate_ds_box_control.py:10` 真实导入 `evaluate_ds_occlusion_control` 的 np/torch/Image/DatasetKeypoints/DenseKP/Hydra/restore_crop_xy/draw_case。
两者有保留用途，原位不改；不是B4删除阻塞，不扩大成evaluator重构。

保持138点/regressor、人体衣物groom标签共用变换、同源隔离、旧NPZ权重和可靠遮挡GT监督；256-square/内部192列无32px偏移；
有效mask不用黑RGB推断，keypoint_value不变置信度；DS/Omni距离/周期/分母分别保持；目标20260320-riding-dataset3不混031/0311；框增强权重仍未获部署替换认可。

## 3. 逐文件决定与完整依赖组

所有下列工具是当前HEAD存在的文件。代码import行号、字面引用及当前导航命中写入inventory；本表补充独立CLI用途与停止能力。
无仓内import不是删除理由。不存在向公共包迁移失败main/体素/修复的计划。


### once_audits

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/experiments/audit_cycling_character_assets.py` | 直接删除；BEDLAM包/许可/预览盘点 | 完成的选材审计；停止此批压缩包盘点和预览复制，保留许可/来源附件。未发现当前制品生产依赖。 |
| `tools/experiments/audit_eye_geometry.py` | 直接删除；首帧眼部网格/像素排错 | 已完成眼部诊断，当前prepare_procedural_eyes仍保留；退出旧病例分析。 |
| `tools/experiments/audit_skin_texture.py` | 保留；UV/component_labels NPZ和皮肤来源审计 | main:26写body_texture_audit/body_uv.npz；prepare_cycling_characters:150、prepare_adult_candidates:68、prepare_commuter_pilot:105、prepare_procedural_eyes:19实际读取。audit_body_uv的JSON不能替代此NPZ生产者。 |
| `tools/experiments/inspect_bedlam2_groom.py` | 保留；Blender Alembic groom数组/report导出 | bind_commuter_groom:52、bind_adult_grooms:53–62读取report cases/export及points/offsets/radius；是实际资产生产者，保留CURVE/CURVES及单位/变换规则。 |
| `tools/experiments/inspect_smplx_rigs.py` | 直接删除；locked-head参考rig对照 | 已完成专用排错；export_groom_reference仍保留；对应AST项退役。 |
| `tools/experiments/prepare_source_trial_config.py` | 暂缓；训练前置recommendation/reference audit和提案status | training_recommendation被train_source_isolated_trial:173读取；regression_reference_audit被compare_source_trial:27读取。仍有过时pending/F04提案和status写出，暂缓拆分，不能删除独有生产者。:58 smoke名仅报告说明不是动态import。 |
| `tools/validate/audit_box_trial.py` | 直接删除；400步历史字段纠正/回放审计 | 完成该次actual_crop_context_v2等历史修订；保留final_audit/活动status和修订来源，不再提供一次性改写工具。 |
| `tools/validate/audit_interrupted_trial.py` | 直接删除；旧中断trial/权重步数排错 | 仅该次不完整400步诊断；保存失败权重/报告，现行gate不变。 |
| `tools/validate/p0_pyrender_crosscheck.py` | 直接删除；三帧EGL/pyrender交叉检查 | 明确退出非主后端诊断协议；Blender验收不是其等价替代，但主线无此必要依赖。 |
| `tools/validate/pyrender_headless_smoke_test.py` | 直接删除；EGL/pyrender headless smoke | 与pyrender专用检查一起退役，保留Blender headless入口，不混同后端。 |
| `tools/visualize/review_eye_background_fix.py` | 直接删除；固定前后眼部/背景画廊 | run_eye_background_fix和rendering.jobs保留；仅退旧画廊，不退出透视能力。 |

### bvh

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/experiments/audit_cyclist_bvh.py` | 直接删除；read_bvh/forward_kinematics及原单位审计 | prepare_cycling_fit_inputs:14、review_cyclist_audit:13真实短名import，按同目录核实；三者同组退出，非当前独立照片链。 |
| `tools/experiments/find_cyclist_clip.py` | 直接删除；BVH十秒窗口筛选 | 消费audit并输出selection；停止原窗口/单位/打分协议，非照片姿态替代。 |
| `tools/experiments/prepare_cycling_fit_inputs.py` | 直接删除；两BVH clip和22-joint映射包 | 真实依赖audit_cyclist_bvh两函数；原.01单位推断不升格为当前契约，整条拟合输入支线退出。 |
| `tools/experiments/review_cyclist_audit.py` | 直接删除；BVH运动/映射图 | read_bvh真实同目录导入；对应frozen main随BVH组退出。 |
| `tools/experiments/search_babel_cycling.py` | 直接删除；BABEL注释骑行检索 | 早期BVH候选来源选择，不参与现有照片/行人准备；明确停止独立检索CLI。 |

### failed_clothing

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/experiments/build_commuter_clothing.py` | 直接删除；旧rest_garment/领口/裤装/碰撞修正 | 失败衣物制作主入口，当前接受链使用clothing.preparation/shells/binding；停止失败分支再制作。 |
| `tools/experiments/build_commuter_polo_collar.py` | 直接删除；失败polo领口制作 | :11真实import repair_commuter_top.edges_of/project；同闭包退出，不迁helper。 |
| `tools/experiments/build_continuous_commuter_pattern.py` | 直接删除；连续衣片smooth_union制作 | 失败分支专用，不是prepare_simple_clothing；停止此配方。 |
| `tools/experiments/commuter_union_volume.py` | 直接删除；capped/scan和体素union | capped内部使用；scan测试已独立到tests。validate_commuter_union同名capped是局部BVH值，不能误判函数依赖；不迁体素实现。 |
| `tools/experiments/diagnose_commuter_clothing.py` | 直接删除；旧衣物rest/bound/final相交诊断 | 产生diagnosis/evidence供commuter_failure_sections；公共crossings及diagnose_adult_contacts保留。 |
| `tools/experiments/extract_commuter_union.py` | 直接删除；旧union envelope提取 | 消费同组体素输出；对应冻结main是迁移期检查，随能力退出。 |
| `tools/experiments/prepare_commuter_cloth_path.py` | 直接删除；旧rest到pose cloth初态 | 失败模拟支线，不是真实连续motion；停止其准备CLI。 |
| `tools/experiments/repair_commuter_top.py` | 直接删除；edges_of/average/project/DQS及失败修复main | build_commuter_polo_collar:11、settle_commuter_top:16、validate_commuter_union:13真实消费者均同组。测试已不导入原DQS；不迁project/tree等失败算法。 |
| `tools/experiments/settle_commuter_cloth.py` | 直接删除；keyed_mesh和Blender cloth模拟 | settle_commuter_top:15真实消费keyed_mesh；整组退役，不保留模拟再制作。 |
| `tools/experiments/settle_commuter_top.py` | 直接删除；失败top settling配方 | 依赖keyed_mesh/edges_of和公共crossings；只退出专用配方，公共相交不删。 |
| `tools/experiments/smooth_commuter_weights.py` | 直接删除；失败衣物权重平滑 | 专用制作及冻结main一起退出，不改当前binding。 |
| `tools/experiments/trim_commuter_neck.py` | 直接删除；失败衣物颈部裁切 | 专用网格修形CLI退出，不影响当前shell或标签。 |
| `tools/experiments/validate_commuter_union.py` | 直接删除；旧union网格/权重/相交核查 | :13依赖repair.edges_of/average；消费同组体积制品；不迁此支线，不删公共crossings。 |
| `tools/render/review_commuter_component.py` | 直接删除；失败组件rest/bound/final预览 | 当前场景索引仍列此能力，须移除推荐行；review_cycling_characters现行角色预览保留。 |
| `tools/validate/audit_commuter_component_contacts.py` | 直接删除；失败组件相交审计 | 已直引公共crossings；12脚本冻结AST之一，移除该项和有限映射。 |
| `tools/validate/audit_commuter_pose_sources.py` | 直接删除；失败姿态/组件相交审计 | 硬编码失败组合，非来源隔离公共模块；已直引crossings；AST项及映射随支线退役。 |
| `tools/validate/audit_commuter_repair_bodies.py` | 直接删除；失败身体对照/SMPL-X重算诊断 | 不是当前身体生成唯一入口；停止此专用重算与AST项，不执行模型。 |
| `tools/validate/probe_commuter_overlap.py` | 直接删除；失败F02/P2 winding/体内外探针 | 专用失败归因，保留既有结果，不迁为公共算法。 |
| `tools/validate/validate_commuter_studio.py` | 直接删除；失败studio Dataset裁剪统计 | 固定外观病例诊断退出，当前裁剪/归一化契约保留，不是唯一数据检查。 |

### duplicate_smoke

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/experiments/train_cycling_smoke.py` | 直接删除；早期同源两人训练/helper再导出 | tests/test_training_audit:61、test_inference:104是实际兼容消费者；删除smoke断言保留active训练入口断言。verify_training_audit.capture动态引用同批退出。 |
| `tools/validate/smoke_densekp_lightning.py` | 直接删除；历史tiny Lightning fit | 会真实更新参数；当前CPU生命周期和真实模型验收保留。删除不授权执行。 |
| `tools/validate/smoke_densekp_training.py` | 直接删除；旧head/full tiny更新 | losses是普通报告字段，不是模块引用；退出早期CLI，保留当前objective/生命周期测试。 |

### old_sweeps

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/fisheye/run_fisheye_seam_sweep.py` | 直接删除；旧固定yaw seam扫描/排名 | 独立CLI有专用能力；建议停止75/90/100/110扫描筛选，不声称现行周期公式测试覆盖该排名。Omni生成保留。 |
| `tools/fisheye/run_human_scale_sweep.py` | 直接删除；旧尺度/距离/可见点范围扫描 | 停止旧固定比例和选择协议，不改当前大占比人体/有效区/Omni算法。 |

### scene_unused

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/scene/build_reference_riverfront.py` | 直接删除；独立河岸场景制作/渲染 | 场景索引明确非主线资产；建议显式停止再制作。scene AST及旧scene验收列表同步处理，已有blend/预览不删。 |

### integration

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/validate/compare_blender_acceptance.py` | 保留；显式before/after制品比较含EXR通道 | 长期离线检查；tests.artifact_comparison不覆盖EXR通道，不机械合并，不扩大路径忽略。 |
| `tools/validate/verify_blender_assembly.py` | 保留；实际evaluated网格/groom/材质/相机/通道快照 | 显式case、assembly/render、output；真实Blender边界长期保留，数组模拟不能替代。 |
| `tools/validate/verify_blender_postprocess.py` | 暂缓；验收EXR连接DS/Omni标签mask框 | 有原ERP射线距离/重映射鱼眼距离独有链；隐含val_000 lens、real masks、Aaron几何。暂缓，不为hardcode丢弃能力，未来原位显式case化需另评。 |
| `tools/validate/verify_densekp_model_lifecycle.py` | 保留；真实DenseKP更新/保存/恢复 | 小网络不能代替；长期原位保留，显式phase/checkpoint/output，内部固定p0_dressed_tiny+trainer=gpu，无--config输入；不加入日常列表，不在本轮执行。 |
| `tools/validate/verify_fixed_validation.py` | 前置迁移后删除；旧新CUDA回调十情景迁移驱动 | 硬编码GPU1/旧source及capture类路径；先提取当前公共回调显式驱动，复用tests.fixed_validation_fixture.run_case，不复制Trainer实现。 先迁至 `tools/validate/check_fixed_validation_runtime.py` 并满足§4。 |
| `tools/validate/verify_refactor_inference.py` | 前置迁移后删除；三权重/两加载路线/四入口AST迁移 | 硬编码GPU1和旧source_snapshot；先提取当前公共加载/预测显式检查，不搬四入口AST或自动三权重循环。 先迁至 `tools/validate/check_inference_runtime.py` 并满足§4。 |

### capture_only

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/validate/verify_candidate_reporting.py` | 直接删除；旧新candidate保存预测制品回放 | 一次性迁移驱动；当前candidate_reporting数值/图片测试和summarize_saved_candidate_predictions保留，退出旧AST路径回放。 |
| `tools/validate/verify_dataset_crop.py` | 直接删除；七样本历史capture拒绝/verify | test_dataset_capture_retirement真实CLI消费者；同组退出。当前31/11合约及原JSON/七样本证据保留，不复录。 |
| `tools/validate/verify_geometry_camera.py` | 直接删除；固定旧人物SMPL-X迁移回放 | 建议明确停止此历史组合回放CLI；保留公共几何数值/单位/拒绝fixture，不称小数组等价于真实SMPL-X重建。 |
| `tools/validate/verify_refactor_metrics_config.py` | 直接删除；历史指标配置capture拒绝/verify | test_import_boundaries首方法是真实CLI消费者；随驱动退出。保留原metrics/config fixture及日常合约。 |
| `tools/validate/verify_scene_recipe_boundaries.py` | 直接删除；旧AST前缀/4场景导入和help对照 | 一次性旧新对照退出；保留3场景惰性导入检查及jobs命令fixture，实际装配另有长期检查。 |
| `tools/validate/verify_training_audit.py` | 直接删除；quiet_contract的capture/verify | 当前test_training_audit直接调用同helper和独立174数组JSON；旧动态smoke capture无继续用途。保留实际CPU生命周期测试，不因删除运行它。 |

### direct

| 路径 | 分类与当前职责 | 依赖、用途和处理依据 |
| --- | --- | --- |
| `tools/visualize/commuter_failure_sections.py` | 直接删除；旧diagnosis截面拼图/组件归因 | 消费diagnose_commuter_clothing evidence，随失败诊断退出，不删现有图/JSON。 |
| `tools/visualize/make_appearance_comparison.py` | 直接删除；P0前后图片拼图 | 独立CLI有功能但不属保留能力，明确退役；不为简单图册增加公共抽象。 |
| `tools/visualize/make_eye_fix_comparison.py` | 直接删除；首帧眼部ROI修正拼图 | 旧固定512像素/材料病例复查完成；当前眼部准备与透视渲染保留。 |
| `tools/visualize/make_skin_texture_review.py` | 直接删除；旧16帧皮肤/关节区域画廊 | 读模型weights仅作旧ROI；连冻结main退役，许可与UV生产者保留。 |


失败衣物19工具整体闭包：repair.edges_of/average/project → polo/settle/validate；settle_cloth.keyed_mesh → settle_top；union.capped/scan及extract/诊断/截面成一支。
不按helper逐批。BVH5文件也完整退出。专用河岸、pyrender、两Omni sweep确有独有能力，须明确同意退役，不能声称公共公式/Blender检查与其协议等价。

## 4. 唯一建议前置批次 B4-P 与长期能力

仅两最小驱动+一份集中安全测试；不建公共执行包/配置框架，不把旧文件改名照搬。默认只展示，--execute才运行；重型import必须在执行分支。

| 旧工具 → 新文件 | 保留的最小实现与显式输入 | 去掉的内容 |
| --- | --- | --- |
| verify_refactor_inference → tools/validate/check_inference_runtime.py | 直调densekp.core.inference的两strict加载路线和predict_batch/predict_item；显式checkpoint、model-config、完整归一化inputs、expected、全新output、route/profile及设备。key/shape/dtype/模式、FP32策略和固定预测对照，原atol=1e-5/rtol=0；两个API各自模式规则不统一 | 旧capture、4入口AST抽取、自动3权重、GPU1/历史目录。直接API检查不假称覆盖4个evaluator所有政策 |
| verify_fixed_validation → tools/validate/check_fixed_validation_runtime.py | 当前公共FixedValidation+已有tests.fixed_validation_fixture.run_case；显式source/before/output/scenario、resumed的resume-checkpoint、设备。保留10情景可选、RNG/参数/optimizer/state_key/异常部分写出、原字段和数组比较 | 旧capture/旧类路径、硬编码source/GPU1；不复制TinyModel/Trainer/helper到新文件，不承诺无tests的wheel可用 |
| 新tests/test_runtime_check_entries.py | RuntimeCheckEntryTests.test_default_plans_are_inert、test_explicit_paths_and_scenarios、test_execution_dispatch_preserves_selected_inputs、test_invalid_execution_refuses_before_writes；小JSON/桩验证无重型导入、资产/权重读取、写出、子进程及正确派发 | 不代替真实加载或CUDA验收，不为每个场景造框架 |

迁移前从旧工具**当前verify**逻辑冻结独立before；旧capture已不是独立旧实现，不使用。必要从源码快照提取当前限定分支，注入明确输入/设备，不能运行原main默认GPU1/自动三权重。新驱动对同一输入，不让两侧共同调用新驱动自证。

后续计算必须单列授权，不属于本轮或B4-D：

- 推理驱动：一个明确checkpoint、两加载路线、最少两个冻结完整输入，必要两profile；只前向不更新。先确认初始化资产、参考精度/输入齐全。保留历史三权重证据，不默认重跑三权重或效果评价。
- CUDA回调驱动：小网络success4步、step2恢复至4再2步、save_failure2步，最小8次更新；其余七失败情景helper字节不变，可先查派发而不声称重新实测。若要重验全部10情景，合计22次更新，须另列范围。无DenseKP更新。
- 每次GPU执行前即时检查设备/进程、明确设备映射；不沿用旧状态。未授权真实计算、缺设备/输入时，只挂起这两项删除，保留旧驱动；纯桩通过不能宣称真实运行完成。

| 长期检查位置 | 触发条件/输入输出 | 分类与限制 |
| --- | --- | --- |
| test_training_audit + training_audit_fixture +原JSON | CheckUpdates/probes或生命周期改动；临时输出、冻结174数组 | CPU真实4次更新/保存恢复，保留但排除B4-D执行；不证明真实DenseKP或CUDA |
| test_trial_callbacks + fixed_validation_fixture | state_key/类路径/接口；按上面显式CUDA驱动选测 | CPU接口无更新；helper真实小GPU网络，禁止整模块意外运行 |
| verify_densekp_model_lifecycle.py（原位） | 真模型/优化器/恢复逻辑改动；显式phase/checkpoint/output-dir，resume需reference-dir；内部固定p0_dressed_tiny+trainer=gpu，无--config参数 | 真实DenseKP/权重/参数更新；固定tiny注册表及初始化资产是本机依赖，fresh最多16步尝试、后续resume；不加入日常，不用CPUmock代替，资产齐备未复验 |
| verify_blender_assembly.py（原位） | 装配/通道改动；显式case JSON、mode assembly/render、output | Blender实际world mesh/groom/UV/材质/矩阵；先装配，仅实际渲染行为变才有限渲染 |
| compare_blender_acceptance.py（原位） | 上述明确before/after/report | CPU选定NPY/JSON/图片/EXR通道，现有窄路径归一化保留；非重渲染 |
| verify_blender_postprocess.py（暂缓改写） | --evidence/--run/--output；另隐含val_000 lens/real masks/Aaron几何 | CPU真实源EXR接DS/Omni；小合成测试不完全替代。未来需要时可原位用一个显式case列清所有输入，不作为B4前置，不造新框架 |
| test_commuter_intersections_blender前4方法 | public crossings或测试导入改动 | Blender小三角形，无人物/场景/渲染，非衣物质量 |
| test_numeric_regression | 投影/历史标注变化，已有透视/Omni保存数组 | CPU；缺资产skip须记未覆盖，保留，不为B4默认读真实资产 |

## 5. T：测试/fixture及精确同步范围

以下5个辅助文件也只是候选，后续授权须点名。两份AST fixture只剩迁移程序全文意义；本轮未删或重录。

| 路径 | 当前用途及处置理由 |
| --- | --- |

| `tests/test_dataset_capture_retirement.py` | 旧CLI/helper capture拒绝；verify_dataset_crop删除后无当前用途；31/11现行合约保留。 |
| `tests/test_geometry_capture_retirement.py` | 旧capture拒绝/verify派发mock；verify_geometry_camera删除后无当前用途；公共几何原数值不动。 |
| `tests/blender_geometry_fixture.py` | 失败DQS/scan独立参考；仅TriangleCrossings后三方法消费；保留生产crossings前四断言。 |
| `tests/fixtures/legacy_import_bodies_v1.json` | 12脚本旧main AST；唯一消费者test_import_boundaries：9脚本退役，保留3脚本导入安全改显式列表；不再维护迁移前全文。 |
| `tests/fixtures/scene_recipe_bodies_v1.json` | 4场景旧main AST；唯一消费者test_scene_recipes：退役程序全文对照，保留3场景惰性导入和scene_jobs数值/命令fixture。 |

现存文件最小同步表：

| 文件 | 准确修改范围 |
| --- | --- |
| `tests/test_import_boundaries.py` | 删metric capture方法和旧main AST方法/两处映射；保留原audit hook及RNG/sys.path检查，显式列表改为audit_skin_texture、prepare_procedural_eyes、blender_headless_smoke_test和build_cycling_forest、prepare_hidden_alley、validate_cycling_forest，不再读AST fixture。 |
| `tests/test_scene_recipes.py` | 删test_original_recipe_statement_order_and_private_helpers_are_unchanged及专属AST解码import；jobs两个数值/命令/只读builder方法原样保留。 |
| `tests/test_commuter_intersections_blender.py` | 只删test_oriented_scan_keeps_overlap_instead_of_xor、test_dqs_exact_rigid_transform、test_dqs_quaternion_sign_alignment及专属import；前四生产crossings断言原样保留。 |
| `tests/dataset_crop_fixture.py` | 仅移除run_dataset的original参数和提前拒绝分支，当前采集/采样/编码/RNG不变。 |
| `tests/geometry_camera_fixture.py` | 仅移除api的original参数和拒绝分支，TARGETS/合成输入/调用顺序/编码不变。 |
| `tests/test_inference.py` | 最后方法仅删除smoke import/assert，保留train_source_isolated_trial.predict检查，改名test_active_caller_is_direct；其它方法不变。 |
| `tests/test_training_audit.py` | 最后方法仅删除smoke import及2assert，保留active CheckUpdates/probes/RunAudit，改名test_active_subclass_uses_public_audit；数值/生命周期方法不变。 |
| `tests/test_trial_callbacks.py` | 仅B4-P完成时更新首行CUDA验收入口说明，方法和state_key/历史类路径断言不变。 |
| `tools/validate/check_workspace.py` | MODEL_TESTS移除两个capture retirement模块；其它保留。可接入已审计test_source_asset_preparation/test_regression_suite小输入；不得纳入更新/模型/GPU/Blender。 |
| `tools/experiments/prepare_source_trial_config.py` | 仅:58报告说明中的train_cycling_smoke.CheckUpdates改指公共CheckUpdates，不改字段/值算法/状态写出流程；业务暂缓不解除。 |
| `docs/COMMANDS.md` | 撤销旧geometry verify当前使用承诺；给当前长期检查显式输入/计算边界，其它现行命令保留。 |
| `docs/READING_GUIDE.md` | 移除smoke旧名导出承诺，指当前公共API/按需检查，不改历史报告。 |
| `docs/rendering/SCENE_RECIPE_INDEX.md` | 移除riverfront/review_commuter_component推荐及失败制作/DQS当前承诺；保留UV/groom制品准备知识和现有其它配方。 |
| `tests/fixtures/README.md` | 旧采集指Git历史；明确两AST fixture退役，保留数值fixture原来源/字节约束，撤销已删capture/verify/DQS用法。 |


12脚本AST列表中9脚本退役、3保留；两处有限AST映射随对应工具退出。保留3脚本再加3惰性场景的导入安全检查，不泛化忽略import/sys.path来蒙混旧AST对照。
DQS/scan后三测试仅有失败参考实现作为被测目标，推荐随能力退休；如评审要留算法参考，可排除1 helper和3方法，不阻碍19工具删除。
公共crossings前4项解析期望/容差/coplanar None原样保留。

**其它全部原fixture字节保持**：legacy_cleanup_v1仍服务归一化，绝不按名字删；metrics/config YAML、dataset_crop、geometry_camera、source_packing、imaging、candidate reporting、scene_jobs、training_audit数值JSON及所有历史outputs证据不重录。
日常检查不依赖ignored验收库；删除旧capture工具也不保留“该工具拒绝capture”的永久测试。

## 6. V：后续验收矩阵（本轮全部未运行）

### 静态、配置和保护

实施前再次记录HEAD/status，字节快照TODO、原数值fixture、配置、公共核心和本批必要源码至新验收目录；不用SHA256。
重新核查import/相对import/TARGETS/importlib/runpy/subprocess/CLI/Hydra与交付闭包；普通字段/历史报告/旧AST不是运行边。
`python -B tools/validate/check_workspace.py --no-tests` 只检查限定源码/config；不启用surface-operator、不遍历outputs。
新增文件check-ignore应可纳入Git；当前文档链接/符号和入口均有效。所有非授权fixture、算法、配置/注册表、活动status与TODO字节不变。

### 有界CPU方法（具体ID）

后续以 `python -B -m unittest <下面具体ID...>` 执行，禁止discover或整生命周期模块。
输入仅原fixture和临时小数组/图像；输出限全新临时/验收目录。允许已有Torch张量/smplx纯公式依赖，不构造真实模型/读权重/更新。
suite子进程全部为桩。下面2个active名称是拟改名，不假称当前已存在。


`tests.test_dataset_crop_contract.DatasetCropContractTests`：

- `test_old_id_and_explicit_sources_match_independent_augmented_outputs`
- `test_named_outputs_mutations_failures_and_rng`
- `test_explicit_fields_override_independently_without_registry_writes`
- `test_dataset_name_default_and_existing_img_dir_attribute`
- `test_evaluation_disables_requested_augmentation`
- `test_flip_rejection_survives_postconstruction_config_changes`

`tests.test_source_packing_contract.SourcePackingContractTests`：

- `test_declared_sources_all_scene_people_and_pair_failures_match_before`
- `test_boxes_rejections_and_bedlam_policy_match_before`
- `test_annotation_contract_masks_and_empty_cases_match_before`
- `test_packing_keeps_index_order_field_dtypes_and_legacy_errors`
- `test_box_gate_thresholds_and_float32_center_scale`
- `test_legacy_weight_and_in_view_adapters_keep_separate_semantics`
- `test_old_imports_remain_compatible`

`tests.test_source_isolation.SourceIsolationTests`：

- `test_excluding_bridge_does_not_discard_unrelated_families`
- `test_unreviewed_old_split_is_rejected`
- `test_possible_source_and_occluder_leakage_are_rejected`

`tests.test_imaging_pipeline.ImagingPipelineTests`：

- `test_full_small_artifacts_and_partial_failure_outputs_match_old_fixture`
- `test_ds_source_keeps_preview_rgb_and_original_erp_range`
- `test_omni_clears_only_remapped_depth_and_keeps_source_range`
- `test_periodic_sampler_wraps_longitude_without_wrapping_poles`
- `test_historical_exports_are_compatibility_only`

`tests.test_geometry_camera_contract.GeometryCameraContractTests`：

- `test_frozen_array_and_rejection_results`
- `test_clothing_lift_uses_body_and_does_not_modify_either_mesh`
- `test_physical_axes_erp_alignment_and_range_units`
- `test_periodic_depth_and_rounded_instance_rules_remain_different`

`tests.test_evaluation_contract.EvaluationContractTests`：

- `test_pre_migration_metric_outcomes`
- `test_pre_migration_configuration_dumps`
- `test_configuration_instances_are_independent_and_extensible`
- `test_historical_imports_remain_available`

`tests.test_image_normalization.ImageNormalizationTests`：

- `test_denormalize_matches_independent_original_bytes`

`tests.test_candidate_reporting.CandidateReportingTests`：

- `test_pairing_matches_independent_old_results`
- `test_panel_pixels_and_historical_keyword_signature`
- `test_json_unicode_order_and_nonfinite_failure`
- `test_input_rejection_preserves_copy_order`

`tests.test_source_asset_preparation.SourceAssetPreparationTests`：

- `test_pose_blocks_preserve_dtype_alias_and_errors`
- `test_clothing_artifacts_and_failure_order`
- `test_recipe_import_is_data_only`
- `test_groom_rigid_transfer_and_binding_contract`
- `test_groom_failure_order_and_partial_outputs`

`tests.test_regression_suite.RegressionSuiteTests`：

- `test_source_plan_argv_and_postprocessing`
- `test_box_plan_order_defaults_and_only`
- `test_display_has_no_write_process_heavy_import_or_weight_read`
- `test_gate_and_existing_directory_order`
- `test_reload_gate_threshold_and_original_nonfinite_rule`
- `test_source_success_environment_and_failure_stopping`
- `test_task_and_source_postprocessing_failure`
- `test_box_only_keeps_order_and_requires_execution_even_unknown`
- `test_gpu_resolution_mismatch_failures_and_explicit_selection`
- `test_relative_paths_follow_cwd_but_fixed_roles_follow_root`
- `test_physical_selection_maps_uuid_and_recheck_stops_before_log`

`tests.test_scene_recipes.SceneRecipeTests`：

- `test_jobs_match_independent_before_commands_paths_failures_and_artifacts`
- `test_job_builder_reads_manifest_without_creating_output_or_launching`

`tests.test_import_boundaries.ImportBoundaryTests`：

- `test_imports_are_inert_without_optional_dependencies`

`tests.test_inference.InferenceTests`：

- `test_active_caller_is_direct`

`tests.test_training_audit.TrainingAuditTests`：

- `test_active_subclass_uses_public_audit`

`tests.test_trial_callbacks.TrialCallbackInterfaceTests`：

- `test_historical_state_key_and_class_resolution`
- `test_zero_and_unscheduled_steps_do_not_touch_model_or_output`


所列Dataset方法仍覆盖原31合成Dataset/11低层及增强、输入变异/异常/RNG；原fixture期待不改。删除original开关前后用同样小合成输入独立对照当前helper结果，公共geometry输入/编码也如此。
保留完整来源、成像、归一化/指标/config回归，不重跑P3全量真实图像制品。
`test_inference`前3方法的RecordingModel/FakeModel是小网络/桩构造，保留但非本批必跑；
TrainingAuditTests.test_independent_lifecycle_and_gradient_failure_contract及Densekp LifecycleTests含真实更新，保留且**排除本批CPU命令**。

### Blender与真实运行分级

只改测试去掉历史helper，生产crossings字节不动。如需要验证导入调整，安排一次同版本Blender后台CPU小检查，显式TestSuite仅如下4方法：

- `tests.test_commuter_intersections_blender.TriangleCrossings.test_edge_pierces_face_while_vertices_are_outside`
- `tests.test_commuter_intersections_blender.TriangleCrossings.test_disjoint`
- `tests.test_commuter_intersections_blender.TriangleCrossings.test_shared_edge_is_not_self_crossing`
- `tests.test_commuter_intersections_blender.TriangleCrossings.test_coplanar_overlap_remains_unresolved`

用新验收目录一次性runner构造上述TestSuite，命令 `blender --background --factory-startup --python-exit-code 1 --python <runner>`。
不读取人物/源blend、不保存制作场景、不渲染；普通Python SkipTest不能记为通过。缺Blender则记未覆盖，不重跑真实装配。
B4-P模型/CUDA验收另按§4单列授权。删除历史工具不默认重跑三权重、400步、SMPL-X或渲染。

### 无资产交付

新源码副本仅含跟踪源码/必要配置/原数值fixture及新增文件，无outputs/资产；设置副本ROOT/src、清除原仓PYTHONPATH并禁止pyc。
从根cwd及临时cwd运行定向导入/CPU检查、新驱动默认计划，不能借本机editable或原仓补齐模块。允许已安装第三方依赖，不称editable重装/空环境安装通过。
只对安全入口测help/计划；不运行无安全help的制作main。未来删除集合与本清单逐项比对；不得清理缓存/失败版本或历史输出。

## 7. 批次、规模与少量取舍

最多两个批次：

1. **B4-P**：2新当前驱动+1集中测试；独立before、安全/派发和明确授权的真实路径验收后才删2旧驱动。更新callback首行说明及相应当前导航。
2. **B4-D**：先同步表T测试/日常/导航，再一次集中删除48工具+5辅助，共53文件；V-static/CPU/无资产交付，必要时仅4项Blender合成。发现新真实依赖暂停相应整组，不擅迁业务。

如果不批准GPU/参数更新，先B4-D完全可行，B4-P两旧驱动暂留；不能为了宣称B4完成跳过实测。这不是按helper拆十几轮。


统计采用当前文件splitlines物理行，包含注释/空行；AST fixture另外列字节，不当算法LOC；规划产物不计未来运行规模。

| 项目 | 数量 |
| --- | --- |
| 原B4现存工具 | 57，5930行 |
| 工具拟删 | 48直接+2前置，共50，5273行 |
| 工具保留/暂缓 | 5+2 |
| 测试/辅助Python额外拟删 | 3 |
| 全部拟删Python | 53文件，5487行 |
| 两AST fixture拟删 | 23行，58421字节 |
| 现存拟修改 | 14文件，表T逐项，不猜净行数 |
| 新增 | 2运行驱动+1测试，约225–365行 |
| 其中迁入/保留实现 | 65–125行；比较/情景驱动，不含旧AST/capture |
| 其中新胶水/测试 | 160–240行；独立列示，不隐去成本 |

新文件分项估计：inference100–160（迁30–65、新70–95），callback65–105（迁35–60、新30–45），测试60–100。
仅Python整文件删减扣新增约5122–5262行，尚不含14个修改文件增删/文档；实施后必须按实际diff重报净数。
原57候选剩7，加2新检查；这不是全仓入口数量，也不作为简洁性指标。

真正需评审：

- 接受独立riverfront、两旧Omni sweep、P0 pyrender及失败DQS/scan参考退役？推荐接受，明确协议退出而非等价替代；资产和主线能力保留。
- 先允许2真实检查前置提取，还是暂留旧驱动先集中删？推荐条件具备时唯一前置批；无明确GPU/更新授权就先B4-D。

真实阻碍仅限2检查的旧硬编码/输入与真实验收条件；3个制品生产者已从删除组排除。未发现必须为B4重构的当前evaluator依赖。

## 8. B5与有效知识保护

本轮不批量删历史报告，也不新增archive。将最小有效知识归已有导航，不为每个退役脚本建新报告：

| 内容 | 保留落点 |
| --- | --- |
| UV/皮肤/groom许可、Alembic版本/单位、绑定制品链 | SCENE_RECIPE_INDEX资产段；引用原skin_texture_audit与groom_import_audit_v2/report，不删许可附件 |
| 失败衣物/未过外观门槛、旧结果位置 | 现行场景说明简短退役声明，B5再归并主题；不暗示框增强部署获准 |
| Blender5.1.1 CPU代表场景与未覆盖分支 | P3_BLENDER_20261002和原execution_manifest；当前长期检查输入说明。不称其它设备/版本已验收 |
| 几何/监督/来源/坐标、原fixture来源 | SAMPLE_WALKTHROUGH、现有契约、tests/fixtures/README；本批只修工具用法，不改数值 |
| 当前真实检查的输入输出与计算边界 | COMMANDS与新help；默认计划禁止计算 |

可只读 `git show ada2295:tools/experiments/repair_commuter_top.py` 获取历史完整源码，其它文件同理。
Git只保存跟踪文件，不备份ignored资产/权重/验收产物，也不保证旧源码在当前环境可直接重跑；本轮及以后代码清理不删、移动或打包这些outputs。

已完成、不得再次列待执行的代表路径：

- `densekp/core/losses.py`
- `densekp/core/datasets/utils.py`
- `densekp/legacy/crop.py`
- `densekp/legacy/__init__.py`
- `tools/validate/verify_source_packing.py`
- `tools/validate/verify_imaging_pipeline.py`
- `tools/validate/evaluate_omni_densekp.py`
- `tools/validate/evaluate_densekp_npz.py`
- `tools/experiments/prepare_mixed_pedestrians.py`
- `tools/validate/commuter_intersections.py`
- `tools/experiments/evaluate_source_trial.py`
- `tools/experiments/evaluate_box_robustness.py`


## 9. 停止点

当前HEAD与57工具闭包复核、制品依赖纠错、测试/fixture处置与分级验收设计已完成；所有运行验收仍为计划。
本轮只有两个规划文件新增；对489个已有跟踪源码/测试/配置/文档文件做内存字节前后比较，TODO包含在内且保持。规划Markdown未被忽略；outputs inventory被忽略，关键要求已在本文完整列出。没有测试、Dataset/模型、训练、几何、GPU、Blender、渲染、下载、安装、SHA256或commit。
**停在B4规划评审点，不实施删除，不进入B5。**

本轮交付复核：489个既有跟踪源码/测试/配置/文档文件已做内存字节前后比较（含TODO），无变化。新规划Markdown未被忽略；outputs inventory被忽略，因此关键要求同时完整写在本文。没有创建实施脚本或运行验收。
