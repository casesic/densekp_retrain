# B4-D 集中退役实施与验收

日期：2026-10-03。**授权53项全部完成，等待本批评审；B4-P未执行，B4整体尚未完成。未提交。**

## 基线和范围

实际起始 HEAD `ada22953bb0696e5a3870d14fb60ccbd715925dc`（master，Complete B3 asset helper migration and regression suite consolidation）。
开始时只有用户 `docs/TODO.md` 修改和未跟踪 B4 规划Markdown；两者保留原字节。未重置、提交或重新实施B3。
删除前将人工列明的53路径与inventory交叉确认，保存 [白名单](../../outputs/acceptance/pruning_b4_delete_20261003/deletion_whitelist.json)，不是按分类直接批删。
独立before与本次必要源码/保护文件放在全新 `outputs/acceptance/pruning_b4_delete_20261003/`；未复制整个历史验收库。

总体证据入口：[execution_manifest.json](../../outputs/acceptance/pruning_b4_delete_20261003/execution_manifest.json)。

## 实际删除清单

48个历史工具和3个专用测试/辅助Python、2份AST fixture全部删除，无暂停项。
这明确停止这些制作/CLI/参考能力，不宣称当前流程与它们全部等价。全部既有资产/失败版本/输出不删除。

| 完整依赖组 | 已删除文件 |
| --- | --- |

| once_audits | `tools/experiments/audit_cycling_character_assets.py`<br>`tools/experiments/audit_eye_geometry.py`<br>`tools/experiments/inspect_smplx_rigs.py`<br>`tools/validate/audit_box_trial.py`<br>`tools/validate/audit_interrupted_trial.py`<br>`tools/validate/p0_pyrender_crosscheck.py`<br>`tools/validate/pyrender_headless_smoke_test.py`<br>`tools/visualize/review_eye_background_fix.py` |
| bvh | `tools/experiments/audit_cyclist_bvh.py`<br>`tools/experiments/find_cyclist_clip.py`<br>`tools/experiments/prepare_cycling_fit_inputs.py`<br>`tools/experiments/review_cyclist_audit.py`<br>`tools/experiments/search_babel_cycling.py` |
| failed_clothing | `tools/experiments/build_commuter_clothing.py`<br>`tools/experiments/build_commuter_polo_collar.py`<br>`tools/experiments/build_continuous_commuter_pattern.py`<br>`tools/experiments/commuter_union_volume.py`<br>`tools/experiments/diagnose_commuter_clothing.py`<br>`tools/experiments/extract_commuter_union.py`<br>`tools/experiments/prepare_commuter_cloth_path.py`<br>`tools/experiments/repair_commuter_top.py`<br>`tools/experiments/settle_commuter_cloth.py`<br>`tools/experiments/settle_commuter_top.py`<br>`tools/experiments/smooth_commuter_weights.py`<br>`tools/experiments/trim_commuter_neck.py`<br>`tools/experiments/validate_commuter_union.py`<br>`tools/render/review_commuter_component.py`<br>`tools/validate/audit_commuter_component_contacts.py`<br>`tools/validate/audit_commuter_pose_sources.py`<br>`tools/validate/audit_commuter_repair_bodies.py`<br>`tools/validate/probe_commuter_overlap.py`<br>`tools/validate/validate_commuter_studio.py` |
| duplicate_smoke | `tools/experiments/train_cycling_smoke.py`<br>`tools/validate/smoke_densekp_lightning.py`<br>`tools/validate/smoke_densekp_training.py` |
| old_sweeps | `tools/fisheye/run_fisheye_seam_sweep.py`<br>`tools/fisheye/run_human_scale_sweep.py` |
| scene_unused | `tools/scene/build_reference_riverfront.py` |
| capture_only | `tools/validate/verify_candidate_reporting.py`<br>`tools/validate/verify_dataset_crop.py`<br>`tools/validate/verify_geometry_camera.py`<br>`tools/validate/verify_refactor_metrics_config.py`<br>`tools/validate/verify_scene_recipe_boundaries.py`<br>`tools/validate/verify_training_audit.py` |
| direct | `tools/visualize/commuter_failure_sections.py`<br>`tools/visualize/make_appearance_comparison.py`<br>`tools/visualize/make_eye_fix_comparison.py`<br>`tools/visualize/make_skin_texture_review.py` |
| supporting_retirement | `tests/test_dataset_capture_retirement.py`<br>`tests/test_geometry_capture_retirement.py`<br>`tests/blender_geometry_fixture.py`<br>`tests/fixtures/legacy_import_bodies_v1.json`<br>`tests/fixtures/scene_recipe_bodies_v1.json` |


B3 已迁出的参数、简单衣物、groom、crossings 未改；失败衣物 edges_of/average/project/keyed_mesh/capped/scan 闭包整组退役，没有再搬入公共包或新tests参考。

## 同步修改

| 文件 | 实际变化 |
| --- | --- |
| tests/dataset_crop_fixture.py | 只去run_dataset的original参数/拒绝分支；当前函数body AST与before剩余部分完全一致 |
| tests/geometry_camera_fixture.py | 只去api的original参数/拒绝分支；TARGETS、合成输入和调用编码不变 |
| tests/test_inference.py、tests/test_training_audit.py | 只撤销旧smoke导入/身份断言，保留当前调用方；所有原数值/生命周期方法源码逐段一致，含更新的方法未执行 |
| tests/test_import_boundaries.py | 删除已无入口的metric capture拒绝、旧main AST和有限映射；普通Python仅三个显式当前脚本，原IO/进程/扫描/RNG/sys.path/重型导入约束保留 |
| tests/test_scene_recipes.py | 删除迁移期整段AST方法；保留scene_jobs两个契约源码原字节；增加显式check_blender_recipe_imports函数，在真实Blender内检查三脚本，不由普通CPU unittest自动触发 |
| tests/test_commuter_intersections_blender.py | 删除DQS/scan后三测试和专属import；前四crossings方法源码/期望/容差原字节保留 |
| tools/validate/check_workspace.py | 移除两个capture retirement模块，接入B3人物准备小输入和regression suite桩；原扫描根/配置范围不扩大 |
| tools/experiments/prepare_source_trial_config.py | 仅implementation_notes一句话改指densekp.core.training_audit.CheckUpdates，字段结构、算法和status写出流程不变 |
| docs/COMMANDS.md、docs/READING_GUIDE.md、docs/rendering/SCENE_RECIPE_INDEX.md、tests/fixtures/README.md | 最小同步退役、当前使用/计算边界、UV/groom生产者与历史来源；不改历史报告正文 |

新增的Blender导入检查放在现有scene测试文件，没有新增运行驱动、测试框架或新测试文件；它允许真实bpy/mathutils导入，
通过审计钩子拒绝文件写出/资产读取/扫描/进程，逐opcode禁止场景脚本访问bpy.ops/data/context，并禁止main被调用；
另比较真实场景对象/变换及场景身份不变。没有模拟bpy，也没有重写三个生产脚本来适配检查。
普通Python三个工具为 audit_skin_texture、prepare_procedural_eyes、blender_headless_smoke_test；
Blender三个脚本为 build_cycling_forest、prepare_hidden_alley、validate_cycling_forest。

## 验收结果与独立性

| 检查 | 本次实际结果 | 限制 |
| --- | --- | --- |
| 独立CPU前测/后测 | 分别57方法通过，无skip；具体ID见test_ids.json及各测试JSON | 禁止discover；没有执行任何生命周期参数更新方法 |
| 当前helper before/after | 31个Dataset用例×旧ID/显式两路线、11低层、33几何记录；2193数组的dtype/shape/顺序/值精确一致，metadata逐字节一致 | before在改代码前调用当时helper；固定共享输入路径，只写新目录，不重录原fixture |
| 现行数值覆盖 | 裁剪/增强/变异/异常/RNG、来源全部人物含遮挡者、打包/框、成像、归一化、指标/config、scene_jobs、B3小输入、调度桩通过 | 不读取真实图片全集或真实模型，不统一历史规则 |
| Blender CPU | 5.1.1，build b70da489d7f4；前/后各4项crossings通过，无skip；三脚本在根/临时cwd实际导入均无副作用 | factory startup、2线程、CUDA/HIP不可见、Cycles CPU；不读源blend/人物，不制作/渲染 |
| 删除闭包 | 当前源码、配置/命令与测试无已删运行模块依赖；显式日常列表有效 | inference驱动的唯一smoke文本读取历史source_snapshot，已人工分类，不是假运行依赖 |
| 无资产源码交付 | 两cwd各57 CPU方法通过，无skip；各124个项目模块实际来自副本；Blender副本4项及三脚本两cwd导入通过 | 只用既有第三方环境，无原仓editable补模块；不是editable重装/空环境安装验收 |
| 日常语法/配置 | check_workspace --no-tests检查267源码/26配置通过，扫描未扩大 | 未启动surface-operator或真实数据/训练路径 |
| 保护 | 426个非修改/删除快照文件字节一致，含TODO、B4规划、公共算法、配置、B4-P驱动、test_trial_callbacks和活动status | 两AST fixture按授权删除；其它原数值fixture不重录 |

四项Blender方法（只运行这四个unittest，三脚本检查是显式受控导入）：

- TriangleCrossings.test_edge_pierces_face_while_vertices_are_outside
- TriangleCrossings.test_disjoint
- TriangleCrossings.test_shared_edge_is_not_self_crossing
- TriangleCrossings.test_coplanar_overlap_remains_unresolved

被忽略的 `tests/fixtures/training_audit_v1.npz` 未进入起始“跟踪文件”快照；收尾单独与已有B3.3独立快照逐字节比较一致，
并交叉核实B3.1/B3.2原件也一致，见ignored_fixture_comparison.json。本批未写该文件；不把事后比较伪称本轮起始快照。
`legacy_cleanup_v1.json` 等原数值fixture继续受保护，归一化仍直接使用独立期望。

### 执行保护和前置问题

- 最初proxy Python缺NumPy，尚未执行项目测试；改用已存在 `/home/yy/conda_envs/camerahmr/bin/python`，未安装升级任何依赖。
- 初版构造保护误拦组件导入时的无参数ReLU默认对象；只为确切ReLU类型允许默认初始化，仍阻断其它Module构造、SMPL-X、真实权重、Trainer.fit/validate/test、optimizer.step、CUDA初始化及业务进程。
- NumPy testing首次惰性导入的CPU `lscpu` 被过宽子进程保护拦住，首次57方法尝试出现98个子项断言失败；记录保留在guard_numpy_before_tests.json/log。
  修正为在项目导入和执行保护前初始化NumPy testing；未开放业务子进程/GPU查询，也未改原测试期望/断言。随后独立有效before在生产编辑前完成。
- 修改说明时发现规划写的smoke符号简写与实际句子不同，替换断言先停止；核对后只改实际报告文字，删除当时尚未开始，没有扩展业务修改。
- 全工作区diff检查报告用户TODO已有尾随空格；保留不修。排除该用户文件后 `git diff --check` 通过。

详细原因/证据见preflight_corrections.json。拒绝输入引起的missing.png警告属于已有错误用例；没有改插值、容差或算法来通过。

## 保留与未运行边界

两个B4-P驱动 `verify_refactor_inference.py`、`verify_fixed_validation.py` 原字节保留，未运行、未新增替代驱动；test_trial_callbacks说明也未改。
UV/groom实际生产者、prepare_source_trial_config、真实DenseKP生命周期、Blender装配/比较/后处理工具、两个DS evaluator、当前两套suite全部保留。
双包/注册表/Hydra/当前算法与模型、监督、评价语义不变，框增强权重仍未获部署替换认可。

未构造DenseKP/SMPL-X或其它参数网络、未读真实权重、未推理/训练/参数更新；没有GPU查询/计算、制作main、源blend、资产生成或渲染。
调度测试的所有评价/summary/compare/GPU命令都是桩，真实子进程只限独立Python安全检查与授权Blender。
不把CPU合约、Blender合成/导入成功称作真实人物重建、衣物质量、完整制作链或模型效果验收。
旧历史源码可用 `git show ada2295:<路径>` 查看；原outputs/权重/资产/保存预测/失败版本未删除、移动或重打包，Git不为它们提供备份。

## 实际规模与收尾


统计相对本批起始HEAD，排除用户TODO和本来已存在的未跟踪B4规划；物理行含注释/空行，不假装算法LOC：

| 项目 | 数量 |
| --- | --- |
| 整文件删除 | 53：48工具、3测试/辅助、2AST fixture |
| 整文件Python毛减少 | 51文件、5136行 |
| AST fixture删除 | 23行、58421字节，单列不算算法代码 |
| 现存文件修改 | 13文件 |
| Python全部实际diff | +93 / -5295，净减少5202行（包含新增导入保护，而非只报毛删除） |
| 当前说明实际diff | +80 / -32，含fixture说明 |
| 已跟踪文件整体净减少 | 5177行，含上述文档和2AST fixture |
| 新增运行/测试文件 | 0；没有B4-P替代驱动 |
| 新增交付报告 | 本文1份；验收目录为ignored运行证据，单独记录文件/字节，不混入源码净数 |

没有因新真实消费者暂停的删除项。53项集合精确匹配白名单，原数值/导入安全覆盖保留；停止的测试只服务已明确退出能力。
**B4-D实现与限定验收完成，等待评审；B4-P暂留、B4整体未完成。停在此处，不继续B4-P/B5，不自动提交。**
