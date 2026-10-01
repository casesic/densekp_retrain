# Documentation

## Start Here

- [当前数据集要求汇总](data/DATASET_REQUIREMENTS_20260929.md) — 目标覆盖、骑行来源、穿衣场景、相机遮罩、138 点监督、划分和验收要求。
- [代码阅读指南](READING_GUIDE.md) — 当前目录职责、阅读顺序和后续可读性整理候选。
- [跟着一个样本读完整条链](SAMPLE_WALKTHROUGH.md) — 从 3D 标注到实际裁剪、模型、loss 和坐标还原。
- [Historical code map](CURRENT_CODE_MAP.md) — September 20 snapshot; follow its current-guide links for updated behavior.
- [Architecture](ARCHITECTURE.md) — end-to-end geometry, rendering, annotation, and training flows.
- [Potential issues](POTENTIAL_ISSUES.md) — known blockers and unresolved risks.
- [TODO](TODO.md) — current work queue.

## Core Architecture

- [Architecture](ARCHITECTURE.md) — implementation-level data flow.
- [Current reading map](READING_GUIDE.md) — current responsibilities and main call chains.

## Training

- [遮挡点监督与评价（2026-09-29）](training/OCCLUDED_SUPERVISION_20260929.md) — 有效视野与可见性分离、增强后的监督 mask、旧预测重新评分和可选 tiny 数据配置。
- [Tiny reproduction](training/TINY_REPRODUCTION.md) — existing rendered samples through the DenseKP loader and CPU optimization check.
- [Omni multi-person pilot](training/OMNI_MULTIPERSON_PILOT.md) — paired ERP/fisheye per-person labels, visibility-aware loss, and checkpoint fine-tuning smoke test.
- [Training audit](training/TRAINING_AUDIT.md) — training implementation and dependency audit.
- [Training data format](training/TRAINING_DATA_FORMAT.md) — expected model-training inputs.

## Synthetic Data

- [Annotation audit](data/ANNOTATION_AUDIT.md) — annotation evidence and limitations.
- [Data format](data/DATA_FORMAT.md) — DenseKP dataset fields and conventions.
- [DenseKP GT pipeline](data/DENSEKP_GT_PIPELINE.md) — ground-truth production flow.
- [BEDLAM documentation](data/bedlam/) — BEDLAM planning, operators, tiny data, and rerendering.

## Camera

- [Coordinate systems](camera/COORDINATE_SYSTEMS.md) — coordinate-frame overview.
- [P0 coordinate convention](camera/P0_COORDINATE_CONVENTION.md) — frozen perspective convention.
- [Fisheye camera P0](camera/FISHEYE_CAMERA_P0.md) — Omni-Radtan validation.
- [Perspective camera stress](camera/PERSPECTIVE_CAMERA_STRESS.md) — edge, scale, and truncation tests.

## Rendering

- [四角色 × 两骑行姿态验收](rendering/CYCLING_FOUR_CHARACTERS_20260929.md) — 2 女 + 2 男、八份衍生资产、48 张预览、双场景 13 行 loader；仅 integration_only。
- [3D 场景＋鱼眼遮罩的新推理基线](rendering/DS_MASKED_BASELINE_20260929.md) — 12 个真实推理视图、可见/遮挡/双视角误差与有效区检查。
- [DS 镜头遮罩更新](rendering/DS_LENS_MASK_20260928.md) — 真实参考图轮廓、带遮罩鱼眼及与原 ERP/标注的对比。
- [DS paired edge-person pilot](rendering/DS_SCENE_PAIR_PILOT_20260928.md) — same-world opposite-facing cameras in the two selected scenes, annotations and actual loader checks.
- [Existing DenseKP predictions on paired scenes](rendering/DS_SCENE_PAIR_PREDICTION_20260928.md) — 12 oracle-box predictions, per-view and shared-visible errors, and occluded-target failure cases.
- [Fixed-target occlusion and box controls](rendering/DS_OCCLUSION_CONTROL_20260928.md) — three occlusion levels with frozen crops, plus same-image box-center sensitivity.
- [Synthetic render baseline](rendering/SYNTHETIC_RENDER_BASELINE.md) — current frozen renderer reference.
- [Synthetic render P0 plan](rendering/SYNTHETIC_RENDER_P0_PLAN.md) — original closed-loop plan.
- [Perspective closed loop](rendering/P0_PERSPECTIVE_CLOSED_LOOP.md) — perspective validation results.
- [Depth and visibility](rendering/P0_DEPTH_VISIBILITY.md) — depth semantics and visibility rules.
- [Real-motion regression](rendering/P0_REAL_MOTION_REGRESSION.md) — real SMPL-X motion validation.
- [Dressed render integration](rendering/P0_DRESSED_RENDER_INTEGRATION.md) — body/clothing render integration.

## Appearance

- [十名成人＋可选儿童角色规划](appearance/CHARACTER_LIBRARY_10PLUS1_PLAN_20260930.md) — 四人升级、六人新增，三维发型/通勤衣型缺项、儿童模板及分阶段验收；仅规划。
- [骑行多角色资产可行性](appearance/CYCLING_MULTI_CHARACTER_FEASIBILITY_20260929.md) — 本地 BEDLAM 归档与官方资源核查、2 女 + 2 男候选、发型/衣物缺口、姿态接入和分组方案。
- [Appearance plan](appearance/APPEARANCE_V1_PLAN.md) — appearance phases and acceptance state.
- [Appearance baseline](appearance/APPEARANCE_BASELINE.md) — initial appearance reference.
- [Body texture](appearance/BODY_TEXTURE_INTEGRATION.md) — body texture and UV evidence.
- [Clothing integration](appearance/CLOTHING_INTEGRATION.md) — clothing asset constraints.
- [Eye geometry](appearance/EYE_GEOMETRY_AUDIT.md) — eye geometry evidence.

## Motion

- [骑行图片重建试验](motion/cycling/CYCLING_IMAGE_RECONSTRUCTION_20260929.md) — 当前主线：12 张公开骑行图片的 SMPL 重建、姿态筛选与 SMPL-X 接入边界。
- [骑行素材与拟合交接](motion/cycling/CYCLING_FIT_HANDOFF_20260929.md) — BVH 备用路线，当前暂停；保留两个片段和 SMPL-X 输出接口。
- [Temporal validation](motion/TEMPORAL_MULTI_SCENARIO_VALIDATION.md) — multi-motion temporal checks.
- [Cycling motion audit](motion/cycling/CYCLING_MOTION_AUDIT.md) — local cycling inventory conclusions.
- [BABEL cycling search](motion/cycling/BABEL_CYCLING_CANDIDATE_SEARCH.md) — BABEL candidate search.
- [Dedicated cycling mocap](motion/cycling/DEDICATED_CYCLING_MOCAP_AUDIT.md) — dedicated BVH/mocap evidence.

## Scenes

- [Urban Street](scenes/URBAN_STREET.md) — modern street environment, markers and validation.
- [Photoreal outdoor assets](scenes/PHOTOREAL_OUTDOOR_ASSETS.md) — outdoor scene selection and validation.

## Environment

- [Headless environment](environment/HEADLESS_ENVIRONMENT.md) — Blender/EGL environment evidence.
- [Renderer selection](environment/HEADLESS_RENDERER_SELECTION.md) — renderer capability comparison.
- [Migration](environment/MIGRATION.md) — environment and data migration notes.

## Refactor History

- [Phase 3 dependency map](refactor/PHASE3_DEPENDENCY_MAP.md) — historical dependency audit.
- [Phase 3 delete candidates](refactor/PHASE3_DELETE_CANDIDATES.md) — historical deletion decisions.
- [Phase 4 delete candidates](refactor/PHASE4_DELETE_CANDIDATES.md) — dead-code decisions.
- [Phase 4 deferred cleanup](refactor/PHASE4_DEFERRED_CLEANUP.md) — work deferred pending regression coverage.
- [Phase 1–2 report](refactor/REFACTOR_REPORT.md) — earlier refactor history.

## DenseKP correctness update

- [2026-09-28 correctness fixes and boundary refactor](refactor/DENSEKP_CORRECTNESS_20260928.md) — current data contract, migration, tests and remaining training gates.

- [Real DenseKP GPU acceptance](refactor/DENSEKP_REAL_MODEL_ACCEPTANCE_20260928.md) — checkpoint lifecycle, AMP skipped updates and edge-truncated person evaluation scope.
