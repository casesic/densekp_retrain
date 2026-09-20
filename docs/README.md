# Documentation

## Start Here

- [Current code map](CURRENT_CODE_MAP.md) — current entry points, ownership, and code roles.
- [Architecture](ARCHITECTURE.md) — end-to-end geometry, rendering, annotation, and training flows.
- [Potential issues](POTENTIAL_ISSUES.md) — known blockers and unresolved risks.
- [TODO](TODO.md) — current work queue.

## Core Architecture

- [Architecture](ARCHITECTURE.md) — implementation-level data flow.
- [Current code map](CURRENT_CODE_MAP.md) — current repository map.

## Training

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

- [Synthetic render baseline](rendering/SYNTHETIC_RENDER_BASELINE.md) — current frozen renderer reference.
- [Synthetic render P0 plan](rendering/SYNTHETIC_RENDER_P0_PLAN.md) — original closed-loop plan.
- [Perspective closed loop](rendering/P0_PERSPECTIVE_CLOSED_LOOP.md) — perspective validation results.
- [Depth and visibility](rendering/P0_DEPTH_VISIBILITY.md) — depth semantics and visibility rules.
- [Real-motion regression](rendering/P0_REAL_MOTION_REGRESSION.md) — real SMPL-X motion validation.
- [Dressed render integration](rendering/P0_DRESSED_RENDER_INTEGRATION.md) — body/clothing render integration.

## Appearance

- [Appearance plan](appearance/APPEARANCE_V1_PLAN.md) — appearance phases and acceptance state.
- [Appearance baseline](appearance/APPEARANCE_BASELINE.md) — initial appearance reference.
- [Body texture](appearance/BODY_TEXTURE_INTEGRATION.md) — body texture and UV evidence.
- [Clothing integration](appearance/CLOTHING_INTEGRATION.md) — clothing asset constraints.
- [Eye geometry](appearance/EYE_GEOMETRY_AUDIT.md) — eye geometry evidence.

## Motion

- [Temporal validation](motion/TEMPORAL_MULTI_SCENARIO_VALIDATION.md) — multi-motion temporal checks.
- [Cycling motion audit](motion/cycling/CYCLING_MOTION_AUDIT.md) — local cycling inventory conclusions.
- [BABEL cycling search](motion/cycling/BABEL_CYCLING_CANDIDATE_SEARCH.md) — BABEL candidate search.
- [Dedicated cycling mocap](motion/cycling/DEDICATED_CYCLING_MOCAP_AUDIT.md) — dedicated BVH/mocap evidence.

## Scenes

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
