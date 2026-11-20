# tools/experiments 索引

134 个脚本、17,158 行，全部是**一次性配方**：准备数据与资产、冻结实验计划、跑一次渲染或拟合、写一份核查。它们不是库——公共实现不反过来依赖它们（模块归属见[代码阅读指南](../../docs/READING_GUIDE.md)）。

## 这张表怎么读

- **做什么**：脚本自己 docstring 的首句，原样截断，没有改写。中英混排是原样，不是风格不统一。
- **结论 / 依据落在**：引用它的那一篇现行文档；没有文档引用但有代码引用的，给出那个测试文件。
- **状态**三档，判定是机械的（按文件名做字符串搜索），不是逐个读代码后的主观判断，边界上会有误判，发现了就改这一行：
  - **在用**：被 `docs/` 下非 `history/` 的现行文档或 README 引用；
  - **在用（被测试引用）**：没有文档引用，但被 `tests/` 直接 import —— 这种**不能移动**；
  - **仅历史（已归档）**：既没有文档引用也没有代码引用，已移到 `archive/`。

**合计：134 个 → 在用 80 个（其中 4 个只被测试引用）、仅历史 54 个（已归档）。**

## 归档时的一次纠错

第一版判定只看文档引用，得出「58 个仅历史」。移动前又扫了一遍 `src/ densekp/ scripts/ configs/ tests/`，发现其中 4 个被测试直接 import：

| 脚本 | 引用它的测试 |
|---|---|
| `audit_identity_texture_sources.py` | `tests/test_source_isolation.py` |
| `prepare_cycling_clothing.py` | `tests/test_cycling_geometry.py` |
| `refine_cycling_character_clothing.py` | `tests/test_character_clothing.py`、`tests/test_adult_candidate_garments.py` |
| `plan_t_junction_local_rigs_a1.py` | `tests/test_t_junction_local_rig_delivery_a1.py` |

这 4 个留在根目录并改判为「在用」，挪走会直接让测试 import 失败。剩下 **54 个**移入 `archive/`，内容一字未改（`git mv`，历史可追），也没有被删除或排除出 Git——只是不再和仍在用的配方混在一起。

### 准备与配方（52 个，其中 38 个在用）

| 脚本 | 做什么 | 结论 / 依据落在 | 状态 |
|---|---|---|---|
| [`adult_candidate_recipe.py`](adult_candidate_recipe.py) | Ten fixed adult clothing palettes and lengths; recipe data only. | [ASSETS.md](../../docs/ASSETS.md) | 在用 |
| [`bind_adult_grooms.py`](bind_adult_grooms.py) | Bind locally available groom variants to new adult candidates, preserving bodies. | [ASSETS.md](../../docs/ASSETS.md) | 在用 |
| [`bind_commuter_groom.py`](bind_commuter_groom.py) | Transfer official LH groom roots onto verified standard SMPL-X scalp semantics. | [ASSETS.md](../../docs/ASSETS.md) | 在用 |
| [`build_dynamic_asset_handoff_a2.py`](archive/build_dynamic_asset_handoff_a2.py) | Create explicit asset combinations, without changing any authoritative source. | —（无引用） | 仅历史（已归档） |
| [`compose_asset_a2.py`](compose_asset_a2.py) | Plan or explicitly build a configured standard-SMPL-X dressed motion. | [KIT11_M03_WALK_ASSET_A2_20261010.md](../../docs/appearance/KIT11_M03_WALK_ASSET_A2_20261010.md) | 在用 |
| [`convert_cycling_smplx.py`](archive/convert_cycling_smplx.py) | Fit independent SMPL-X bodies to saved HMR2 meshes via surface correspondences. | —（无引用） | 仅历史（已归档） |
| [`generate_cyclic_validation_motion.py`](archive/generate_cyclic_validation_motion.py) | Generate a deterministic cyclic gait articulation for validation only. | —（无引用） | 仅历史（已归档） |
| [`label_composed_road_video.py`](archive/label_composed_road_video.py) | Add component and original-time captions to an existing ROI-only preview. | —（无引用） | 仅历史（已归档） |
| [`plan_road_pass.py`](plan_road_pass.py) | Model/render-free recipe entry point; optional manifest geometry preflight. | [PEDESTRIAN_SEEDED_PILOT_A1_20261009.md](../../docs/training/PEDESTRIAN_SEEDED_PILOT_A1_20261009.md) | 在用 |
| [`plan_t_junction_local_rigs_a1.py`](plan_t_junction_local_rigs_a1.py) | Freeze and evaluate bounded local camera translations plus rig yaw. | [test_t_junction_local_rig_delivery_a1.py](../../tests/test_t_junction_local_rig_delivery_a1.py) | 在用（被测试引用） |
| [`plan_t_junction_rigs_a1.py`](plan_t_junction_rigs_a1.py) | Freeze, screen, and precisely preflight three paired city-junction rigs. | [T_JUNCTION_MULTIRIG_A1_20261011.md](../../docs/rendering/T_JUNCTION_MULTIRIG_A1_20261011.md) | 在用 |
| [`prepare_adult_candidates.py`](prepare_adult_candidates.py) | Versioned simple-clothing candidates.  | [READING_GUIDE.md](../../docs/READING_GUIDE.md) | 在用 |
| [`prepare_adult_comparison.py`](archive/prepare_adult_comparison.py) | Common outfit, illumination and no-groom comparison inputs; never new identities. | —（无引用） | 仅历史（已归档） |
| [`prepare_apd_6526_dressed.py`](prepare_apd_6526_dressed.py) | Bounded, frozen-body clothes and rigid-head groom pilot; no pose optimization. | [APD_6526_DRESSED_PILOT_20261008.md](../../docs/motion/cycling/APD_6526_DRESSED_PILOT_20261008.md) | 在用 |
| [`prepare_bedlam2_walk_motion_a2.py`](prepare_bedlam2_walk_motion_a2.py) | Normalize one already-acquired BEDLAM2 SMPL-X walk for the composition CLI. | [KIT11_M03_WALK_ASSET_A2_20261010.md](../../docs/appearance/KIT11_M03_WALK_ASSET_A2_20261010.md) | 在用 |
| [`prepare_bedlam_m03_transfer_preview_a2.py`](prepare_bedlam_m03_transfer_preview_a2.py) | Freeze two-view comparison preview inputs for one BEDLAM2-to-M03 garment. | [BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md](../../docs/appearance/BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md) | 在用 |
| [`prepare_bedlam_native_walk_a2.py`](prepare_bedlam_native_walk_a2.py) | Evaluate the original B2 neutral locked-head body, without addon registration. | [BEDLAM_NATIVE_WALK_PILOT_A2_20261009.md](../../docs/data/bedlam/BEDLAM_NATIVE_WALK_PILOT_A2_20261009.md) | 在用 |
| [`prepare_bedlam_walk_appearance.py`](archive/prepare_bedlam_walk_appearance.py) | Attach one existing BEDLAM groom and native eye materials to accepted caches. | —（无引用） | 仅历史（已归档） |
| [`prepare_box_robustness.py`](prepare_box_robustness.py) | Freeze box-only derivatives of accepted images.  | [DUAL_VIEW_TEMPORAL_AUDIT_20261008.md](../../docs/training/DUAL_VIEW_TEMPORAL_AUDIT_20261008.md) | 在用 |
| [`prepare_bvh_p2_dressed_a2.py`](prepare_bvh_p2_dressed_a2.py) | Derive M01 simple clothing and rigid groom matrices from accepted BVH body. | [BVH_P2_DRESSED_ASSET_A2_20261010.md](../../docs/motion/cycling/BVH_P2_DRESSED_ASSET_A2_20261010.md) | 在用 |
| [`prepare_commuter_comparison.py`](prepare_commuter_comparison.py) | Same sports outfit and studio setup for old/new identity shape comparison only. | [COMMUTER_TWO_CHARACTER_PILOT_20260930.md](../../docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md) | 在用 |
| [`prepare_commuter_pilot.py`](prepare_commuter_pilot.py) | Two preserved identities: full-space regression, bounded shape fit, four poses. | [COMMUTER_TWO_CHARACTER_PILOT_20260930.md](../../docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md) | 在用 |
| [`prepare_commuter_scalp.py`](archive/prepare_commuter_scalp.py) | A versioned shader-only scalp mask; original face/skin PNG remains unchanged. | —（无引用） | 仅历史（已归档） |
| [`prepare_cycling_asset_a2.py`](prepare_cycling_asset_a2.py) | A2 frozen cycling delivery and bounded BEDLAM offset-transfer experiment. | [CYCLING_SCENE_INTERFACE_A2_20261009.md](../../docs/data/CYCLING_SCENE_INTERFACE_A2_20261009.md) | 在用 |
| [`prepare_cycling_characters.py`](archive/prepare_cycling_characters.py) | Instantiate fixed local characters and independent cycling derivatives. | —（无引用） | 仅历史（已归档） |
| [`prepare_cycling_clothing.py`](prepare_cycling_clothing.py) | Build fitted jersey/shorts shells for each independent cycling body. | [test_cycling_geometry.py](../../tests/test_cycling_geometry.py) | 在用（被测试引用） |
| [`prepare_cycling_female_asset_a2.py`](prepare_cycling_female_asset_a2.py) | Reuse one accepted APD rotation sequence with the existing female character. | [CYCLING_FEMALE_ASSET_A2_20261009.md](../../docs/motion/cycling/CYCLING_FEMALE_ASSET_A2_20261009.md) | 在用 |
| [`prepare_cycling_images.py`](archive/prepare_cycling_images.py) | Select explicit V-COCO ride+bicycle pairs and download a small COCO image pilot. | —（无引用） | 仅历史（已归档） |
| [`prepare_cycling_team_relative_layout_a1.py`](prepare_cycling_team_relative_layout_a1.py) | Read-only reuse of two frozen combinations; bounded relative river layout. | [CYCLING_TEAM_RELATIVE_LAYOUT_A1_20261010.md](../../docs/rendering/CYCLING_TEAM_RELATIVE_LAYOUT_A1_20261010.md) | 在用 |
| [`prepare_dispersed_cyclists_a1.py`](prepare_dispersed_cyclists_a1.py) | Freeze and preflight 4/5 distinct riders in the existing city scene. | [DISPERSED_CYCLISTS_4_5_MULTIRIG_A1_20261011.md](../../docs/rendering/DISPERSED_CYCLISTS_4_5_MULTIRIG_A1_20261011.md) | 在用 |
| [`prepare_f02_mixed_scene.py`](prepare_f02_mixed_scene.py) | Freeze the accepted F02 + M04 asset inputs for the public mixed planner. | [F02_MTB_MIXED_SCENE_A1_20261010.md](../../docs/rendering/F02_MTB_MIXED_SCENE_A1_20261010.md) | 在用 |
| [`prepare_first_dataset_a1.py`](prepare_first_dataset_a1.py) | CPU metadata planner.  | [FIRST_DATASET_PREPARATION_A1_20261011.md](../../docs/training/FIRST_DATASET_PREPARATION_A1_20261011.md) | 在用 |
| [`prepare_first_four_episodes_a1.py`](archive/prepare_first_four_episodes_a1.py) | Freeze first-trial road inputs; use the existing mixed road planner for CPU checks. | —（无引用） | 仅历史（已归档） |
| [`prepare_front_back_side_a1.py`](prepare_front_back_side_a1.py) | Freeze camera-only candidates around an unchanged five-rider city episode. | [FRONT_BACK_SIDE_MULTIRIG_A1_20261011.md](../../docs/rendering/FRONT_BACK_SIDE_MULTIRIG_A1_20261011.md) | 在用 |
| [`prepare_native_ground_repairs.py`](archive/prepare_native_ground_repairs.py) | Bounded world-placement repair using native terrain probes, never pose edits. | —（无引用） | 仅历史（已归档） |
| [`prepare_pedestrian_seeded_assets.py`](prepare_pedestrian_seeded_assets.py) | Bounded local walking adapter: fixed target shape, source root, fixed bindings. | [PEDESTRIAN_SEEDED_PILOT_A1_20261009.md](../../docs/training/PEDESTRIAN_SEEDED_PILOT_A1_20261009.md) | 在用 |
| [`prepare_procedural_eyes.py`](prepare_procedural_eyes.py) | Generate material coordinates from native SMPL-X shaped rest vertices. | [ASSETS.md](../../docs/ASSETS.md) | 在用 |
| [`prepare_second_environment_a1.py`](prepare_second_environment_a1.py) | Place a frozen boundary-pass episode into a native road environment. | [SECOND_ENVIRONMENT_BOUNDARY_PASS_A1_20261010.md](../../docs/rendering/SECOND_ENVIRONMENT_BOUNDARY_PASS_A1_20261010.md) | 在用 |
| [`prepare_source_isolated_assets.py`](prepare_source_isolated_assets.py) | Reuse unchanged assets or regenerate body, simple clothes and bound hair per pose. | [READING_GUIDE.md](../../docs/READING_GUIDE.md) | 在用 |
| [`prepare_source_isolated_split.py`](archive/prepare_source_isolated_split.py) | Freeze conservative identity/source groups before producing new combinations. | —（无引用） | 仅历史（已归档） |
| [`prepare_source_trial_config.py`](prepare_source_trial_config.py) | Prepare a bounded next-run recipe and audit immutable regression references. | [SCENE_RECIPE_INDEX.md](../../docs/rendering/SCENE_RECIPE_INDEX.md) | 在用 |
| [`prepare_t_junction_a1.py`](prepare_t_junction_a1.py) | Freeze a bounded three-rider T-junction episode from accepted city assets. | [T_JUNCTION_MULTIRIG_A1_20261011.md](../../docs/rendering/T_JUNCTION_MULTIRIG_A1_20261011.md) | 在用 |
| [`prepare_urban_three_multirig_a1.py`](prepare_urban_three_multirig_a1.py) | Freeze and preflight a second paired rig for a fixed three-rider city scene. | [URBAN_THREE_CYCLISTS_MULTIRIG_A1_20261011.md](../../docs/rendering/URBAN_THREE_CYCLISTS_MULTIRIG_A1_20261011.md) | 在用 |
| [`reconstruct_cycling_hmr2.py`](archive/reconstruct_cycling_hmr2.py) | Recover independent SMPL poses from supplied cyclist boxes using local 4D-Humans. | —（无引用） | 仅历史（已归档） |
| [`reconstruct_mtb_video_a2.py`](reconstruct_mtb_video_a2.py) | Bounded local MTB MHR reconstruction and fixed-shape SMPL-X adaptation; external code re… | [MTB_VIDEO_SMPLX_PILOT_A2_20261009.md](../../docs/motion/cycling/MTB_VIDEO_SMPLX_PILOT_A2_20261009.md) | 在用 |
| [`seed_asset_composition_a1.py`](seed_asset_composition_a1.py) | Seed compatible components, then explicitly build and preflight road episodes. | [SEEDED_ASSET_COMPOSITION_A1_20261010.md](../../docs/rendering/SEEDED_ASSET_COMPOSITION_A1_20261010.md) | 在用 |
| [`seed_cycling_team_a1.py`](seed_cycling_team_a1.py) | Seed distinct cycling components, then explicitly build and preflight teams. | [SEEDED_CYCLING_TEAM_A1_20261010.md](../../docs/rendering/SEEDED_CYCLING_TEAM_A1_20261010.md) | 在用 |
| [`seed_fixed_person_side_pass_a1.py`](seed_fixed_person_side_pass_a1.py) | Freeze a rider trajectory and seed only the *whole* paired rig pose. | [MULTIRIG_SCENE_RENDER_A1_20261010.md](../../docs/rendering/MULTIRIG_SCENE_RENDER_A1_20261010.md) | 在用 |
| [`seed_scene_modes.py`](seed_scene_modes.py) | Freeze bounded road scene recipes; rendering is a separate explicit step. | [SEEDED_SCENE_MODES_A1_20261010.md](../../docs/rendering/SEEDED_SCENE_MODES_A1_20261010.md) | 在用 |
| [`select_garment_window_a2.py`](select_garment_window_a2.py) | Describe an existing garment or select a contiguous cache-consumption view. | [GARMENT_RENDER_CONTRACT_A2_20261010.md](../../docs/appearance/GARMENT_RENDER_CONTRACT_A2_20261010.md) | 在用 |
| [`select_t_junction_local_rigs_a1.py`](archive/select_t_junction_local_rigs_a1.py) | Apply frozen edge-first/ordinary-fallback exact support policy. | —（无引用） | 仅历史（已归档） |
| [`transfer_bedlam2_cloth_to_m03_a2.py`](transfer_bedlam2_cloth_to_m03_a2.py) | Transfer one original BEDLAM2 simulated garment onto a fixed target SMPL-X body. | [BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md](../../docs/appearance/BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md) | 在用 |

### 冻结・打包・交付（19 个，其中 12 个在用）

| 脚本 | 做什么 | 结论 / 依据落在 | 状态 |
|---|---|---|---|
| [`exact_t_junction_rig_repair_a1.py`](archive/exact_t_junction_rig_repair_a1.py) | Run at most three already frozen exact support candidates per view. | —（无引用） | 仅历史（已归档） |
| [`export_groom_reference.py`](export_groom_reference.py) | Read the official locked-head neutral mesh as a groom reference, never a label body. | [COMMUTER_TWO_CHARACTER_PILOT_20260930.md](../../docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md) | 在用 |
| [`export_mamma_observations_a3.py`](export_mamma_observations_a3.py) | Export an existing MAMMA ROI run; does not start rendering. | [MAMMA_GENERATION_REFACTOR_A3_20261022.md](../../docs/data/MAMMA_GENERATION_REFACTOR_A3_20261022.md) | 在用 |
| [`finalize_adult_candidates.py`](finalize_adult_candidates.py) | Record reviewed candidates separately from diagnostic rows and superseded versions. | [PRUNING_FINAL_20261003.md](../../docs/refactor/PRUNING_FINAL_20261003.md) | 在用 |
| [`finalize_bvh_p1_dressed_a2.py`](archive/finalize_bvh_p1_dressed_a2.py) | Finalize two M02 dressed reviews after both complete, preserving A2 catalog. | —（无引用） | 仅历史（已归档） |
| [`finalize_cycling_characters.py`](archive/finalize_cycling_characters.py) | Summarize completed integration evidence and preserve intermediate versions. | —（无引用） | 仅历史（已归档） |
| [`finalize_first_four_episodes_a1.py`](archive/finalize_first_four_episodes_a1.py) | Read-only accounting and viewing index for the bounded four-episode run. | —（无引用） | 仅历史（已归档） |
| [`finalize_mamma_512_label_probe_a3.py`](finalize_mamma_512_label_probe_a3.py) | Finalize the sparse e03 MAMMA 512 label probe without sequence packing. | [MAMMA_512_LABEL_PILOT_A3_20261015.md](../../docs/data/MAMMA_512_LABEL_PILOT_A3_20261015.md) | 在用 |
| [`finalize_t_junction_local_delivery_a1.py`](finalize_t_junction_local_delivery_a1.py) | Summarize the frozen junction run and link its existing ROI videos. | [T_JUNCTION_LOCAL_RIG_DELIVERY_A1_20261011.md](../../docs/rendering/T_JUNCTION_LOCAL_RIG_DELIVERY_A1_20261011.md) | 在用 |
| [`freeze_bvh_p1_dressed_a2.py`](freeze_bvh_p1_dressed_a2.py) | Freeze two accepted M02 body sequences for simple garment/groom review. | [BVH_P1_DRESSED_ASSETS_A2_20261010.md](../../docs/motion/cycling/BVH_P1_DRESSED_ASSETS_A2_20261010.md) | 在用 |
| [`freeze_bvh_p1_smplx_a2.py`](freeze_bvh_p1_smplx_a2.py) | Freeze the two reviewed p1 BVH windows before any SMPL-X optimization. | [BVH_P1_SMPLX_PILOT_A2_20261010.md](../../docs/motion/cycling/BVH_P1_SMPLX_PILOT_A2_20261010.md) | 在用 |
| [`freeze_cycling_characters.py`](archive/freeze_cycling_characters.py) | Freeze candidate metadata after a recorded visual review and geometry check. | —（无引用） | 仅历史（已归档） |
| [`freeze_direct_erp_roi_trial.py`](freeze_direct_erp_roi_trial.py) | Freeze one existing scene state and bounded same-pixel A/B/C comparisons. | [DIRECT_ERP_ROI_TRIAL_20261009.md](../../docs/rendering/DIRECT_ERP_ROI_TRIAL_20261009.md) | 在用 |
| [`freeze_f02_mixed_repair.py`](freeze_f02_mixed_repair.py) | Freeze one bounded F02 + M04 repair layout list before CPU preflight. | [F02_MTB_MIXED_SCENE_REPAIR_A1_20261010.md](../../docs/rendering/F02_MTB_MIXED_SCENE_REPAIR_A1_20261010.md) | 在用 |
| [`package_bvh_p1_dressed_a2.py`](archive/package_bvh_p1_dressed_a2.py) | Package one M02 BVH dressed sequence and exercise A1 read-only adapters. | —（无引用） | 仅历史（已归档） |
| [`package_bvh_p1_smplx_a2.py`](archive/package_bvh_p1_smplx_a2.py) | Write body-only handoff for the two frozen p1 windows; no scene assets. | —（无引用） | 仅历史（已归档） |
| [`package_bvh_p2_dressed_asset_a2.py`](package_bvh_p2_dressed_asset_a2.py) | Deliver one M01/BVH dressed asset and exercise A1's read-only adapter. | [BVH_P2_DRESSED_ASSET_A2_20261010.md](../../docs/motion/cycling/BVH_P2_DRESSED_ASSET_A2_20261010.md) | 在用 |
| [`prune_commuter_groom.py`](prune_commuter_groom.py) | Physically remove a bounded number of invalid strands, consistently across poses. | [COMMUTER_TWO_CHARACTER_PILOT_20260930.md](../../docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md) | 在用 |
| [`resume_roi_only.py`](resume_roi_only.py) | One bounded consumer-failure recovery; reuse frame0, never rerender it. | [ROI_ONLY_RENDER_A1_20261009.md](../../docs/rendering/ROI_ONLY_RENDER_A1_20261009.md) | 在用 |

### 运行・渲染・拟合（27 个，其中 20 个在用）

| 脚本 | 做什么 | 结论 / 依据落在 | 状态 |
|---|---|---|---|
| [`consume_continuous_roi.py`](consume_continuous_roi.py) | Synchronous compact consumption and bounded sequence Dataset/video delivery. | [PERSISTENT_ROI_CONSUMER_A1_20261009.md](../../docs/rendering/PERSISTENT_ROI_CONSUMER_A1_20261009.md) | 在用 |
| [`consume_direct_erp_roi_trial.py`](consume_direct_erp_roi_trial.py) | Three-route trial consumers and fixed-crop comparisons; never loads a model. | [DIRECT_ERP_ROI_TRIAL_20261009.md](../../docs/rendering/DIRECT_ERP_ROI_TRIAL_20261009.md) | 在用 |
| [`consume_road_pass_probe.py`](consume_road_pass_probe.py) | Final-raster DS consumer + per-episode packing + bounded real Dataset checks. | [PEDESTRIAN_SEEDED_PILOT_A1_20261009.md](../../docs/training/PEDESTRIAN_SEEDED_PILOT_A1_20261009.md) | 在用 |
| [`fit_apd_6526_pilot.py`](fit_apd_6526_pilot.py) | Isolated, budgeted APD 6526 pose adaptation.  | [APD_6526_SMPLX_PILOT_20261008.md](../../docs/motion/cycling/APD_6526_SMPLX_PILOT_20261008.md) | 在用 |
| [`fit_bvh_p2_smplx_a2.py`](fit_bvh_p2_smplx_a2.py) | Bounded native BVH -> fixed-shape SMPL-X cycling body experiment. | [BVH_P2_SMPLX_PILOT_A2_20261010.md](../../docs/motion/cycling/BVH_P2_SMPLX_PILOT_A2_20261010.md) | 在用 |
| [`preview_mixed_road_pass.py`](preview_mixed_road_pass.py) | CPU-only physical-domain projection review; no RGB render or model. | [MIXED_WALK_CYCLING_EDGE_A1_20261009.md](../../docs/rendering/MIXED_WALK_CYCLING_EDGE_A1_20261009.md) | 在用 |
| [`preview_multiperson_event.py`](archive/preview_multiperson_event.py) | Three label/ID evidence overlays from final ROI data, no renderer or model. | —（无引用） | 仅历史（已归档） |
| [`probe_cuda_device.py`](archive/probe_cuda_device.py) | Small isolated device check; never loads DenseKP or resets a GPU. | —（无引用） | 仅历史（已归档） |
| [`refine_cycling_character_clothing.py`](refine_cycling_character_clothing.py) | Clip garment boundaries within body triangles instead of selecting whole faces. | [test_character_clothing.py](../../tests/test_character_clothing.py) | 在用（被测试引用） |
| [`refine_first_four_e00_rig_a1.py`](archive/refine_first_four_e00_rig_a1.py) | Bounded rig-only refinement of the already frozen e00 layout attempt 1. | —（无引用） | 仅历史（已归档） |
| [`relax_commuter_groom.py`](relax_commuter_groom.py) | Bounded control-point/body collision correction; originals and bindings remain intact. | [COMMUTER_TWO_CHARACTER_PILOT_20260930.md](../../docs/appearance/COMMUTER_TWO_CHARACTER_PILOT_20260930.md) | 在用 |
| [`render_adult_candidates_ds.py`](render_adult_candidates_ds.py) | Small integration-only DS layouts; numerical pass never implies appearance pass. | [COMMANDS.md](../../docs/COMMANDS.md) | 在用 |
| [`render_cycling_characters_ds.py`](render_cycling_characters_ds.py) | Two bounded layouts covering eight independent character/pose derivatives. | [COMMANDS.md](../../docs/COMMANDS.md) | 在用 |
| [`render_cycling_ds.py`](render_cycling_ds.py) | Render two independent cycling bodies in masked, paired DS scene pilots. | [COMMANDS.md](../../docs/COMMANDS.md) | 在用 |
| [`render_source_ground_repairs.py`](render_source_ground_repairs.py) | Render only bounded, already-probed placement repairs into new directories. | [DUAL_VIEW_TEMPORAL_AUDIT_20261008.md](../../docs/training/DUAL_VIEW_TEMPORAL_AUDIT_20261008.md) | 在用 |
| [`repair_front_rig_a1.py`](archive/repair_front_rig_a1.py) | Plan a closer front paired rig for the frozen city five-rider episode. | —（无引用） | 仅历史（已归档） |
| [`repair_t_junction_rigs_a1.py`](repair_t_junction_rigs_a1.py) | Bounded, replayable position-and-physical-domain search for a fixed junction scene. | [T_JUNCTION_RIG_REPAIR_A1_20261011.md](../../docs/rendering/T_JUNCTION_RIG_REPAIR_A1_20261011.md) | 在用 |
| [`revise_adult_skin.py`](archive/revise_adult_skin.py) | Preserve similar-face failures; revise skin within the same synthetic identity family. | —（无引用） | 仅历史（已归档） |
| [`revise_cycling_eye_materials.py`](archive/revise_cycling_eye_materials.py) | Version the four-character eye material correction; preserve all geometry. | —（无引用） | 仅历史（已归档） |
| [`run_continuous_roi.py`](run_continuous_roi.py) | One Blender process, explicit bounded timeline; synchronous CPU consumption. | [CYCLING_ROI_INTEGRATION_A1_20261009.md](../../docs/rendering/CYCLING_ROI_INTEGRATION_A1_20261009.md) | 在用 |
| [`run_direct_erp_roi_stage.py`](run_direct_erp_roi_stage.py) | Run only this trial's Blender child with budget/resource accounting. | [DIRECT_ERP_ROI_TRIAL_20261009.md](../../docs/rendering/DIRECT_ERP_ROI_TRIAL_20261009.md) | 在用 |
| [`run_ds_source_resolution.py`](run_ds_source_resolution.py) | Render the frozen heavy DS scene at three source ERP resolutions. | [DUAL_VIEW_TEMPORAL_AUDIT_20261008.md](../../docs/training/DUAL_VIEW_TEMPORAL_AUDIT_20261008.md) | 在用 |
| [`run_multirig_roi_a1.py`](run_multirig_roi_a1.py) | Bounded scene episode with a frozen list of paired ROI rigs in one frame loop. | [SCENE_EDITOR_A3_20261013.md](../../docs/tools/SCENE_EDITOR_A3_20261013.md) | 在用 |
| [`run_source_isolated_batch.py`](run_source_isolated_batch.py) | Bounded layouts from a frozen split; pilots must be reviewed before expansion. | [DUAL_VIEW_TEMPORAL_AUDIT_20261008.md](../../docs/training/DUAL_VIEW_TEMPORAL_AUDIT_20261008.md) | 在用 |
| [`run_source_occlusion_pairs.py`](run_source_occlusion_pairs.py) | Four preselected controls: remove only foreground person 1, retain all world transforms. | [DUAL_VIEW_TEMPORAL_AUDIT_20261008.md](../../docs/training/DUAL_VIEW_TEMPORAL_AUDIT_20261008.md) | 在用 |
| [`train_source_isolated_trial.py`](train_source_isolated_trial.py) | Next-turn bounded trial.  | [COMMANDS.md](../../docs/COMMANDS.md) | 在用 |
| [`visualize_seeded_boundary.py`](archive/visualize_seeded_boundary.py) | Small CPU diagrams from an already preflighted single-person episode. | —（无引用） | 仅历史（已归档） |

### 核查・验收・审计（36 个，其中 10 个在用）

| 脚本 | 做什么 | 结论 / 依据落在 | 状态 |
|---|---|---|---|
| [`audit_body_uv.py`](archive/audit_body_uv.py) | 审计 SMPL-X body mesh 与 UV OBJ 的 face-corner 拓扑兼容性。 | —（无引用） | 仅历史（已归档） |
| [`audit_clothing_uv.py`](archive/audit_clothing_uv.py) | 全量核对 BEDLAM clothing NPZ 与参考 OBJ 的几何及 corner UV 索引。 | —（无引用） | 仅历史（已归档） |
| [`audit_first_four_box_readback_a1.py`](archive/audit_first_four_box_readback_a1.py) | Limited CPU readback of shared native parents with box augmentation off/on. | —（无引用） | 仅历史（已归档） |
| [`audit_identity_texture_sources.py`](audit_identity_texture_sources.py) | Reproducible texture comparisons; similarity is evidence, never an identity score. | [test_source_isolation.py](../../tests/test_source_isolation.py) | 在用（被测试引用） |
| [`audit_new_pedestrian_source.py`](archive/audit_new_pedestrian_source.py) | Review a complete new SMPL-X motion on unchanged target identities. | —（无引用） | 仅历史（已归档） |
| [`audit_pedestrian_poses.py`](archive/audit_pedestrian_poses.py) | Inspect existing full SMPL-X motion frames without inventing action labels. | —（无引用） | 仅历史（已归档） |
| [`audit_second_environment_a1.py`](archive/audit_second_environment_a1.py) | Read-only numerical audit and compact review drawings for A1's environment probe. | —（无引用） | 仅历史（已归档） |
| [`audit_seeded_asset_composition_a1.py`](archive/audit_seeded_asset_composition_a1.py) | Small manifest/annotation audit for the two frozen composed road episodes. | —（无引用） | 仅历史（已归档） |
| [`audit_skin_texture.py`](audit_skin_texture.py) | Record official texture provenance and component/eye UV compatibility evidence. | [ASSETS.md](../../docs/ASSETS.md) | 在用 |
| [`check_first_dataset_box_support_a1.py`](archive/check_first_dataset_box_support_a1.py) | Bounded read-only audit of old ROI packs; writes only small metadata overlays. | —（无引用） | 仅历史（已归档） |
| [`check_front_back_side_clearance_a1.py`](archive/check_front_back_side_clearance_a1.py) | Short static-ray clearance audit for the selected paired city rig centers. | —（无引用） | 仅历史（已归档） |
| [`check_repaired_front_clearance_a1.py`](archive/check_repaired_front_clearance_a1.py) | Check the selected new paired camera centers against static city geometry. | —（无引用） | 仅历史（已归档） |
| [`check_t_junction_ground_a1.py`](archive/check_t_junction_ground_a1.py) | Check the frozen three-rider turn against the delivered blend's native road. | —（无引用） | 仅历史（已归档） |
| [`check_t_junction_rig_repair_clearance_a1.py`](archive/check_t_junction_rig_repair_clearance_a1.py) | Evaluate frozen paired camera centers against native city geometry. | —（无引用） | 仅历史（已归档） |
| [`evaluate_bedlam2_locked_body_a2.py`](evaluate_bedlam2_locked_body_a2.py) | Evaluate one published BEDLAM2 locked-head body sequence for garment transfer. | [BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md](../../docs/appearance/BEDLAM_M03_CLOTH_TRANSFER_A2_20261010.md) | 在用 |
| [`evaluate_ds_source_resolution.py`](evaluate_ds_source_resolution.py) | Freeze and evaluate fixed DenseKP crops for the source-resolution experiment. | [DS_SOURCE_RESOLUTION_20261008.md](../../docs/rendering/DS_SOURCE_RESOLUTION_20261008.md) | 在用 |
| [`fetch_bedlam_motion_member.py`](archive/fetch_bedlam_motion_member.py) | Read only TAR headers and one walking member using an authenticated session. | —（无引用） | 仅历史（已归档） |
| [`fetch_bedlam_native_walk_assets.py`](archive/fetch_bedlam_native_walk_assets.py) | Interactive, bounded official BEDLAM2 retrieval; credentials stay in memory. | —（无引用） | 仅历史（已归档） |
| [`inspect_bedlam2_groom.py`](inspect_bedlam2_groom.py) | Blender Alembic import audit; preserves raw curves without claiming a head binding. | [ASSETS.md](../../docs/ASSETS.md) | 在用 |
| [`inspect_t_junction_a1.py`](archive/inspect_t_junction_a1.py) | Read-only inspection of the two end junctions in the delivered city blend. | —（无引用） | 仅历史（已归档） |
| [`review_f02_preflight.py`](archive/review_f02_preflight.py) | CPU-only physical-domain illustration of the first frozen F02 candidate. | —（无引用） | 仅历史（已归档） |
| [`review_source_layouts.py`](archive/review_source_layouts.py) | Contact sheets at full 256-square scale; does not approve any candidate. | —（无引用） | 仅历史（已归档） |
| [`summarize_front_back_side_a1.py`](archive/summarize_front_back_side_a1.py) | Small SVG/JSON review plots for a frozen five-person, three-rig scene. | —（无引用） | 仅历史（已归档） |
| [`summarize_multiperson_road_pass.py`](archive/summarize_multiperson_road_pass.py) | Summarize a completed single multiperson run; no rendering, models or Dataset reads. | —（无引用） | 仅历史（已归档） |
| [`summarize_seeded_scene_modes.py`](archive/summarize_seeded_scene_modes.py) | Bounded manifest-only acceptance for two explicit seeded ROI episodes. | —（无引用） | 仅历史（已归档） |
| [`summarize_t_junction_a1.py`](archive/summarize_t_junction_a1.py) | Small read-only geometric delivery for a bounded T-junction preflight. | —（无引用） | 仅历史（已归档） |
| [`validate_direct_erp_roi_trial.py`](validate_direct_erp_roi_trial.py) | Read-only buffer/point audit plus cost summary for the completed trial. | [DIRECT_ERP_ROI_TRIAL_20261009.md](../../docs/rendering/DIRECT_ERP_ROI_TRIAL_20261009.md) | 在用 |
| [`validate_dispersed_cyclists_a1.py`](archive/validate_dispersed_cyclists_a1.py) | Full-slot validation for a 3–5-person city scene with paired rig lists. | —（无引用） | 仅历史（已归档） |
| [`validate_front_back_side_a1.py`](archive/validate_front_back_side_a1.py) | Check three actual paired rigs against one frozen five-rider city scene. | —（无引用） | 仅历史（已归档） |
| [`validate_front_rig_size_repair_a1.py`](archive/validate_front_rig_size_repair_a1.py) | Validate the replacement front paired rig and quantify actual cam4 scale. | —（无引用） | 仅历史（已归档） |
| [`validate_mamma_512_handoff_a3.py`](validate_mamma_512_handoff_a3.py) | Read at most six rows through A1's Dataset without importing its model stack. | [MAMMA_512_LABEL_PILOT_A3_20261015.md](../../docs/data/MAMMA_512_LABEL_PILOT_A3_20261015.md) | 在用 |
| [`validate_multirig_scene_a1.py`](archive/validate_multirig_scene_a1.py) | Bounded post-run checks for the two-rig single-person scene episode. | —（无引用） | 仅历史（已归档） |
| [`validate_persistent_roi.py`](validate_persistent_roi.py) | Bounded same-buffer process comparison and short ROI sequence delivery. | [PERSISTENT_ROI_CONSUMER_A1_20261009.md](../../docs/rendering/PERSISTENT_ROI_CONSUMER_A1_20261009.md) | 在用 |
| [`validate_roi_only.py`](validate_roi_only.py) | Bounded six same-buffer comparisons plus production packing/ROI preview. | [ROI_ONLY_RENDER_A1_20261009.md](../../docs/rendering/ROI_ONLY_RENDER_A1_20261009.md) | 在用 |
| [`validate_t_junction_local_delivery_a1.py`](validate_t_junction_local_delivery_a1.py) | Validate one frozen three-rig junction delivery without model inference. | [T_JUNCTION_LOCAL_RIG_DELIVERY_A1_20261011.md](../../docs/rendering/T_JUNCTION_LOCAL_RIG_DELIVERY_A1_20261011.md) | 在用 |
| [`validate_urban_three_multirig_a1.py`](archive/validate_urban_three_multirig_a1.py) | Bounded postproduction check for the fixed three-cyclist, two-rig episode. | —（无引用） | 仅历史（已归档） |
## 新增脚本时的约定

1. 第一行 docstring 写清「这个脚本验证什么」，一句话，别写「临时脚本」。
2. 产出写进 `outputs/`，不要写进仓库。
3. 如果它是某个阶段结论的证据，就在对应的 `docs/` 报告里引用它的文件名——否则下次做这张表时它会被判成「仅历史」而被归档。
