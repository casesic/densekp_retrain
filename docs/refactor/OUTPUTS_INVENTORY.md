# Outputs inventory

Inventory date: 2026-09-20. Sizes are the pre-move `du -sh` values. Runtime and
test references were searched in `src`, `tools`, `tests`, `configs`, and scripts.

| Current path before Phase 5 | Size | Type | Runtime reference | Test reference | Docs reference | Destination / retained path | Action |
|---|---:|---|---|---|---|---|---|
| `p0_dressed_eye_background_fixed` | 170M | BASELINE | default appearance, stress geometry | perspective regression | baseline docs | `baselines/` | MOVE |
| `fisheye_camera_p0` | 1.1G | BASELINE | fisheye scene/sweep fingerprint | fisheye regression | fisheye docs | `baselines/` | MOVE |
| `temporal_case_aaron_1095` | 430M | BASELINE | fisheye default geometry | fisheye regression | temporal docs | `baselines/` | MOVE |
| `p0_perspective` | 57M | BASELINE | generator/finalizer defaults | indirect | rendering docs | `baselines/` | MOVE |
| `p0_real_motion` | 57M | BASELINE | no current default | none | formal regression | `baselines/` | MOVE |
| `p0_appearance_baseline` | 60M | BASELINE | no current default | none | frozen reference | `baselines/` | MOVE |
| `p0_dressed_dataset` | 98M | BASELINE | eye/skin generators | none | dressed validation | `baselines/` | MOVE |
| `p0_dressed_textured` | 93M | BASELINE | review input | none | frozen appearance | `baselines/` | MOVE |
| `p0_dressed_skin_textured` | 94M | BASELINE | review input | none | frozen appearance | `baselines/` | MOVE |
| `p0_dressed_hdri` | 100M | BASELINE | no current default | none | frozen appearance | `baselines/` | MOVE |
| `p0_dressed_hdri_eye_fixed` | 96M | BASELINE | CLI default/review input | none | appearance history | `baselines/` | MOVE |
| `cycling_forest` | 12M | SCENE | build/validate defaults | none | scene docs | `scenes/` | MOVE |
| `photoreal_hidden_alley` | 13M | SCENE | scene preparation default | none | scene docs | `scenes/` | MOVE |
| `riverfront` | 5.7M | SCENE | build preview default | none | asset README | `scenes/` | MOVE |
| `appearance_v1c_audit` | 12K | AUDIT | none | none | appearance docs | `audits/` | MOVE |
| `bedlam_tiny_mapping` | 92K | AUDIT | none | none | BEDLAM docs | `audits/` | MOVE |
| `clothing_texture_audit` | 14M | AUDIT | none | none | appearance docs | `audits/` | MOVE |
| `cycling_motion_audit` | 1.9M | AUDIT | BABEL search output | none | cycling docs | `audits/` | MOVE |
| `dedicated_cycling_mocap_audit` | 76M | AUDIT | review output | none | cycling docs | `audits/` | MOVE |
| `body_texture_audit` | 1.2M | GENERATED_ASSET | `body_uv.npz` read by several tools | none | appearance docs | root unchanged | DEFER |
| `eye_fix_v2` | 362M | GENERATED_ASSET | procedural mapping read by render jobs | baseline acceptance | baseline docs | root unchanged | DEFER |
| `perspective_camera_stress` | 1.7G | EXPERIMENT | stress tools | none | camera docs | `experiments/` | MOVE |
| `human_scale_sweep` | 1.5G | EXPERIMENT | sweep output | none | limited | `experiments/` | MOVE |
| `fisheye_scene_p0` | 16K | EXPERIMENT | scene CLI output | none | scene/fisheye docs | `experiments/` | MOVE |
| `background_fix` | 1.4M | EXPERIMENT | comparison output | none | appearance evidence | `experiments/` | MOVE |
| `eye_fix` | 14M | EXPERIMENT | none | none | appearance evidence | `experiments/` | MOVE |
| `simple_mode_regression` | 6.5M | EXPERIMENT | none | none | historical | `experiments/` | MOVE |
| `temporal_case_kit_bend` | 323M | EXPERIMENT | none | none | temporal docs | `experiments/` | MOVE |
| `temporal_case_procedural_cyclic` | 281M | EXPERIMENT | none | none | temporal docs | `experiments/` | MOVE |
| `temporal_validation_summary.csv` | 4K | EXPERIMENT | none | none | temporal docs | `experiments/` | MOVE |
| `hdri_preview_*` | 33M | PREVIEW | none | none | appearance evidence | `previews/hdr/<setting>/` | MOVE; DELETE_CANDIDATE |
| `headless_smoke` | 1.5M | SMOKE | smoke defaults | none | environment docs | `smoke/headless/` | MOVE |
| `ppt_materials_phase17_18` + zip | 92M | TEMP | none | none | none | `archive/` | MOVE; DELETE_CANDIDATE |

No output was deleted. `datasets/` is intentionally empty and reserved for future
generated dataset deliverables.
