# Phase 4 deletion audit

Audit date: 2026-09-20. Runtime references were checked with repository-wide
search. Generated outputs are ignored and were treated as dependencies only when
current code or current documentation names them.

## DELETE

| Files | Runtime references | Docs references | Generated artifact dependency | Replacement | Reason |
|---|---|---|---|---|---|
| `tools/scene/inspect_downloaded_scene.py`, `inspect_forest_nodes.py`, `inspect_pine_forest.py` | none | old code/dependency maps only | none used by build or validation | scene READMEs and active build/validate tools | one-off downloaded-scene inspection is complete; no unique pipeline behavior |
| `tools/experiments/audit_cycling_motion.py` | none | refactor maps and `motion/cycling/CYCLING_MOTION_AUDIT.md` | obsolete `outputs/audits/cycling_motion_audit/local_inventory.txt` with stale absolute paths | cycling audit document; active BABEL/BVH tools | the complete conclusions and limitations are already preserved in the audit document |
| `tools/experiments/audit_eye_realism.py` | none after deletion | old refactor/code map | historical audit output only; not required by rendering | `prepare_procedural_eyes.py`, `audit_eye_geometry.py`, current appearance validation/docs | historical replay assertion is known to fail; current eye asset and baseline have independent generators/evidence |
| `tools/common/` | none | none | none | `src/densekp_retrain/` packages | contained only ignored `__pycache__`; source files and imports were absent |

## KEEP_ACTIVE

| Files | Runtime references | Docs references | Generated artifact dependency | Replacement | Reason |
|---|---|---|---|---|---|
| `tests/**`, `tools/validate/**`, `tools/scene/validate_cycling_forest.py` | regression entry points | validation documents | validates current artifacts | none | evidence chain, not redundant code |
| `tools/experiments/audit_eye_geometry.py` | imported by other eye tooling | eye audit docs | eye evidence | none | active code dependency |
| `tools/experiments/audit_body_uv.py` | imported by other audits | texture docs | UV evidence | none | active code dependency |
| `tools/experiments/audit_cyclist_bvh.py` | imported by `review_cyclist_audit.py` | cycling docs | BVH review | none | active code dependency |
| `tools/experiments/prepare_procedural_eyes.py` | current asset generator | appearance baseline docs | `procedural_eye_coordinates.npz` | none | only reproducible source of current eye asset |

## KEEP_PROVENANCE

| Files | Runtime references | Docs references | Generated artifact dependency | Replacement | Reason |
|---|---|---|---|---|---|
| `tools/scene/build_reference_riverfront.py`, `assets/scenes/riverfront/` | no current render pipeline import | asset README, outdoor-assets history, current scene map | reproducible `riverfront.blend` recipe and retained `.blend1` | current production scenes are cycling forest and hidden alley | not active production, but still a self-contained, documented reference-scene recipe; delete only with an explicit asset-retention decision |

## DEFER

| Files | Runtime references | Docs references | Generated artifact dependency | Replacement | Reason |
|---|---|---|---|---|---|
| `tools/fisheye/render_equirectangular_source.py`, `render_equirectangular_scene_source.py` | called by fisheye pipelines | fisheye/scene docs | ERP RGB/depth/IndexOB and manifests | none | large duplication, but safe consolidation requires Blender E2E output regression |
| `tools/fisheye/run_fisheye_camera_p0.py::warp_case` | imported by three fisheye CLIs | fisheye architecture docs | RGB/depth/mask/visibility outputs | future `src` module | current numeric test covers projection only, not complete warp behavior |
