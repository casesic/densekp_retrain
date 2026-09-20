# Phase 3 delete candidates

每一项均先搜索了文件名、顶层函数/类、imports、subprocess、shell、configs、README、
docs 和 tests。文件名本身不作为删除理由。

| file | reason | replacement | references found | decision |
|---|---|---|---|---|
| `tools/common/{__init__,paths,runtime}.py` | Phase 2 临时共享模块，已被正式 src package 取代 | `densekp_retrain.runtime` | 6 render/fisheye wrappers，imports 已迁移 | **DELETE** |
| `tools/fisheye/omni_radtan.py` | reusable camera implementation 不应位于 CLI tree | `densekp_retrain.camera.omni_radtan` | fisheye drivers、package init、test、docs，imports 已迁移且 regression PASS | **DELETE** |
| `tools/fisheye/test_omni_radtan.py` | 真正单测应位于 `tests/` | `tests/test_omni_radtan.py` | two subprocess drivers、docs，callers 已迁移 | **DELETE** |
| `tools/scene/scene_loader.py` | reusable scene spec，不是 CLI | `densekp_retrain.scene.spec` | fisheye scene driver，import 已迁移 | **DELETE** |
| `tools/experiments/audit_body_uv.py` | 名称像 audit，但被 eye/skin audits import，且有资产审计价值 | none | two imports、body texture docs | KEEP |
| `tools/experiments/audit_clothing_uv.py` | clothing UV evidence generator | none | refactor/code-map docs | KEEP |
| `tools/experiments/audit_eye_geometry.py` | `audit_eye_realism` imports its math; generated eye evidence | none | runtime import、docs | KEEP |
| `tools/experiments/audit_eye_realism.py` | historical validation, known replay limitation | none | audit docs | KEEP |
| `tools/experiments/audit_skin_texture.py` | imports body UV parser and records texture evidence | none | runtime import、docs | KEEP |
| `tools/experiments/prepare_procedural_eyes.py` | generated current `procedural_eye_coordinates.npz` | no equivalent generator | render baseline docs/output dependency | KEEP |
| `tools/experiments/audit_cyclist_bvh.py` | parser/FK is imported by review script | none | runtime import、mocap audit docs | KEEP |
| `tools/experiments/find_cyclist_clip.py` | current dedicated mocap window ranker | none | mocap audit docs | KEEP |
| `tools/experiments/review_cyclist_audit.py` | current mapping/kinematics evidence generator | none | runtime import、docs | KEEP |
| `tools/experiments/search_babel_cycling.py` | unique BABEL candidate search | none | audit/code-map docs | KEEP |
| `tools/experiments/audit_cycling_motion.py` | one-off inventory audit, but no proven equivalent and retained historical evidence | none | refactor/code-map docs and output inventory | KEEP |
| `tools/experiments/generate_cyclic_validation_motion.py` | deterministic validation fixture generator | none | code-map docs | KEEP |
| `tools/scene/inspect_*.py` | one-off debug/inspection, but tied to current scene asset audit and no replacement | none | scene outputs/docs | KEEP |
| `tools/scene/validate_cycling_forest.py` | active scene validator explicitly protected by Phase 3 scope | none | builder/scene docs | KEEP |

结论：除被 src/tests 完全替代的 6 个 implementation/test 文件外，没有高置信业务或实验脚本删除候选。
