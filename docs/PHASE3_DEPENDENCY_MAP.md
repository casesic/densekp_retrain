# Phase 3 dependency map

基线：`fa7fbfd`。本图由 Phase 3 开始时对 `tools/`、`densekp/`、`configs/`、
`scripts/`、`docs/` 和 `pyproject.toml` 的实际扫描得到。`densekp/` 是原 CameraHMR
训练边界；下表只规划本项目新增 infrastructure。

## 分类

- A：CLI/参数解析；B：通用工程、路径和 IO；C：渲染编排；D：annotation/GT；
  E：camera 数学；F：一次性实验、审计或场景 recipe。
- `tools/` 中的正式入口应最终只保留 A；B-D 放入 `src/densekp_retrain/`。
- E 只有在单测和落盘数值回归覆盖下才能原样移动。F 保持明确脚本，不强行 class 化。

## 正式 entry points

| entry point | imports / imported by | subprocess target | config / asset dependency | shared or duplicated logic | class |
|---|---|---|---|---|---|
| `tools/generate/load_real_smplx_motion.py` | NumPy；无代码反向 import | 无 | motion NPZ | frame selection、NPZ serialization | A+D |
| `tools/generate/generate_real_motion_geometry.py` | smplx、torch、NumPy；无反向 import | 无 | SMPL-X model、regressor、motion NPZ | regressor load、fingerprint、geometry manifest | A+D |
| `tools/generate/generate_p0_toy_motion.py` | smplx、torch、NumPy；无反向 import | 无 | SMPL-X model、regressor | 与 real generator 重复 geometry output | A+D |
| `tools/generate/prepare_p0_dressed_geometry.py` | NumPy、SciPy；无反向 import | 无 | clothing/body NPZ | fingerprint、manifest output | A+D |
| `tools/render/run_eye_background_fix.py` | runtime helper | Blender → `render_p0_blender.py` | frozen baseline、texture、eye mapping | Blender invocation、render log | A+B+C |
| `tools/render/run_temporal_case.py` | runtime helper | Blender → `render_p0_blender.py` | appearance manifest、geometry、HDRI | 同上 | A+B+C |
| `tools/render/run_camera_stress.py` | stdlib；无反向 import | Python → `run_eye_background_fix.py` | stress JSON、placements | Python invocation、completion checks | A+B+C |
| `tools/render/render_p0_blender.py` | bpy、`eye_background_render` | 无 | geometry/textures/HDRI | Blender render implementation；projection math 内嵌 | A+C+E |
| `tools/render/finalize_p0_outputs.py` | OpenEXR、NumPy、PIL | 无 | raw EXR、manifest、annotations | EXR channel IO、visibility/output writing | A+B+D |
| `tools/render/finalize_temporal_case.py` | NumPy、PIL | ffmpeg | finalized render directory | video invocation、report writing | A+B |
| `tools/render/solve_camera_stress_placement.py` | SciPy、NumPy | 无 | stress JSON、baseline geometry | perspective projection/solver | A+E |
| `tools/render/finalize_camera_stress.py` | OpenEXR、NumPy、PIL | 无 | all stress renders | EXR IO、copy/sheet/annotation assembly | A+B+D |
| `tools/fisheye/run_fisheye_camera_p0.py` | omni-radtan、OpenCV、OpenEXR、PIL | Blender → ERP renderer | geometry、appearance manifest | pose、EXR IO、warp/output writing；`warp_case` 被三个 drivers import | A+B+C+D+E |
| `tools/fisheye/run_fisheye_scene_p0.py` | `warp_case`、scene loader | Blender → scene ERP renderer；Python unittest | scene JSON/.blend、geometry、baseline | fingerprint、render invocation、test invocation | A+B+C+D |
| `tools/fisheye/run_fisheye_seam_sweep.py` | `warp_case`、pose、omni-radtan | Blender → ERP renderer | geometry、baseline | render invocation、report/contact sheet | A+B+C+D |
| `tools/fisheye/run_human_scale_sweep.py` | `warp_case`、pose、omni-radtan | Blender → ERP renderer；Python unittest | geometry、phase17 report、baseline | fingerprint、render/test invocation | A+B+C+D |
| `tools/fisheye/render_equirectangular_source.py` | bpy、NumPy | 无（在 Blender 内） | geometry/camera/textures/HDRI | ERP Blender implementation | A+C |
| `tools/fisheye/render_equirectangular_scene_source.py` | bpy、scene bridge | 无（在 Blender 内） | 以上 + scene .blend | 与 source renderer 重复 setup | A+C |
| `tools/annotation/generate_bedlam_densekp.py` | OpenCV、NumPy | 无 | intermediate NPZ、images、regressor | operator checks、projection、bbox、NPZ serialization | A+D+E |
| `tools/annotation/build_bedlam_tiny_frame_manifest.py` | csv/regex | 无 | BEDLAM CSV/metadata/images | parsing and manifest writing | A+B+D |
| `tools/annotation/inspect_annotation.py` | NumPy | 无 | annotation NPZ | reusable numeric summaries | A+B+D |
| `tools/annotation/inspect_surface_operator.py` | NumPy | 无 | regressor `.npy` | operator sanity validation | A+D |
| `tools/annotation/visualize_annotation.py` | original `densekp.core.datasets.utils` | 无 | images + annotation NPZ | visualization only; intentionally crosses training boundary read-only | A+D |
| `tools/scene/build_reference_riverfront.py` | bpy | 无（在 Blender 内） | scene assets | concrete scene recipe | A+F |
| `tools/scene/build_cycling_forest.py` | bpy | 无（在 Blender 内） | downloaded forest blend/assets | concrete scene recipe | A+F |
| `tools/scene/prepare_hidden_alley.py` | bpy | 无（在 Blender 内） | downloaded alley blend/assets | concrete scene recipe | A+F |
| `tools/scene/validate_cycling_forest.py` | bpy | 无（在 Blender 内） | built scene | current validator, regression value | F |
| `tools/scene/scene_loader.py` | stdlib；被 fisheye scene driver import | 无 | scene spec JSON | reusable spec parsing/validation | B |
| `tools/scene/blender_scene_bridge.py` | bpy；被 scene ERP renderer import | 无 | `.blend` collection | reusable scene linking/bounds | C |
| `tools/scene/inspect_downloaded_scene.py` | bpy | 无 | downloaded scene | one-off inspection | F |
| `tools/scene/inspect_forest_nodes.py` | bpy | 无 | forest scene | one-off inspection | F |
| `tools/scene/inspect_pine_forest.py` | bpy | 无 | forest scene | one-off inspection | F |

## Critical dependency edges

```text
run_{eye_background_fix,temporal_case}
  -> Blender -> tools/render/render_p0_blender.py

run_fisheye_* -> Blender -> render_equirectangular_{source,scene_source}.py
run_{scene,human_scale}_p0 -> subprocess -> camera unit test
run_{scene,seam,human_scale}_p0 -> import warp_case/pose from run_fisheye_camera_p0.py

render_equirectangular_scene_source.py -> tools/scene/blender_scene_bridge.py
run_fisheye_scene_p0.py -> tools/scene/scene_loader.py

visualize_annotation.py / validate_bedlam_densekp_tiny.py
  -> densekp.core.datasets (read-only compatibility validation)
```

## Extraction order

1. Runtime paths/Blender/baseline helpers: no numeric behavior.
2. Process invocation, render logs, JSONL/EXR IO: non-mathematical shared behavior.
3. Annotation operator validation/serialization where exact bodies can be lifted.
4. Omni-Radtan only after preserving the existing test and adding disk-baseline numeric regression.
5. Keep Blender render bodies, visibility loops, perspective solver, bbox math and scene recipes in
   place unless direct regression coverage exists.

