# Phase 4 deferred cleanup

## ERP Blender renderers

`render_equirectangular_source.py` and
`render_equirectangular_scene_source.py` duplicate Cycles/pass/compositor setup,
HDRI, body and clothing construction, panoramic cameras, EXR renaming, and source
manifests. They were deliberately not merged in Phase 4.

The environment does not provide a frozen Blender comparison covering both
renderers with identical geometry/cameras and byte- or numeric-level RGB, depth,
and IndexOB checks. Static similarity is insufficient for changing compositor
and render-pass code. This item is `DEFERRED_UNTIL_BLENDER_E2E_REGRESSION_EXISTS`.

## Fisheye warp implementation

`warp_case` remains in `run_fisheye_camera_p0.py`. Three CLI tools still import
it, which is the one known tool-to-tool implementation dependency. Moving it now
would combine a module relocation with untested ERP remapping, interpolation,
depth, IndexOB mask, visibility, and output-writing behavior. Existing numeric
regression covers the Omni-Radtan DenseKP projection but not that complete path.

This item is `DEFERRED_UNTIL_WARP_REGRESSION_EXISTS`. The small, fully
deterministic `pose` implementation was moved independently to
`src/densekp_retrain/camera/extrinsics.py`.

## Render job Namespace API

`rendering/jobs.py` continues to accept an `argparse.Namespace`. Each job needs a
coherent group of CLI values and immediately translates them into one subprocess
command. Replacing this with roughly a dozen keyword parameters would lengthen
both caller and implementation without adding reuse or removing mutable state.
No config/factory hierarchy was introduced.
