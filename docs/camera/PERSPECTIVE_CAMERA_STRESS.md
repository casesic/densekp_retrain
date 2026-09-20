# Phase 15A — Perspective Camera Stress P0

This phase builds 128 deterministic perspective samples from the existing 16
motion frames and eight camera configurations. Appearance, local SMPL-X and
clothing vertices, motion, textures, HDRI, lighting, projection type, lens,
sensor, resolution, and principal point remain fixed. Only the camera world
translation changes. No image-space crop is used.

## Configuration and placement

The source definition is
`configs/perspective_camera_stress.json`. The original appearance baseline
framing is about 76% high and its feet extend roughly two pixels below the
image. It therefore cannot serve simultaneously as an untruncated C0 and make
`near_70` larger. C0 is defined as an automatically centered 60% full-body
reference. This decision preserves the requested ordering and full-body C0;
the appearance and all local geometry remain the same.

| ID | Name | Geometry target |
|---|---|---|
| C0 | baseline | body bbox height 60%, centered |
| C1 | near_70 | height 70%, centered |
| C2 | near_90 | height 90%, centered |
| C3 | near_110 | height 110%, centered; natural top/bottom truncation |
| C4 | edge_left | height 60%; left margin 3.5% of image width |
| C5 | edge_right | height 60%; right margin 3.5% |
| C6 | trunc_left_25 | height 60%; 25% of full bbox width left of image |
| C7 | trunc_bottom_25 | height 60%; 25% of full bbox height below image |

`tools/render/solve_camera_stress_placement.py` solves three camera-local translation
components per frame against the full 10,475-vertex SMPL-X body projection.
It uses neither clothing, visible pixels, nor masks to define the bbox. Clothing
is included only in the near-plane safety constraint. Least-squares residuals
are checked below 1e-7 normalized image units. Solver outputs are stored in
`outputs/experiments/perspective_camera_stress/configs/`.

The camera remains a 512×512 pinhole camera with 50 mm lens, 36 mm horizontal
sensor, centered principal point, and 0.05 m near plane. All body and clothing
depths exceed the near plane by more than the required 0.10 m margin. The
smallest body depth is approximately 1.64 m in near_110, so its truncation is
caused by image framing rather than near-plane clipping.

## Bounding boxes and truncation

`full_body_bbox=[xmin,ymin,xmax,ymax]` is the axis-aligned bbox of every
analytically projected SMPL-X body vertex, before clipping. Coordinates may be
outside the image. `in_frame_body_bbox` intersects that bbox with `[0,W]×[0,H]`.

Visible fractions divide the clipped bbox width/height by the full bbox
width/height. Each side's truncation fraction is the amount beyond that image
edge divided by the full bbox extent on that axis. These are geometric framing
metrics, not percentages of visible pixels or mesh surface area.

Every frame meets its target: near_70/90/110 mean heights are exactly
0.70/0.90/1.10 within numerical precision; edge margins are 3.5%; both requested
25% truncations are 0.25. C0 and near_70/90 are fully in frame. near_110 has
both top and bottom truncation in all frames.

## DenseKP semantics

Each config saves arrays for all 16×138 points under `annotations/<config>/`:

- `densekp_3d`: unchanged body-local `R @ smplx_body_vertices`.
- `densekp_2d`: un-clipped analytic pinhole coordinates; values outside the
  image remain outside.
- `densekp_depth`: positive camera-space Z depth.
- `densekp_valid`: finite 3D/projection and in front of camera. It does not
  encode image bounds or visibility.
- `densekp_in_frame`: valid, camera-front, and `0 <= u < W`, `0 <= v < H`.
- `densekp_visible`: binary, in frame, and consistent with rendered scene depth.
- `densekp_visibility_state`: diagnostic categorical state retained from the
  existing visibility evaluator.

For out-of-frame points, `valid=1`, `in_frame=0`, `visible=0`; their 2D values
are retained. Visibility includes body self-occlusion and clothing occlusion.
No loss, training policy, point definition, or adapter is changed.

Overlays draw green for visible/in-frame and red for occluded/in-frame. They do
not draw out-of-frame points at an artificial edge. The text panel gives counts.

## Validation

The renderer saves both direct analytic coordinates and Blender's independent
`world_to_camera_view` result. Across all eight configs, the worst discrepancy
is 4.809970366226838e-5 px, well below the 0.5 px requirement.

Rendered Z was additionally compared with independent Blender scene ray casts
at stable interior pixel neighborhoods. Across 749 comparisons, mean error is
0.0001133 m, p95 is 0.0004781 m, and maximum is 0.0010424 m. This check is
separate from DenseKP projection and confirms that placement is reflected in
the rasterized depth scene. All depth arrays are 512×512 finite float arrays
with zero background; all person masks are binary 512×512 images.

Renderer readback confirms exact local vertex arrays for each frame. The common
body BLAKE2b-128 fingerprint is `00dd073faa497d91ab55de498e6af6e7`;
clothing is `adbde4fb78df12de51d5b93fb7b948b5`. Object matrices remain the same;
camera matrices record the solved relative placements. Intrinsics are identical
for all 128 samples.

The machine-readable acceptance report is
`outputs/experiments/perspective_camera_stress/stress_statistics.json`. Sample manifests
are in `manifests/`; the combined manifest has 128 lines. RGB, depth, masks,
overlays, and annotations use per-config directories. The two contact sheets
show configs by column and configs by row respectively.

## Reproduction

Run the placement solver, then `tools/render/run_camera_stress.py`, finalize each raw
render with the existing `tools/render/finalize_p0_outputs.py`, and run
`tools/finalize_camera_stress.py`. Rendering is headless Blender. This phase
does not implement fisheye; Phase 15B is the next isolated projection task.
