# Phase 16 — Temporal Continuity + Multi-Scenario Validation

Baseline commit: `1ad425978882ac96fbdaacec6fa15d1e21ec2a1b`.

## Asset audit and selection

The bounded local audit found one matched BEDLAM motion/clothing pair (Aaron 1095),
one additional real SMPL-X sample in the checked-in BEDLAM renderer assets (KIT
`314/bend_left01`), and one local HDRI (`abandoned_church_4k.hdr`). No second
matched clothing simulation or real locomotion SMPL-X file was locally available.

The additional real case therefore uses frames 56–87 of the KIT motion without
clothing. This is the highest body-pose-energy 32-frame window and exercises bend,
turn and self-occlusion. The third case is explicitly a deterministic procedural
fallback with two cycles of alternating hips, knees, ankles and arms; it is validation
data, not a claim of captured locomotion.

Two minimal 1K HDRIs were added from Poly Haven: `sunset_forest` (outdoor, 1.84 MB)
and `blocky_photo_studio` (indoor, 1.59 MB). Both asset pages identify the HDRIs as
CC0. SHA-256 values are `189502addad02ba43449b895de66bcdf115e7150536d9150d9a7ba05e1d15b11`
and `45ecf08f0d6e3f11ccfa70c5b34f478833940a4784d6ccb5842938e90e6e66ac`.

## Case summary

| case_id | subject | motion_id | state | frames | GT fps | preview fps | scene | scene type | body texture | clothing texture | projection max px | mean visible | mean in-frame | render ms/frame | status |
|---|---|---|---|---:|---:|---:|---|---|---|---|---:|---:|---:|---:|---|
| aaron_1095_dressed_church | rp_aaron_posed_009 | 1095 | dressed | 39 | 6 | 24 | abandoned_church | outdoor/church | skin_m_white_01_ALB | texture_01 | 3.47903e-5 | 53.282 | 137.0 | 828.71 | PASS |
| kit_bend_real_sunset | KIT 314 male | bend_left01 | undressed | 32 | 30 | 30 | sunset_forest | warm outdoor | skin_m_white_01_ALB | N/A | 3.11735e-5 | 63.031 | 138.0 | 739.14 | PASS |
| procedural_cyclic_indoor | SMPL-X neutral | procedural_cyclic_gait_v1 | undressed, procedural fallback | 32 | 16 | 24 | blocky_photo_studio | indoor studio | skin_m_white_01_ALB | N/A | 3.20257e-5 | 76.125 | 138.0 | 706.79 | PASS |

Preview videos contain only repeated source frames. The Aaron preview has 156 frames
at 24 fps from 39 GT frames at 6 fps. **Preview only, GT temporal sampling remains
6 fps.** The same rule applies to the procedural 16-to-24 fps preview.

## Temporal findings

Aaron 1095 is temporally stable. Clothing indices 0–38 map exactly to motion-local
indices 0,5,...,190 and source indices 101,106,...,291. There are no missing,
repeated or reordered GT samples, and no body-clothing slip. All numeric geometry,
projection and visibility arrays are finite. In-frame DenseKP count is exactly 137
for every frame; visible count is 50–58 with maximum adjacent change 4. Background
mean absolute frame difference is 0.000136 (maximum 0.000304), so no background or
shadow flicker was detected. Projected trajectories are coherent on manual contact
sheet review; the robust 99th-percentile displacement test does not flag overlay
jitter.

The original 16 sampled body vertices, clothing vertices and DenseKP 3D arrays are
bit-identical to the corresponding samples in the new 39-frame artifact. Frame 0
RGB, depth and mask are also pixel/value-identical to the frozen appearance output.
See `outputs/temporal_case_aaron_1095/baseline_regression.json`.

## Implementation notes

The DenseKP regressor, camera model, projection, depth, mask and visibility rules are
unchanged. The renderer only accepts an arbitrary temporal length now. The undressed
shadow-catcher branch preserves the transparent person pass separately from the
ground depth/shadow pass; this fixed a full-frame-mask failure found during validation.

No blocker remains for this three-case validation set. The only scope limitation is
that case C is procedural because no second local real locomotion motion was found.
