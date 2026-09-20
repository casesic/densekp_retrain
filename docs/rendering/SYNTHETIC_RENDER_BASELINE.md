# synthetic-render-baseline-before-cycling

Baseline source snapshot, 2026-09-06. This directory had no valid Git history;
the baseline is an initial source commit, not a continuation of an older commit.
External dependencies and dataset assets remain local and ignored.

Validated output: `outputs/baselines/p0_dressed_eye_background_fixed` (16 frames).
`texture_regression.json`: body/clothing geometry, camera, DenseKP 2D/camera XYZ,
depth, person masks and visibility are identical to `p0_dressed_hdri_eye_fixed`.
Projection maximum error: 3.3463941410797083e-5 px. Eye and background visual
acceptance: `outputs/eye_fix_v2/acceptance.json`.

Eye shader uses native rest-space procedural iris/pupil and local correction of
duplicated eyes in the supplied skin albedo. Ground mode is `shadow_catcher`,
implemented as linear differential shadow compositing. Same abandoned church
HDRI, strength 1, rotation 0. Source texture files and geometry are unchanged.

Entry points: `tools/experiments/prepare_procedural_eyes.py`, `tools/render/run_eye_background_fix.py`,
`tools/render/finalize_p0_outputs.py`, `tools/validate/validate_p0_textured_regression.py`.
The reproducibility notes under `outputs/eye_fix_v2/README.md` are local artifacts.
No RGB, motion/model arrays, texture archives, HDRI, or credentials belong in Git.
