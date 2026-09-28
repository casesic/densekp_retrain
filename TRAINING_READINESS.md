# DenseKP training readiness

> **Post-audit implementation update (2026-09-28):** The archaeology below is a frozen
> report of baseline `8553ccd`. A subsequently authorized correctness/refactor batch
> is now present in the working tree. Current contracts, fixed/guarded issue IDs,
> configuration migration, verification evidence and remaining training gates are in
> [the implementation report](docs/refactor/DENSEKP_CORRECTNESS_20260928.md).
> Statements below that production code was unchanged refer to the archaeology phase;
> baseline bug descriptions and unchecked checklists are not the current fix status.

Baseline: `8553ccd2d3117755e595ae97c8522c303bd05ceb`; audited 2026-09-28. This document is a handoff to the separate training conversation. It records verification/fix gates; it does not authorize or claim that this audit performed training or refactoring.

**Assessment: not yet demonstrated ready for a full DenseKP retraining run.** Core local geometry, the small regression suite, and the subsequent same-bbox production preprocessing/restoration comparison pass. Reproducible configuration/data provenance, a formal deployment-targeted corpus and independent evaluation, the original-BEDLAM RGB correspondence chain if used, and actual model/backend inference parity remain unresolved. Some confirmed bugs are dormant under the default DenseKP augmentation settings and can be excluded by an explicit supported configuration while later fixes are reviewed.

**Follow-up status (2026-09-28):** The user has not selected a formal NPZ. All five available candidate pairs, totaling 33 sample rows, were subsequently checked against their saved producers and through the loader with augmentation disabled. The synthetic body/138-row/projection chain was also numerically reconstructed. This accepts the local data as regression fixtures, not a formal training corpus. Full evidence and annotation hashes are in [DATA_FLOW section 9](DATA_FLOW.md#9-follow-up-candidate-contract-verification).

The original BEDLAM route has a concrete data blocker: its 24-image manifest requires 157 motion paths, none present under the inspected local motion root. The available aaron/1095 motion belongs to the synthetic fixtures and is not a substitute for those people. Thus the next data milestone is to define the formal source/split/supervision policy and obtain matching inputs, before generating or selecting a formal NPZ. Keep the original-image route and the newly rendered route distinct in that decision.

Findings are explained with triggers and falsifiers in [BUG_AUDIT.md](BUG_AUDIT.md). The actual contracts are in [DATA_FLOW.md](DATA_FLOW.md).

## Training direction determined from pano-stitch

The user's target is better multi-person, severely distorted-body and occlusion performance in `pano-stitch`. Its inspected production code supports the following scope: **crop-conditioned 138-point prediction on normal ERP from the actual DS camera/remap, with the correct target person retained under crowding and occlusion**. The model still receives one person's bbox per prediction. Keep the 138 semantic row order and the 256-square output coordinate contract. The internal 192-wide slice remains part of the model. ERP_ROTATE is a later coordinate conversion in the inspected paths, not the training raster target.

This is a recommended data/evaluation specification for the separate training conversation, not a claim that a formal NPZ has been chosen or that the model has improved.

**User-confirmed scope: DenseKP given an existing person box.** Person detection and missed detections are outside this retraining task. Freeze source images, detector/guided boxes, crop transforms and intended person IDs for baseline/candidate comparisons; improve only the 138-point predictions and their coordinate correctness. Missing-box cases are recorded as unavailable upstream inputs and excluded from the conditional DenseKP denominator. Keep every eligible supplied-box case, including failed/non-finite predictions and people later rejected by cross-camera matching. Detection, matching, tracking and fitting components remain fixed; downstream diagnostics are supporting evidence, not separate optimization targets or prerequisites for this task.

| Priority | Proposed data/evaluation requirement | Why the inspected consumer needs it |
|---|---|---|
| Primary image domain | Diverse rendered/captured **normal ERP**, reproducing selected DS calibration, lens masks, resampling, overlap-region boundaries and padding | FULL and TRACK_GUIDED both predict from normal-ERP raster crops |
| Crowding | Multiple identities and body shapes in the same source frame; overlapping people, similar appearances, crossing limbs, varied distance/size; one explicit target ID per bbox | Independent DenseKP crops can contain distractor bodies; model does not jointly identify all people |
| Distortion | Cover latitude, longitude/seam, camera orientation, near-camera large bodies, broad bboxes and strong nonrigid poses; measure these in the actual normal-ERP frame | Generic affine augmentation does not reproduce all camera-induced deformation |
| Occlusion | Preserve full 3D-derived 138 xy plus independent visibility and validity states; recommend amodal supervision of reliably projected, in-domain hidden points, with a visible-only control | Geometric deployment validity does not remove anatomically occluded predictions |
| Bboxes | Mix realistic detector boxes and guided boxes from previous matched observations, including truncation, center/scale errors and black padding; keep GT boxes as a separate diagnostic | Current GT-mask ×1.2 crops do not match either production bbox source; guided eligibility requires a successful preceding fit |
| Generalization | Hold out actors, motions and captures/scenes; preserve a separate real deployment set and easy/ordinary-pose set | All 33 current tiny rows reuse one actor/motion and are insufficient for retraining or a generalization claim |
| Paired/temporal evaluation | True synchronized two-camera scenes and identity-labeled sequences | DS pilot uses independent yaw placements, so its two cameras cannot validate matching/tracking |

**Target fixed by the user:** The primary difficult real scene is **20260320-riding-dataset3**, stored as `20260320_riding_dataset3` in pano-stitch. Use **20260320 / new**, with distinct cam2/cam4 DS intrinsics, for its data generation, remapping and evaluation. The 0311 capture uses **20260311 / old** and different parameters; it may be an additional regression domain, with its own calibration and masks. It is not the target calibration for dataset3. This supersedes the earlier open question about which camera group to target. The existing DS pilot's camera parameters match 20260320 numerically. Production defaults still require an explicit `--camera-group 20260320` (or `--camera 20260320` in the pair/legacy CLI); defaults do not describe this capture. Omni/P0 fixtures can support geometry checks and additional domains.

**Located target assets:** `PANO/data_process/tmp/production_regression_20260320_dataset3_multiperson/` contains a prepared **4325–4575** normal-ERP window: 251 PNGs per camera, no missing frame IDs. `prepare_selected.py` explicitly uses the 20260320 profile, video `.0`→cam2 and `.1`→cam4, and writes `prepared/valid_masks_20260320.npz`. This is the first concrete candidate window for difficult-scene evaluation. Existing review flags nominate 4461/4474 (low identity margin), 4472 (many output persons), 4480 (recovery with no output person), and 4560/4573 (suspected identity switch). These are historical pipeline flags, **not verified errors, real-person counts or ground-truth annotations**. Verify them visually and annotate failure causes before using them to rank models. The packaged 651–659 clip is labeled a single-person example and is useful as a control, not sufficient difficult-scene coverage.

**Mask provenance remains separate from intrinsics:** Local `assets_external/real_erp_valid_masks.npz` exactly matches the older `densekp_phaseA_erp_input` mask file, but differs from this target window's masks by 7,834 cam2 and 13,074 cam4 pixels. Lens masks are image-derived, so this difference alone does not demonstrate a calibration bug. Use the selected window's matching masks and record their hashes. DATA_FLOW section 11 records paths, identities and limitations. Reserve validation/test windows before using dataset3 for tuning; adjacent frames are not independent held-out scenes.

**Occlusion decision:** Favor explicit supervision for synthetic hidden points whose target geometry is reliable and whose projection belongs to the intended camera/input domain. Record self-occlusion, mutual occlusion, lens exclusion and crop truncation separately. Freeze how off-crop labels are treated, including the internal 192-wide visible region; do not silently reuse `visible` as the only mask or label all projections valid. The present loader/loss consumes one channel as the supervision weight, so this policy must be deliberately encoded by a future reviewed adapter. Never derive it from DenseKP `keypoint_value` magnitude. No policy change is implemented during this audit.

### Next concrete milestone before a full NPZ or long run

1. **Freeze a representative deployment manifest for 20260320 dataset3:** start with the located 4325–4575 window and its own 20260320 masks; record resolved package/checkpoint/backend, ERP dimensions and raw/prepared-frame provenance. Verify the flagged cases, add single-person controls and other independent clips covering crowded overlap, severe distortion, occlusion and seam/lens boundaries. Existing production outputs are not ground truth. Keep any 0311 regression samples under their separate calibration.
2. **Establish a frozen supplied-box baseline with the training owner:** record existing detector/guided bbox, target person identity and restored raw 138 xy before matching/fitting filters. Reuse exactly those boxes for every checkpoint. Use manually verified real annotations where available and synthetic geometry where dense ground truth is needed. Keep all DenseKP failures on eligible supplied boxes in the denominator; record absent-box cases separately, without making detection improvement a gate.
3. **Produce a versioned annotation contract:** stable IDs; xy frame; operator/model hashes; distinct geometric validity, surface visibility and supervision mask; bbox source; crop/roll provenance; source actor/motion/camera IDs; independent split rules. Start from the verified DS branch for matching captures. Original BEDLAM RGB is optional for this route; if used, its missing motion inputs and reconstruction adapter remain blockers.
4. **Validate a small diverse pilot against that baseline and contract**, then choose/build the formal NPZ. Current tiny sets are regression fixtures only. Keep implementation and actual experiments in the training conversation, with the audit findings as gates.

### Required evaluation layers

- **Landmark quality given a known target box:** fixed-ID xy error in normalized crop pixels, periodic original-ERP pixels and spherical angle; report visible, self-occluded, mutually occluded and truncated subsets, and person-scale/distortion bins. Report fraction of finite outputs and failure counts.
- **Within-crop person identity:** check whether each semantic point belongs to the intended person, especially with overlapping bodies and crossed limbs. A neighboring person's plausible pose is a DenseKP error when a target box/identity is supplied. Report errors on all eligible boxes, including unmatched detections; final matched-only JSON cannot supply a complete evaluation population.
- **Fixed input-box conditions:** retain realistic existing detector/guided crops and label center/scale/truncation conditions. Optional controlled bbox perturbations evaluate DenseKP robustness to its input; they do not tune the detector or require recovering people with no box. Wrong-target or unassignable input boxes need explicit annotation and separate accounting.
- **Optional downstream checks with unchanged components:** cross-camera matching, tracking, fitting and final alignment may indicate consequences, but raw DenseKP localization and target-person correctness are the primary criteria. Fitted coordinates are not landmark ground truth, and successful matching alone is not proof of accurate landmarks.
- **Deployment compatibility:** same selected crops through checkpoint/ONNX/TRT, full restoration and final coordinate conversions. Freeze tolerances before comparing candidates; this audit has not run those networks.

Do not choose sampling ratios, error thresholds or a winning occlusion loss from the tiny fixtures alone. Establish them from the frozen baseline and user-visible downstream requirements. The recommended direction is now concrete; the empirical winner remains the responsibility of the training experiments.

### Result of the first supplied-box real-scene review

Completed structural checks on 251 frames / 775 retained pair observations and visually inspected **53 pair observations / 106 single-camera panels** across 10 frames, including adjacent-frame checks. Source, counts and case identifiers are in DATA_FLOW section 12 and BUG_AUDIT's real-scene review. This is qualitative evidence for data design, not a measured model ranking. The current retained coordinates are all finite; this does not prove correct localization.

Priorities supported by the inspected images:

1. **Crowded target-conditioned crops:** same-image overlapping people, background targets partially covered by foreground people, crossed arms and similar clothing. Preserve the intended target's full 138 semantic identities in synthetic labels. Do not define identity from historic cross-camera pair/track IDs, which visibly change or mismatch in some cases.
2. **Realistic occluders and partial observations:** umbrellas, other people, bicycles/motorbikes, helmet/visor and self-occluding seated/riding poses. Separate hidden-but-projectable landmarks from lens-invalid targets. The visible-only tiny mask policy still needs the explicit amodal-versus-visible-only experiment described above.
3. **Source resolution and boundary coverage:** include small upper-body regions, severe partial bodies, 0320 lens-edge deformation/black masks and real square-crop padding. Train/evaluate at the actual 256-square/192-retained-width contract, with recorded original bbox size. Treat bbox perturbation only as a controlled input condition for DenseKP; detection remains out of scope.
4. **Controls and trustworthy labels:** retain grossly coherent existing cases alongside difficult cases, and obtain independently checked visible-point labels or reliable synthetic geometry before computing errors. Image motion, changed guided boxes, lower pair-valid counts and fitted-point jumps are not model-error ground truth.

**Next data milestone is now specific:** turn the reviewed cases into a frozen per-camera, per-existing-box evaluation manifest containing target identity, source image/ROI, actual bbox/affine, all 138 raw outputs before matching, and label availability by visibility/validity state. This requires complete prediction capture by the training/evaluation owner, not detector retraining. The inspected matched-only artifact omits 2,393 of the 3,263 reported FULL/RECOVERY camera-result entries, so it cannot serve as the complete baseline denominator (R12). Guided records additionally require the accepted candidate's actual source crop (R13). Use the saved FULL/RECOVERY cases as initial reproducible inspection fixtures while closing those recording gaps. No formal NPZ was selected or generated, and no network or production source was changed.

## Evidence already obtained

| Check | Audit result | Limit |
|---|---|---|
| Baseline identity | `8553ccd2d3117755e595ae97c8522c303bd05ceb` | Local ignored assets are additional state |
| Syntax | 119 tracked Python files parsed | Does not validate optional imports/runtime behavior |
| Existing tests | 18 passed, zero skipped | Small tests; no full model/training; optional local baselines available here |
| Loader / restoration | P0, DS, near ERP representative validation samples; inverse xy error <`2.35e-5` original pixels | No flip, no full external BEDLAM corpus |
| Torch vs NumPy affine | Exact agreement on those samples | The Torch helper expects batch affine dtype consistent with its float64 homogeneous coordinates |
| Surface operator | Local `[138,10475] float32`, 204 nonzeros, row sums ≈1; hash in DATA_FLOW | Not tracked; historical semantic/checkpoint parity not established |
| Tiny NPZ structure | Inspected local P0/Omni/DS/boundary/near ERP train/val files finite, float32, correct K=138 | Cannot generalize to imported data |
| Clean config composition | Default fails; explicit committed tiny data override composes | Data files/weights still require provisioning |
| Tiny batching | DS 2/2 samples at default B=64 gives zero train and val batches | Smoke scripts explicitly use B=1 |
| External operator assets | Exact equality to pano-stitch NumPy `138×10475` and expanded C++ `138×204` assets | Historical checkpoint training provenance still unverified |
| External crop/decode | All 33 same-bbox samples: normalized tensor error ≤`2.39e-7`, inverse xy error ≤`1.53e-5` pixels | No network/detector inference or bbox-distribution test |
| External camera | DS 20260320 calibration/domain agreement; both reduced-resolution LUTs exactly equal; user target fixed to 20260320 dataset3 | Record matching capture masks and artifact provenance; shared predicate does not prove physical domain |

Test command used (no production changes, no training):

```bash
PYTHONDONTWRITEBYTECODE=1 /home/yy/conda_envs/camerahmr/bin/python -B -m unittest discover -s tests -v
```

The shell-default Python was a different environment without NumPy/Torch/OpenCV. The tested environment reported Torch `2.0.0+cu118` and NumPy `1.26.4`, which already differs from the inherited requirement list's NumPy pin. Record the actual environment rather than treating `requirements.upstream.txt` as a lock.

## Blockers before a full run

### 1. Fix the data identity and third-field contract (R01, R02, R06)

- [ ] Identify the exact full training and validation annotation producers and inputs. Supply the missing raw BEDLAM manifest→camera-consistent geometry step if original BEDLAM RGB is used.
- [ ] Record source actor/motion/frame IDs, camera row, frame-rate mapping, units, root/pelvis conventions, gender/body-model variant, hand pose conventions and topology hashes. Check original RGB overlays independently of the same projection implementation used to generate labels.
- [ ] Establish correspondence of every output row `0..137` to the intended DenseKP semantic point/operator and checkpoint. Keep BEDLAM joints/downsample rows separate. Include operator checksum and vertex support checks; equal shape is insufficient.
- [ ] Specify annotation channel 2 for each dataset. The current consumer always clamps it as a weight; accept only explicitly compatible semantics or arrange a reviewed conversion/loss change. **Do not infer that external `keypoint_value` is confidence or a weight from its numeric magnitude.**
- [ ] Reject or sanitize non-finite xy before arithmetic, including masked points (C06). Validate `[N,138,3]` floating targets, `[N,2]` center, supported scale shape, positive finite sizes, equal sample counts, readable images and any auxiliary-joint order over the entire corpus.

**Acceptance evidence:** A versioned schema/provenance record, coverage counts, no invalid numeric targets, verified selected semantic rows and independent overlays across views/body regions. No unexplained third-field conversion.

### 2. Make the chosen experiment reproducible (C03, C04, R05, R07, R08)

- [ ] Compose the intended config using only committed configuration plus an explicit documented external-data manifest. Avoid dependence on ignored `data/train.yaml` or other local-only data configs.
- [ ] Verify every resolved registry path. The JSON example is not consumed. Distinguish repository `densekp/data`, CWD `data`, and generated `datasets/` roots.
- [ ] Record exact versions for Torch/Lightning/Hydra/OpenCV/Albumentations/SMPL-X and renderer tools, selected checkpoint/operator/model hashes, git revision and resolved run settings.
- [ ] Make actual Trainer construction match the intended accelerator/device count/precision/validation limits/sanity behavior. At this baseline, YAML CPU/MPS/DDP overrides do not implement those modes.
- [ ] Confirm the network remains the intended fixed ViT + 138×3 head, with 256-square label space and central-width input slice. Verify effective loss branch and manual optimization behavior; `LOSS_TYPE: l1` does not change the implemented squared-distance objective.
- [ ] Implement or demonstrate complete seed/worker/resume reproducibility. Distinguish fresh initialization (`init_checkpoint`) from full-state resume (`ckpt_path`); avoid ambiguous simultaneous settings.

**Acceptance evidence:** Reproducible composition/provisioning from the baseline, actual-runtime settings compared to saved config, repeated short-run/resume results with known data order. No full training is required to expose the current composition/config mismatches.

### 3. Establish a meaningful validation set and metric (C05, R04)

- [ ] Split by intended independence units (such as identity, motion, scene/capture), not only adjacent frames or reused synthetic camera/person placements. The tiny adapters explicitly share source motion/scene and are smoke datasets.
- [ ] Ensure nonempty train/validation loaders and account for every intended validation row. Default B=64 is incompatible with the local tiny splits. Handle zero/partial batches and aggregation together.
- [ ] Decide whether checkpoint ranking is loader-0-only; otherwise report/select across the required domains. Record per-dataset counts and the meaning of `val_loss`.
- [ ] Exclude/report zero-supervision samples and fail/report non-finite predictions rather than silently improving the scored subset. Decide person-average versus point-average weighting explicitly.
- [ ] Report normalized crop error or crop-pixel error with correct units; add original-image/detection-aware evaluation if that is the target use. Oracle-mask/GT-hull crops do not establish deployment accuracy.

**Acceptance evidence:** A split manifest, loader/sample counts, finite prediction counts, explicit metric formula/units, and successful checkpoint monitoring over the intended validation scope.

### 4. Validate the exact preprocessing and supervision policy (C01, C02, C08, C10, R03, H01)

- [ ] Keep horizontal flips disabled until both geometric reflection and the verified 138-point left/right permutation are implemented and checked. The 44-joint permutation cannot substitute for this.
- [ ] Specify whether hidden/off-crop landmarks should be regressed. For visible-input supervision, reconcile generated masks with random crop changes and the 192-wide region visible to ViT. P0 all-one targets are a different policy from the multiperson binary masks.
- [ ] Compare image/label overlays after random scale, translation, rotation and extreme cropping for the actual data. Check H/W ordering, border fill and non-square original images. Verify auxiliary-joint semantics if extreme cropping is enabled.
- [ ] Establish portrait/closeup label raster orientation before enabling those datasets; the loader rotates images only.
- [ ] Ensure intended augmentation disable controls actually disable it. `use_augmentation=False` and `TRANS_AUG_RATE=0` alone do not do so at this baseline. Keep the optional inverted antialias path disabled until reviewed if it is not required.

**Acceptance evidence:** Reversible affine/corner tests plus image overlays and semantic checks across the actual enabled augmentation distribution. A no-augmentation smoke batch is necessary but insufficient for this gate.

### 5. Verify enabled synthetic camera branches (C07, R09, H02–H04, H07)

- [ ] For Omni-derived training, enforce camera model-domain validity in annotation output and preserve the generating intrinsics through fisheye→ERP conversion.
- [ ] For DS-derived training, resolve the rejected-but-invertible boundary-ray question against the intended camera domain and deployment masks. Existing mask coverage passing is evidence for tested coverage, not a general domain proof.
- [ ] Check positive depth/front-facing assumptions for original perspective BEDLAM generation. Preserve differences between camera-Z and ray-range depth.
- [ ] Make P0 state codes versus boolean visibility explicit at each artifact boundary. Audit depth tolerance and instance matching against independent occlusion examples, including clothing/silhouettes/multiple people.
- [ ] Verify periodic ERP longitude, per-person roll provenance, camera-specific yaw placement and inverse coordinate ordering. Do not evaluate independent DS yaw placements as synchronized stereo.
- [ ] Capture color-management settings and compare corresponding preview/fisheye RGB before interpreting a paired result as a projection-only difference.

**Acceptance evidence:** Camera/visibility overlays and numeric domain/round-trip checks for the chosen camera branch. These gates are conditional: a perspective-only run need not implement unused fisheye branches, but its scope must be explicit.

### 6. Demonstrate model/runtime and inference compatibility (R08, H05, H06)

- [ ] Provision both backbone and DenseKP checkpoints where relevant; this baseline loads the backbone even when restoring a complete checkpoint. Verify schema/state compatibility and checksums.
- [ ] Run a bounded real-model forward/backward and validation check in the actual intended precision/device environment. Require `[B,138,3]` finite output, finite loss/gradients and a verified parameter update for the intended trainable parameters. Check uncertainty arithmetic at plausible extreme outputs and mask counts.
- [ ] Confirm the separate training workflow can save/reload/resume with the intended checkpoint monitor and reproducible data selection before committing a long budget.
- [ ] Compare the actual downstream application's crop/input normalization and prediction restoration to this repository. Apply inverse affine to `(xy+.5)*256`, then inverse raster orientation/ERP roll as applicable. Do not add a 32-pixel ViT-slice offset or spatially transform a third value.
- [x] Trace the inspected external third-value consumer: preserved as raw output, omitted from v2 pair matching/fitting weights. Revalidate any different deployed snapshot; local log-sigma training still does not establish calibrated confidence.

**Acceptance evidence:** Real-model smoke results and an end-to-end preprocessing/restoration parity report using the target inference application. For research without a deployment target, record that exclusion and defer deployment readiness rather than claiming it.

## Non-blocking improvements after correctness gates

- [ ] Profile full-resolution image decoding, repeated per-person ERP images, full-image augmentation, display tensors in batches, worker/prefetch memory and transfers (P01). Optimize only measured bottlenecks.
- [ ] Profile/chunk Omni LUT generation and trajectory loading; reuse camera LUTs where parameters match (P02).
- [ ] Measure redundant initialization/checkpoint I/O and GPU activation memory before changing loading or backbone memory policy (P03).
- [ ] Improve training metric persistence (`train/loss` currently sets `logger=False`) and per-domain diagnostics.
- [ ] In a later authorized refactor, consolidate transform contracts, isolate experiment-specific asset assumptions, remove only proven-unused code, and align package/dependency boundaries (M02–M03).
- [ ] Reconcile historical docs with current semantics and mark version-specific claims; retain experiment provenance (M01).

## Handoff and final consistency pass

The archaeology deliverables are documentation only. Confirmed issues have stated triggers and are not blanket claims that current outputs are corrupt. Tests support the checked baseline geometry but do not close the raw BEDLAM mapping, full-training stability or external deployment questions. Before a full run, record each applicable blocker as passed, fixed-and-reverified, or explicitly excluded by the run's scope; a successful tiny overfit does not replace those decisions.

The second pass checked these gates against actual call sites and the other three documents, including local-versus-committed assets, active versus inherited losses, visibility codes, fixed network dimensions, and the absence of a restoration implementation in this training repository. The external follow-up adds the actual restoration implementation and numeric comparison, while retaining model inference as an open gate. Normal-ERP input, rotated output, both bbox producers, camera defaults and R10/R11 were rechecked against source. Production source was left unchanged.
