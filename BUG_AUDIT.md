# DenseKP technical audit

> **Post-audit implementation update (2026-09-28):** The archaeology below is a frozen
> report of baseline `8553ccd`. A subsequently authorized correctness/refactor batch
> is now present in the working tree. Current contracts, fixed/guarded issue IDs,
> configuration migration, verification evidence and remaining training gates are in
> [the implementation report](docs/refactor/DENSEKP_CORRECTNESS_20260928.md).
> Statements below that production code was unchanged refer to the archaeology phase;
> baseline bug descriptions and unchecked checklists are not the current fix status.

Baseline: `8553ccd2d3117755e595ae97c8522c303bd05ceb`, 2026-09-28. No production source edits. “Confirmed” means a source-level contradiction or reproduced failure under the stated trigger, **not** proof that an existing training run encountered it. Risks and hypotheses retain their uncertainty. Architecture and representation details are in [ARCHITECTURE.md](ARCHITECTURE.md) and [DATA_FLOW.md](DATA_FLOW.md).

Follow-up evidence: [DATA_FLOW section 9](DATA_FLOW.md#9-follow-up-candidate-contract-verification) records all 33 available candidate rows and a CPU SMPL-X reconstruction, performed after this initial audit. It narrows R02 for the local synthetic artifacts (row-order consistency demonstrated), concretizes R03 (29 selected observations outside the retained ViT width), and found no C06 non-finite targets or C07 invalid-domain rays in the inspected candidate chains. The bugs' conditional triggers remain. The original-image manifest still lacks 157 matching motion files in the inspected local motion root; historical checkpoint correspondence and formal corpus selection remain unresolved.

## Evidence collected

- Parsed all 119 tracked Python files with `ast.parse`; no syntax errors.
- Ran the existing test suite in the local CameraHMR environment: **18 passed, zero skipped**. This included optional local frozen perspective/fisheye outputs and DS mask checks. These assets are not tracked.
- Loaded P0, DS and near-ERP validation samples: `[3,256,256] float32` images, `[138,3] float32` labels, float64 affine matrices. NumPy and Torch affine results agreed exactly in these samples. Inverse restoration maximum absolute errors were approximately `1.38e-5`, `1.62e-5`, `2.34e-5` original pixels respectively. A rotated rectangular affine round trip was within `1.43e-14` pixels.
- Probed flips, masked NaNs, empty validation aggregation, worker count zero, camera-domain disagreement, and a baseline-only config tree in temporary storage. Results below distinguish these reproductions from static inspection.
- Inspected the local surface matrix and tiny NPZ schemas. No full-model training/inference, GPU experiment, renderer execution or download was performed. An attempted CPU float16 loss probe stopped because this installed Torch does not implement CPU Half clamp; it is **not** evidence of CUDA mixed-precision failure.
- Subsequent authorized read-only inspection of `pano-stitch` narrowed R02/H02/H05: all 138 operator rows match its NumPy and C++ assets exactly, the `20260320` DS LUTs match, and the actual ORT preprocessing/restoration helpers agree numerically with all 33 local samples. See DATA_FLOW section 10. Model/backend inference was not executed. The external checkout has pre-existing changes; the audit distinguishes its working tree from clean HEAD.

## Confirmed correctness bugs

### C01 — Horizontal flip changes the raster without changing the labels

**Chain/location:** `DatasetKeypoints.__getitem__` → `datasets/utils.py::get_example_projverts` (line 520) → `generate_image_patch_cv2` (354), then label updates at 616–621.

**Input/output:** Original xy labels and image enter; `do_flip=True` reverses image columns and reflects crop center. Returned `trans` maps the flipped raster, but original `proj_verts` and auxiliary joints are transformed directly. `flip_kp_permutation` is unused; no DenseKP 138-point permutation exists in this call.

**Proof/consequence:** A 100×100 image with a bright point at `(20,50)`, center `(50,50)`, identity-sized crop and forced flip returned label pixel approximately `(21,50)` while the image point was `(80,50)`. Training would supervise unrelated locations and identities. **Dormant under the Hydra DenseKP default**, which disables flipping.

**Falsifier/boundary:** An executed pre-reflection and verified 138-index permutation before this transform would invalidate the diagnosis; none is present. A dataset that never enables flips avoids the bug but does not correct it.

### C02 — `use_augmentation=False` is ignored

**Chain/location:** `dataset_keypoints.py::DatasetKeypoints.__init__` (17) accepts the flag but does not retain it; `__getitem__` passes `self.is_train` as `do_augment`.

**Input/output:** An explicitly non-augmented training dataset still produces stochastic crops/colors if its augmentation config enables them. `validate_bedlam_densekp_tiny.py` passes this flag but separately zeroes augmentation, hiding the problem. The NPZ evaluator correctly works around it by setting `dataset.is_train=False` after selecting the split.

**Consequence:** Callers relying on the argument cannot reproduce or compare preprocessing reliably.

**Falsifier:** Use of the argument in the actual augmentation predicate, or a different executed adapter. Zero augmentation settings only remove the trigger.

### C03 — The default Hydra configuration cannot compose from the baseline alone

**Chain/location:** `configs/hydra/train.yaml` defaults to `data: train.yaml`; `.gitignore` ignores all Hydra data files except two named tiny configs; `git ls-files configs/hydra/data` confirms the default file is absent.

**Input/output:** A baseline-only config tree with `experiment=densekp` raises `MissingConfigException: Could not find 'data/train.yaml'` before model/data construction. Reproduced by copying only committed config contents to a temporary directory. The local ignored file masks this failure in the working tree. Explicit `data=ds_multiperson_tiny` composes successfully from committed configs.

**Consequence:** Default training and evaluator commands are not reproducible from a clean checkout.

**Falsifier:** A committed/default-resolvable data config or a documented command explicitly selecting an available config eliminates this trigger. Local ignored files do not falsify the clean-checkout finding.

### C04 — Advertised trainer settings do not control the actual trainer

**Chain/location:** `configs/hydra/trainer/{cpu,mps,gpu,ddp,default,default_hmr}.yaml` → `densekp/train_keypoints.py::train` (Trainer construction).

**Input/output:** Configured accelerator/device counts/strategy/determinism/validation limits/sanity steps enter config logging, but runtime is explicitly `accelerator='gpu', devices=1, precision='16-mixed', log_every_n_steps=500`. Only max steps and validation interval are forwarded. The `seed`, `train`, and `test` top-level controls are also unused by `train`/`main`.

**Consequence:** CPU/MPS/DDP selection is misleading, runs and saved configuration disagree, validation may be much larger than configured, and `test=True` does not execute a test phase. This is established by the constructor call, without relying on framework defaults.

**Falsifier:** An actual configuration-to-Trainer forwarding path or an entry point with different semantics. The separate smoke script does construct its own trainer and is not affected in the same way.

### C05 — Validation drops samples; empty epoch aggregation has an undefined variable

**Chain/location:** `DataModule.val_dataloader/test_dataloader` (42/48) use `drop_last=True` → `DenseKP.validation_step` → `on_validation_epoch_end` (177).

**Input/output:** A validation set of N rows yields only `floor(N/B)*B` scored rows. For N<B it yields zero batches. The hook assigns `avg_val_loss` only when loader-0 outputs exist, then logs it unconditionally for the default argument 0.

**Proof/consequence:** Local DS split has 2 train/2 val samples; DenseKP default B=64 produces zero train and val batches. Direct empty-hook invocation raises `UnboundLocalError`. Lightning may skip validation/hook execution for an empty loader, so the observed training symptom can instead be no validation metric or no training progress; a crash in every such run is not claimed. Partial validation silently excludes the tail.

**Falsifier:** A loader/aggregation contract covering every row and empty epochs. B=1 avoids the current tiny-data trigger. Removing `drop_last` alone is insufficient: `torch.stack` of variable-sized per-batch error vectors would also need attention.

### C06 — Zero supervision weights do not mask non-finite coordinates

**Chain/location:** `build_omni_multiperson_tiny.build` (30) can set validity false while retaining xy → loader affine → `DenseKP.compute_loss` (85), and similarly `validation_step`.

**Input/output:** A finite visible point and a zero-weight point with NaN xy enter the objective. Squared differences are computed first; multiplication by zero yields NaN, not exclusion.

**Proof/consequence:** Both `with_var=False` and `True` returned NaN for `GT=[(0,0,1),(NaN,0,0)]` and zero predictions. At least ten other valid points can allow a generated sample through the multiperson adapter without sanitizing its invalid xy. Non-finite loss/gradients can poison training. All inspected local tiny NPZs were finite, so exposure in those files was not observed.

**Falsifier:** Enforced finite placeholders at every producer/loader boundary, or explicit indexing/sanitization before arithmetic. A zero third field alone is insufficient. The existing tests only mask large **finite** errors.

### C07 — Omni annotation validity ignores the model's own domain test

**Chain/location:** `camera/omni_radtan.py::project_points_omni_radtan` returns `details['model_domain']`/`inside`; `tools/fisheye/run_fisheye_camera_p0.py::warp_case` recomputes `inside` from finite xy and image bounds only, saves it as `densekp_valid`, and uses it for depth checks. Multiperson caller packages these arrays into NPZ.

**Input/output:** A world point in an invalid camera branch can produce in-image xy and be labeled valid. For identity pose and default intrinsics, `(0,0,-1)` projects approximately to `(1500,1500)`; the model returns domain false, while the writer's predicate is true.

**Consequence:** Backward rays can alias forward pixels; if a matching range/instance neighborhood happens to exist, visibility can also be falsely true. The case report's scalar formula comparison does not test branch validity.

**Falsifier:** Intersecting the writer's predicate with the model-domain mask or a proven upstream restriction excluding such rays. This is a branch-handling bug, not evidence that all current rendered landmarks are affected.

### C08 — The optional antialias condition is inverted relative to image reduction

**Chain/location:** `DatasetKeypoints.use_skimage_antialias` → `get_example_projverts`, around line 575 (duplicated in `get_example`).

**Input/output:** The code computes `downsampling_factor = patch_width/(width*scale)` and blurs only when it exceeds 1.1. A 1024-pixel box reduced to 256 has factor .25 and receives no Gaussian; a 128-pixel box enlarged to 256 has factor 2 and is blurred.

**Consequence:** The named antialias option blurs enlargement and fails to prefilter reduction before bilinear warp. This is inactive when the default flag is false, and it is an image-quality correctness issue rather than label misalignment.

**Falsifier:** Evidence that this option intentionally specifies enlargement blur, or a different ratio/condition in the executed path. The separate skimage crop helper is not called by this loader.

### C09 — Zero-worker training configuration fails loader construction

**Chain/location:** `DataModule.train_dataloader` (38) always passes `prefetch_factor=cfg.GENERAL.PREFETCH_FACTOR` with configured `num_workers`.

**Input/output:** `NUM_WORKERS=0, PREFETCH_FACTOR=2` → `torch.utils.data.DataLoader` raises ValueError in the audited environment: prefetch can only be specified with multiprocessing.

**Consequence:** A common CPU/debugging override fails before iteration. Default 16 workers and the one-worker smoke workflow avoid this.

**Falsifier:** Conditional prefetch handling, an explicit compatible value, or a runtime version demonstrably accepting this combination. This finding is runtime-version-scoped to the reproduced environment.

### C10 — `TRANS_AUG_RATE` does not gate translation

**Chain/location:** `datasets/utils.py::do_augmentation` (57) samples `tx/ty` from `TRANS_FACTOR` unconditionally; `get_example_projverts` adds them to crop center. No `TRANS_AUG_RATE` read occurs there.

**Input/output:** Nonzero `TRANS_FACTOR` with `TRANS_AUG_RATE=0` still produces translated crops. Smoke scripts set the rate to zero as though disabling translation, but their inherited factor is also zero, so they happen to be unaffected.

**Consequence:** Configuration-dependent augmentation differs from the advertised rate. Current Hydra DenseKP factor 0 avoids this case; YACS defaults have a nonzero factor.

**Falsifier:** A rate gate in the executed augmentation code. A zero factor prevents the effect but does not establish rate support.

## High-confidence risks

### R01 — Annotation channel 2 has multiple schemas but one unconditional consumer

**Chain/location:** BEDLAM tiny generator/P0 builder append homogeneous 1; multiperson builder writes explicit binary selection; `DenseKP.compute_loss` and validation clamp every channel 2 to `[0,1]`. `evaluate_densekp_npz` instead thresholds at `.5`.

**Input/output:** Unspecified historical third fields become loss weights; their numeric values change sampling contributions and train/evaluation selection. No schema tag is checked. The binary adapter and loss tests are explicit evidence that masking is intended **for those new datasets**. There is no tracked `keypoint_value` symbol or producer contract authorizing its generic confidence interpretation.

**Consequence:** Importing other annotation semantics could silently alter the objective; fractional fields yield different train/evaluator policies. Constant 1 and documented binary masks behave consistently with their producers.

**Falsifier:** A traced, versioned schema for every actual input proving the consumed third field is the intended mask/weight. Numeric range or a name such as `conf` is not sufficient evidence. No semantic corruption of an unseen dataset is classified as confirmed.

### R02 — Raw BEDLAM reconstruction and semantic correspondence are not closed

**Chain/location:** `build_bedlam_tiny_frame_manifest` emits person/frame/placement records; `generate_bedlam_densekp` starts from reconstructed vertices/joints/translations. No tracked connecting producer was found. Local `R` has shape/row-sum support but no committed derivation tying its row order to historical checkpoint labels.

**Input/output:** Raw RGB/CSV/motion identities must become camera-consistent `[10475,3]` vertices and then `[138,3]` projections. The gap can hide frame offsets, unit conversion, root/pelvis shifts, body orientation, or row permutation mistakes.

**Consequence:** Perfect downstream affine math can still train on wrong labels. BEDLAM index equality must not be assumed; SMPL/SMPL-X joint maps do not solve this surface-point correspondence.

**Falsifier:** A reproducible producer with model/operator hashes and independently checked source-frame RGB overlays, including selected vertex-support rows and camera transforms. Existing synthetic-render tests do not close this original-data gap.

### R03 — Visibility selection is frozen before random crops

**Chain/location:** Multiperson builder creates `valid = visible & within_crop & finite` for its original square; loader later rescales/translates/rotates/extreme-crops xy while preserving channel 2; loss consumes the old selection. P0 builder instead selects all points deliberately.

**Input/output:** A selected point can leave the augmented crop while remaining selected; an initially excluded but otherwise visible point entering a larger crop remains excluded. The ViT further sees only square-crop x columns 32–223, though targets span the square.

**Consequence:** Supervision depends on a pre-augmentation visibility policy, potentially inconsistent with a visible-in-input objective. Full-body/amodal regression may intentionally supervise unseen points, so this is not automatically wrong.

**Falsifier:** An explicit decision to keep fixed/amodal labels under cropping, validated against the intended inference behavior; otherwise an augmentation-aware mask specification and overlays. The central-width slice itself is not proof of incorrect coordinate normalization.

### R04 — Validation can obscure important failures or optimize a different sample weighting

**Chain/location:** `compute_loss` normalizes over all selected points in the batch; `validation_step` averages per person; `on_validation_epoch_end` uses only loader 0. `evaluate_omni_densekp.evaluate_case` excludes non-finite errors from `scored`.

**Input/output:** Persons with many selected landmarks contribute more to training, but equally to validation; an all-zero-mask person reports zero error; additional validation datasets produce no final monitored metric. Paired evaluator predictions that become NaN can reduce the scored set rather than force evaluation failure.

**Consequence:** Checkpoint ranking can fail to represent all domains or hide invalid samples. Crop-pixel error is also not an original-image or detector-aware metric.

**Falsifier:** A documented loader-0-only selection objective and explicit valid-sample/finite-prediction gates with reported denominators. Different train and validation objectives alone are not bugs; their intended aggregation must be agreed.

### R05 — Configuration does not capture a reproducible training experiment

**Chain/location:** `train_keypoints.py` calls only `torch.manual_seed(0)` at import; augmentation uses Python random and NumPy, ViT uses stochastic depth; top-level `seed` is unused. `dataset_config()` returns an empty frozen node, saved beside model config; actual dataset paths are globals in `core/configs`.

**Input/output:** A run's saved YAML can omit effective dataset mappings/hashes and misrepresent active trainer settings. No explicit `seed_everything(...,workers=True)` is called by this entry, unlike the Lightning smoke tool.

**Consequence:** Reproducing augmentation, worker behavior, resume ordering and exact data membership is not guaranteed. Framework worker seeding may make some runs repeatable; universal nondeterminism is not claimed.

**Falsifier:** Repeated run/resume equivalence with recorded Python/NumPy/Torch/worker states, resolved paths, data/config hashes, environment versions and actual trainer settings. An unchanged `seed` field alone is not evidence.

### R06 — Loader accepts unchecked shape, dtype and auxiliary-joint semantics

**Chain/location:** `DatasetKeypoints.__init__` loads arrays without schema checks → `get_example_projverts` writes transformed floats into their original arrays → loss assumes `[B,138,3]`. `part` has no enforced shape; `gtkps[:,:44]` is assumed to match the legacy crop helpers' indices.

**Input/output:** Integer labels lose fractional coordinates; wrong landmark count fails downstream or interacts with reshaping; malformed auxiliary joints can change extreme crops or cause indexing errors. NaN/negative bbox sizes may reach OpenCV or `breakpoint()` rather than a clear validation error.

**Consequence:** External dataset incompatibilities are caught late or can be silent. The tiny producers inspected here use the expected float32 layout.

**Falsifier:** Full ingestion validation of sample counts, shapes, dtypes, positive finite scales, image paths and joint ordering, plus semantic metadata. A successfully loaded first batch alone does not validate all rows.

### R07 — Runtime architecture and loss are substantially fixed despite configurable-looking YAML

**Chain/location:** `create_backbone()`/`build_keypoints_head()` take no cfg; `constants.TRANSFORMER_DECODER` supplies head dimensions; ViT positional embeddings and `forward_step` width slice assume the existing input geometry. YAML `SMPL_HEAD`, `FINETUNE`, `LOSS_TYPE`, `LOSS_REDUCTION` and clipping-related inherited defaults are not consumed by these paths.

**Input/output:** Editing architecture or loss fields can leave the network/objective unchanged. Changing `MODEL.IMAGE_SIZE` changes the loader while fixed positional tokens still expect the ViT grid; default 256 works.

**Consequence:** Experiments can be mislabeled or fail on shape addition. The correct response is to verify effective code behavior, not infer support from config keys.

**Falsifier:** A traced config read reaching model construction/loss selection or an explicit validated supported-shapes contract. This audit did not instantiate the full large network to test alternate sizes.

### R08 — Data and checkpoint provisioning are machine-dependent

**Chain/location:** `core/configs.base_dir` resolves to repository `densekp/`; many legacy paths are under `densekp/data`, while SMPL-X/BEDLAM2 strings are CWD-relative `data/...`. Tiny registrations use repository-relative absolute paths. `configs/datasets.example.json` is never read. `VITPOSE_BACKBONE` is read from env/default and unconditionally loaded by `DenseKP.__init__`.

**Input/output:** The same command in a different working directory or clean checkout can resolve different/missing data; loading a complete DenseKP checkpoint still first needs the separate backbone checkpoint. No operator/weight checksum gate is used by construction.

**Consequence:** Migration/fresh-run/inference failures and ambiguous provenance; not proof that this machine's tiny paths are wrong.

**Falsifier:** Resolved-path inventories and an explicit provisioning recipe with hashes for each selected dataset/operator/model. An unused JSON example does not configure the loader.

### R09 — Near-ERP builder can silently use a different camera calibration

**Chain/location:** `run_fisheye_multiperson_pilot` reads custom `camera_config['intrinsics']`; `build_omni_erp_pilot.build` checks only the camera model string and image shape, then constructs `OmniRadtanIntrinsics()` defaults.

**Input/output:** A custom-intrinsics 3000×3000 source is inverse-warped with the default calibration. RGB/points may agree under the wrong inverse yet disagree with the intended physical ERP mapping.

**Consequence:** Systematic angular distortion for non-default calibration. Current checked-in near camera uses the defaults, so this is a supported-input generalization risk, not an observed current artifact failure.

**Falsifier:** Restricting source provenance to identical hashed intrinsics, or consuming the generating calibration. Merely retaining image dimensions does not establish it.

### R10 — Visible-only multiperson targets leave deployed hidden landmarks unsupervised

**Chain/location:** `build_omni_multiperson_tiny.py::build`, used for both Omni and DS fixtures, sets `valid = visible & in_crop & finite` → NPZ `proj_verts[...,2]` → `DenseKP.compute_loss` zeroes hidden-point contributions. External `pano-stitch/src/densekp_delivery/person_match.py::_compute_valid` / `match_all_persons` instead select finite, in-bounds, lens-valid predictions → `densekp_runtime_export.py::materialize_runtime_fixture` sends them to C++ fitting. Neither path has a predicted surface-occlusion mask.

**Input/output:** Synthetic geometry supplies full fixed-row xy plus surface visibility; the training mask retains visible points only. Deployment retains geometrically eligible predictions even when the physical landmark is hidden. DenseKP `raw_scores`/`keypoint_value` do not gate these observations.

**Evidence/consequence:** The policy difference is explicit in code; actual hidden-point accuracy degradation is **not measured**. Visible-only fine-tuning may improve visible points while failing to improve or degrading hidden-point localization that still affects matching/fitting. Initial pretrained knowledge may mitigate this. P0 all-one supervision is different again and cannot be mixed without an explicit objective.

**Falsifier/boundary:** A deployed surface-visibility filter, or held-out occlusion-stratified evidence that visible-only training meets the intended hidden-point/identity/fitting targets, would remove this concern. Recommended experiment: reliable in-domain occluded-point supervision with separate visibility metadata versus visible-only control, keeping person identity and data splits fixed. This audit changes neither policy nor loss.

### R11 — Camera parity does not establish deployment crop, scene or evaluation parity

**Chain/location:** Local `tools/fisheye/build_ds_multiperson_pilot.py::build` fixes `DS_20260320` and allows independent per-camera longitude placement → `build_omni_multiperson_tiny.py` uses GT instance masks with 1.2 bbox expansion and per-person ERP rolls → local oracle-crop evaluation. External `DenseKPConfig` / sequence CLI default to `20260311`; production FULL uses overlap ROIs and detector xyxy boxes; TRACK_GUIDED requires a successful preceding fit but derives its region from `track.last_person.pairs` / `.valid` (previous matched observations), with its own margin policy, then both enter `ORTRunner` square crops.

**Input/output:** A small synthetic corpus with known persons/masks and a fixed camera model becomes crop-space training/evaluation. Deployment supplies noisy detector/tracker regions, real overlapping bodies and camera-specific masks. A missed detector person has no DenseKP crop and cannot be recovered by a better landmark model alone.

**Evidence/consequence:** `20260320` parameters/LUTs and same-bbox preprocessing pass direct comparisons; no incorrect affine or row mapping is inferred. All current tiny data reuse one actor/motion, and DS pilot camera placements are explicitly not synchronized stereo. Good tiny/oracle results therefore do not demonstrate crowded-image recall, identity preservation, genuine cross-camera matching or tracking robustness. Using the config default without freezing the camera group can also select a different calibration from the synthetic corpus.

**Falsifier/boundary:** Actual resolved calibration/assets, representative detector and guided-crop distributions, independent identities/motions/captures, and end-to-end held-out evaluation would close this risk. The specific calibration mismatch is avoided by explicit matching `20260320` settings, or deliberate coverage of both supported groups; that does not close the other distribution gaps.

**Target clarification:** The user selected `20260320_riding_dataset3`; local preparation code explicitly selects `20260320`, and both camera folders contain all 251 frames in its 4325–4575 window. Camera-group selection is now resolved for the primary target. The separate 0311 calibration is not interchangeable. Local synthetic masks and this prepared window's masks differ despite matching intrinsics; because masks are extracted from image content, the difference is a provenance/coverage check, not a newly confirmed calibration error. Historical review flags are candidate failure cases, not ground truth. See DATA_FLOW section 11.

**Scope clarification:** Detection improvement is explicitly outside the user's task. R11 is a data/crop representativeness risk for DenseKP **conditioned on existing boxes**, not a requirement to fix detector recall. Freeze boxes across checkpoint comparisons and retain raw predictions for all eligible boxes before matching. Missing detections are separate upstream exclusions; matched-only output would omit some actual DenseKP failures. Matching/tracking/fitting results may support diagnosis but are not optimization targets for this retraining.

### R12 — Matched-only artifacts bias a supplied-box DenseKP benchmark

**Chain/location:** External `_run_densekp` produces per-camera person results → `match_all_persons` retains eligible pairs → `sequence_orchestrator._full_delivery` records counts and matched boxes → `matched_person_tracks.py` serializes matched `raw_pairs`. `B/standalone_sequence/sequence_result.json` is therefore a selected subset of available predictions; see DATA_FLOW section 12 for its checksum and counts.

**Input/output and evidence:** Across 138 FULL/RECOVERY frames, reported per-camera result counts total 3,263 entries, whereas the retained JSON represents only 870 single-camera outputs. Frame 4480 records 8/16 per-camera persons and zero matched pairs. It has no raw pairs to diagnose; its absence is not proof of a missing detection.

**Consequence:** Scoring only this JSON could omit hard DenseKP outputs, while recovering frames from final success flags could incorrectly label failures. This is a confirmed artifact-selection property, **not a production bug in an intentionally matched-pair delivery API** and not an observed accuracy estimate.

**Falsifier/remedy boundary:** A complete pre-matching per-box dump with consistent image/ROI/bbox IDs, checked against runner counts, would remove this evaluation limitation. Such a dump was not found in the inspected target regression directory. Future offline evaluation needs this evidence; it does not require detector improvement. Existing retained records remain useful qualitative fixtures.

### R13 — Guided logs do not always identify the source crop of the assigned prediction

**Chain/location:** `track_guided_delivery.run_track_guided_densekp` builds a crop per source track, retains `GuidedPersonCandidate.source_track_ids/source_guided_regions`, performs global identity assignment, and then assigns `row.person = replace(candidate.person, person_id=track.track_id)`. It does not replace the destination row's `crop_cam2/crop_cam4`. `sequence_orchestrator.process_frame` serializes `row.crop_*` as `guided_crops` and candidate ID in `identity`, but the inspected sequence JSON lacks the candidate-to-source-region mapping.

**Input/output:** A prediction produced in one source region can be assigned to another destination track; its saved proposal bbox still describes the destination track. A plot joining only on destination `track_id` can therefore show the wrong input crop even though restored full-image predictions remain inspectable.

**Consequence:** Exact fixed-input attribution or a bbox-jitter claim is not justified from these guided logs alone. This does not prove a faulty inference transform or incorrect assignment. The reviewed FULL/RECOVERY records have a direct matched-person-ID-to-bbox join and do not share this ambiguity.

**Falsifier:** Persisted final candidate source-region/ROI/affine IDs, or verified identity assignment that leaves source and destination identical for the exact case, would establish crop provenance. A one-track/one-candidate frame removes cross-track ambiguity but still needs artifact/version provenance for exact reproduction.

## Unverified hypotheses

### H01 — Name-triggered portrait rotation may disagree with imported labels

**Chain/location:** `DatasetKeypoints` → `get_example_projverts` rotates image clockwise if dataset name includes `portrait`/`closeup`, leaving `center` and `proj_verts` unchanged. `visualize_annotation` duplicates this convention; BEDLAM tiny generation excludes these orientations.

**Representation/consequence:** Unrotated-raster labels would be wrong after rotation; already-rotated labels would be correct. The historical producer was not available in the tracked chain, so neither case can be inferred from the unusual code alone.

**Falsifier:** Verified raw-image overlays and metadata showing which raster the coordinates reference, including non-square images and corner/center examples. Keep this out of confirmed-bug counts until then.

### H02 — Double Sphere's domain predicate may discard invertible boundary rays

**Chain/location:** `double_sphere.project_double_sphere` computes `valid = z_shifted > -w2*d1`; LUT builders discard false values; `unproject_double_sphere` supplies an independent inverse path in the same module.

**Representation/evidence:** A 10,001-angle sweep of unit rays found 548 cam2 and 685 cam4 rays rejected by forward validity despite inverse projection recovering the input to <1e-7. No accepted rays failed the tested inverse. Current measured-mask test passed.

**Hypothesis/consequence:** Validity may be more conservative than the intended DS domain, clipping peripheral coverage. It may deliberately reproduce an external deployment restriction, and invertibility alone does not prove the intended physical/model domain.

**Falsifier:** A domain derivation and actual deployment/reference-lens comparison showing the rejected region is intentionally excluded. The external follow-up now verifies that `pano-stitch` uses the same predicate and yields identical masks on 20,000 seeded rays per camera and identical reduced-resolution LUTs. This closes implementation disagreement for the inspected `20260320` branch, but shared code agreement does not prove the intended physical domain.

### H03 — Raw BEDLAM negative-depth projections may become supervised targets

**Chain/location:** `generate_bedlam_densekp.project_official` permits negative Q.z; `official_bbox` checks 2D bounds only; serializer writes homogeneous 1; trainer selects those rows.

**Representation/evidence:** `(0,0,-1)` with zero translation/identity intrinsics returns `(0,0,1)`, with no error. A camera-space sign error upstream could therefore look numerically valid.

**Consequence:** Behind-camera points could become labels, but all valid original input records may have positive depth by construction. No such corrupted real record was established here.

**Falsifier:** Positive-depth invariants measured over all source geometry or an explicit signed-domain requirement enforced before serialization.

### H04 — Depth tolerance and neighborhood tests may overstate surface visibility

**Chain/location:** Interpolated surface landmarks → P0 3×3 minimum-depth-error comparison or ERP/Omni 5×5 comparison (<4 cm) → binary visibility → selected training points. P0 sidecar uses state codes, other branches use booleans.

**Representation/consequence:** A point under clothing or just behind a neighboring surface can pass tolerance. Interpolated body points are not necessarily on the outer rendered clothing surface; silhouettes and seam neighborhoods need special care. Known consumers of the P0 state array correctly use `state==0`; this audit did not find a proven current boolean-cast inversion.

**Falsifier:** Independent instance-aware ray intersections and a tolerance sensitivity study covering clothing, self-occlusion, mutual occlusion, silhouettes, and image edges. Existing formula agreement does not establish visibility ground truth.

### H05 — Model/backend parity remains unproved after crop/restoration comparison

**Chain/location:** Local evaluators end at normalized/crop-pixel predictions. Inspected external `ORTRunner` and inherited `TensorRTRunner` use square-crop affine inversion; `overlap_roi.py` restores periodic full-ERP xy; `erp_rotate.py` rotates output rays. Loader `box_size` excludes random scale, and training ERP shift stays in a separate manifest.

**Representation/evidence:** The inspected path correctly decodes `(xy+.5)*256`, with no erroneous 192-width or 32-pixel offset, and preserves raw third values. Numeric tests on all 33 same-bbox samples give tensor discrepancy ≤`2.39e-7` and inverse error ≤`1.53e-5` pixels; a seam/rotation example also passes. Thus those previously hypothetical affine/normalization errors were **not found** in this path. Actual checkpoint/ONNX/TRT output equivalence, exact-pole behavior and different deployment snapshots remain unverified.

**Falsifier:** Fixed real-crop samples through the actual selected checkpoint, ONNX/TRT artifact and restoration chain, including off-center/non-square source images, seams and boundary crops, with explicit third-channel transport semantics and numeric tolerances. Reusing the current helper comparison alone does not test network/export differences.

### H06 — Mixed-precision variance arithmetic may become unstable at extreme outputs

**Chain/location:** Head's unconstrained channel 2 → `compute_loss`: lower-only clipping, `exp`, square, reciprocal and log → manually optimized mixed-precision training. Weights and their normalizer are converted to prediction dtype.

**Representation/consequence:** Very large log sigma can overflow even float32; sufficiently small half-precision sigma squared can underflow if operations remain half, and large half-precision selected-point sums can overflow. Actual autocast promotion and CUDA execution determine exposure. Zero-weight points do not automatically exclude invalid variance arithmetic.

**Falsifier:** Finite-loss/gradient probes on the actual precision/device path over the intended output/mask/batch ranges and recorded dtype promotion, or numerically stable bounded arithmetic. The CPU Half probe failed at an unsupported clamp before reaching these operations; it provides no confirmation of GPU instability.

### H07 — Panorama preview and fisheye RGB may use different display transforms

**Chain/location:** `render_equirectangular_source` saves Blender preview PNG without explicitly fixing `view_transform`; `warp_case` reads linear EXR and applies `linear_to_srgb`; paired evaluator compares these imaging paths. DS pilot also starts from the preview image.

**Representation/consequence:** Identical scene radiance can become different displayed intensities, confounding a projection-only comparison or training-domain matching. Blender version/defaults and output history determine the size of the difference.

**Falsifier:** Captured color-management settings and a corresponding linear/PNG pixel comparison demonstrating equivalent transfer for the actual assets. Source syntax alone does not establish an observed color mismatch.

## Maintainability issues

### M01 — Historical documents contradict the baseline code/artifact inventory

**Chain/location:** README/older `docs/training/TRAINING_DATA_FORMAT.md` guide setup and annotation interpretation; active code now uses channel 2 in loss, includes generators and has local pretrained files. `assets/README.md` says the operator is included, while `git ls-files` excludes it.

**Input/output/consequence:** Readers can infer that GT channel 2 is ignored, generation tooling is absent, or ignored assets will be available in a clone. The four root audit documents describe the baseline without changing historical records.

**Falsifier:** A documentation/version policy aligning those statements with their intended commit and explicit local-versus-tracked inventory. Earlier experimental results are not automatically invalidated by stale prose.

### M02 — Duplicated geometry and inherited unused helpers invite divergent fixes

**Chain/location:** Active NumPy crop affine and separate batched Torch transform; `get_example` versus `get_example_projverts`; old skimage crop; separate ERP projection formulas, world transforms and visibility implementations; unused inherited losses/SMPL helpers in `losses.py` and `train_utils.py`.

**Representation/consequence:** Similar names can refer to different coordinate spaces or third-field policies. Search found no in-repository calls to `get_example`, `generate_image_patch_skimage`, `keypoint_3d_processing`, `convert_to_full_img_cam`, or `load_valid`; that establishes local non-use, not proof that external consumers do not import them. `load_valid` even computes a filtered dict then updates with the unfiltered original, but it is not on the active path. `DenseKP.visualize` scales all three output fields and calls the third “confidences”; it is not called by forward and the value is unused.

**Falsifier:** External call inventory and contract tests showing intentional distinctions. Preserve the active-path evidence before any later consolidation or deletion.

### M03 — Experiment-specific assumptions cross the reusable boundary

**Chain/location:** Reusable runtime defaults read frozen output manifests; render jobs import tool entry paths; clothing adapter asserts one pair's topology/timing; scene/camera tools hard-code baseline directories, resolutions or revision strings. `camera.extrinsics.pose(target,origin,yaw)` accepts but ignores `target` and constructs yaw-only orientation.

**Representation/consequence:** New assets or experiments can be constrained by assumptions hidden in old workflows; absolute paths inside saved manifests hinder relocation. Current pose callers intentionally use yaw sweeps, so ignored target is not by itself a proven camera error. The pip package exposes reusable `src` code while tools and trainer still rely on checkout paths.

**Falsifier:** An explicit supported-input matrix and caller requirements that justify these constraints, with provenance rather than hard-coded historical revision labels. Refactoring is deferred by the user.

## Performance issues

### P01 — Loader transfers and constructs more data than optimization consumes

**Chain/location:** `DatasetKeypoints` materializes NPZ arrays, decodes the original image per person/sample, creates full-image Albumentations objects on each call, and returns display crops/original labels/affine metadata; DataModule collates them with 16 workers and prefetch 2 by default.

**Input/output/consequence:** High-resolution repeated-person ERP images are re-decoded; display/original arrays may be transferred by Lightning despite the training loss using only image and cropped target. Full-frame image augmentation adds CPU work. Worker memory behavior depends on fork/spawn and copy-on-write; replication magnitude was not profiled.

**Falsifier:** Profiling showing these costs are negligible at the intended data resolution/throughput. Do not optimize based on a tiny one-person batch alone.

### P02 — Fisheye warp allocates full-resolution float64 intermediates

**Chain/location:** `warp_case` creates `np.indices`, flattened pixel arrays, rays and inverse iterations for all default 3000×3000 pixels, then full RGB/depth/index maps. DS and reusable ERP LUT builders already support row chunks; the Omni warp does not.

**Input/output/consequence:** A single `[9,000,000,3] float64` array is 216 MB; several coexist, plus iterative inverse arrays and rendered buffers. Render pipelines also load complete body/clothing trajectories in several entry points. This limits scale and can increase wall time/memory substantially; peak RSS was not measured here.

**Falsifier:** Representative memory/throughput measurements showing acceptable headroom, or a caller that uses smaller dimensions/chunked implementation. The code-level allocation sizes remain facts.

### P03 — Full checkpoint initialization has unnecessary backbone I/O and memory overlap

**Chain/location:** `DenseKP.load_from_checkpoint` constructs `DenseKP`, which always creates ViT and loads `VITPOSE_BACKBONE`, before restoring DenseKP state. Evaluators and `init_checkpoint` use this route; `ckpt_path` resume likewise starts from an initialized model.

**Input/output/consequence:** Large backbone checkpoint load is repeated even when the complete DenseKP state will overwrite it. Local file sizes are about 2.52 GB for backbone and 8.05 GB for DenseKP. ViT activation checkpointing is disabled in its factory; full finetuning cost needs GPU measurement. This audit did not load the large model or benchmark it.

**Falsifier:** A demonstrated loading path that bypasses redundant initialization, or profiling establishing the overhead is acceptable for the workload. No architecture/precision change should be justified solely by this static estimate.

## Real-scene supplied-box case review

Reviewed raw predictions on 106 single-camera panels from 10 frames in 20260320 dataset3, plus two empty-output frame records. All 775 retained observations in the whole saved sequence have finite `[138,4]` raw pairs and IDs `0..137`. DATA_FLOW section 12 describes the exact artifact, joins, transformations and limitations. **This pass does not establish a quantitative DenseKP accuracy or occlusion failure rate, and adds no source-correctness bug based on visual appearance.**

| Case | Observed input/output | Interpretation and likely consequence | Evidence needed to confirm/falsify a DenseKP failure |
|---|---|---|---|
| 4472 cam2 P2/P3/P4; 4473 cam2 P1 | Red-shirt background person overlaps a gray-shirt foreground person and black-clothed neighbor; the returned point sets extend into occluded regions | Target attribution under overlapping bodies is a real input condition; a point on the foreground body is not automatically a wrong-person point | Explicit target identity and visible semantic labels; reliable synthetic hidden-point GT for amodal error. Coherent projection of the intended hidden body would falsify a mixing claim |
| 4472 cam2 P5 and cam4 P4; 4572 cam4 P3 | Very small/partial source bboxes: respectively 38.25×87.75 and 56.25×85.5 pixels in the first two cases; model emits all 138 rows | Upsampling cannot restore missing detail; severe partial-body cases need their own supervised/evaluation stratum. This is a DenseKP input condition, not a request to fix boxes | Known target and crop-aware visible/hidden labels; compare at fixed boxes and stratify by original source resolution |
| 4472/4473 cam2 umbrella region, and motorbike rider P9/P8 respectively | Umbrella hides torso/limbs; helmet/visor, gloves, handlebars and seated legs obscure anatomy. Predictions include hidden-body locations | Ordinary standing-body examples and simple rectangular erasing do not cover these occluder/pose combinations. Exact hand/foot errors remain unmeasured | Instance-aware synthetic geometry and visible-point labels; do not use umbrella silhouettes or optimized SMPL-X projections as point GT |
| 4559 cam2 P0 → 4560 T45; 4573 cam2 P0 | Existing supplied regions intersect large black lens-invalid areas. Frame 4560 shows a narrow visible cyclist strip and predictions extending over road/black area; pair-valid count is 18 | Underconstrained input and potential extrapolation instability are visible. This is not a missing detection. An amodal prediction outside the visible lens is not directly verifiable from that camera | Freeze the exact existing crop; separate lens exclusion from body occlusion; obtain target GT from geometry/independent evidence. Changing frame/crop alone does not establish a model regression |
| 4331 cam2 P1 versus cam4 P1 | The saved pair corresponds visually to a checkered-shirt pedestrian on cam2 and a black-jacket pedestrian on cam4; each side has its own supplied bbox | Clear example where a paired-result anomaly must not automatically become a DenseKP wrong-person label. Matching improvement is outside scope | Assess each side against its own bbox's intended person before labeling model error; exclude cross-camera identity mismatch as the sole error evidence |
| 4572→4573 track 53 | Saved boxes/visible target change from checkered-shirt pedestrian to masked cyclist; prior `ID_SWITCH_SUSPECT` is computed from optimized coordinates | Track continuity is not a stable target-person annotation. Temporal jumps cannot serve as DenseKP GT | Fixed frame/crop/target comparisons and explicit person identity independent of historic track IDs |
| 4331 cam4 P0; 4473 cam2 P5; 4573 cam4 P1 | Visually coherent gross localization on cyclist, visor-wearing person and checkered-shirt person, with remaining occlusion/detail uncertainty | Retain as controls so a hard-case training mix does not lose existing gross localization | Accurate visible-point annotations are still required before claiming small numerical error |

Temporary review figures are described in DATA_FLOW section 12. FULL case IDs are frame-local saved `person_id`, not globally stable identities; T IDs are historical track labels, not human-verified identity. Guiding-frame proposal plots carry the R13 caveat. Head/body/hand-looking clusters were inspected qualitatively; no unverified BEDLAM index mapping or numeric third-channel score was used as an annotation.

## Things checked that are not established bugs

- `DATASET_FILES[False]`/`[True]` selects val/train correctly.
- Affine math, `(W,H)` OpenCV sizes, RGB normalization and active square xy normalization are consistent in inspected samples. Rotation uses the same affine for image and labels when flipping is disabled.
- Validation's recomputed target agrees with loader crop coordinates for no augmentation; its metric differs from training by design and needs an explicit reporting policy.
- The 256→192 feature slice does not imply a 192-pixel prediction frame. Applying a 32-pixel restoration offset would contradict the actual target normalization.
- The 138-point head preserves batch size 1. Its third output is log sigma only under the corresponding loss; neither the head name nor numeric magnitude establishes external confidence semantics.
- Synthetic body→Blender/CV axis conversions and ERP latitude sign alternatives are consistent with the inspected renderer and existing stored projection regressions.
- P0 all-one targets are an explicit producer policy. They are not evidence that occluded landmarks were accidentally considered visible, nor proof that the policy suits the intended deployment.

## Second-pass consistency review

After drafting, rechecked active call sites, config tracking, loss branches, validation aggregation, camera predicates, model token dimensions and generated-data consumers against the baseline. Specifically kept the unsupported CPU Half result out of confirmed findings; kept portrait/DS/deployment concerns as hypotheses; distinguished state-coded P0 visibility from binary ERP masks; and identified the missing raw-BEDLAM adapter and external restoration as boundaries rather than invented implementations. Readiness gates refer to the IDs above. No proposed fix has been applied in this archaeology phase.

The external follow-up's second pass additionally reconciled the original absence-of-deployment statements with newly inspected source; distinguished normal-ERP input from ERP_ROTATE output; distinguished lens validity from body visibility; checked both FULL and TRACK_GUIDED bbox producers; and retained checkpoint/backend performance as unmeasured. R10/R11 describe target alignment risks, not newly confirmed corrupt-label bugs.
