# DenseKP data and coordinate flow

> **Post-audit implementation update (2026-09-28):** The archaeology below is a frozen
> report of baseline `8553ccd`. A subsequently authorized correctness/refactor batch
> is now present in the working tree. Current contracts, fixed/guarded issue IDs,
> configuration migration, verification evidence and remaining training gates are in
> [the implementation report](docs/refactor/DENSEKP_CORRECTNESS_20260928.md).
> Statements below that production code was unchanged refer to the archaeology phase;
> baseline bug descriptions and unchecked checklists are not the current fix status.

Baseline: `8553ccd2d3117755e595ae97c8522c303bd05ceb`; 2026-09-28. Read alongside [ARCHITECTURE.md](ARCHITECTURE.md) and [BUG_AUDIT.md](BUG_AUDIT.md). `N` denotes annotation samples, `T` motion frames, `B` batch size, `K=138`. Array rows are semantic identities; equal array lengths never establish matching point orders.

## 1. Original BEDLAM source to intermediate geometry

`tools/annotation/build_bedlam_tiny_frame_manifest.py::parse_state_machine` reads `be_seq.csv`. A `sequence_name` comment changes current sequence; each subsequent `start_frame` body row supplies actor placement and motion identity. `main` matches PNG filenames and camera CSV rows, then records:

```text
smplx_param_ind = img_ind + start_frame
motion_relpath = person_id / sequence_id / motion_seq.npz
camera_row = img_ind
body_X, body_Y, body_Z, body_yaw = source CSV values
```

This script verifies camera row names and required columns but does not reconstruct the body/camera. The tracked code has no connected adapter consuming this manifest to produce the intermediate NPZ used below. Unit conversion of original BEDLAM placement, source-camera rotations, pelvis offsets, and 6/30-fps image-to-motion correspondence therefore remain upstream obligations. Do not substitute the synthetic canonical-motion path for this missing reconstruction step: it can deliberately zero translation and uses a newly rendered camera.

`tools/annotation/generate_bedlam_densekp.py::main` requires:

| Field | Shape | Meaning required by consumer |
|---|---|---|
| `imgname` | `[N]` | Image path, relative to `--image-root` or absolute |
| `vertices` / `smplx_vertices` | `[N,10475,3]` | Correct-topology SMPL-X vertices already oriented for the projection convention |
| `joints3d` / `body_joints3d` | `[N,J,3]`, J≥22 | First 22 joints used for bbox, not DenseKP identities |
| `camera_translation` / `cam_trans` | `[N,3]` or per-record `[1,3]` | Translation added before projection, same length units as vertices |
| `camera_intrinsics` / `cam_int` | `[N,3,3]` | Pixel intrinsics, homogeneous bottom row consistent with projected channel 2 = 1 |
| `orientation` | `[N]` | Named portrait/closeup samples are skipped |

There is no extrinsic rotation input to `project_official`; any required rotation must already be represented in the supplied geometry. It computes `P = R_surface @ V`, `Q = P + t`, `h = K_camera @ (Q / Q.z)`, and writes `h.astype(float32)`. Division is before intrinsics and after 3D landmark interpolation. Projection of an interpolated 3D point is generally **not** the interpolation of separately projected vertices.

`project_official` rejects non-finite/zero-depth inputs and non-unit homogeneous output, but allows negative depth. `official_bbox` uses finite first-22-joint xy inside inclusive `[0,W]×[0,H]`, requires at least 8 joints, and computes center `(min+max)/2`, scalar `scale = 1.2*max(xmax-xmin,ymax-ymin)/200`. Debug image bounds use `<W,<H` instead. Neither operation proves front-facing visibility.

`write_bedlam_tiny_dataset` copies images, writes the four-field NPZ, and writes a separate debug NPZ with 22 joint projections, source indices, counts and operator checks. It does not generate the legacy 44-joint `part` field.

## 2. Landmark indexing and synthetic geometry

The concrete definition used by the local generators is `assets/densekp_regressor_R.npy`, `[138,10475] float32`, multiplied by SMPL-X vertices in their existing index order. Local inspection found 204 nonzero entries and row sums from `0.99999994` to `1.0`; SHA-256:

```text
9187dd25d7024eb47f052d7df08e81ae25cf622a62e92325cba40069efc4a9f4
```

This ignored local file defines vertex support/weights, not anatomical names. `annotation.surface::surface_operator_rows` can enumerate every row. The baseline does not contain the artifact or a derivation establishing parity with the checkpoint's historical training order. The loader preserves `proj_verts` rows verbatim; no BEDLAM-to-DenseKP index conversion occurs during training. The BEDLAM downsample asset is separate; none of the traced generation paths uses it as the DenseKP operator. SMPL joint maps and the 44-entry flip permutation in `constants.py` are not a 138-point mapping. No active SMPL→SMPL-X or SMPL-X→SMPL vertex conversion was found in this path.

For newly rendered data, `load_real_smplx_motion.py` splits `poses[T,165]` into global orientation `[T,3]`, body `[T,63]`, jaw/left-eye/right-eye `[T,3]` each, left/right hands `[T,45]` each. It checks static betas, frame bounds/uniqueness and finite arrays. Translation is zero by default; `--use-source-translation` retains source translation. Original translation and selected frame IDs are saved either way.

`generate_real_motion_geometry.py` calls `body.geometry.load_smplx_model` with `use_pca=False`, `flat_hand_mean=True`, source gender and beta count; forwards the selected parameters to SMPL-X; saves vertices `[T,10475,3]`, triangular faces, and `compute_densekp_3d` output `[T,138,3] float32`. Default expression is supplied by the body model rather than the adapter. Model variant, hand convention, units and topology must match the source motion contract.

`prepare_p0_dressed_geometry.py` is specific to a checked actor/motion pair. It asserts clothing shape `[T,13027,3]`, 26328 faces and metadata `{fps:6, source_fps:30, start_frame:101, end_frame:292}`. Local motion indices are `clothing_index*5`; source 30-fps indices add 101. The script checks body `source_frame_ids` against local motion indices. This is explicit alignment for that pair, not a general BEDLAM timing rule.

## 3. Synthetic projection and visibility representations

### Perspective

`render_p0_blender.py` uses the same geometry arrays for mesh vertices and landmark projection. Object-to-world converts `(x,y,z)` to `(x,-z,y)` and applies a fixed floor lift `-min(vertices[0,:,1])` from the first frame; clothing receives the same matrix. This differs from the panorama pilot's per-selected-frame lift and should be preserved when comparing their world coordinates. Camera-local Blender axes are +x right, +y up, -z forward. With `q = inverse(camera_matrix_world) @ object_matrix_world @ [P,1]`:

```text
depth_z = -q.z
u = fx*q.x/depth_z + cx
v = cy - fy*q.y/depth_z
```

The current perspective renderer uses square images, unit pixel aspect, zero shift, horizontal sensor fit, `fx=fy=lens/sensor_width*W`, `cx=cy=W/2`. It compares analytical xy with Blender `world_to_camera_view`. General non-square or shifted-camera support is not demonstrated by that renderer.

`finalize_p0_outputs.py` performs a rendered plane check to choose camera-Z depth versus Euclidean ray range before comparing depth. The closest absolute error in a 3×3 depth neighborhood is compared against `max(0.002, 2*median_landmark_depth/fx)`. It writes:

| Artifact | Actual meaning |
|---|---|
| `densekp_2d.npy` | `[T,138,2]` original rendered pixel xy |
| `densekp_camera_xyz.npy` | `[T,138,3]` Blender camera coordinates |
| `densekp_depth.npy` | `[T,138]`, chosen depth convention |
| `densekp_visible.npy` in P0 finalizer | **State codes**: 0 visible, 1 occluded, 2 outside frame, 3 behind camera |
| `densekp_valid.npy` in P0 finalizer | All ones; not a recomputed in-frame/front-facing mask |
| `densekp_visible.npy` in camera-stress finalizer | Boolean-like uint8; `state==0 AND in_frame`; original state saved separately |

`build_rendered_tiny.py` uses rendered mask extrema including `+1` pixel extent to form a 1.2-expanded square, and appends **ones** to all 138 points. It intentionally does not transfer the above visibility states. Every fourth frame goes to validation, so adjacent motion frames are shared across the split's source sequence.

### Omni-Radtan / panorama

The reusable camera model takes world points `[M,3]`, camera-to-world `[4,4]` with CV axes (+x right,+y down,+z forward), and intrinsics. Row-vector camera coordinates are `(P - pose.translation) @ pose.rotation`. Unit rays determine angles; range is separate. `project_points_omni_radtan(..., return_details=True)` returns xy plus `model_domain` and `inside`.

The Blender panorama renderer converts CV camera basis to Blender by negating columns y and z. `world_points` applies the matching body conversion, per-frame floor lift, world yaw, then person location. `project_erp` uses:

```text
longitude = atan2(ray.x, ray.z)
latitude_down = asin(ray.y)
u = (longitude/(2*pi) + 0.5)*W
v = (latitude_down/pi + 0.5)*H
```

Some functions express the same v formula as `0.5 - asin(-ray.y)/pi`; this is algebraically equivalent. ERP is periodic only in longitude. `erp_visibility` checks 5×5 neighborhoods with horizontal wrap, matching person ID and range error <0.04 m. `warp_case` uses inverse Omni rays to resample panorama RGB/depth/index, then projects landmarks analytically and checks 5×5 depth neighborhoods (with instance matching in multiperson mode). Its saved validity currently omits the camera model-domain mask; see C07.

Original multiperson ERP RGB is the Blender preview PNG. Omni fisheye RGB is generated from linear EXR through `linear_to_srgb`. `build_omni_erp_pilot.py` converts the actual fisheye image/masks/points back to ERP; it does not reuse the pre-warp preview as training RGB. It currently instantiates default Omni intrinsics rather than reading the generating configuration.

### Double Sphere

`double_sphere.py` has named cam2/cam4 calibration constants. `ROTATION_MATRIX=I`; `CAMERA_EXTRINSIC['cam2']=I`, cam4 is `diag(-1,1,-1)`. Forward LUTs map output ERP pixels to fisheye input pixels; inverse LUTs map synthetic fisheye pixels to source ERP. These are sampling maps, not transforms to apply directly to keypoint arrays.

`build_ds_multiperson_pilot.py` rolls each camera's source ERP by a configured shift, warps ERP→fisheye→ERP, applies measured ERP valid masks and optional measured lens footprint, and transports annotations via the equivalent rolled ERP coordinates with width/height scaling. It intersects source visibility with remapped instance-mask support. DS `valid` is finite/image-bounds validity, whereas `visible` additionally includes actual usable image/instance support. It labels the cameras as independent yaw placements, not synchronized stereo views.

`build_omni_multiperson_tiny.py::roll_person` cuts the largest empty longitude gap in the person mask, rolls image and mask equally, and transforms `x=(x+shift)%W`. Bbox center/size come from the rolled mask. Annotation channel 2 is explicitly `source_visible AND within_square_crop AND finite_xy`, encoded 0/1. This is a documented supervision selection for this adapter, not generic confidence. `erp_roll_pixels` remains in the dataset manifest, but is not returned by `DatasetKeypoints`.

## 4. NPZ to crop

`DatasetKeypoints.__init__` uses `DATASET_FILES[is_train][dataset]`; Python bool indexing correctly chooses `[val,train]`. Required fields are `imgname[N]`, `center[N,2]`, `scale[N]` (also vector scales are accepted by crop logic), `proj_verts[N,138,3]`. The loader does not enforce these shapes/dtypes/finiteness. `part`, else first 44 rows of `gtkps`, supplies optional joint labels for extreme cropping; otherwise `[N,44,3]` zeros are synthesized. These auxiliary joints are not the 138 prediction targets.

For each sample:

1. `scale*200` is treated as bbox size in image pixels. Optional `BBOX_SHAPE` expands a width/height vector to an aspect ratio; `.max()` then selects a square side `s`. Scalar scales bypass aspect expansion.
2. OpenCV loads BGR while ignoring EXIF orientation. If dataset name contains `closeup` or `portrait`, the **image only** is rotated 90° clockwise. Labels/center must already refer to that rotated raster; this has not been proved for arbitrary imported datasets.
3. Training samples draw scale, rotation, translation, color, flip and extreme-crop choices. `is_train` controls augmentation; the constructor's `use_augmentation` argument is ignored. Hydra DenseKP defaults disable flips/rotation/translation, use scale factor 0.1, color 0.2, extreme-crop rate 0.1 and Albumentations. Missing `part`/`gtkps` means extreme cropping's full/upper-body predicates fail and the bbox stays unchanged.
4. Extreme cropping, when applicable, changes center and square side from auxiliary joints. Translation adds `width*tx`, `height*ty`. The affine generator then applies scale and rotation.
5. `generate_image_patch_cv2` uses three source/destination points to construct `A[2,3]`, then `cv2.warpAffine(..., (patch_width,patch_height))`. For no rotation/flip, destination square width `S=256`, augmented side `s*a`, center c: `p_crop = S/(s*a)*(p-c) + S/2`. With rotation, use the returned A, not this simplification.
6. Labels use the same A when no flip: `q = A @ [p.x,p.y,1]`; `normalized_xy=q/256-0.5`. xy is not clamped; off-crop targets can be outside `[-0.5,0.5]`. Channel 2 is unchanged and no crop validity is recomputed. For flipping, A refers to the flipped image while labels remain unflipped (C01).
7. Image channels reverse BGR→RGB, convert HWC→CHW float32, apply clipped color multipliers, and normalize `(value-255*mean)/(255*std)`. This equals `(value/255-mean)/std`. `img_disp` remains BGR and lacks the later per-channel color multiplier.

Width/height conventions: OpenCV destination size is `(W,H)`; image array shape and `img_size` are `(H,W)`; bbox aspect ratio is `(W,H)`; point coordinates and centers are `(x,y)`. The active loader always uses square output, so division of both xy components by `patch_width` is correct for it. The generic helper's non-square normalization would require a separate y denominator.

| Returned key | Per-sample shape/type | Space/meaning |
|---|---|---|
| `img` | `[3,256,256]`, float32 | RGB normalized model input |
| `img_disp` | `[256,256,3]`, normally uint8 | BGR crop for display |
| `proj_verts` | `[138,3]`, source dtype | Original annotation xy and original third field |
| `proj_verts_cropped` | `[138,3]`, source dtype | Normalized square-crop xy; unchanged third field |
| `_trans` | `[2,3]`, float64 | Image-to-crop affine; flipped-image domain if flip enabled |
| `box_center` | `[2]` | Center after crop/translation selection, not complete inverse metadata |
| `box_size` | scalar | Width before random scale is applied inside A |
| `img_size` | `[2]`, floating | `[H,W]` after any name-triggered image rotation |
| `_scale` | scalar or vector | Original NPZ scale, not final augmentation scale |
| `imgname`, `dataset` | strings | Source identity |

Numpy assignment retains the input label dtype: integer labels would truncate transformed xy. Supported producers write float32; imported NPZs need explicit schema validation.

## 5. Forward and third-channel semantics

Default batch flow:

```text
train joint_batch['img']['img']: [B,3,256,256]
  -> slice width 32:224: [B,3,256,192]
  -> ViT: [B,1280,16,12]
  -> rearrange: [B,192,1280]
  -> query [B,1,1], embedding/decoder: [B,1,1024]
  -> squeeze only dim 1: [B,1024]
  -> linear: [B,414]
  -> pred_keypoints: [B,138,3]
```

The central-width slice does **not** redefine target coordinates. Targets and predictions are still referenced to the full 256-square crop. There is no evidence for adding 32 pixels to predictions or scaling predicted x by 192. Batch size 1 is preserved by `squeeze(1)` and explicit reshape.

| Field/source | Proven semantics | Permitted conclusion |
|---|---|---|
| BEDLAM tiny `proj_verts[...,2]` | Homogeneous projection coordinate 1 | All points contribute under current loss; 1 does not establish visibility |
| Rendered P0 tiny channel 2 | Explicitly appended constant 1 | All 138 targets selected, including occluded points |
| Multiperson/ERP tiny channel 2 | Explicit binary supervision flag | May be used as mask for this adapter's known schema |
| Arbitrary historical NPZ third field | Unknown until producer traced | Numeric range alone cannot establish confidence/weight semantics |
| Model `pred_keypoints[...,2]`, `with_var=True` | Learned log sigma in this loss | Uncertainty parameter under that training objective; not a calibrated confidence |
| Model channel 2, `with_var=False` | No direct supervision | No established inference meaning |
| External `keypoint_value` | No symbol/API in this training baseline; external consumer traced in section 10 | Preserved raw third value; no confidence weighting in inspected pair matching/fitting |

The code **does** currently clamp every GT channel 2 into a weight. That describes consumer behavior; it does not prove this interpretation is valid for arbitrary DenseKP data. This distinction is a retraining gate (R01).

## 6. Loss, optimization and validation

Let `e_bk = sum_xy((target-prediction)^2)`, `w_bk=clamp(GT_channel2,0,1)`, `D=max(sum_bk(w),1)`.

```text
with_var=False:
    loss = 60 * sum(w * e) / D

with_var=True:
    l = max(pred_channel2, log(1e-6))
    sigma = exp(l)
    loss = 60 * [sum(w * 1000*e/(2*sigma^2))/D
                 + sum(w * log(sigma^2))/D]
```

The log-sigma term can make total loss negative; this alone is not a correctness bug. Sigma is related to the error scaled by 1000, not directly a pixel confidence. Masks multiply already-computed errors, so zero weight does not eliminate NaN/Inf arithmetic.

`training_step` obtains the Lightning optimizer, runs forward/loss, zeros gradients, calls `manual_backward` and steps AdamW. There is no scheduler, explicit clipping, or accumulation in this loop. `TRAIN.LOSS_TYPE: l1`, `LOSS_REDUCTION`, and several inherited options do not select this implemented objective. `train/loss` uses `logger=False`; it is not sent to TensorBoard by that call.

Validation recomputes cropped GT from original xy and `_trans` using the Torch helper, divides by `MODEL.IMAGE_SIZE`, and compares xy only. With augmentation disabled, this matches the loader's cropped xy. Metric per sample is `sum(w*sqrt(e))/max(sum(w),1)`; dataset mean equally weights samples, including all-zero-mask samples whose result is 0. It is a normalized-crop Euclidean error, not training NLL and not original-image pixel error. Multiplying by 256 yields the evaluator's crop-pixel units.

Only loader index 0 contributes to monitored `val_loss` and the epoch summary. Other validation loaders execute forward and store results but are not reported by the final aggregation. Train/val/test loaders all use `drop_last=True`; validation discards a partial final batch. Epoch aggregation stacks equal-sized retained batches. Fixing partial-batch handling later also requires changing the stack/aggregation logic.

## 7. Inference and restoration contract

`evaluate_densekp_npz.py` selects a train/val NPZ, sets `dataset.is_train=False`, runs model evaluation and measures crop error on `channel2>.5`. `evaluate_omni_densekp.py` uses a rendered-mask fisheye bbox or visible-GT-hull ERP bbox and scores visible, in-crop points with finite prediction errors. These are oracle-crop diagnostics. They cannot establish detector robustness, original-image accuracy, or deployment parity.

No prediction restoration implementation is present in the tracked runtime. For a no-flip sample, the required inverse is:

```text
crop_xy = (predicted_normalized_xy + 0.5) * 256
image_xy = invertAffineTransform(A) @ [crop_xy.x,crop_xy.y,1]
```

This undoes scale/translation/rotation recorded in A. A center/box-only shortcut is valid only for the corresponding unrotated crop and actual augmented side. The internal ViT width slice requires **no** additional inverse offset.

If the original raster was rotated clockwise before cropping, undo that after affine restoration: for unrotated image height H, `(x_original,y_original)=(y_rotated,H-1-x_rotated)`. If ERP was rolled, undo `(x_rolled-shift)%W` in the same ERP resolution. A later resolution change needs its inverse scaling in the right order. The loader does not return ERP shift or an explicit raster-orientation transform. A future mirrored inference path would additionally need coordinate unreflection and a verified semantic inverse permutation; current augmentation does not provide either correctly.

Only xy is transformed as coordinates. Do not add `.5`, multiply by image dimensions, apply an affine matrix, or attach a weight meaning to external `keypoint_value` without its explicit producer contract.

## 8. Local checks and limits

Existing tests: 18 passed, zero skipped, using `/home/yy/conda_envs/camerahmr/bin/python` (Torch `2.0.0+cu118`, NumPy `1.26.4`). The shell's default `proxy` Python lacked the scientific dependencies. Local sample NPZs for P0, Omni, DS, boundary DS and near ERP were float32 `[N,138,3]`, finite, with channel 2 either all ones (P0) or binary masks (others). This does not validate all external BEDLAM labels.

Representative P0/DS/near-ERP validation samples were loaded without model construction; image/target shapes, NumPy-versus-Torch affine agreement, and inverse restoration were checked. Maximum inverse errors were approximately `1.38e-5`, `1.62e-5`, and `2.34e-5` original-image pixels, respectively. A separate rectangular rotated affine round trip was within `1.43e-14` pixels; that does not certify non-square use of the full loader/model. No source data or production code was rewritten.

## 9. Follow-up candidate contract verification

Following the initial archaeology, the user confirmed that **no formal training NPZ has been selected**. This follow-up therefore audits all five available registered candidate datasets, without declaring one the formal training corpus. Baseline remains `8553ccd2d3117755e595ae97c8522c303bd05ceb`. Checks below were executed on 2026-09-28; local artifacts and their hashes are a separate evidence layer from the commit.

### Available data and appropriate use

The registry contains 137 training entries and 10 validation entries. From the repository working directory, only the five tiny dataset pairs below resolve to existing annotation files: 5/137 training entries and 5/10 validation entries. No other formal NPZ was found in the inspected BEDLAM directory. This is a bounded local inventory, not a claim about all disks or future data.

| Dataset registry name | Train / val samples | Selected landmark observations, both splits | Third-field producer | Suitable verification use |
|---|---:|---:|---|---|
| `p0-dressed-tiny` | 12 / 4 | 2208 | Builder appends 1 to every point | Perspective crop and full-body target contract |
| `omni-multiperson-tiny` | 4 / 2 | 493 | Source visibility ∩ square crop ∩ finite xy | Panorama-source ERP, multiple people/cameras |
| `ds-multiperson-tiny` | 2 / 2 | 324 | Same explicit binary selection | DS warp/ERP roll and masks |
| `ds-multiperson-boundary-tiny` | 2 / 2 | 233 | Same explicit binary selection | DS boundary-mask cases |
| `omni-near-erp-tiny` | 2 / 1 | 209 | Same explicit binary selection | Actual Omni fisheye→ERP, near people |

Total: **33 sample rows, 4554 landmark observations, 3467 selected observations**. These are repeated views/frames of limited synthetic content, not 33 independent subjects. All five are suitable as a regression fixture set for their stated contracts; none demonstrates a diverse, independent formal training/validation split. The near-ERP person holdout still reuses the same underlying actor/motion source.

### Checks performed on every sample

For all ten NPZs, loaded with `allow_pickle=False`, checked sample counts, unique image names within each split, exact `[N,138,3]` float32 target layout, finite targets/centers/scales, `[N,2]` centers, scalar positive scales, binary third fields, nonempty selection and readable images. Then independently reconstructed the adapter output from its saved source:

- For P0: source render-frame selection, projected xy, all-one third field, mask extrema and 1.2-expanded inclusive-pixel bbox; compared decoded RGB with the source frame.
- For ERP candidates: matched person/camera IDs and manifest split indices; applied the saved longitude shift to source xy and full decoded RGB; derived bbox from the rolled source instance mask; recomputed `visible & within_crop & finite` at every row index. Checked the manifest's selected-point counts.
- Compared expected xy/third-field arrays with NPZ arrays **exactly**. Every row matched, without a point permutation. Decoded output images matched source images after the specified copy/roll **pixel for pixel**. Bbox centers matched exactly; side differences were at most `2.90e-5` pixels from stored float32 scale.
- Ran every sample through the actual `DatasetKeypoints` path with `is_train=False` after selecting the NPZ split. Verified unchanged original targets/third field, independently evaluated normalized xy as `(original_xy-center)/(200*scale)`, restored xy with the inverse expression, and checked the actual BGR→RGB/mean/std input normalization. This explicitly avoids relying on the ignored `use_augmentation` flag.
- Checked each NPZ's SHA-256 before and after its audit; unchanged. Random augmentation and depth-derived visibility truth were not independently revalidated in this follow-up.

| Candidate | Max normalized crop discrepancy | Max restored original-pixel discrepancy | Selected outside 256-square crop | Selected outside ViT's retained x interval |
|---|---:|---:|---:|---:|
| P0 dressed | `6.98e-8` | `3.24e-5` | 0 | 2 |
| Omni multiperson | `2.17e-7` | `4.23e-5` | 0 | 12 |
| DS multiperson | `1.39e-7` | `7.61e-5` | 0 | 8 |
| DS boundary | `1.95e-7` | `1.08e-4` | 0 | 0 |
| Omni near ERP | `1.13e-7` | `1.11e-4` | 0 | 7 |

The inverse here uses an independently evaluated center/side formula and float32 stored scale, rather than the returned affine inverse used by the initial spot check; error magnitudes need not be identical. Both results are small numerical discrepancies. The ViT interval check is `-.375 <= normalized_x < .375`, corresponding to columns `32:224` in the square crop. The 29 selected observations outside that interval make the fixed/amodal-versus-visible-input policy concrete (R03); they are not evidence of an incorrect output coordinate frame.

### Source motion → body → fixed 138 rows

Inspected original `rp_aaron_posed_009/1095/motion_seq.npz`: poses `[192,165]`, with source betas/orientation/translation. Compared all selected canonical pose slices, betas and translations against that source for both 16-frame P0 and 39-frame temporal geometry. Maximum absolute parameter discrepancy was below `5.88e-8` (a numeric discrepancy across fields with different physical units, not a distance metric).

Instantiated the existing local male SMPL-X model using the generator's options (`num_betas=10`, full hand axis-angle, flat hand mean) and ran a **CPU body-model forward only**, without gradients. Compared every vertex in all 39 frames and the 16-frame subset: maximum discrepancy `2.384185791015625e-7` m; saved faces equal the model's faces exactly. This extends the initial audit, which did not reconstruct the body model. No DenseKP network was instantiated or trained.

Independently recomputed every surface row using its nonzero vertex supports and double-precision sums, instead of the generator's batched einsum. Against saved `[T,138,3]` arrays, maximum discrepancy was `1.475e-7` m for P0 and `1.548e-7` m for the temporal geometry. Also compared saved clothing vertices/faces with the original clothing NPZ selections: exact after the producer's float32 conversion, for all 16 and 39 frames. Clothing-index×5 agrees with the saved local motion indices.

This establishes **artifact-level row-order consistency for the inspected synthetic chain**. It does not identify anatomical names for each row, prove the historical DenseKP checkpoint used this operator, or establish correspondence with any other BEDLAM point set. Those remain separate gates.

### Body points → camera annotations

Computed projection formulas from saved camera/object matrices and person placement, without calling the corresponding camera projection helper under audit. Compared all point rows, not just selected/visible rows:

| Chain | Coverage | Maximum absolute xy discrepancy |
|---|---|---:|
| P0 body/object/camera → perspective xy | 16 frames × 138 | `3.85e-5` px |
| Hires multiperson placement → source ERP | 2 persons × 3 cameras × 138 | `6.11e-5` px |
| Same placement → Omni fisheye | 2 persons × 3 cameras × 138 | `1.22e-4` px |
| Near multiperson placement → source ERP | 3 persons × 1 camera × 138 | `6.08e-5` px |
| Near placement → Omni fisheye | 3 persons × 1 camera × 138 | `1.22e-4` px |
| Near placement → final 5760×2880 ERP | 3 persons × 138 | `2.22e-4` px |
| DS source roll/resolution transport → final ERP | Pilot and boundary, both cameras, both people | 0 px |

For multiperson projection, the inspected 39-frame geometry is the candidate input used for reconstruction; saved `source_manifest.json` records placements/cameras but not a geometry hash. Numerical agreement supports this lineage, while the missing explicit hash remains a provenance limitation. All tested hires/near landmark rays were inside the Omni model domain; C07's invalid-domain bug was **not observed in these artifacts**. DS checks above cover point transport through the recorded ERP coordinate conventions; they do not independently certify the camera model's boundary domain or re-render image warps.

The P0 saved visibility state counts over all 2208 targets are: 861 visible, 1331 occluded, 16 outside the source image, zero behind-camera. Its NPZ selects all 2208 by explicit producer policy. A point outside the source image can still be inside the expanded, padded crop, so this is consistent with the zero outside-crop count above. Treat this as all-point supervision; do not label its channel 2 as measured visibility. For other candidates, agreement with saved visibility-derived flags does not prove that the underlying depth tolerance is physically correct.

### Original BEDLAM RGB route: concrete missing inputs

Rechecked `outputs/audits/bedlam_tiny_mapping/frame_manifest.csv` against `be_seq.csv`, the actual camera CSVs and image paths. All **186 person-frame records**, referring to **24 PNGs**, **20 camera files**, and **157 distinct motion paths**, match the saved sequence/person/start-frame/placement/camera-row conventions; every listed PNG exists. This validates the recorded local mapping against these CSVs, not the external producer's geometric convention.

None of those 157 required motion paths exists beneath the inspected `/home/yy/datasets/BEDLAM/motion_data/tiny` root. The available `rp_aaron_posed_009/1095` motion is not among them. It supports the synthetic chain above but cannot supply poses for these original BEDLAM people/images. No generated DenseKP annotation NPZ for this original-image set was found in the inspected paths.

Consequently, the original-image path still needs its matching motion files **and** the tracked reconstruction adapter described in section 1. It is not ready to become a formal NPZ merely by reusing synthetic aaron geometry or assuming an index correspondence.

### Frozen annotation/model identifiers

Paths below are relative to repository root unless stated otherwise. Hashes identify inspected files, not their semantic correctness. These files remain local/ignored assets.

| Dataset directory | `train.npz` SHA-256 | `val.npz` SHA-256 |
|---|---|---|
| `datasets/p0_dressed_tiny` | `b1234118c703107a5447543b3622b6fccdcd180a59efb7d8d7eb102b1529d2f7` | `9dd9ef3345b69247443282eb1d035d9db10dafae0b296cbcd0af82c5ea5aa179` |
| `datasets/omni_multiperson_tiny` | `17d164054c0376d8e4d8185a59c9b8eb2c48d1211f6e93603bce626377564dd8` | `7d70ce57d2a63c9a3a97c957d5a27bf9a4d544a44dcbb372bb8bfed34f153445` |
| `datasets/ds_multiperson_tiny` | `e134e919e9ea1726e5e559138471f10fd5464b68c1dea07a0656060f31fd80ba` | `a8b5401de9295f442ed897dea28217a46afeb74f869c1ef623ab1fb8d5ff9311` |
| `datasets/ds_multiperson_boundary_tiny` | `be43492b6e60e1e8a4df9682fe243f229a65e43f2575f4db7cead548080b05b2` | `ef5a82c409f435a2aecd524a0c4ced7036f610cf4f4e13049777a5274d5801c5` |
| `datasets/omni_near_erp_tiny` | `988785c6c583ce36854935823495443be3e4e852a5461e9e23f58b5aae674e11` | `1792df502d7a1066461247f82a3b9636d212737617c1fc96a00d7ffc649dbbc6` |

- Original motion `/home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz`: `01ce870d62408c66c640148b35429fb14ce9e6ef6f4f094e099a13b5d416e184`.
- Body model `/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx/SMPLX_MALE.npz`: `adeecd0cd798da7cb6dcff0c05d0b49074f4558e8b811bec0ac9138be6635b17`.
- P0 `geometry/vertices.npy`: `b55a04869b754648bc5a5c6ee4871d214a7108ec545726ad7d319a67dccf51dd`; `geometry/densekp_3d.npy`: `0d223960d1fc4a0149f0acf9f6d55feabb4795d865c4e343cd70ef4267454c27`.
- Temporal `geometry/vertices.npy`: `60989503da358bde9a26c2adaa3d1f132c2f590415163587c6b8b08177d1aff2`; `geometry/densekp_3d.npy`: `7c6ef6a1a875a567fdd35eff88f1c347b3d27337c1f5121b4a9d2207a1b035da`.
- Both `geometry/faces.npy`: `c08e8e7e0c8ee38f4409fffe169556f510e194c7eae9f5ec2ca12959902c76e4`. Surface-operator hash is recorded in section 2.

### Contract decision

Accept these five pairs as **verified local adapter/coordinate fixtures under disabled augmentation**. Do not promote any pair to a formal training corpus or independent benchmark based on these results. Local synthetic row order, producer-specific third-field semantics and saved-coordinate transport are now supported by full-sample evidence. Historical checkpoint semantic parity, formal data selection/diversity, raw BEDLAM reconstruction, augmentation policy and depth visibility truth remain unresolved. The later external inspection below establishes how third values are transported and excluded from matching/fitting; it does not establish a calibrated confidence meaning.

## 10. External deployment contract and numeric comparison

External root `PANO=/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch`; HEAD `830dc51cce7467a99278a72d33ae3d9d8fffe5c2`, working tree inspected 2026-09-28. Existing changes include fitting/sequence adapters; this section describes the inspected working tree, not a claim that every statement holds at clean HEAD or an arbitrary exported deployment snapshot.

### Actual image and coordinate chain

1. `PANO/src/densekp_delivery/pipeline.py::_get_or_build_erp` calls `select_camera_params`, constructs DS LUTs and remaps the two BGR fish-eye rasters into **normal ERP**, default `W=5760,H=2880`. Supported calibration groups are `20260311` and `20260320`; config/sequence CLI default to `20260311`. Training `DS_20260320` matches only the latter group. Parameters and remap masks must come from the selected capture; output coordinate selection does not select input orientation.
2. `build_detection_rois` uses the intersection of per-camera lens-valid ERP masks, with a default 160-pixel context margin and periodic horizontal extraction. ROI image `[h_roi,w_roi,3]` retains other people/background. `crop_erp_roi` joins longitude segments when a ROI crosses the seam. Detector bboxes are local `[x1,y1,x2,y2]`, not center/scale and not rotated-ERP bboxes.
3. `ORTRunner::_build_densekp_batches` / `_run_detected_batch` compute center `(xy1+xy2)/2`, scale `(xy2-xy1)/200`, square side `max(width,height)`, no automatic 1.2 expansion. An OpenCV affine produces `256×256`, constant-zero padding, BGR→RGB, CHW float32 and ImageNet normalization. Compare training tiny bboxes: derived from rendered person masks then expanded 1.2, so transform agreement does not imply bbox-distribution agreement.
4. The inspected export wrapper slices `img[:,:,:,32:-32]` internally. Input remains `[B,3,256,256]`; output remains `[B,138,3]` in the full **256-square** normalization. TRT explicitly requires B=1 and float32 input/output buffers, even with FP16 internal inference. ORT batching retains an item/person restoration key.
5. Runner decode applies `A^-1((pred_xy+.5)*256)` to yield ROI-local pixels. No extra 32-pixel x offset is added. `restore_points_to_full_erp` then computes `x=(x_local+roi.x_start)%W`, `y=y_local+roi.y_start`. Bboxes retain unwrapped endpoints plus a seam flag. Third values are copied as `raw_scores`, unaffected by the coordinate transforms.
6. `match_all_persons` checks finite/in-bounds coordinates and the two camera lens masks, then compares same-ID rays with median angular error. Default eligibility: at least 8 valid points and median angle ≤12 degrees; exact bipartite assignment gives one-to-one pairs. Neither DenseKP third values nor an anatomical visibility estimate enters this cost. Geometric eligibility is not proof of a correct identity match.
7. For fitting/output rotation, `rotate_erp_points` maps xy→unit ray, applies `R=[[1,0,0],[0,0,-1],[0,1,0]]`, and maps back to ERP_ROTATE xy. `matched_persons_from_full_erp` keeps bboxes in **normal ERP**, but pairs in **ERP_ROTATE**, explicitly carrying point IDs `0..137`. `densekp_runtime_export` selects rows with `valid` and exports targets plus IDs to C++; the regressor maps 204 supporting vertices to the 138 semantic rows. Selected targets receive global normalization `1/valid_count`; residual-dependent IRLS weights in C++ are distinct from network third values.
8. `correspondence_output.py` emits fitted/optimized pairs by default, optionally raw DenseKP pairs, in requested ERP_ROTATE/ERP/fisheye space. It derives output validity from base validity plus transformed coordinates, and a separate CPW mask from image masks. These are still not point confidence or surface-occlusion masks.

The formal sequence entry receives **already remapped normal ERP** rather than calling the fisheye conversion itself. Its TRACK_GUIDED branch first requires `track.optimized.success`, but gets crop geometry from `track.last_person.pairs` and `.valid`: previous **matched observation** in ERP_ROTATE → inverse sphere rotation → `build_guided_crop_spec` minimum circular longitude interval → expanded region (working-tree defaults margin ratio .30, minimum margin 32 pixels) → `run_bboxes` → the same square affine/normalization/decode. It explicitly avoids deriving the bbox from possibly collapsed optimized projections. These crops have a different distribution from detector boxes and GT instance-mask boxes. FULL/TRACK_GUIDED scheduling, identity gating and recovery remain separate components.

### Third-field and occlusion decision

The runner preserves the third prediction as `raw_scores`; the v2 pair JSON and C++ observation export do not include it as a weight. `tools/CameraHMR/core/densekp_trainer.py` interprets it as log sigma only in the `with_var` objective. The older `densekp_io.py` calls `keypoint_value` a heatmap value but computes validity from xy bounds/finite checks, not its magnitude. Naming/comments are not a calibrated confidence specification.

The production geometric `valid` can include self-occluded or mutually occluded anatomical points whose predicted xy falls inside the lenses. The current multiperson NPZ instead sets channel 2 to `visible & inside_crop & finite`, suppressing hidden-point direct supervision. This is an objective choice with a deployment consequence (R10), not evidence that visibility masks themselves are malformed. For the requested occlusion robustness, the recommended new data contract separates **projection/lens validity**, **surface visibility**, **crop inclusion**, **supervision eligibility**, and **inference raw third output**. With reliable synthetic 3D truth, supervise occluded in-domain semantic landmarks as an explicit amodal target and report them separately; do not treat all off-lens/off-crop points as visible or silently turn every mask into ones. No NPZ or loss was changed here.

### Numeric checks performed without network execution

CPU scratch script `/tmp/densekp_pano_parity_20260928.py`, report `/tmp/densekp_pano_parity_20260928.json`, using the previously recorded scientific environment:

| Check | Result | Scope |
|---|---|---|
| All 33 candidate images, training no-augmentation loader vs actual ORT crop/normalization helper | Maximum normalized tensor difference `2.384185791015625e-7` | Same centers/scales; proves crop transport, not detector bbox quality or network equivalence |
| Training transformed GT through actual ORT inverse helper | Maximum original-image error `1.52587890625e-5` pixels | Synthetic network outputs equal to transformed GT, not model predictions |
| Local `R[138,10475]` vs external NumPy asset | Arrays and file SHA256 exactly equal | SHA256 `9187dd25d7024eb47f052d7df08e81ae25cf622a62e92325cba40069efc4a9f4` |
| External C++ `R[138,204]` scattered by its 204 vertex IDs vs local `R` | Exactly equal | Confirms all 138 row/support identities across these assets; not historical checkpoint training provenance |
| 20,000 seeded unit rays per camera, DS `20260320` | Parameters and model-domain masks exactly equal; maximum pixel discrepancy `<1.221e-4` | External projection casts to float32; local retains float64 |
| Two 576×288 ERP→DS LUT comparisons | Every x/y element exactly equal | Same calibration/extrinsics; shared domain policy is not an independent physical calibration proof |
| Four seam-crossing points, full normal ERP→ERP_ROTATE→normal ERP | Maximum periodic-coordinate error `1.220703125e-4` pixels | Away from exact poles; not an exhaustive pole-boundary test |

No detector, DenseKP checkpoint, ONNX session, TensorRT engine, GPU fitting or rendering was run. Export metadata names a historical checkpoint but is not proof that a future runtime loads that artifact. Full model/backend accuracy and real capture distribution remain verification gates.

Selected external source SHA256 values at inspection:

| PANO-relative file | SHA256 |
|---|---|
| `src/densekp_delivery/runners/ort_runner.py` | `386078730006c0335d464b87cefc555d84318f2baee2a56eaef2fc95658848de` |
| `src/densekp_delivery/pipeline.py` | `298c239453c2b401091b43865f5a6e1fd2f463ed034b8cf95ee4e39ebf5d721a` |
| `src/densekp_delivery/camera/ds_camera_params.py` | `ba4a2175b50e1737e96f54fefbd41d9dac50568200326c3f65c99739efee9308` |
| `src/lib_pano/fitting_io/densekp_delivery_boundary.py` (dirty) | `13fd1ca694e1311c61b907885309a8a9aa9a1432f711e0d671d560dcd0a16a76` |
| `src/lib_pano/human_fitting/sequence_orchestrator.py` (dirty) | `444c19d56d0ba744f70a94484807af2be8cb7c00493f269a8ba9ca532e1c4863` |
| `src/script/human_fitting/run_densekp_track_guided_sequence.py` (dirty) | `13e1b538cd2810ff9655bedcc5e1e7bf5f0e292c1ad3d1df0a8f6762c43c88a4` |
| `tools/CameraHMR/deploy/densekp_onnx/export_densekp_onnx.py` (local dependency) | `1f004b93ac879573f27c3da21aaed8a9d0287e6fc7922105b5aa372040924c24` |

## 11. Target scene fixed: 20260320 riding dataset3

The user identified **20260320-riding-dataset3** as the difficult real target. The local directory spelling is `20260320_riding_dataset3`. This resolves the camera-choice question in section 10: primary target **20260320/new**; **20260311/old** is a separate capture/calibration domain. In `PANO/data_process/Output/config.json`, the full target path is under key `dataset4` with `camera: new`; key `dataset3` points to riding dataset2. Do not select the wrong capture by short-key coincidence.

### Distinct calibration contracts

`PANO/src/densekp_delivery/camera/fisheye_equirect.py::select_camera_params` returns `DS_PARAMS, DS_PARAMS` for 20260311 and `DS_PARAMS_CAM2, DS_PARAMS_CAM4` for 20260320. Source constants in `ds_camera_params.py`:

| Group / camera | xi | alpha | fx | fy | cx | cy |
|---|---:|---:|---:|---:|---:|---:|
| 20260311 cam2/cam4 | 0.02541467 | 0.58707823 | 819.659819122500 | 819.389361330000 | 1455.78851018250 | 1435.04276513250 |
| 20260320 cam2 | -0.12464126 | 0.56540615 | 681.49808088 | 682.11759351 | 1445.73197064 | 1444.25576211 |
| 20260320 cam4 | -0.15331803 | 0.55745717 | 656.78496421 | 655.70192322 | 1440.93901552 | 1443.33089179 |

These are verified code selections, not a new physical calibration measurement. Target training camera constants already match the two 20260320 rows. Keep separate group/camera keys for LUTs and masks. The CLI's 20260311 default must be overridden for the target scene.

### Concrete local target assets and evidence limits

Let `B=PANO/data_process/tmp/production_regression_20260320_dataset3_multiperson`.

- `B/prepare_selected.py` reads `Circular_Video/01_send/20260320_riding/dataset3/VID_20260317_112543_00_013.0.mp4` as cam2 and `.1.mp4` as cam4, seeks frame 4325, explicitly selects `20260320`, and remaps 2880×2880 fisheyes into 5760×2880 normal ERP through frame 4575. Its imported `dist/.../vendor/densekp_delivery` camera parameter and remap source files are byte-identical to the corresponding audited `src/densekp_delivery` files.
- `B/prepared/preparation_summary.json` records group 20260320, 251 frames, no missing IDs, and the same camera mapping. Directory inspection independently found all **251 PNGs in each of `prepared/erp/cam2` and `cam4`**. This check did not rerun decoding/remapping or inspect every image's content.
- The preparation script builds lens masks from the first selected frame and derives ERP masks from those masks and the selected LUTs. It writes `prepared/fisheye_lens_masks_20260320.npz` and `prepared/valid_masks_20260320.npz`. The target ERP mask file contains two boolean `[2880,5760]` arrays; SHA256 `ecdd908299a6063cb8437469309fc20f6b7a0cc2887eff35b6809f7cdd7d59e1`.
- Training's current `assets_external/real_erp_valid_masks.npz` is byte-identical to `PANO/data_process/tmp/densekp_phaseA_erp_input/valid_masks.npz`, SHA256 `f13d0269734e17683ce3d19339a78293bf6fa7b5627a4dd2844481109ff2c37a`. Compared to the target window masks, cam2 differs at **7,834** pixels and cam4 at **13,074** pixels. Identical calibration does not guarantee identical image-derived masks. These differences require deliberate asset selection, not a claim of incorrect projection.
- `PANO/artifacts/human_fitting_delivery/examples/20260320-dataset3/README.md` explicitly selects `--camera-group 20260320` but describes **651–659 as a single-person example**. Its mask file SHA256 is `744d1651743503f8cb5307af3ab06122ea07c88e9029d51cfea9347c54e3695d`, also different from the local training mask. Use it as a separate control with its own mask provenance.
- `B/frames_to_review.json` has **102 flagged frames**. Useful initial review candidates: 4461/4474 (`LOW_IDENTITY_MARGIN`), 4472 (`NEW`, `RECONNECT`, `PERSON_COUNT_CHANGE`, ten pipeline output persons), 4480 (`RECOVERY`, no output person), 4560/4573 (`ID_SWITCH_SUSPECT`). These stored algorithmic labels/counts are **not manually established error types or real-person counts**. Likewise, the existing sequence's fitting success or accepted identity gates do not establish DenseKP accuracy.
- `B/temporal_sync_audit.json` compares each video's decoded images to historical same-index JPEGs at five positions. This supports that artifact's frame-index convention; it does not independently prove physical cross-camera synchronization. Preserve that distinction for stereo ground-truth evaluation.

Recommended use under the user's DenseKP-only scope: reserve independent validation/test clips, freeze existing target-person boxes, visually classify their raw predictions and annotate person identity/landmark evidence before tuning. Evaluate within-crop wrong-person landmarks and distortion/occlusion localization errors on the same boxes for all checkpoints. Missing boxes are outside the DenseKP denominator; DenseKP failures on supplied boxes remain in it, even if downstream matching rejects them. Matching/fitting output is optional contextual evidence, not ground truth or a component to improve here. No target frames, masks, run outputs, NPZ labels or production code were modified, and no inference was run in this follow-up.

## 12. Supplied-box real-scene review performed

### Artifact, scope and reconstruction

Read `B/standalone_sequence/sequence_result.json` from section 11; SHA256 **`a3b7bf1558f5adc1626d6d614f0fc07a3205be200bae105d2e045d9aee43baaf`**. It records TRT inference, relative engine path `artifacts/densekp/densekp_b1_256x256_fp16.plan`, detector backend `rtmlib_yolox_trt`, and the prepared target ERP root. These are recorded run metadata, not a verified checksum of the historical loaded engine. No inference, detector tuning or training was performed in this review.

All **251 frame records**, comprising 104 FULL_DETECTION, 113 TRACK_GUIDED and 34 RECOVERY frames, were structurally checked. The **775 retained person-pair observations** each have `raw_pairs[138,4]`, IDs exactly `0..137`, and no non-finite raw coordinates. Of these, 435 observations come from FULL/RECOVERY and 340 from GUIDED. Finite output is not proof of accurate localization. `raw_pairs` are pre-fitting predictions **after pair matching and ERP_ROTATE conversion**, not the complete pre-matching network output.

For visual review, loaded normal-ERP source images and mapped `raw_pairs` back with `rotate_erp_points(..., R.T)`. FULL/RECOVERY boxes are joined by `person_id` to `delivery.matched_bboxes`; `_full_delivery` builds those boxes in the same match order as `matched_persons_from_full_erp`, and `matched_person_tracks.py` preserves that person ID when serializing `raw_pairs`. Built the two component ROIs from the matching target masks: `(x_start,width,y_start,height)=(118,2624,0,2880)` and `(3019,2624,0,2880)`. All visually reviewed FULL/RECOVERY boxes map to a unique component ROI. Used the actual ORT affine helper to recreate 256-square image crops, including ROI-local padding, then overlaid the restored points. Yellow lines at x=32/224 show the retained ViT width; point color denotes semantic row index only. All 138 rows were shown, including pair-invalid rows.

The guided panels instead show the recorded proposal for the destination track. The current caller can assign another source crop's candidate to that track without updating `row.crop_cam2/cam4`; the serialized proposal is therefore **not guaranteed to be the crop that produced the assigned prediction**. These panels support image-space inspection but cannot establish exact guided-input errors without source-candidate provenance (R13). A single candidate/track, as in 4560, avoids competing-track reassignment, but still has no saved input-tensor checksum. Also, inverse sphere rotation cannot recover information discarded by earlier modulo/clipping for out-of-range predictions; these are not exact pre-rotation raw tensors.

### Review coverage

| Frame | Mode | Retained pairs visually inspected | Main purpose |
|---|---|---:|---|
| 4331 | FULL | 3 | Cyclist control; occluded checkered-shirt person; distinguish pairing from single-view prediction |
| 4461 | GUIDED | 6 | Crowded proposals and ambiguous source-crop provenance |
| 4472 | FULL | 10 | Dense crowd, small/partly visible people, umbrella, visor and motorbike |
| 4473 | RECOVERY/FULL | 9 | Adjacent-frame comparison with actual saved detection boxes |
| 4474 | GUIDED | 9 | Proposal changes, partial/lens-boundary inputs; not a fixed-box temporal experiment |
| 4559 | FULL | 1 | Cyclist close to lens boundary before flagged frame |
| 4560 | GUIDED | 1 | Mostly road/black region with a narrow visible person strip |
| 4572 | RECOVERY/FULL | 6 | Small background people, partial bodies, close distractors |
| 4573 | RECOVERY/FULL | 4 | Adjacent-frame target/track changes and lens truncation |
| 4574 | FULL | 4 | Follow-up single-view raw predictions |

Total: **53 retained pair observations / 106 single-camera panels across 10 frames**, not 53 unique people. Also inspected metadata for 4480 and 4561, where no person pairs remain. Qualitative visual review is not dense ground truth, a measured error rate, or an independent benchmark; all frames belong to one previously used clip.

Scratch scientific figures and the reconstruction script are in `/tmp/densekp_real_review/` and `/tmp/densekp_real_review.py`; inventory and temporal checks are JSON files there. These are temporary analysis artifacts, not new repository source files. Key figures: `4331_0.jpg`, `4472_0.jpg`, `4472_4.jpg`, `4472_8.jpg`, `4473_4.jpg`, `4559_0.jpg`, `4560_0.jpg`, `4572_0.jpg`, `4573_0.jpg`, `4574_0.jpg`. They compare unannotated and raw-point crops side by side; existing `review_overlays/*_tracking.jpg` instead draw optimized points and were not used as DenseKP ground truth.

### What the current logs cannot establish

In the 138 FULL/RECOVERY records, the per-camera runner counts sum to **3,263 entries**, but only **870 single-camera predictions** survive as 435 paired observations in this sequence JSON. The remaining **2,393 entries are not represented there**. These are output-entry counts, not unique people or detector recall. In frame 4480 specifically, the saved counts are cam2=8, cam4=16, matched=0; the file therefore contains no raw predictions to inspect despite nonzero upstream results. Searching the selected regression directory found masks and final correspondence files, not a complete per-box pre-matching dump. Do not classify this frame as a detector miss or a proved DenseKP failure.

`analyze_regression.py` creates `ID_SWITCH_SUSPECT` from the angular change of **optimized**, valid-selected point-set medians, with changing selected rows. It is not a DenseKP error label. For 4559→4560 track 45, raw same-ID angular changes have medians 8.64°/4.71° over all 138 IDs, and 7.05°/3.47° over the 18 jointly valid IDs (cam2/cam4). These are temporal displacements, not localization errors: image/camera motion, changing crops and visibility remain involved. For 4572→4573 track 53, the visible target changes from the checkered-shirt pedestrian to the masked cyclist; this track-level event does not establish that either independently supplied crop was decoded onto the wrong person. Neither temporal comparison holds its input box fixed.

No manual 138-point GT or independently verified hidden-point geometry was available in the inspected run artifacts. Points lying on an umbrella, another person's silhouette or black pixels do not by themselves prove a semantic localization error: projected hidden points may legitimately overlap an occluder, and lens-excluded points have no visible image evidence. The concrete observations and training implications are in BUG_AUDIT's real-scene case review and TRAINING_READINESS's resulting priorities.
