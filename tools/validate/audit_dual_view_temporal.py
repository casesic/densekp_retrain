"""Bounded saved-evidence audit; no video decoding, model loading or production writes.

Reads one explicitly selected 251-frame JSON, saved masks, six existing PNGs,
two video stream headers (ffprobe), and small saved synthetic/asset arrays.
The output directory must not exist. External imports are CPU ROI geometry only.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from densekp_retrain.camera.domain import point_in_view


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def distribution(values):
    if not values:
        return {"n": 0}
    values = np.asarray(values, float)
    return dict(n=len(values), minimum=float(values.min()),
                median=float(np.median(values)), p90=float(np.percentile(values, 90)),
                maximum=float(values.max()))


def rectangle_pixels(box, width=5760, height=2880):
    """Conservative floor/ceil rectangle, periodic x; not a person segmentation."""
    x0, y0, x1, y1 = box
    xs = np.arange(int(np.floor(x0)), int(np.ceil(x1))) % width
    ys = np.arange(max(0, int(np.floor(y0))), min(height, int(np.ceil(y1))))
    return ys, xs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    external = args.external.resolve()
    base = external / "data_process/tmp/production_regression_20260320_dataset3_multiperson"
    sequence_path = base / "standalone_sequence/sequence_result.json"
    sequence = read(sequence_path)
    frames = sequence["frames"]
    assert [f["frame"] for f in frames] == list(range(4325, 4576))
    assert sequence["num_frames"] == 251
    selected = [4325, 4450, 4575]  # fixed start/middle/end, not selected by model error
    manifest = dict(
        schema="dual_view_temporal_audit_v1", status="audit_only_review_pending",
        scope=dict(video_frames_decoded=0, existing_pngs_decoded=6,
                   detector_calls=0, densekp_calls=0, renders=0, training_updates=0),
        selected_frames=selected, inputs=dict(sequence=str(sequence_path),
        preparation=str(base / "prepared/preparation_summary.json")),
        code_versions={}, video_metadata={}, limitations=[
            "Nominal times are index*1001/30000, not measured PTS or exposure synchronization.",
            "Saved matched detector boxes are a selected subset, not all detections or GT.",
            "Track-guided boxes are NOT detector observations; tracks are not verified identities.",
            "Mask coverage of a box is NOT body visibility or truncation ground truth.",
            "Synthetic 3D check uses a synthetic 0.18m baseline, not measured real extrinsics.",
        ])
    for name, repo in [("densekp_retrain", ROOT), ("pano_stitch", external)]:
        manifest["code_versions"][name] = subprocess.check_output(
            ["git", "-c", f"safe.directory={repo}", "-C", str(repo), "rev-parse", "HEAD"],
            text=True).strip()
    compared = {}
    for rel in ["densekp_delivery/overlap_roi.py", "densekp_delivery/camera/fisheye_equirect.py",
                "densekp_delivery/camera/erp_rotate.py"]:
        compared[rel] = ((external / "src" / rel).read_bytes() ==
                         (external / "dist/human_fitting_delivery/vendor" / rel).read_bytes())
    manifest["src_dist_byte_equality_selected_files"] = compared
    assert all(compared.values())
    for camera, suffix in [("cam2", 0), ("cam4", 1)]:
        video = external / ("Circular_Video/01_send/20260320_riding/dataset3/"
                            f"VID_20260317_112543_00_013.{suffix}.mp4")
        command = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                   "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,time_base,start_time,duration,nb_frames:format=start_time,duration",
                   "-of", "json", str(video)]
        manifest["video_metadata"][camera] = dict(path=str(video), command=command,
            metadata=json.loads(subprocess.check_output(command, text=True)))

    with np.load(base / "prepared/valid_masks_20260320.npz") as z:
        masks = {c: z[f"valid_masks_{c}"].astype(bool) for c in ("cam2", "cam4")}
    assert all(m.shape == (2880, 5760) for m in masks.values())
    sys.path.insert(0, str(external / "src"))
    from densekp_delivery.overlap_roi import build_detection_rois
    strict, detection, rois = build_detection_rois(masks["cam2"], masks["cam4"])
    manifest["roi"] = dict(mask_source=str(base / "prepared/valid_masks_20260320.npz"),
        mask_reference_frame=4325, strategy="component_bbox", margin_px=160,
        mask_pixels={c: int(m.sum()) for c, m in masks.items()},
        strict_same_raster_overlap_pixels=int(strict.sum()),
        detection_mask_pixels=int(detection.sum()), rectangles=[asdict(r) for r in rois],
        rectangle_pixel_fraction=sum(r.pixel_count for r in rois) / strict.size,
        note="Same-ray ROI proposal, not the common support of a finite-distance person.")

    rows, frame_rows = [], []
    for f in frames:
        fid, mode, info = f["frame"], f["mode"], f["delivery"]
        entry = dict(frame=fid, nominal_time_s=fid * 1001 / 30000,
                     camera_frame_indices={"cam2": fid, "cam4": fid},
                     measured_pts=None, sync_offset_s=None, mode=mode,
                     matched_people=len(f["persons"]),
                     detections={c: info.get(f"{c}_person_count") for c in masks},
                     images={c: str(base / f"prepared/erp/{c}/{fid:05d}.png") for c in masks})
        entry["images_exist"] = all(Path(p).is_file() for p in entry["images"].values())
        frame_rows.append(entry)
        candidates = []
        if mode == "TRACK_GUIDED":
            for crop in info.get("guided_crops", []):
                candidates.append((crop["camera"], crop["predicted_bbox_erp"],
                    "track_guided_prediction", crop["track_id"], None, crop["status"]))
        else:
            for index, pair in enumerate(info.get("matched_bboxes", [])):
                for camera in masks:
                    # Do not infer a global track identity from matched-bbox array order.
                    candidates.append((camera, pair[f"bbox_{camera}"],
                        "matched_detector_box", None, index, "matched_subset"))
        for camera, box, kind, track, index, status in candidates:
            b = np.asarray(box, float)
            assert b.shape == (4,) and np.isfinite(b).all() and np.all(b[2:] > b[:2])
            ys, xs = rectangle_pixels(b)
            frac = float(masks[camera][np.ix_(ys, xs)].mean()) if len(xs) and len(ys) else None
            rows.append(dict(frame=fid, camera=camera, mode=mode, kind=kind,
                track_id=track, match_index=index, status=status, bbox_xyxy=b.tolist(),
                width_px=float(b[2]-b[0]), height_px=float(b[3]-b[1]),
                max_side_px=float(max(b[2:]-b[:2])), box_mask_coverage=frac,
                crosses_valid_domain=frac is not None and 0 < frac < 1,
                crosses_erp_seam=bool(b[0] < 0 or b[2] > 5760)))
    assert all(f["images_exist"] for f in frame_rows)
    manifest["sequence"] = dict(frame_count=len(frames),
        mode_counts=dict(Counter(f["mode"] for f in frames)),
        matched_person_observations=sequence["matched_person_observations"],
        saved_track_count=sequence["unique_tracks"],
        nominal_start_s=frame_rows[0]["nominal_time_s"],
        nominal_end_s=frame_rows[-1]["nominal_time_s"],
        detector_backend=sequence["detector_backend"],
        detector_threshold=sequence["detector_diagnostics"]["detector_threshold"],
        saved_correspondence_space=sequence["correspondence_output"]["coord_space"],
        saved_correspondence_is_gt=False)
    stats = {}
    for mode in sorted(set(r["mode"] for r in rows)):
        for camera in masks:
            group = [r for r in rows if r["mode"] == mode and r["camera"] == camera]
            stats[f"{mode}/{camera}"] = dict(count=len(group),
                crosses_domain=sum(r["crosses_valid_domain"] for r in group),
                crosses_seam=sum(r["crosses_erp_seam"] for r in group),
                **{key: distribution([r[key] for r in group if r[key] is not None])
                   for key in ["width_px", "height_px", "max_side_px", "box_mask_coverage"]})
    manifest["box_statistics"] = stats
    # Consecutive guided records only: descriptive tracking change, never detector error.
    changes = []
    prior = {}
    for r in rows:
        if r["kind"] != "track_guided_prediction":
            continue
        key = (r["camera"], r["track_id"])
        previous = prior.get(key)
        if previous and r["frame"] == previous["frame"] + 1:
            a, b = np.array(previous["bbox_xyxy"]), np.array(r["bbox_xyxy"])
            delta = (b[:2]+b[2:]-a[:2]-a[2:])/2
            delta[0] = (delta[0]+2880) % 5760-2880
            changes.append(dict(frame=r["frame"], camera=r["camera"], track_id=r["track_id"],
                center_delta_over_previous_max_side=(delta/previous["max_side_px"]).tolist(),
                max_side_ratio=r["max_side_px"]/previous["max_side_px"]))
        prior[key] = r
    save(out / "frames.json", frame_rows)
    save(out / "boxes.json", rows)
    save(out / "guided_changes_not_detector_error.json", changes)

    # Six existing images only; source scale 1:5, no model crop claim.
    small_masks = {c: cv2.resize(m.astype(np.uint8), (1152, 576), interpolation=cv2.INTER_NEAREST)
                   for c, m in masks.items()}
    for fid in selected:
        panels = []
        for camera in masks:
            path = base / f"prepared/erp/{camera}/{fid:05d}.png"
            with Image.open(path) as source:
                assert source.size == (5760, 2880)
                rgb = np.array(source.convert("RGB").resize((1152, 576)))
            contours, _ = cv2.findContours(small_masks[camera], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (0, 255, 255), 2)
            panel = Image.new("RGB", (1152, 626))
            panel.paste(Image.fromarray(rgb), (0, 50))
            draw = ImageDraw.Draw(panel)
            mode = next(f["mode"] for f in frames if f["frame"] == fid)
            draw.text((8, 4), f"{camera} frame {fid} | {mode} | ERP 5760x2880, display 1:5", fill="white")
            draw.text((8, 23), "cyan=explicit valid boundary; yellow=ROI; red=saved matched/guided boxes (NOT GT)", fill="white")
            for roi in rois:
                for shift in [-5760, 0]:
                    x0, x1 = max(0, roi.x_start+shift), min(5760, roi.x_start+roi.width+shift)
                    if x1 > x0:
                        draw.rectangle((x0/5, 50+roi.y_start/5, x1/5, 50+(roi.y_start+roi.height)/5), outline="yellow", width=2)
            for r in rows:
                if r["frame"] != fid or r["camera"] != camera:
                    continue
                x0, y0, x1, y1 = np.array(r["bbox_xyxy"])/5
                draw.rectangle((x0, y0+50, x1, y1+50), outline="red", width=2)
                label = f"T{r['track_id']}" if r["track_id"] is not None else f"M{r['match_index']}"
                draw.text((x0, max(50, y0+38)), label, fill="white")
            panels.append(panel)
        sheet = Image.new("RGB", (2304, 626))
        for i, panel in enumerate(panels):
            sheet.paste(panel, (1152*i, 0))
        sheet.save(out / f"real_roi_{fid:05d}.png")

    synthetic = ROOT / "outputs/experiments/ds_source_resolution_20261008/C"
    with np.load(synthetic / "pair_annotations.npz") as z:
        world, ids = z["world_densekp"], z["person_ids"]
    by_camera, checks = {}, {}
    for camera in masks:
        with np.load(synthetic / camera / "annotations.npz") as z:
            a = {k: z[k] for k in ["xy", "in_view", "visible", "camera_to_world_erp"]}
        with Image.open(synthetic / camera / "valid_mask.png") as im:
            mask = np.array(im) > 0
        xy, _, _, _, inside = point_in_view(world.reshape(-1, 3), a["camera_to_world_erp"], camera, mask)
        xy, inside = xy.reshape(4, 138, 2), inside.reshape(4, 138)
        error = xy-a["xy"]
        error[..., 0] = (error[..., 0]+2880) % 5760-2880
        checks[camera] = dict(max_erp_difference_px=float(np.abs(error).max()),
                             in_view_mismatch=int((inside != a["in_view"]).sum()))
        assert checks[camera]["max_erp_difference_px"] < .001 and checks[camera]["in_view_mismatch"] == 0
        by_camera[camera] = a
    pairs = []
    for i, pid in enumerate(ids):
        a, b = by_camera["cam2"], by_camera["cam4"]
        common = a["in_view"][i] & b["in_view"][i]
        pairs.append(dict(person_id=int(pid),
            in_view={c: int(by_camera[c]["in_view"][i].sum()) for c in masks},
            common_in_view=int(common.sum()), common_semantic_ids=np.flatnonzero(common).tolist(),
            missing_semantic_ids={c: np.flatnonzero(~by_camera[c]["in_view"][i]).tolist() for c in masks},
            both_visible=int((common & a["visible"][i] & b["visible"][i]).sum()),
            one_visible=int((common & (a["visible"][i] ^ b["visible"][i])).sum()),
            both_occluded=int((common & ~a["visible"][i] & ~b["visible"][i]).sum())))
    manifest["synthetic_projected_support"] = dict(source=str(synthetic), baseline_m=.18,
        checks=checks, persons=pairs, crop_support_applied=False,
        note="Actual saved synthetic GT only; common domain before reference-crop intersection.")
    assets = ROOT / "outputs/experiments/cycling_ten_adults_candidates_20260930"
    manifest["existing_assets"] = dict(counts=read(assets / "character_manifest.json")["counts"],
        combination_counts={k: len(v) for k, v in read(assets / "combination_status.json").items() if isinstance(v, list)},
        static_pose_shapes={})
    for name in ["coco_292226_person_1330527", "coco_306800_person_225160"]:
        with np.load(ROOT / "outputs/experiments/cycling_smplx_20260929" / name / "canonical_pose.npz") as z:
            manifest["existing_assets"]["static_pose_shapes"][name] = {k: list(z[k].shape) for k in z.files}
    manifest["elapsed_s"] = time.monotonic()-started
    manifest["generated_bytes_before_audit_json"] = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    save(out / "audit.json", manifest)
    print(json.dumps(dict(output=str(out), seconds=manifest["elapsed_s"], boxes=len(rows),
                          synthetic_checks=checks, modes=manifest["sequence"]["mode_counts"])))


if __name__ == "__main__":
    main()
