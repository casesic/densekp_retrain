#!/usr/bin/env python3
"""Bounded CPU-only compact consumer acceptance: <=6 parents, four variants each."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import cv2
import numpy as np
from PIL import Image
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.keypoint_supervision import supervision_for_crop
from densekp_retrain.data.loading import load_box_dataset
from densekp_retrain.io.json import write_json
from tools.experiments.prepare_box_robustness import prepare_sized_derivatives


def inventory(root):
    return {str(p.relative_to(root)): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", type=Path, default=ROOT / "outputs/acceptance/ds_output_modes_20261008/attempt_4/compact")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cv2.setNumThreads(1)
    started = time.monotonic()
    cpu_started = time.process_time()
    source, out = args.compact.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    before = inventory(source)
    write_json(out / "source_inventory_before.json", before)
    forbidden = [p for p in source.rglob("*") if p.name in
                 ("source", "_temporary_ds_source", "fisheye.png") or p.suffix.lower() == ".exr"]
    assert not forbidden, forbidden
    original_path = source / "diagnostic_loader.npz"
    parents = []
    with np.load(original_path) as archive:
        for i, (camera, person_id) in enumerate(zip(archive["camera_id"], archive["person_id"])):
            camera, person_id = str(camera), int(person_id)
            box = next(row for row in json.loads((source / camera / "boxes.json").read_text())
                       if row["person_id"] == person_id)
            if box["bbox_xyxy_exclusive"] is None:
                continue
            parents.append(dict(
                image=str(source / camera / "normal_erp.png"), manifest=str(source / "report.json"),
                valid_mask=str(source / camera / "valid_mask.png"),
                person_mask=str(source / camera / f"person_{person_id}_mask.png"),
                mask_person_id=person_id, person_id=person_id, camera=camera,
                bbox_kind="visible_instance_tight", bbox_source=box["source"],
                given_bbox_xyxy=box["bbox_xyxy_exclusive"],
                loader_input=str(original_path), loader_row=i,
            ))
            if len(parents) == 6:
                break
    assert parents and any(p["given_bbox_xyxy"][0] > 1536 and p["given_bbox_xyxy"][1] > 768 for p in parents)
    variants = [dict(name="center_x", parameters=dict(dx=.03), crop_policy="deployment"),
                dict(name="center_y", parameters=dict(dy=-.02), crop_policy="historical_trial"),
                dict(name="scale", parameters=dict(factor=.9), crop_policy="deployment"),
                dict(name="single_edge", parameters=dict(edge=2, fraction=.15), crop_policy="historical_trial")]
    write_json(out / "config.json", dict(parents=parents, variants=variants,
        scope="Interface validation only; these are not approved training augmentation ranges",
        max_parents=6, max_candidates_per_parent=4, max_total_candidates=24,
        min_side_source_px=2, min_supervised_points=10, min_target_full_crop_pixels=16,
        shift_reference_expansion=1.2, boundary_policy="erp_reject_seam"))
    index = prepare_sized_derivatives(parents, variants, out / "consumer", min_side_px=2,
        min_points=10, min_target_crop_pixels=16, shift_reference_expansion=1.2,
        boundary_policy="erp_reject_seam")
    assert len(index["rows"]) <= 24
    assert {r["box_variant"] for r in index["rows"]} == {v["name"] for v in variants}
    dataset = load_box_dataset(out / "consumer/packed.npz")
    checks = []
    for i, row in enumerate(index["rows"]):
        item = dataset[i]
        image = cv2.imread(row["image"], cv2.IMREAD_COLOR)
        assert [image.shape[1], image.shape[0]] == row["image_size_wh"] == [5760, 2880]
        warped = cv2.warpAffine(image, item["_trans"], (256, 256), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
        np.testing.assert_array_equal(warped, item["img_disp"])
        Image.fromarray(warped[:, :, ::-1]).save(out / f"crop_{i:03d}.png")
        xy = item["proj_verts"][:, :2]
        assert xy.shape == (138, 2)
        # Independent homogeneous multiplication, with the historical float32 assignment.
        pixels = (np.c_[xy, np.ones(138)] @ item["_trans"].T).astype(np.float32)
        expected_xy = (pixels / np.array([256, 256]) - .5).astype(np.float32)
        np.testing.assert_array_equal(expected_xy, item["proj_verts_cropped"][:, :2])
        active = dataset.data["in_view"][i]
        supervised = supervision_for_crop(active, expected_xy)
        np.testing.assert_array_equal(supervised, item["proj_verts_cropped"][:, 2] > 0)
        restored = restore_crop_xy(expected_xy, item["_trans"], (256, 256))
        error = float(np.abs(restored - xy).max())
        tolerance = max(2e-4, image.shape[1] * np.finfo(np.float32).eps)
        assert error <= tolerance, (error, tolerance)
        # Analytic square crop affine, allowing OpenCV's float32 control-point rounding.
        center, side = dataset.center[i], np.float32(dataset.scale[i] * 200)
        expected_affine = np.array([[256 / side, 0, 128 - center[0] * 256 / side],
                                    [0, 256 / side, 128 - center[1] * 256 / side]])
        np.testing.assert_allclose(item["_trans"], expected_affine, atol=2e-4, rtol=1e-6)
        checks.append(dict(row=i, camera=row["camera"], person_id=row["person_id"],
            variant=row["box_variant"], crop_policy=row["crop_policy"],
            roundtrip_max_source_px=error, roundtrip_tolerance_source_px=tolerance,
            supervised_points=int(supervised.sum()),
            supervised_occluded_points=int((supervised & ~dataset.data["visible"][i]).sum()),
            affine=item["_trans"].tolist(), image_affine_exact=True, keypoint_affine_exact=True))
    after = inventory(source)
    assert before == after, "Source file size/mtime inventory changed"
    write_json(out / "source_inventory_after.json", after)
    assert sum(c["supervised_occluded_points"] for c in checks) > 0
    report = dict(status="passed", selected_parents=len(parents), proposed_boxes=len(parents)*len(variants),
        accepted_boxes=len(index["rows"]), rejected=index["excluded_rows"],
        packed_and_dataset_readback_rows=len(dataset), image_and_138_point_affine="exactly matched",
        source_file_size_mtime_unchanged=True, source_or_exr_or_fisheye_dependencies=[],
        total_supervised=sum(c["supervised_points"] for c in checks),
        total_supervised_occluded=sum(c["supervised_occluded_points"] for c in checks),
        max_roundtrip_source_px=max(c["roundtrip_max_source_px"] for c in checks),
        checks=checks, elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        bytes_before_report=sum(p.stat().st_size for p in out.rglob("*") if p.is_file()),
        cpu_threads=dict(opencv=1, omp="environment=1", openblas="environment=1"),
        render_count=0, model_count=0, inference_count=0, optimizer_updates=0,
        limitation="One existing static compact scene; no periodic crop, sequence or real detector validation")
    write_json(out / "acceptance.json", report)
    print(json.dumps({k:v for k,v in report.items() if k != "checks"}, indent=2))


if __name__ == "__main__":
    main()
