#!/usr/bin/env python3
"""Convert a paired multi-person ERP pilot into per-person DenseKP NPZ samples."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from densekp.core.keypoint_supervision import SUPERVISION_POLICY, make_targets, points_in_mask


def roll_person(rgb: np.ndarray, mask: np.ndarray, xy: np.ndarray):
    """Place the person's shortest cyclic longitude interval near image center."""
    occupied = np.flatnonzero(mask.any(axis=0))
    if not len(occupied):
        raise ValueError("person mask is empty")
    width = mask.shape[1]
    gaps = np.diff(np.r_[occupied, occupied[0] + width])
    largest_gap = int(np.argmax(gaps))
    arc_start = int(occupied[(largest_gap + 1) % len(occupied)])
    arc_width = width - int(gaps[largest_gap]) + 1
    shift = int(round(width / 2 - (arc_start + arc_width / 2)))
    shifted = xy.copy()
    shifted[:, 0] = (shifted[:, 0] + shift) % width
    return np.roll(rgb, shift, axis=1), np.roll(mask, shift, axis=1), shifted, shift


def build(source: Path, output: Path, val_camera: str = "cam_right",
          val_person_id: int | None = None) -> dict:
    report = json.loads((source / "report.json").read_text())
    ids = [int(p["id"]) for p in report["people"]]
    image_dir = output / "images"
    output.mkdir(parents=True, exist_ok=False)
    image_dir.mkdir()
    samples = []
    for camera in report["results"]:
        camera_id = camera["camera"]
        camera_dir = source / camera_id
        with Image.open(source / "source" / f"{camera_id}_preview.png") as image:
            rgb = np.asarray(image.convert("RGB"))
        with np.load(camera_dir / "erp_annotations.npz") as data:
            if data["xy"].shape != (len(ids), 138, 2) or list(data["person_ids"]) != ids:
                raise ValueError(f"annotation layout mismatch for {camera_id}")
            points, visible = data["xy"].copy(), data["visible"].copy()
            in_view = data['valid'].astype(bool)
        # DS legacy 'valid' only checks image bounds. Its measured lens mask is essential.
        mask_path = camera_dir / 'erp_valid_mask.png'
        valid_mask = None
        if mask_path.exists():
            valid_mask = np.asarray(Image.open(mask_path).convert('L')) > 0
        elif report.get('valid_masks'):
            if report.get('fisheye_reference_root') and camera.get('reference_lens_coverage') != 1.0:
                raise ValueError('Regenerate DS source with erp_valid_mask.png to include its lens footprint')
            with np.load(report['valid_masks']) as archive:
                valid_mask = archive[f'valid_masks_{camera_id}'].astype(bool)
        elif 'Double Sphere' in report.get('camera_model', ''):
            raise ValueError('DS supervision requires an explicit lens-valid ERP mask')
        if valid_mask is not None:
            if valid_mask.shape != rgb.shape[:2]:
                raise ValueError(f'Lens mask and RGB dimensions differ for {camera_id}')
            in_view &= points_in_mask(points, valid_mask)
        visible &= in_view
        for person_index, person_id in enumerate(ids):
            with Image.open(camera_dir / f"erp_person_{person_id}_mask.png") as image:
                mask = np.asarray(image.convert("L")) >= 128
            if mask.shape != rgb.shape[:2] or mask.sum() < 100:
                raise ValueError(f"missing or tiny person mask: {camera_id}/{person_id}")
            rolled_rgb, rolled_mask, xy, shift = roll_person(rgb, mask, points[person_index])
            ys, xs = np.where(rolled_mask)
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            box_size = max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * 1.2
            target = make_targets(xy, in_view[person_index], [cx, cy], box_size / 200)
            valid = target[:, 2] > 0
            if valid.sum() < 10:
                raise ValueError(f"fewer than 10 supervised points: {camera_id}/{person_id}")
            filename = f"{camera_id}_person_{person_id}.png"
            Image.fromarray(rolled_rgb).save(image_dir / filename)
            samples.append({"imgname": filename, "center": [cx, cy], "scale": box_size / 200,
                            "proj_verts": target,
                            "in_view": in_view[person_index], "visible": visible[person_index],
                            "person_id": person_id, "camera_id": camera_id,
                            "visible_count": int((valid & visible[person_index]).sum()),
                            "occluded_count": int((valid & ~visible[person_index]).sum()),
                            "supervised_count": int(valid.sum()), "erp_roll_pixels": shift})
    if val_person_id is None:
        splits = {"train": [i for i, row in enumerate(samples) if row["camera_id"] != val_camera],
                  "val": [i for i, row in enumerate(samples) if row["camera_id"] == val_camera]}
    else:
        splits = {"train": [i for i, row in enumerate(samples) if row["person_id"] != val_person_id],
                  "val": [i for i, row in enumerate(samples) if row["person_id"] == val_person_id]}
    if not all(splits.values()):
        raise ValueError("train/val split requires samples on both sides")
    for name, indices in splits.items():
        selected = [samples[i] for i in indices]
        np.savez(output / f"{name}.npz",
                 imgname=np.asarray([s["imgname"] for s in selected]),
                 center=np.asarray([s["center"] for s in selected], dtype=np.float32),
                 scale=np.asarray([s["scale"] for s in selected], dtype=np.float32),
                 proj_verts=np.stack([s["proj_verts"] for s in selected]).astype(np.float32),
                 supervision_policy=np.array(SUPERVISION_POLICY),
                 in_view=np.stack([s['in_view'] for s in selected]),
                 visible=np.stack([s['visible'] for s in selected]),
                 person_id=np.asarray([s["person_id"] for s in selected], dtype=np.int32),
                 camera_id=np.asarray([s["camera_id"] for s in selected]))
    manifest = {"source": str(source.resolve()), "people": len(ids), "samples": samples,
                "val_camera": val_camera,
                "val_person_id": val_person_id,
                "train_indices": splits["train"], "val_indices": splits["val"],
                "note": "smoke-only split: one synthetic motion/scene shared across splits"}
    # The sample records contain NumPy arrays; keep the manifest compact.
    manifest['supervision_policy'] = SUPERVISION_POLICY
    manifest["samples"] = [{k: v for k, v in sample.items()
                            if k not in ('proj_verts', 'in_view', 'visible')} for sample in samples]
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/experiments/fisheye_multiperson_pilot")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/omni_multiperson_tiny")
    parser.add_argument("--val-camera", default="cam_right")
    parser.add_argument("--val-person-id", type=int,
                        help="for one-camera pilots, hold out one person (smoke only)")
    args = parser.parse_args()
    report = build(args.source, args.output, args.val_camera, args.val_person_id)
    print(json.dumps({"samples": len(report["samples"]), "train": len(report["train_indices"]),
                      "val": len(report["val_indices"]),
                      "visible_counts": [s["visible_count"] for s in report["samples"]]}, indent=2))


if __name__ == "__main__":
    main()
