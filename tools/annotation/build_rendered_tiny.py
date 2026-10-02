#!/usr/bin/env python3
"""Convert a completed single-person P0 render into a tiny DenseKP training set."""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from densekp.core.keypoint_supervision import legacy_weight_targets


def build(source: Path, output: Path) -> dict:
    rows = [json.loads(line) for line in (source / "manifest.jsonl").read_text().splitlines()]
    points = np.load(source / "annotations/densekp_2d.npy", allow_pickle=False)
    if not rows or points.shape != (len(rows), 138, 2) or not np.isfinite(points).all():
        raise ValueError(f"invalid render annotations: frames={len(rows)}, points={points.shape}")

    image_dir = output / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for index, row in enumerate(rows):
        if row["frame_id"] != index:
            raise ValueError(f"unexpected frame order at {index}: {row['frame_id']}")
        image = source / row["rgb_path"]
        mask = source / (row.get("person_mask_path") or row["mask_path"])
        with Image.open(image) as rgb, Image.open(mask) as mask_image:
            width, height = rgb.size
            segmentation = np.asarray(mask_image.convert("L")) >= 128
        if segmentation.shape != (height, width) or not segmentation.any():
            raise ValueError(f"empty or mismatched person mask: {mask}")
        ys, xs = np.where(segmentation)
        xmin, xmax = int(xs.min()), int(xs.max())
        ymin, ymax = int(ys.min()), int(ys.max())
        box_size = max(xmax - xmin + 1, ymax - ymin + 1) * 1.2
        output_name = f"{index:06d}.png"
        shutil.copy2(image, image_dir / output_name)
        samples.append({
            "imgname": output_name,
            "center": [(xmin + xmax) / 2, (ymin + ymax) / 2],
            "scale": box_size / 200,
            "proj_verts": legacy_weight_targets(points[index], np.ones(138, np.float32)),
        })

    # Every fourth frame is held out for a loader/validation smoke check.
    # Adjacent frames share a motion sequence; this is not an independent benchmark.
    splits = {"train": [i for i in range(len(samples)) if i % 4 != 3],
              "val": [i for i in range(len(samples)) if i % 4 == 3]}
    if not all(splits.values()):
        raise ValueError("need at least four frames for train/val split")
    for name, indices in splits.items():
        selected = [samples[i] for i in indices]
        np.savez(
            output / f"{name}.npz",
            imgname=np.asarray([sample["imgname"] for sample in selected]),
            center=np.asarray([sample["center"] for sample in selected], dtype=np.float32),
            scale=np.asarray([sample["scale"] for sample in selected], dtype=np.float32),
            proj_verts=np.stack([sample["proj_verts"] for sample in selected]).astype(np.float32),
        )
    report = {"source": str(source.resolve()), "frames": len(samples),
              "train_frames": splits["train"], "val_frames": splits["val"],
              "bbox": "person mask square, expanded 1.2x",
              "proj_verts": "rendered original pixels; homogeneous channel=1"}
    (output / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/baselines/p0_dressed_eye_background_fixed")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/p0_dressed_tiny")
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output), indent=2))


if __name__ == "__main__":
    main()
