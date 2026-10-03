#!/usr/bin/env python3
"""通过 DatasetKeypoints 验证 tiny BEDLAM DenseKP NPZ、裁剪与可视化。"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from densekp.core.datasets.configuration import audit_cfg  # noqa: E402; historical re-export
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints  # noqa: E402
from densekp.core.crop_geometry import trans_points2d_parallel  # noqa: E402


DATASET_NAME = "bedlam-densekp-tiny-audit"


def draw_points(image: np.ndarray, points: np.ndarray, color, radius: int, label_every=0):
    canvas = image
    for point_id, point in enumerate(points):
        if not np.isfinite(point[:2]).all():
            continue
        xy = tuple(np.rint(point[:2]).astype(int))
        cv2.circle(canvas, xy, radius, color, -1, cv2.LINE_AA)
        if label_every and point_id % label_every == 0:
            cv2.putText(canvas, str(point_id), (xy[0] + 3, xy[1] - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
    return canvas


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs" / "bedlam_generator_audit"
    )
    args = parser.parse_args()

    image_root = args.dataset_root / "images"
    annotation = args.dataset_root / "annotations" / "bedlam_densekp_tiny.npz"
    debug_annotation = args.dataset_root / "annotations" / "bedlam_densekp_tiny_debug.npz"
    for path in (image_root, annotation, debug_annotation):
        if not path.exists():
            raise SystemExit(f"missing required input: {path}")

    dataset = DatasetKeypoints(
        audit_cfg("supervision_weight"), DATASET_NAME, use_augmentation=False, is_train=True,
        annotation_path=annotation, image_root=image_root,
    )

    original_dir = args.output / "original"
    cropped_dir = args.output / "cropped"
    original_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)
    reports = []

    with np.load(annotation, allow_pickle=False) as data, np.load(debug_annotation, allow_pickle=False) as debug:
        if data["proj_verts"].shape != (len(dataset), 138, 3):
            raise SystemExit(f"invalid proj_verts shape: {data['proj_verts'].shape}")
        if data["center"].shape != (len(dataset), 2) or data["scale"].shape != (len(dataset),):
            raise SystemExit("invalid center or scalar scale layout")

        for index in range(len(dataset)):
            item = dataset[index]
            original_xy = np.asarray(item["proj_verts"][:, :2])
            expected = trans_points2d_parallel(original_xy, np.asarray(item["_trans"])) / 256.0 - 0.5
            returned = np.asarray(item["proj_verts_cropped"][:, :2])
            error = np.abs(expected - returned)
            crop_xy = (returned + 0.5) * 256.0

            image = cv2.imread(item["imgname"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise SystemExit(f"cannot read {item['imgname']}")
            center = np.asarray(data["center"][index])
            bbox_size = float(data["scale"][index]) * 200.0
            half = bbox_size / 2.0
            cv2.rectangle(image,
                          tuple(np.rint(center - half).astype(int)),
                          tuple(np.rint(center + half).astype(int)),
                          (0, 255, 255), 2, cv2.LINE_AA)
            draw_points(image, debug["joints2d"][index], (255, 0, 0), 4)
            draw_points(image, original_xy, (0, 0, 255), 3, label_every=10)
            cv2.imwrite(str(original_dir / f"sample_{index:03d}.png"), image)

            crop = np.asarray(item["img_disp"]).copy()
            draw_points(crop, crop_xy, (0, 0, 255), 3, label_every=10)
            cv2.imwrite(str(cropped_dir / f"sample_{index:03d}.png"), crop)

            normalized_outside = np.any((returned < -0.5) | (returned > 0.5), axis=1)
            reports.append(
                {
                    "index": index,
                    "max_abs_error": float(error.max()),
                    "mean_abs_error": float(error.mean()),
                    "normalized_xy_min": returned.min(axis=0).tolist(),
                    "normalized_xy_max": returned.max(axis=0).tolist(),
                    "normalized_outside_count": int(normalized_outside.sum()),
                    "image_bounds_count": int(debug["densekp_in_image_count"][index]),
                    "bbox_bounds_count": int(debug["densekp_in_bbox_count"][index]),
                    "surface_operator_max_abs_error": float(
                        debug["surface_operator_max_abs_error"][index]
                    ),
                }
            )

    all_max = max(report["max_abs_error"] for report in reports)
    all_mean = float(np.mean([report["mean_abs_error"] for report in reports]))
    result = {"samples": reports, "overall_max_abs_error": all_max,
              "overall_mean_abs_error": all_mean, "augmentation": "disabled"}
    report_path = args.output / "verification.json"
    report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"verification report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
