#!/usr/bin/env python3
"""可视化真实 DenseKP NPZ 的原图坐标与无增强裁剪坐标。"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from densekp.core.datasets.utils import (  # noqa: E402
    expand_to_aspect_ratio,
    get_example_projverts,
    trans_points2d_parallel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("npz", type=Path)
    parser.add_argument("image_root", type=Path)
    parser.add_argument("--index", type=int, action="append", dest="indices")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--bbox-shape", type=float, nargs=2, metavar=("W", "H"))
    parser.add_argument(
        "--dataset-name",
        default="",
        help="与 DatasetKeypoints 一致；名称含 closeup/portrait 时顺时针旋转图像",
    )
    parser.add_argument("--label-every", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "annotation_audit",
    )
    return parser.parse_args()


def scalar_bbox_size(scale: np.ndarray, bbox_shape: tuple[float, float] | None) -> float:
    expanded = expand_to_aspect_ratio(np.asarray(scale) * 200.0, bbox_shape)
    return float(np.asarray(expanded).max())


def read_dataset_image(path: Path, dataset_name: str) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if not isinstance(image, np.ndarray):
        raise OSError(f"无法读取图像: {path}")
    if "closeup" in dataset_name or "portrait" in dataset_name:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def draw_overlay(
    image: np.ndarray,
    points: np.ndarray,
    label_every: int,
    bbox: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    canvas = image.copy()
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(
            canvas,
            (int(round(x0)), int(round(y0))),
            (int(round(x1)), int(round(y1))),
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    for point_id, (x, y) in enumerate(points):
        if not np.isfinite([x, y]).all():
            continue
        xy = (int(round(float(x))), int(round(float(y))))
        cv2.circle(canvas, xy, 3, (0, 0, 255), -1, cv2.LINE_AA)
        if label_every > 0 and point_id % label_every == 0:
            cv2.putText(
                canvas,
                str(point_id),
                (xy[0] + 3, xy[1] - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
    return canvas


def main() -> int:
    args = parse_args()
    bbox_shape = tuple(args.bbox_shape) if args.bbox_shape else None
    args.output.mkdir(parents=True, exist_ok=True)
    original_dir = args.output / "original"
    cropped_dir = args.output / "cropped"
    original_dir.mkdir(parents=True, exist_ok=True)
    cropped_dir.mkdir(parents=True, exist_ok=True)

    reports = []
    with np.load(args.npz, allow_pickle=True) as data:
        required = ("imgname", "center", "scale", "proj_verts")
        missing = [key for key in required if key not in data]
        if missing:
            raise SystemExit(f"NPZ 缺少字段: {missing}")
        n = len(data["imgname"])
        indices = args.indices if args.indices is not None else list(range(min(args.count, n)))

        for ordinal, index in enumerate(indices):
            if not 0 <= index < n:
                raise SystemExit(f"sample index {index} 超出 [0, {n - 1}]")
            relative_name = str(data["imgname"][index])
            image_path = Path(relative_name)
            if not image_path.is_absolute():
                image_path = args.image_root / image_path

            image = read_dataset_image(image_path, args.dataset_name)
            center = np.asarray(data["center"][index], dtype=np.float64)
            scale = np.asarray(data["scale"][index], dtype=np.float64)
            proj_verts = np.asarray(data["proj_verts"][index]).copy()
            bbox_size = scalar_bbox_size(scale, bbox_shape)
            half = bbox_size / 2.0
            bbox = (center[0] - half, center[1] - half, center[0] + half, center[1] + half)

            original = draw_overlay(
                image, proj_verts[:, :2], args.label_every, bbox=bbox
            )
            original_path = original_dir / f"sample_{ordinal:03d}.png"
            if not cv2.imwrite(str(original_path), original):
                raise OSError(f"无法写入: {original_path}")

            keypoints = np.zeros((44, 3), dtype=np.float32)
            _, crop_bgr, _, _, _, final_width, final_height, _, transformed, trans = (
                get_example_projverts(
                    image,
                    float(center[0]),
                    float(center[1]),
                    bbox_size,
                    bbox_size,
                    keypoints,
                    [],
                    proj_verts.copy(),
                    args.image_size,
                    args.image_size,
                    None,
                    None,
                    False,
                    None,
                    is_bgr=True,
                    use_skimage_antialias=False,
                    border_mode=cv2.BORDER_CONSTANT,
                    dataset=None,
                )
            )
            crop_xy = (transformed[:, :2] + 0.5) * args.image_size
            expected_crop_xy = trans_points2d_parallel(proj_verts[:, :2], trans)
            expected_normalized = expected_crop_xy / args.image_size - 0.5
            error = np.abs(expected_normalized - transformed[:, :2])

            cropped = draw_overlay(crop_bgr, crop_xy, args.label_every)
            cropped_path = cropped_dir / f"sample_{ordinal:03d}.png"
            if not cv2.imwrite(str(cropped_path), cropped):
                raise OSError(f"无法写入: {cropped_path}")

            height, width = image.shape[:2]
            xy = proj_verts[:, :2]
            finite = np.isfinite(xy).all(axis=1)
            in_frame = finite & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
            report = {
                "ordinal": ordinal,
                "index": index,
                "imgname": relative_name,
                "image_path": str(image_path),
                "image_size_hw": [height, width],
                "center_xy": center.tolist(),
                "scale": scale.tolist(),
                "bbox_size": bbox_size,
                "final_width": float(final_width),
                "final_height": float(final_height),
                "finite_points": int(finite.sum()),
                "in_frame_points": int(in_frame.sum()),
                "out_of_frame_points": int((finite & ~in_frame).sum()),
                "max_abs_error": float(error.max()),
                "mean_abs_error": float(error.mean()),
                "trans": np.asarray(trans).tolist(),
            }
            reports.append(report)
            print(json.dumps(report, ensure_ascii=False))

    report_path = args.output / "transform_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2) + "\n")
    print(f"audit report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
