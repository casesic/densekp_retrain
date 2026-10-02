#!/usr/bin/env python3
"""从少量 BEDLAM 几何记录生成 DenseKP 138 点监督，不负责重建 SMPL-X。"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from densekp_retrain.annotation import write_bedlam_tiny_dataset

DEFAULT_OPERATOR = ROOT / "assets" / "densekp_regressor_R.npy"
DEFAULT_OUTPUT = ROOT / "datasets" / "densekp_bedlam_tiny"
ROWS_TO_CHECK = (0, 1, 10, 50, 100, 137)


def print_operator_rows(R: np.ndarray) -> None:
    """打印人工复算使用的 operator 行及其精确 support。"""
    for row_id in ROWS_TO_CHECK:
        support = np.flatnonzero(R[row_id] != 0)
        weights = R[row_id, support]
        print(
            f"operator_row[{row_id}]: vertex_ids={support.tolist()}, "
            f"weights={weights.tolist()}, row_sum={float(weights.sum())}"
        )


def project_official(points: np.ndarray, camera_translation: np.ndarray, K: np.ndarray) -> np.ndarray:
    """复现 BEDLAM 官方 project(): (points + t) / z，再左乘 K。"""
    points_camera = np.asarray(points, dtype=np.float64) + np.asarray(
        camera_translation, dtype=np.float64
    ).reshape(1, 3)
    z = points_camera[:, 2:3]
    if not np.isfinite(points_camera).all() or np.any(z == 0):
        raise ValueError("camera-space points contain non-finite values or zero depth")
    normalized = points_camera / z
    projected = np.einsum("ij,kj->ki", np.asarray(K, dtype=np.float64), normalized)
    if not np.isfinite(projected).all():
        raise ValueError("projection produced non-finite values")
    if not np.allclose(projected[:, 2], 1.0, rtol=1e-6, atol=1e-6):
        raise ValueError("projection channel 2 is not homogeneous coordinate 1")
    return projected


from densekp_retrain.data.boxes import official_bbox  # historical import compatibility


def surface_row_error(R: np.ndarray, vertices: np.ndarray, densekp_3d: np.ndarray) -> float:
    errors = []
    for row_id in ROWS_TO_CHECK:
        support = np.flatnonzero(R[row_id] != 0)
        manual = np.sum(R[row_id, support, None] * vertices[support], axis=0)
        errors.append(np.max(np.abs(manual - densekp_3d[row_id])))
    return float(max(errors))


def get_array(data, *names):
    for name in names:
        if name in data:
            return data[name]
    raise KeyError(f"missing input field; expected one of {names}")


def decode_text(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_npz", type=Path, help="少量 BEDLAM 中间几何记录")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--operator", type=Path, default=DEFAULT_OPERATOR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=32)
    args = parser.parse_args()

    if not 1 <= args.min_samples <= args.max_samples <= 32:
        raise SystemExit("require 1 <= min-samples <= max-samples <= 32")

    R = np.load(args.operator)
    if R.shape != (138, 10475) or R.dtype != np.float32 or not np.isfinite(R).all():
        raise SystemExit(f"invalid DenseKP operator: shape={R.shape}, dtype={R.dtype}")
    print_operator_rows(R)

    records = []
    with np.load(args.input_npz, allow_pickle=False) as data:
        names = get_array(data, "imgname")
        vertices_all = get_array(data, "vertices", "smplx_vertices")
        joints_all = get_array(data, "joints3d", "body_joints3d")
        translations = get_array(data, "camera_translation", "cam_trans")
        intrinsics = get_array(data, "camera_intrinsics", "cam_int")
        orientations = get_array(data, "orientation")

        n = len(names)
        leading = {
            "vertices": len(vertices_all),
            "joints3d": len(joints_all),
            "camera_translation": len(translations),
            "camera_intrinsics": len(intrinsics),
            "orientation": len(orientations),
        }
        if any(count != n for count in leading.values()):
            raise SystemExit(f"inconsistent sample counts: imgname={n}, {leading}")

        for source_index in range(n):
            if len(records) >= args.max_samples:
                break
            orientation = decode_text(orientations[source_index]).lower()
            if "portrait" in orientation or "closeup" in orientation:
                print(f"skip[{source_index}]: rotated orientation is excluded in v1 ({orientation})")
                continue

            image_name = decode_text(names[source_index])
            source_image = Path(image_name)
            if not source_image.is_absolute():
                source_image = args.image_root / source_image
            image = cv2.imread(str(source_image), cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                print(f"skip[{source_index}]: cannot read RGB {source_image}")
                continue
            image_height, image_width = image.shape[:2]

            vertices = np.asarray(vertices_all[source_index], dtype=np.float64)
            joints3d = np.asarray(joints_all[source_index], dtype=np.float64)
            translation = np.asarray(translations[source_index], dtype=np.float64)
            K = np.asarray(intrinsics[source_index], dtype=np.float64)
            if vertices.shape != (10475, 3):
                raise SystemExit(f"sample[{source_index}] vertices shape {vertices.shape}, expected (10475,3)")
            if joints3d.ndim != 2 or joints3d.shape[0] < 22 or joints3d.shape[1] != 3:
                raise SystemExit(f"sample[{source_index}] joints3d shape {joints3d.shape}")
            if translation.shape not in ((3,), (1, 3)) or K.shape != (3, 3):
                raise SystemExit(f"sample[{source_index}] invalid camera shapes: t={translation.shape}, K={K.shape}")
            if not np.isfinite(vertices).all():
                raise SystemExit(f"sample[{source_index}] vertices are not finite")

            densekp_3d = np.asarray(R, dtype=np.float64) @ vertices
            if densekp_3d.shape != (138, 3) or not np.isfinite(densekp_3d).all():
                raise SystemExit(f"sample[{source_index}] invalid densekp_3d")
            proj_verts = project_official(densekp_3d, translation, K)
            joints2d = project_official(joints3d, translation, K)
            bbox = official_bbox(joints2d, image_height, image_width)
            if bbox is None:
                print(f"skip[{source_index}]: fewer than 8 valid body joints or invalid bbox")
                continue
            center, scale, valid_count, tight_bbox = bbox

            xy = proj_verts[:, :2]
            in_image = (xy[:, 0] >= 0) & (xy[:, 0] < image_width)
            in_image &= (xy[:, 1] >= 0) & (xy[:, 1] < image_height)
            bbox_size = float(scale) * 200.0
            half = bbox_size / 2.0
            in_bbox = (xy[:, 0] >= center[0] - half) & (xy[:, 0] <= center[0] + half)
            in_bbox &= (xy[:, 1] >= center[1] - half) & (xy[:, 1] <= center[1] + half)

            records.append(
                {
                    "source_index": source_index,
                    "source_image": source_image,
                    "source_imgname": image_name,
                    "center": center,
                    "scale": scale,
                    "proj_verts": proj_verts.astype(np.float32),
                    "joints2d": joints2d[:22].astype(np.float32),
                    "valid_joint_count": valid_count,
                    "tight_bbox": tight_bbox,
                    "image_size": np.array([image_height, image_width], dtype=np.int32),
                    "in_image_count": np.int32(in_image.sum()),
                    "in_bbox_count": np.int32(in_bbox.sum()),
                    "surface_error": surface_row_error(R, vertices, densekp_3d),
                }
            )

    if len(records) < args.min_samples:
        raise SystemExit(
            f"only {len(records)} valid landscape samples; need at least {args.min_samples}; no dataset written"
        )

    output_npz, debug_npz = write_bedlam_tiny_dataset(records, args.output)
    print(f"wrote {len(records)} samples: {output_npz}")
    print(f"debug metadata: {debug_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
