#!/usr/bin/env python3
"""验证纹理集成没有改变 P0 的几何、相机和监督产物。"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def same_array(a: Path, b: Path) -> bool:
    return np.array_equal(np.load(a), np.load(b), equal_nan=True)


def same_images(a: Path, b: Path, count: int) -> bool:
    return all(np.array_equal(np.asarray(Image.open(a / f"{i:06d}.png")),
                              np.asarray(Image.open(b / f"{i:06d}.png")))
               for i in range(count))


def fingerprint(path: Path) -> str:
    return hashlib.blake2b(np.ascontiguousarray(np.load(path)).tobytes(), digest_size=16).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    args = p.parse_args()
    before, after = args.before, args.after
    br = json.loads((before / "p0_render_report.json").read_text())
    ar = json.loads((after / "p0_render_report.json").read_text())
    bo = json.loads((before / "p0_output_report.json").read_text())
    ao = json.loads((after / "p0_output_report.json").read_text())
    bm = [json.loads(x) for x in (before / "manifest.jsonl").read_text().splitlines()]
    am = [json.loads(x) for x in (after / "manifest.jsonl").read_text().splitlines()]
    camera_fields = ("frame_id", "geometry_frame_id", "resolution_width", "resolution_height",
                     "lens_mm", "sensor_width_mm", "sensor_height_mm", "sensor_fit",
                     "pixel_aspect", "shift_x", "shift_y", "near_clip", "far_clip",
                     "camera_location_world", "camera_rotation_world", "camera_matrix_world",
                     "object_location_world", "object_rotation_world", "object_scale_world",
                     "object_matrix_world", "projection", "intrinsics")
    camera_same = len(bm) == len(am) and all(
        all(x.get(k) == y.get(k) for k in camera_fields) for x, y in zip(bm, am)
    )
    report = {
        "frames": len(am),
        "body_geometry_unchanged": ar["geometry_fingerprint"]["identical"]
            and ar["geometry_fingerprint"]["before"] == br["geometry_fingerprint"]["before"],
        "clothing_geometry_unchanged": ar["clothing_geometry_fingerprint"]["identical"]
            and ar["clothing_geometry_fingerprint"]["before"] == br["clothing_geometry_fingerprint"]["before"],
        "camera_unchanged": camera_same,
        "densekp_2d_unchanged": same_array(before / "annotations/densekp_2d.npy", after / "annotations/densekp_2d.npy"),
        "densekp_camera_xyz_unchanged": same_array(before / "annotations/densekp_camera_xyz.npy", after / "annotations/densekp_camera_xyz.npy"),
        "depth_unchanged": all(same_array(before / "depth" / f"{i:06d}.npy", after / "depth" / f"{i:06d}.npy") for i in range(16)),
        "mask_unchanged": same_images(before / "mask", after / "mask", 16),
        "visibility_unchanged": same_array(before / "annotations/densekp_visible.npy", after / "annotations/densekp_visible.npy"),
        "visibility_counts_unchanged": bo["visibility_counts"] == ao["visibility_counts"],
        "projection_error_px": ar["projection_error_px"],
        "render_ms_per_frame": ar["render_seconds"]["mean"] * 1000.0,
    }
    checks = [v for k, v in report.items() if k.endswith("_unchanged")]
    report["pass"] = all(checks) and ar["projection_error_px"]["max"] < 1e-3
    (after / "texture_regression.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
