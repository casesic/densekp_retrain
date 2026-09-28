#!/usr/bin/env python3
"""Pass a rendered whole-sphere ERP through production DS fisheye -> ERP geometry.

Uses the real cam2/cam4 ERP valid masks from a provided NPZ. This is a
geometry/annotation pilot, not a substitute for diverse training scenes.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from densekp_retrain.camera.double_sphere import (  # noqa: E402
    DS_20260320, build_erp_to_fisheye_lut, build_fisheye_to_source_erp_lut,
)


def visible_at_instance_points(xy: np.ndarray, instance_mask: np.ndarray,
                               valid_mask: np.ndarray) -> np.ndarray:
    height, width = instance_mask.shape
    visible = np.zeros(len(xy), dtype=bool)
    for index, (x, y) in enumerate(xy):
        if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
            continue
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < width and 0 <= iy < height) or not valid_mask[iy, ix]:
            continue
        region = instance_mask[max(0, iy - 3):iy + 4, max(0, ix - 3):ix + 4]
        visible[index] = bool(region.any())
    return visible


def remap_periodic_erp(image: np.ndarray, map_x: np.ndarray, map_y: np.ndarray,
                       interpolation: int) -> np.ndarray:
    """Sample longitude periodically without wrapping the ERP poles."""
    padded = np.concatenate((image[:, -1:], image, image[:, :1]), axis=1)
    padded_x = np.where(map_x >= 0, map_x + 1, -1).astype(np.float32)
    return cv2.remap(padded, padded_x, map_y, interpolation,
                     borderMode=cv2.BORDER_CONSTANT)


def lens_mask_from_reference(path: Path, expected_size: tuple[int, int]) -> np.ndarray:
    """Keep the real lens footprint, without copying any scene RGB into training."""
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        raise FileNotFoundError(f"reference fisheye not readable: {path}")
    if frame.shape[:2] != expected_size:
        raise ValueError(f"reference fisheye size {frame.shape[:2]} != {expected_size}: {path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rough = (gray > 5).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51, 51))
    rough = cv2.morphologyEx(rough, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise ValueError(f"cannot find lens footprint: {path}")
    mask = np.zeros_like(rough)
    cv2.drawContours(mask, [max(contours, key=cv2.contourArea)], -1, 255, -1)
    return mask != 0


def build(source: Path, masks_path: Path, output: Path,
          camera_roll_px: dict[str, int],
          fisheye_reference_root: Path | None = None,
          reference_frame: str = "02214") -> dict:
    source_cam = source / "fisheye_center"
    with Image.open(source / "source/fisheye_center_preview.png") as image:
        source_rgb = np.asarray(image.convert("RGB"))
    with np.load(source_cam / "erp_annotations.npz") as data:
        ids = data["person_ids"].astype(int).tolist()
        source_xy = data["xy"].copy()
        source_visible = data["visible"].copy()
    if source_xy.shape != (len(ids), 138, 2):
        raise ValueError(f"invalid source annotations: {source_xy.shape}")
    source_height, source_width = source_rgb.shape[:2]
    person_masks = {}
    for person_id in ids:
        with Image.open(source_cam / f"erp_person_{person_id}_mask.png") as image:
            mask = np.asarray(image.convert("L")) >= 128
        if mask.shape != source_rgb.shape[:2]:
            raise ValueError(f"source mask size mismatch for person {person_id}")
        person_masks[person_id] = mask.astype(np.uint8) * 255
    output.mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    results = []
    with np.load(masks_path) as masks:
        for camera in ("cam2", "cam4"):
            roll_source_px = camera_roll_px[camera]
            shifted_rgb = np.roll(source_rgb, roll_source_px, axis=1)
            shifted_xy = source_xy.copy()
            shifted_xy[..., 0] = (shifted_xy[..., 0] + roll_source_px) % source_width
            intrinsics = DS_20260320[camera]
            valid_mask = masks[f"valid_masks_{camera}"].astype(bool)
            out_height, out_width = valid_mask.shape
            fish_x, fish_y = build_fisheye_to_source_erp_lut(
                intrinsics, camera, source_width, source_height)
            fish_rgb = remap_periodic_erp(shifted_rgb, fish_x, fish_y, cv2.INTER_LINEAR)
            fish_person_masks = {person_id: remap_periodic_erp(
                                    np.roll(mask, roll_source_px, axis=1),
                                    fish_x, fish_y, cv2.INTER_NEAREST)
                                 for person_id, mask in person_masks.items()}
            lens_mask = None
            if fisheye_reference_root is not None:
                reference_path = fisheye_reference_root / camera / f"{reference_frame}.jpg"
                lens_mask = lens_mask_from_reference(
                    reference_path, (intrinsics.height, intrinsics.width))
                fish_rgb[~lens_mask] = 0
                for mask in fish_person_masks.values():
                    mask[~lens_mask] = 0
            del fish_x, fish_y
            lut_x, lut_y = build_erp_to_fisheye_lut(intrinsics, camera, out_width, out_height)
            erp_rgb = cv2.remap(fish_rgb, lut_x, lut_y, cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT)
            erp_rgb[~valid_mask] = 0
            cam_dir = output / camera
            cam_dir.mkdir(exist_ok=True)
            Image.fromarray(fish_rgb).save(cam_dir / "synthetic_fisheye.png")
            if lens_mask is not None:
                Image.fromarray(lens_mask.astype(np.uint8) * 255).save(
                    cam_dir / "reference_lens_mask.png")
            Image.fromarray(erp_rgb).save(output / "source" / f"{camera}_preview.png")
            final_xy = shifted_xy.copy()
            final_xy[..., 0] *= out_width / source_width
            final_xy[..., 1] *= out_height / source_height
            final_visible = np.zeros_like(source_visible)
            instance_pixel_counts = []
            for person_index, person_id in enumerate(ids):
                person_erp = cv2.remap(fish_person_masks[person_id], lut_x, lut_y,
                                       cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
                person_erp[~valid_mask] = 0
                instance = person_erp >= 128
                Image.fromarray(person_erp).save(cam_dir / f"erp_person_{person_id}_mask.png")
                instance_pixel_counts.append(int(instance.sum()))
                final_visible[person_index] = (
                    source_visible[person_index]
                    & visible_at_instance_points(final_xy[person_index], instance, valid_mask))
            np.savez(cam_dir / "erp_annotations.npz", person_ids=np.asarray(ids),
                     xy=final_xy.astype(np.float32),
                     visible=final_visible,
                     valid=np.isfinite(final_xy).all(axis=-1) &
                     (final_xy[..., 0] >= 0) & (final_xy[..., 0] < out_width) &
                     (final_xy[..., 1] >= 0) & (final_xy[..., 1] < out_height))
            predicted_valid = ((lut_x >= 0) & (lut_x < intrinsics.width) &
                               (lut_y >= 0) & (lut_y < intrinsics.height))
            measured_coverage = float(predicted_valid[valid_mask].mean())
            if measured_coverage < .99:
                raise ValueError(f"{camera}: projection covers only {measured_coverage:.1%} "
                                 "of measured valid pixels; check camera axes/model")
            lens_coverage = None
            if lens_mask is not None:
                projected_lens = cv2.remap(lens_mask.astype(np.uint8), lut_x, lut_y,
                                           cv2.INTER_NEAREST,
                                           borderMode=cv2.BORDER_CONSTANT) != 0
                lens_coverage = float(projected_lens[valid_mask].mean())
                if lens_coverage < .99:
                    raise ValueError(f"{camera}: reference lens covers only "
                                     f"{lens_coverage:.1%} of measured ERP valid pixels")
            results.append({"camera": camera, "roll_source_px": roll_source_px,
                            "erp_visible": final_visible.sum(axis=1).tolist(),
                            "person_mask_pixels": instance_pixel_counts,
                            "valid_fraction": float(valid_mask.mean()),
                            "measured_mask_coverage": measured_coverage,
                            "reference_lens_coverage": lens_coverage,
                            "output_size": [out_width, out_height]})
            del lut_x, lut_y, fish_rgb, fish_person_masks, erp_rgb
    report = {"camera_model": "Double Sphere / densekp_delivery 20260320 (new)",
              "source": str(source.resolve()), "valid_masks": str(masks_path.resolve()),
              "fisheye_reference_root": (str(fisheye_reference_root.resolve())
                                         if fisheye_reference_root else None),
              "reference_frame": reference_frame if fisheye_reference_root else None,
              "camera_roll_source_px": camera_roll_px,
              "people": [{"id": person_id} for person_id in ids], "results": results,
              "note": "independent camera-specific yaw placements, not a synchronized stereo pair; "
                      "one motion sequence, real masks and DS geometry, synthetic content"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/experiments/fisheye_multiperson_hires")
    parser.add_argument("--valid-masks", type=Path, default=ROOT / "assets_external/real_erp_valid_masks.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/ds_multiperson_pilot")
    parser.add_argument("--cam2-roll-source-px", type=int, default=0)
    parser.add_argument("--cam4-roll-source-px", type=int, default=-600)
    parser.add_argument("--fisheye-reference-root", type=Path,
                        help="real cam2/cam4 fisheye root; extract lens footprint only")
    parser.add_argument("--reference-frame", default="02214")
    args = parser.parse_args()
    rolls = {"cam2": args.cam2_roll_source_px, "cam4": args.cam4_roll_source_px}
    print(json.dumps(build(args.source, args.valid_masks, args.output, rolls,
                           args.fisheye_reference_root, args.reference_frame), indent=2))


if __name__ == "__main__":
    main()
