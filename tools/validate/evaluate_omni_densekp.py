#!/usr/bin/env python3
"""Evaluate the original DenseKP checkpoint on paired P0 ERP/fisheye renders.

This is a three-camera diagnostic, not an independent validation set.  Fish
crops use rendered person masks; ERP crops use visible GT landmark extents.
Detector errors are excluded. Errors use the 256px DenseKP input crop.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from densekp.core.datasets.utils import get_example_projverts  # noqa: E402
from densekp.core.densekp_trainer import DenseKP  # noqa: E402


def tight_square(mask: np.ndarray) -> tuple[float, float, float]:
    ys, xs = np.where(mask)
    if len(xs) < 100:
        raise ValueError("person mask has fewer than 100 pixels")
    size = max(int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)) * 1.2
    return float((xs.min() + xs.max()) / 2), float((ys.min() + ys.max()) / 2), float(size)


def center_erp_person(rgb: np.ndarray, xy: np.ndarray, valid: np.ndarray):
    """Roll the periodic longitude axis so a seam-crossing person stays whole."""
    occupied = np.unique(np.rint(xy[valid, 0]).astype(int) % rgb.shape[1])
    if len(occupied) == 0:
        raise ValueError("no visible ERP keypoints")
    width = rgb.shape[1]
    gaps = np.diff(np.r_[occupied, occupied[0] + width])
    gap_index = int(np.argmax(gaps))
    arc_start = int(occupied[(gap_index + 1) % len(occupied)])
    arc_size = width - int(gaps[gap_index]) + 1
    shift = int(round(width / 2 - (arc_start + arc_size / 2)))
    shifted_xy = xy.copy()
    shifted_xy[:, 0] = (shifted_xy[:, 0] + shift) % width
    mask = np.zeros(rgb.shape[:2], dtype=np.uint8)
    hull_points = np.rint(shifted_xy[valid]).astype(np.int32)
    cv2.fillConvexPoly(mask, cv2.convexHull(hull_points), 1)
    mask = cv2.dilate(mask, np.ones((9, 9), np.uint8)) > 0
    return np.roll(rgb, shift, axis=1), mask, shifted_xy, shift


def erp_points(world: np.ndarray, camera: np.ndarray, width: int, height: int) -> np.ndarray:
    q = (world - camera[:3, 3]) @ camera[:3, :3]
    ray = q / np.linalg.norm(q, axis=1, keepdims=True)
    lon = np.arctan2(ray[:, 0], ray[:, 2])
    lat = np.arcsin(np.clip(-ray[:, 1], -1, 1))
    return np.column_stack(((lon / (2 * np.pi) + .5) * width,
                            (.5 - lat / np.pi) * height))


def make_cases(source: Path):
    config = json.loads((source / "camera_config.json").read_text())
    source_report = json.loads((source / "report.json").read_text())
    geometry = Path(source_report["geometry"])
    if not (geometry / "vertices.npy").is_file():
        geometry = ROOT / "outputs/baselines" / geometry.parent.name / "geometry"
    if not (geometry / "vertices.npy").is_file():
        raise FileNotFoundError(f"render source geometry not found: {geometry}")
    frame = int(source_report["geometry_frame"])
    verts = np.load(geometry / "vertices.npy", mmap_mode="r")[frame]
    dense = np.load(geometry / "densekp_3d.npy", mmap_mode="r")[frame]
    lift = -float(verts[:, 1].min())
    world = np.column_stack((dense[:, 0], -dense[:, 2], dense[:, 1] + lift))
    locations = {"center": source / "center", "cam_left": source / "seam/cam_left",
                 "cam_right": source / "seam/cam_right"}
    for camera in config["cameras"]:
        name = "center" if camera["id"] == "fisheye_center" else camera["id"]
        pose = np.asarray(camera["camera_to_world"], dtype=np.float64)
        fisheye_dir = locations[name]
        with Image.open(fisheye_dir / "rgb.png") as image:
            fish_rgb = np.asarray(image.convert("RGB"))
        with Image.open(fisheye_dir / "mask.png") as image:
            fish_mask = np.asarray(image.convert("L")) >= 128
        fish_xy = np.load(fisheye_dir / "densekp_fisheye_2d.npy")[0]
        fish_visible = np.load(fisheye_dir / "densekp_visible.npy")[0]
        yield name, "fisheye", fish_rgb, fish_mask, fish_xy, fish_visible, 0

        with Image.open(source / "equirectangular_source" / f"{camera['id']}_preview.png") as image:
            erp_rgb = np.asarray(image.convert("RGB"))
        erp_xy = erp_points(world, pose, erp_rgb.shape[1], erp_rgb.shape[0])
        # Same camera rays: fish-visible 3D points are also ERP-visible.  The
        # subset is conservative because the fisheye has a smaller FOV.
        visible = fish_visible.copy()
        erp_rgb, erp_mask, erp_xy, shift = center_erp_person(erp_rgb, erp_xy, visible)
        yield name, "equirectangular", erp_rgb, erp_mask, erp_xy, visible, shift


def evaluate_case(model, cfg, device, case, image_dir):
    name, projection, rgb, mask, xy, visible, shift = case
    cx, cy, box_size = tight_square(mask)
    gt = np.column_stack((xy, np.ones(138, dtype=np.float32))).astype(np.float32)
    mean = 255 * np.asarray(cfg.MODEL.IMAGE_MEAN)
    std = 255 * np.asarray(cfg.MODEL.IMAGE_STD)
    patch, display, _, _, _, _, _, _, cropped_gt, _ = get_example_projverts(
        cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), cx, cy, box_size, box_size,
        np.zeros((1, 3), dtype=np.float32), [0], gt, 256, 256,
        mean, std, do_augment=False, is_bgr=True, border_mode=cv2.BORDER_CONSTANT)
    with torch.inference_mode():
        output = model({"img": torch.from_numpy(patch[None]).to(device)})["pred_keypoints"]
    prediction = output[0, :, :2].float().cpu().numpy()
    target = cropped_gt[:, :2]
    error = np.linalg.norm(prediction - target, axis=1) * 256
    in_crop = ((target >= -.5) & (target < .5)).all(axis=1)
    scored = visible & in_crop & np.isfinite(error)
    overlay = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(overlay)
    for points, color in ((target, "#28d750"), (prediction, "#ff4040")):
        for j, (x, y) in enumerate(points):
            if not scored[j]:
                continue
            px, py = (np.array([x, y]) + .5) * 256
            if np.isfinite(px) and np.isfinite(py):
                draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
    image_dir.mkdir(parents=True, exist_ok=True)
    overlay.save(image_dir / f"{name}_{projection}.png")
    return {"camera": name, "projection": projection, "bbox_center": [cx, cy],
            "bbox_size": box_size, "erp_roll_pixels": shift,
            "visible_points": int(visible.sum()), "scored_points": int(scored.sum()),
            "mean_crop_error_px": float(error[scored].mean()) if scored.any() else None,
            "median_crop_error_px": float(np.median(error[scored])) if scored.any() else None,
            "p90_crop_error_px": float(np.percentile(error[scored], 90)) if scored.any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/baselines/fisheye_camera_p0")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "assets_external/densekp.ckpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/omni_densekp_baseline/report.json")
    args = parser.parse_args()
    with initialize_config_dir(version_base="1.2", config_dir=str(ROOT / "configs/hydra")):
        cfg = compose(config_name="train", overrides=["experiment=densekp"])
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("medium")
    model = DenseKP.load_from_checkpoint(str(args.checkpoint), cfg=cfg, map_location="cpu")
    model.to(device).eval()
    rows = [evaluate_case(model, cfg, device, case, args.output.parent / "overlays")
            for case in make_cases(args.source)]
    report = {"checkpoint": str(args.checkpoint.resolve()), "source": str(args.source.resolve()),
              "description": "three paired rendered cameras, no detector; fish boxes from person mask, ERP boxes from visible GT landmarks",
              "visibility": "same fisheye-visible 3D point subset in each camera's ERP/fisheye pair",
              "error_unit": "pixels in the 256x256 DenseKP crop", "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
