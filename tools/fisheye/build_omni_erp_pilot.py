#!/usr/bin/env python3
"""Convert an actual rendered Omni-Radtan fisheye sample back to ERP.

The RGB, instance masks and 138-point annotations all originate from the
fisheye image, not from the pre-warp Blender panorama.
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from densekp_retrain.camera import (  # noqa: E402
    OmniRadtanIntrinsics, build_erp_to_omni_lut, omni_pixels_to_erp,
)
from tools.fisheye.build_ds_multiperson_pilot import visible_at_instance_points  # noqa: E402


def build(source: Path, output: Path, camera_id: str, width: int, height: int) -> dict:
    source_report = json.loads((source / "report.json").read_text())
    if source_report["camera_model"] != "Omni-Radtan":
        raise ValueError("source is not an Omni-Radtan pilot")
    if camera_id not in [row["camera"] for row in source_report["results"]]:
        raise ValueError(f"camera not in source report: {camera_id}")
    ids = [int(person["id"]) for person in source_report["people"]]
    intrinsics = OmniRadtanIntrinsics()
    fish_dir = source / camera_id / "fisheye"
    fish_rgb = cv2.imread(str(fish_dir / "rgb.png"), cv2.IMREAD_COLOR)
    if fish_rgb is None or fish_rgb.shape[:2] != (intrinsics.height, intrinsics.width):
        raise ValueError(f"invalid fisheye RGB: {fish_dir / 'rgb.png'}")
    with np.load(fish_dir / "person_annotations.npz") as data:
        if data["xy"].shape != (len(ids), 138, 2) or data["person_ids"].tolist() != ids:
            raise ValueError("fisheye annotation layout mismatch")
        fish_xy = data["xy"].copy()
        fish_visible = data["visible"].copy()
    map_x, map_y = build_erp_to_omni_lut(intrinsics, width, height)
    valid_mask = (map_x >= 0) & (map_y >= 0)
    erp_rgb = cv2.remap(fish_rgb, map_x, map_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT)
    erp_xy, xy_valid = omni_pixels_to_erp(fish_xy, intrinsics, width, height)
    output_cam = output / camera_id
    output_cam.mkdir(parents=True, exist_ok=True)
    (output / "source").mkdir(exist_ok=True)
    if not cv2.imwrite(str(output / "source" / f"{camera_id}_preview.png"), erp_rgb):
        raise RuntimeError("failed to write ERP RGB")
    Image.fromarray(valid_mask.astype(np.uint8) * 255).save(output_cam / "erp_valid_mask.png")
    erp_visible = np.zeros_like(fish_visible)
    mask_pixels = []
    for index, person_id in enumerate(ids):
        fish_mask = cv2.imread(str(fish_dir / f"person_{person_id}_mask.png"), cv2.IMREAD_GRAYSCALE)
        if fish_mask is None or fish_mask.shape != fish_rgb.shape[:2]:
            raise ValueError(f"missing fisheye person mask: {person_id}")
        erp_mask = cv2.remap(fish_mask, map_x, map_y, cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT)
        mask_pixels.append(int((erp_mask >= 128).sum()))
        Image.fromarray(erp_mask).save(output_cam / f"erp_person_{person_id}_mask.png")
        erp_visible[index] = (
            fish_visible[index] & xy_valid[index]
            & visible_at_instance_points(erp_xy[index], erp_mask >= 128, valid_mask))
    np.savez(output_cam / "erp_annotations.npz", person_ids=np.asarray(ids),
             xy=erp_xy.astype(np.float32),
             valid=xy_valid & np.isfinite(erp_xy).all(axis=-1),
             visible=erp_visible)
    report = {"camera_model": "Omni-Radtan fisheye -> ERP",
              "source": str(source.resolve()), "people": [{"id": person_id} for person_id in ids],
              "results": [{"camera": camera_id,
                           "fisheye_visible": fish_visible.sum(axis=1).tolist(),
                           "erp_visible": erp_visible.sum(axis=1).tolist(),
                           "person_mask_pixels": mask_pixels,
                           "valid_fraction": float(valid_mask.mean()),
                           "output_size": [width, height]}],
              "note": "RGB and labels originate from rendered Omni-Radtan fisheye; "
                      "single synthetic motion sequence, not an independent validation set"}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "outputs/experiments/fisheye_near_multiperson")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/omni_near_erp")
    parser.add_argument("--camera-id", default="fisheye_center")
    parser.add_argument("--width", type=int, default=5760)
    parser.add_argument("--height", type=int, default=2880)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.camera_id,
                           args.width, args.height), indent=2))


if __name__ == "__main__":
    main()
