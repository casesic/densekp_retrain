#!/usr/bin/env python3
"""Score a registered DenseKP dataset with visibility-aware per-person metrics."""

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

from densekp.core.configs import DATASET_FILES  # noqa: E402
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints  # noqa: E402
from densekp.core.densekp_trainer import DenseKP  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="omni-multiperson-tiny")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "assets_external/densekp.ckpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/fisheye_multiperson_hires/baseline.json")
    parser.add_argument("--target-value-semantics", required=True, choices=["supervision_weight"],
                        help="Confirm producer-defined target semantics; not model keypoint_value")
    args = parser.parse_args()
    with initialize_config_dir(version_base="1.2", config_dir=str(ROOT / "configs/hydra")):
        cfg = compose(config_name="train", overrides=["experiment=densekp"])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS = args.target_value_semantics
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("medium")
    model = DenseKP.load_from_checkpoint(str(args.checkpoint), cfg=cfg, map_location="cpu")
    model.to(device).eval()
    rows = []
    overlay_dir = args.output.parent / "baseline_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for split, is_train in (("train", True), ("val", False)):
        dataset = DatasetKeypoints(cfg, args.dataset, is_train=is_train, use_augmentation=False)
        with np.load(DATASET_FILES[is_train][args.dataset]) as data:
            person_ids = data["person_id"] if "person_id" in data else np.full(len(dataset), -1)
            camera_ids = data["camera_id"] if "camera_id" in data else np.full(len(dataset), "unknown")
        for index in range(len(dataset)):
            item = dataset[index]
            with torch.inference_mode():
                output = model({"img": torch.from_numpy(item["img"][None]).to(device)})["pred_keypoints"]
            predicted = output[0, :, :2].float().cpu().numpy()
            target = item["proj_verts_cropped"][:, :2]
            weights = item['proj_verts_cropped'][:, 2]
            if not np.isin(weights, [0., 1.]).all():
                raise ValueError('This diagnostic evaluator requires binary supervision masks')
            if not np.isfinite(predicted).all():
                raise ValueError(f"nonfinite model prediction: {item['imgname']}")
            visible = weights > 0
            if not visible.any():
                raise ValueError(f"no visible points: {item['imgname']}")
            error = np.linalg.norm(predicted - target, axis=1) * cfg.MODEL.IMAGE_SIZE
            rows.append({"split": split, "camera": str(camera_ids[index]),
                         "person_id": int(person_ids[index]), "imgname": item["imgname"],
                         "visible_points": int(visible.sum()),
                         "mean_crop_error_px": float(error[visible].mean()),
                         "median_crop_error_px": float(np.median(error[visible])),
                         "p90_crop_error_px": float(np.percentile(error[visible], 90))})
            overlay = Image.fromarray(cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(overlay)
            for coords, color in ((target, "#23e05b"), (predicted, "#ff4040")):
                for point_id, (x, y) in enumerate(coords):
                    if visible[point_id] and np.isfinite(x) and np.isfinite(y):
                        px, py = (np.array([x, y]) + .5) * cfg.MODEL.IMAGE_SIZE
                        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=color)
            overlay.save(overlay_dir / f"{split}_{index}_{camera_ids[index]}_person_{person_ids[index]}.png")
    report = {"checkpoint": str(args.checkpoint.resolve()), "dataset": args.dataset,
              "description": "oracle instance-mask crops; no detector; GT-visible points only",
              "error_unit": "pixels in the 256x256 DenseKP crop", "cases": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
