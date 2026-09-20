#!/usr/bin/env python3
"""Create a frame-0 eye-material comparison using verified eye component bounds."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    row = json.loads((args.before / "manifest.jsonl").read_text().splitlines()[0])
    vertices = np.load(args.geometry / "vertices.npy")[0]
    labels = np.load("outputs/body_texture_audit/body_uv.npz")["component_labels"]
    matrix = np.linalg.inv(row["camera_matrix_world"]) @ np.asarray(row["object_matrix_world"])
    q = np.c_[vertices, np.ones(len(vertices))] @ matrix.T
    k = row["intrinsics"]
    xy = np.c_[k["fx"] * q[:, 0] / -q[:, 2] + k["cx"],
               k["cy"] - k["fy"] * q[:, 1] / -q[:, 2]]
    eye_xy = xy[labels != np.argmax(np.bincount(labels))]
    lo = np.floor(eye_xy.min(0) - 38).astype(int); hi = np.ceil(eye_xy.max(0) + 38).astype(int)
    box = (max(0, lo[0]), max(0, lo[1]), min(512, hi[0]), min(512, hi[1]))
    before = Image.open(args.before / "rgb" / "000000.png").convert("RGB")
    after = Image.open(args.after / "rgb" / "000000.png").convert("RGB")
    before.save(args.output / "before.png"); after.save(args.output / "after.png")
    before_crop, after_crop = before.crop(box), after.crop(box)
    before_crop.save(args.output / "eye_crop_before.png")
    after_crop.save(args.output / "eye_crop_after.png")
    scale = 6
    canvas = Image.new("RGB", (before_crop.width * scale * 2, before_crop.height * scale + 30), (18, 18, 18))
    canvas.paste(before_crop.resize((before_crop.width*scale, before_crop.height*scale), Image.Resampling.NEAREST), (0, 30))
    canvas.paste(after_crop.resize((after_crop.width*scale, after_crop.height*scale), Image.Resampling.NEAREST), (before_crop.width*scale, 30))
    draw = ImageDraw.Draw(canvas); draw.text((8, 8), "Before: reddish legacy material", fill="white")
    draw.text((before_crop.width*scale+8, 8), "After: neutral eye fallback; geometry identical", fill="white")
    canvas.save(args.output / "closeup_before_after.png")


if __name__ == "__main__":
    main()
