#!/usr/bin/env python3
"""Generate native-render before/after reviews for an environment-only change."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def project(points, row):
    transform = np.linalg.inv(row["camera_matrix_world"]) @ np.asarray(row["object_matrix_world"])
    q = np.c_[points, np.ones(len(points))] @ transform.T
    k = row["intrinsics"]
    return np.c_[k["fx"] * q[:, 0] / -q[:, 2] + k["cx"],
                 k["cy"] - k["fy"] * q[:, 1] / -q[:, 2]]


def bounds(xy, width, height, pad):
    lo = np.floor(xy.min(0) - pad).astype(int)
    hi = np.ceil(xy.max(0) + pad).astype(int)
    return (max(0, lo[0]), max(0, lo[1]), min(width, hi[0]), min(height, hi[1]))


def pair_canvas(before, after, frames, boxes, output, title):
    cell_w, cell_h, bar = 320, 320, 24
    canvas = Image.new("RGB", (2 * cell_w, len(frames) * (cell_h + bar)), (18, 18, 18))
    draw = ImageDraw.Draw(canvas)
    for row_index, frame in enumerate(frames):
        for side, directory in enumerate((before, after)):
            image = Image.open(directory / "rgb" / f"{frame:06d}.png").convert("RGB").crop(boxes[frame])
            scale = min(cell_w / image.width, cell_h / image.height)
            image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.NEAREST)
            x = side * cell_w + (cell_w - image.width) // 2
            y = row_index * (cell_h + bar) + bar + (cell_h - image.height) // 2
            canvas.paste(image, (x, y))
            label = "simple environment" if side == 0 else "HDRI + weak fill"
            draw.text((side * cell_w + 8, row_index * (cell_h + bar) + 5),
                      f"{title} frame {frame}: {label}", fill="white")
    canvas.save(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    args = parser.parse_args()
    before, after = args.before, args.after
    rows = [json.loads(line) for line in (after / "manifest.jsonl").read_text().splitlines()]
    vertices = np.load(after / "geometry" / "vertices.npy")
    clothing = np.load(after / "geometry" / "clothing_vertices.npy")
    geometry_manifest = json.loads((after / "geometry" / "geometry_manifest.json").read_text())
    model = np.load(Path(geometry_manifest["model_dir"]) / "SMPLX_MALE.npz", allow_pickle=True)
    face_ids = model["weights"][:, [15, 22]].sum(1) > .5
    frames = [0, 5, 10, 15]
    width, height = Image.open(after / "rgb" / "000000.png").size
    face_boxes = {frame: bounds(project(vertices[frame, face_ids], rows[frame]), width, height, 7)
                  for frame in frames}
    clothing_boxes = {frame: bounds(project(clothing[frame], rows[frame]), width, height, 8)
                      for frame in frames}
    full_boxes = {}
    for frame in frames:
        mask = np.asarray(Image.open(after / "masks" / "person" / f"{frame:06d}.png"))
        yy, xx = np.where(mask > 0)
        full_boxes[frame] = (max(0, int(xx.min()) - 8), max(0, int(yy.min()) - 8),
                             min(width, int(xx.max()) + 9), min(height, int(yy.max()) + 9))
    pair_canvas(before, after, frames, face_boxes,
                after / "face_environment_before_after.png", "face")
    pair_canvas(before, after, frames, clothing_boxes,
                after / "clothing_environment_before_after.png", "clothing")
    pair_canvas(before, after, frames, full_boxes,
                after / "fullbody_environment_before_after.png", "fullbody")

    contact = Image.new("RGB", (1024, 8 * 256), (18, 18, 18))
    draw = ImageDraw.Draw(contact)
    for index, frame in enumerate(range(16)):
        row, column = divmod(index, 2)
        for side, directory in enumerate((before, after)):
            image = Image.open(directory / "rgb" / f"{frame:06d}.png").convert("RGB").resize((256, 256))
            x = column * 512 + side * 256
            contact.paste(image, (x, row * 256))
            draw.text((x + 6, row * 256 + 6), f"{frame}: {'simple' if side == 0 else 'HDRI'}",
                      fill="white", stroke_width=1, stroke_fill="black")
    contact.save(after / "contact_sheet_before_after.png")


if __name__ == "__main__":
    main()
