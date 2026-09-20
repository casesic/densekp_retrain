#!/usr/bin/env python3
"""用 pyrender EGL 对 P0 的 0/8/15 帧做 silhouette 交叉检查。"""

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/baselines/p0_perspective"))
    args = parser.parse_args()
    geometry = args.output_dir / "geometry"
    vertices = np.load(geometry / "vertices.npy")
    faces = np.load(geometry / "faces.npy")
    records = [json.loads(line) for line in (args.output_dir / "manifest.jsonl").read_text().splitlines()]
    cross_dir = args.output_dir / "crosscheck"
    cross_dir.mkdir(exist_ok=True)
    ious = {}
    renderer = pyrender.OffscreenRenderer(records[0]["resolution_width"], records[0]["resolution_height"])
    for frame in [0, 8, 15]:
        record = records[frame]
        object_matrix = np.asarray(record["object_matrix_world"], dtype=np.float64)
        camera_pose = np.asarray(record["camera_matrix_world"], dtype=np.float64)
        homogeneous = np.column_stack((vertices[frame], np.ones(len(vertices[frame]))))
        world_vertices = (object_matrix @ homogeneous.T).T[:, :3]
        tri = trimesh.Trimesh(vertices=world_vertices, faces=faces, process=False)
        scene = pyrender.Scene(bg_color=[14, 19, 28, 255], ambient_light=[0.3] * 3)
        node = scene.add(pyrender.Mesh.from_trimesh(tri, smooth=True))
        intr = record["intrinsics"]
        camera = pyrender.IntrinsicsCamera(intr["fx"], intr["fy"], intr["cx"], intr["cy"],
                                           znear=record["near_clip"], zfar=record["far_clip"])
        scene.add(camera, pose=camera_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=camera_pose)
        color, _ = renderer.render(scene)
        seg, _ = renderer.render(scene, flags=pyrender.RenderFlags.SEG, seg_node_map={node: [255, 255, 255]})
        Image.fromarray(color).save(cross_dir / f"pyrender_{frame:06d}.png")
        pyrender_mask = seg[..., 0] > 0
        blender_mask = np.asarray(Image.open(args.output_dir / "mask" / f"{frame:06d}.png")) > 0
        intersection = np.count_nonzero(pyrender_mask & blender_mask)
        union = np.count_nonzero(pyrender_mask | blender_mask)
        ious[str(frame)] = float(intersection / union) if union else 1.0
    renderer.delete()
    report = {"frames": [0, 8, 15], "silhouette_iou": ious, "mean_iou": float(np.mean(list(ious.values()))),
              "pass": min(ious.values()) > 0.95}
    (cross_dir / "pyrender_crosscheck.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
