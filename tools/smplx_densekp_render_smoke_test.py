#!/usr/bin/env python3
"""用同一组 SMPL-X 顶点渲染并生成 DenseKP 138 点 overlay。"""

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import smplx
import torch
import trimesh
from PIL import Image, ImageDraw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--regressor", type=Path, default=Path("assets/densekp_regressor_R.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/headless_smoke"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs = dict(gender="neutral", ext="npz", num_betas=10, use_pca=False, flat_hand_mean=True)
    if (args.model_dir / "SMPLX_NEUTRAL.npz").is_file():
        model = smplx.SMPLX(str(args.model_dir), **model_kwargs)
    else:
        model = smplx.create(str(args.model_dir), model_type="smplx", **model_kwargs)
    with torch.no_grad():
        output = model(return_verts=True)
    vertices = output.vertices[0].cpu().numpy()
    if vertices.shape != (10475, 3):
        raise RuntimeError(f"顶点 shape 不符合预期：{vertices.shape}")
    regressor = np.load(args.regressor)
    if regressor.shape != (138, 10475):
        raise RuntimeError(f"regressor shape 不符合预期：{regressor.shape}")

    # 单一几何源：渲染 mesh 和 DenseKP 都由 vertices 派生。
    center = (vertices.min(axis=0) + vertices.max(axis=0)) / 2.0
    vertices_gl = vertices - center
    vertices_gl[:, 2] -= 3.0
    densekp_gl = regressor @ vertices_gl

    size = args.size
    focal = 0.9 * size
    scene = pyrender.Scene(bg_color=[235, 235, 235, 255], ambient_light=[0.35] * 3)
    tri = trimesh.Trimesh(vertices=vertices_gl, faces=model.faces, process=False)
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.72, 0.48, 0.35, 1.0), metallicFactor=0.0, roughnessFactor=0.75
    )
    scene.add(pyrender.Mesh.from_trimesh(tri, material=material, smooth=True))
    scene.add(pyrender.IntrinsicsCamera(fx=focal, fy=focal, cx=size / 2, cy=size / 2, znear=0.05), pose=np.eye(4))
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0.0, 0.0, 1.0]
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=light_pose)
    renderer = pyrender.OffscreenRenderer(size, size)
    rgb, depth = renderer.render(scene)
    renderer.delete()

    # pyrender/OpenGL 相机朝 -Z；转成像素坐标时令 depth=-Z、图像 v 轴向下。
    z = -densekp_gl[:, 2]
    pixels = np.column_stack((focal * densekp_gl[:, 0] / z + size / 2,
                              -focal * densekp_gl[:, 1] / z + size / 2))
    image = Image.fromarray(rgb)
    image.save(args.output_dir / "smplx_rgb.png")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)
    for u, v in pixels:
        draw.ellipse((u - 2, v - 2, u + 2, v + 2), fill=(255, 20, 20))
    overlay.save(args.output_dir / "smplx_densekp_overlay.png")
    np.save(args.output_dir / "smplx_vertices.npy", vertices)
    np.save(args.output_dir / "smplx_densekp_2d.npy", pixels)
    np.save(args.output_dir / "smplx_depth.npy", depth)
    print(f"PASS vertices={vertices.shape} densekp={pixels.shape}")


if __name__ == "__main__":
    main()
