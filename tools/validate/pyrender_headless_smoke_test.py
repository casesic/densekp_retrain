#!/usr/bin/env python3
"""在无 DISPLAY 环境下验证 pyrender EGL 的最小离屏渲染。"""

import argparse
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import pyrender
import trimesh
from OpenGL import GL
from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/smoke/headless"))
    parser.add_argument("--frames", type=int, default=10)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    renderer = pyrender.OffscreenRenderer(args.size, args.size)
    startup_seconds = time.perf_counter() - started
    renderer._platform.make_current()
    gl_vendor = GL.glGetString(GL.GL_VENDOR).decode("utf-8", errors="replace")
    gl_renderer = GL.glGetString(GL.GL_RENDERER).decode("utf-8", errors="replace")
    gl_version = GL.glGetString(GL.GL_VERSION).decode("utf-8", errors="replace")
    renderer._platform.make_uncurrent()

    scene = pyrender.Scene(bg_color=[24, 32, 48, 255], ambient_light=[0.08, 0.08, 0.08])
    primitive = trimesh.creation.icosphere(subdivisions=3, radius=0.75)
    colors = np.empty((len(primitive.vertices), 4), dtype=np.uint8)
    colors[:] = [70, 155, 240, 255]
    colors[primitive.vertices[:, 2] > 0] = [245, 150, 55, 255]
    primitive.visual.vertex_colors = colors
    mesh_node = scene.add(pyrender.Mesh.from_trimesh(primitive, smooth=True))

    camera = pyrender.IntrinsicsCamera(
        fx=220.0, fy=220.0, cx=args.size / 2, cy=args.size / 2, znear=0.05, zfar=20.0
    )
    camera_pose = np.eye(4)
    camera_pose[2, 3] = 3.0
    scene.add(camera, pose=camera_pose)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
    scene.add(light, pose=camera_pose)

    render_times = []
    color = depth = None
    for frame in range(args.frames):
        angle = 2.0 * np.pi * frame / max(args.frames, 1)
        pose = np.eye(4)
        pose[:3, :3] = trimesh.transformations.rotation_matrix(angle, [0, 1, 0])[:3, :3]
        scene.set_pose(mesh_node, pose)
        before = time.perf_counter()
        color, depth = renderer.render(scene)
        render_times.append(time.perf_counter() - before)

    assert color is not None and depth is not None
    Image.fromarray(color).save(args.output_dir / "pyrender_rgb.png")
    np.save(args.output_dir / "pyrender_depth.npy", depth)
    segmentation, _ = renderer.render(
        scene, flags=pyrender.RenderFlags.SEG, seg_node_map={mesh_node: [255, 255, 255]}
    )
    Image.fromarray(segmentation).save(args.output_dir / "pyrender_mask.png")
    renderer.delete()

    valid_depth = depth[depth > 0]
    report = {
        "platform": os.environ.get("PYOPENGL_PLATFORM"),
        "display": os.environ.get("DISPLAY"),
        "gl_vendor": gl_vendor,
        "gl_renderer": gl_renderer,
        "gl_version": gl_version,
        "size": [args.size, args.size],
        "frames": args.frames,
        "startup_seconds": startup_seconds,
        "mean_render_seconds": float(np.mean(render_times)),
        "min_render_seconds": float(np.min(render_times)),
        "max_render_seconds": float(np.max(render_times)),
        "max_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "depth_finite": bool(np.isfinite(depth).all()),
        "depth_nonzero_pixels": int(np.count_nonzero(depth)),
        "depth_min_nonzero": float(valid_depth.min()) if valid_depth.size else None,
        "depth_max": float(depth.max()),
    }
    (args.output_dir / "pyrender_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
