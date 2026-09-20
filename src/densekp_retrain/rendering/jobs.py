"""Frozen perspective render jobs called by thin CLI wrappers."""

import json
from pathlib import Path

from densekp_retrain.runtime import (
    find_blender_executable,
    load_appearance_baseline,
)

from .process import run_logged


def run_eye_background_fix(args, root: Path) -> None:
    baseline = args.baseline_dir if args.baseline_dir.is_absolute() else root / args.baseline_dir
    record = json.loads((baseline / "manifest.jsonl").read_text().splitlines()[0])
    clothing_texture_dir = args.clothing_texture_dir.expanduser().resolve()
    command = [
        str(find_blender_executable(args.blender_bin)),
        "-b", "-t", "4", "--python", str(root / "tools/render/render_p0_blender.py"), "--",
        "--input-geometry", str(baseline / "geometry"),
        "--clothing-geometry", str(baseline / "geometry"),
        "--output-dir", str(args.output_dir),
        "--clothing-texture-id", "texture_01",
        "--clothing-texture-dir", str(clothing_texture_dir),
        "--body-texture", record["body_diffuse_texture_path"],
        "--body-uv", record["body_uv_path"],
        "--environment-mode", "hdri",
        "--hdri-path", record["hdri_path"],
        "--hdri-strength", "1.0",
        "--hdri-rotation-deg", "0",
        "--manual-lights", "fill",
        "--eye-render-mode", "procedural",
        "--eye-mapping", str(root / "outputs/eye_fix_v2/procedural_eye_coordinates.npz"),
        "--ground-mode", "shadow_catcher",
        "--render-frame-count", str(args.frames),
        "--resolution", str(args.resolution),
    ]
    if args.camera_placement:
        command += ["--camera-placement", str(args.camera_placement.resolve())]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_logged(command, cwd=root, log_path=args.output_dir / "render.log")
    if not (args.output_dir / "p0_render_report.json").is_file():
        raise RuntimeError(
            f"Blender did not finish rendering; inspect {args.output_dir / 'render.log'}"
        )
    geometry_link = args.output_dir / "geometry"
    if not geometry_link.exists():
        geometry_link.symlink_to((baseline / "geometry").resolve(), target_is_directory=True)


def run_temporal_case(args, root: Path) -> None:
    baseline, _ = load_appearance_baseline(args.appearance_manifest)
    command = [
        str(find_blender_executable(args.blender_bin)),
        "-b", "-t", "4", "--python", str(root / "tools/render/render_p0_blender.py"), "--",
        "--input-geometry", str(args.geometry.resolve()),
        "--output-dir", str(args.output_dir.resolve()),
        "--render-frame-count", str(args.frames),
        "--resolution", str(args.resolution),
        "--appearance-baseline",
        "--body-texture", baseline["body_diffuse_texture_path"],
        "--body-uv", baseline["body_uv_path"],
        "--environment-mode", "hdri",
        "--hdri-path", str(args.hdri.resolve()),
        "--hdri-strength", "1.0",
        "--hdri-rotation-deg", str(args.hdri_rotation),
        "--manual-lights", "fill",
        "--eye-render-mode", "procedural",
        "--eye-mapping", str((root / "outputs/eye_fix_v2/procedural_eye_coordinates.npz").resolve()),
        "--ground-mode", "shadow_catcher",
    ]
    if args.clothing_geometry:
        command += [
            "--clothing-geometry", str(args.clothing_geometry.resolve()),
            "--clothing-texture-id", "texture_01",
            "--clothing-texture-dir", str(args.clothing_texture_dir.resolve()),
        ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_logged(command, cwd=root, log_path=args.output_dir / "render.log")
    if not (args.output_dir / "p0_render_report.json").is_file():
        raise RuntimeError(f"Blender did not finish; inspect {args.output_dir / 'render.log'}")
    geometry_link = args.output_dir / "geometry"
    if not geometry_link.exists():
        geometry_link.symlink_to(args.geometry.resolve(), target_is_directory=True)

