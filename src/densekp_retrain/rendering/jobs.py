"""Two frozen perspective recipes and their shared command execution.

Builders read inputs and return a reviewable command without creating output or
launching Blender. Recipe flags remain explicit here; no universal renderer or
scene recipe registry is introduced. Legacy run_* functions preserve their API.
"""

from dataclasses import dataclass
import json
from pathlib import Path
from densekp_retrain.runtime import find_blender_executable, load_appearance_baseline
from .process import run_logged


@dataclass(frozen=True)
class GeometryRenderJob:
    """Exact argv plus the geometry link source (resolved only after render success)."""

    command: list[str]
    geometry: Path


def build_eye_background_job(args, root: Path) -> GeometryRenderJob:
    baseline = (
        args.baseline_dir
        if args.baseline_dir.is_absolute()
        else root / args.baseline_dir
    )
    record = json.loads((baseline / "manifest.jsonl").read_text().splitlines()[0])
    clothing_texture_dir = args.clothing_texture_dir.expanduser().resolve()
    command = [
        str(find_blender_executable(args.blender_bin)),
        "-b",
        "-t",
        "4",
        "--python",
        str(root / "tools/render/render_p0_blender.py"),
        "--",
        "--input-geometry",
        str(baseline / "geometry"),
        "--clothing-geometry",
        str(baseline / "geometry"),
        "--output-dir",
        str(args.output_dir),
        "--clothing-texture-id",
        "texture_01",
        "--clothing-texture-dir",
        str(clothing_texture_dir),
        "--body-texture",
        record["body_diffuse_texture_path"],
        "--body-uv",
        record["body_uv_path"],
        "--environment-mode",
        "hdri",
        "--hdri-path",
        record["hdri_path"],
        "--hdri-strength",
        "1.0",
        "--hdri-rotation-deg",
        "0",
        "--manual-lights",
        "fill",
        "--eye-render-mode",
        "procedural",
        "--eye-mapping",
        str(root / "outputs/eye_fix_v2/procedural_eye_coordinates.npz"),
        "--ground-mode",
        "shadow_catcher",
        "--render-frame-count",
        str(args.frames),
        "--resolution",
        str(args.resolution),
    ]
    if args.camera_placement:
        command += ["--camera-placement", str(args.camera_placement.resolve())]
    return GeometryRenderJob(command, baseline / "geometry")


def build_temporal_job(args, root: Path) -> GeometryRenderJob:
    baseline, _ = load_appearance_baseline(args.appearance_manifest)
    command = [
        str(find_blender_executable(args.blender_bin)),
        "-b",
        "-t",
        "4",
        "--python",
        str(root / "tools/render/render_p0_blender.py"),
        "--",
        "--input-geometry",
        str(args.geometry.resolve()),
        "--output-dir",
        str(args.output_dir.resolve()),
        "--render-frame-count",
        str(args.frames),
        "--resolution",
        str(args.resolution),
        "--appearance-baseline",
        "--body-texture",
        baseline["body_diffuse_texture_path"],
        "--body-uv",
        baseline["body_uv_path"],
        "--environment-mode",
        "hdri",
        "--hdri-path",
        str(args.hdri.resolve()),
        "--hdri-strength",
        "1.0",
        "--hdri-rotation-deg",
        str(args.hdri_rotation),
        "--manual-lights",
        "fill",
        "--eye-render-mode",
        "procedural",
        "--eye-mapping",
        str((root / "outputs/eye_fix_v2/procedural_eye_coordinates.npz").resolve()),
        "--ground-mode",
        "shadow_catcher",
    ]
    if args.clothing_geometry:
        command += [
            "--clothing-geometry",
            str(args.clothing_geometry.resolve()),
            "--clothing-texture-id",
            "texture_01",
            "--clothing-texture-dir",
            str(args.clothing_texture_dir.resolve()),
        ]
    return GeometryRenderJob(command, args.geometry)


def execute_geometry_job(
    job: GeometryRenderJob, *, root: Path, output_dir: Path, missing_report_message: str
) -> None:
    """Create output, launch, require the existing report file, then link geometry.

    Preserve is_file-only report acceptance and the old dangling-link failure;
    this layer does not repair artifacts or reinterpret a render report.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    run_logged(job.command, cwd=root, log_path=output_dir / "render.log")
    if not (output_dir / "p0_render_report.json").is_file():
        raise RuntimeError(missing_report_message)
    geometry_link = output_dir / "geometry"
    if not geometry_link.exists():
        geometry_link.symlink_to(job.geometry.resolve(), target_is_directory=True)


def run_eye_background_fix(args, root: Path) -> None:
    job = build_eye_background_job(args, root)
    execute_geometry_job(
        job,
        root=root,
        output_dir=args.output_dir,
        missing_report_message=f"Blender did not finish rendering; inspect {args.output_dir / 'render.log'}",
    )


def run_temporal_case(args, root: Path) -> None:
    job = build_temporal_job(args, root)
    execute_geometry_job(
        job,
        root=root,
        output_dir=args.output_dir,
        missing_report_message=f"Blender did not finish; inspect {args.output_dir / 'render.log'}",
    )
