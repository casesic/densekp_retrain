"""Independent command/path/failure observations; never starts Blender."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import patch

CASES = (
    "relative",
    "absolute",
    "camera",
    "no_clothing",
    "existing_directory",
    "existing_link",
    "dangling_link",
    "missing_report",
    "invalid_report_json",
    "launch_failure",
    "resolver_failure",
    "missing_manifest",
    "empty_manifest",
    "missing_field",
    "missing_texture",
)


def normalize(value, root):
    if isinstance(value, dict):
        return {k: normalize(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [normalize(v, root) for v in value]
    if isinstance(value, str):
        return value.replace(str(root.resolve()), "<ROOT>").replace(
            str(Path.home()), "<HOME>"
        )
    return value


@contextmanager
def cwd(path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def capture(output, jobs):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for route in ("eye", "temporal"):
        for case in CASES:
            base = output / f"{route}_{case}"
            root = base / "project"
            work = base / "cwd"
            root.mkdir(parents=True)
            work.mkdir()
            baseline = root / "baseline data"
            (baseline / "geometry").mkdir(parents=True)
            geometry = work / "geometry space"
            geometry.mkdir()
            record = dict(
                body_diffuse_texture_path="textures/body color.png",
                body_uv_path="uv relative.npz",
                hdri_path="environment relative.exr",
            )
            if case == "missing_field":
                del record["body_uv_path"]
            manifest = baseline / "manifest.jsonl"
            if case != "missing_manifest":
                manifest.write_text(
                    "" if case == "empty_manifest" else json.dumps(record) + "\n"
                )
            # Each job has one real input manifest; no model, scene or render data needed.
            args = SimpleNamespace(
                baseline_dir=baseline if case == "absolute" else Path("baseline data"),
                geometry=geometry if case == "absolute" else Path("geometry space"),
                output_dir=(
                    base / "output" if case == "absolute" else Path("render out")
                ),
                clothing_texture_dir=(
                    None if case == "missing_texture" else Path("~/fixture clothing")
                ),
                camera_placement=(
                    Path("camera relative.json") if case == "camera" else None
                ),
                blender_bin=Path("blender relative"),
                frames=3,
                resolution=256,
                appearance_manifest=manifest,
                hdri=Path("environment relative.exr"),
                hdri_rotation=17,
                clothing_geometry=(
                    None if case == "no_clothing" else Path("cloth relative")
                ),
            )
            events = []

            def find_blender(value):
                events.append(dict(event="resolve_blender", input=str(value)))
                if case == "resolver_failure":
                    raise RuntimeError("fixture resolver failure")
                return Path("/fixture/blender")

            def run_logged(command, *, cwd, log_path):
                events.append(
                    dict(
                        event="launch",
                        command=list(command),
                        cwd=str(cwd),
                        log_path=str(log_path),
                        output_exists=args.output_dir.is_dir(),
                        geometry_exists=(args.output_dir / "geometry").exists(),
                    )
                )
                if case == "launch_failure":
                    raise subprocess.CalledProcessError(7, command)
                if case != "missing_report":
                    (args.output_dir / "p0_render_report.json").write_text(
                        "not json"
                        if case == "invalid_report_json"
                        else '{"fixture":true}\n'
                    )

            with globals()["cwd"](work), patch.object(
                jobs, "find_blender_executable", find_blender
            ), patch.object(jobs, "run_logged", run_logged):
                if case in ("existing_directory", "existing_link", "dangling_link"):
                    args.output_dir.mkdir(parents=True)
                    link = args.output_dir / "geometry"
                    if case == "existing_directory":
                        link.mkdir()
                    else:
                        link.symlink_to(
                            geometry if case == "existing_link" else base / "missing",
                            target_is_directory=True,
                        )
                try:
                    function = (
                        jobs.run_eye_background_fix
                        if route == "eye"
                        else jobs.run_temporal_case
                    )
                    value = function(args, root)
                    exception = None
                except Exception as error:
                    value = None
                    exception = dict(type=type(error).__name__, message=str(error))
                output_state = []
                if args.output_dir.exists():
                    for p in sorted(args.output_dir.iterdir()):
                        output_state.append(
                            dict(
                                name=p.name,
                                symlink=p.is_symlink(),
                                target=os.readlink(p) if p.is_symlink() else None,
                                directory=p.is_dir(),
                                text=p.read_text() if p.is_file() else None,
                            )
                        )
            records.append(
                dict(
                    route=route,
                    case=case,
                    events=events,
                    exception=exception,
                    return_value=value,
                    output=output_state,
                )
            )
    result = normalize(records, output)
    (output / "records.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
