"""Render all eight deterministic configurations through the existing renderer."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Optional single config smoke run")
    parser.add_argument("--blender-bin", type=Path)
    parser.add_argument("--clothing-texture-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configs = json.loads((ROOT / "configs/perspective_camera_stress.json").read_text())
    output_dir = ROOT / "outputs/experiments/perspective_camera_stress"
    for config in configs["configs"]:
        if args.config and config["name"] != args.config:
            continue
        target = output_dir / "_renders" / config["name"]
        if (target / "p0_render_report.json").is_file() and not args.config:
            print("Keeping complete render", config["name"], flush=True)
            continue
        print("Rendering", config["name"], flush=True)
        command = [
            sys.executable,
            str(ROOT / "tools/render/run_eye_background_fix.py"),
            "--output-dir",
            str(target),
            "--frames",
            "16",
            "--camera-placement",
            str(output_dir / "configs" / f"{config['name']}.json"),
        ]
        if args.blender_bin:
            command += ["--blender-bin", str(args.blender_bin)]
        if args.clothing_texture_dir:
            command += [
                "--clothing-texture-dir",
                str(args.clothing_texture_dir),
            ]
        subprocess.run(command, check=True, cwd=ROOT)
        report = json.loads((target / "p0_render_report.json").read_text())
        if report["projection_error_px"]["max"] >= 0.1:
            raise RuntimeError("Unexpected projection error; stop before further configs")
        print("PASS projection", report["projection_error_px"]["max"], flush=True)


if __name__ == "__main__":
    main()
