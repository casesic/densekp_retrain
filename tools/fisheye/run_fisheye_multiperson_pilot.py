#!/usr/bin/env python3
"""Render a paired ERP/fisheye multi-person pilot with per-person 138-point GT."""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "assets_external/python_deps"))

from densekp_retrain.camera import OmniRadtanIntrinsics  # noqa: E402
from densekp_retrain.io import read_combined_depth_index  # noqa: E402
from densekp_retrain.rendering.process import run_logged  # noqa: E402
from densekp_retrain.runtime import find_blender_executable, load_appearance_baseline  # noqa: E402
from densekp_retrain.imaging.omni import warp_case


from densekp_retrain.body.transforms import world_points  # historical export


from densekp_retrain.camera.erp import project_erp  # historical export


from densekp_retrain.camera.visibility import erp_visibility  # historical export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry",
        type=Path,
        default=ROOT / "outputs/baselines/temporal_case_aaron_1095/geometry",
    )
    parser.add_argument(
        "--clothing-geometry",
        type=Path,
        default=ROOT / "outputs/baselines/temporal_case_aaron_1095/clothing_geometry",
    )
    parser.add_argument(
        "--people-config", type=Path, default=ROOT / "configs/fisheye_people_pilot.json"
    )
    parser.add_argument(
        "--cameras",
        type=Path,
        default=ROOT / "outputs/baselines/fisheye_camera_p0/camera_config.json",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "outputs/experiments/fisheye_multiperson_pilot"
    )
    parser.add_argument("--blender-bin", type=Path)
    parser.add_argument("--appearance-manifest", type=Path)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--skip-render", action="store_true")
    args = parser.parse_args()
    people = json.loads(args.people_config.read_text())["people"]
    ids = [int(person["id"]) for person in people]
    if len(people) < 2 or len(ids) != len(set(ids)) or min(ids) < 1:
        parser.error("people-config needs at least two people with unique positive IDs")
    vertices = np.load(args.geometry / "vertices.npy", mmap_mode="r")
    dense = np.load(args.geometry / "densekp_3d.npy", mmap_mode="r")
    for person in people:
        if not 0 <= int(person["frame"]) < len(vertices):
            parser.error(f"frame out of range for person {person['id']}")
    points = np.stack(
        [world_points(dense[int(p["frame"])], vertices[int(p["frame"])], p) for p in people]
    )
    if points.shape != (len(people), 138, 3):
        raise ValueError(f"unexpected 3D points shape: {points.shape}")
    camera_config = json.loads(args.cameras.read_text())
    intrinsics = OmniRadtanIntrinsics.from_dict(camera_config["intrinsics"])
    source = args.output / "source"
    source.mkdir(parents=True, exist_ok=True)
    if not args.skip_render:
        appearance, _ = load_appearance_baseline(args.appearance_manifest)
        command = [
            str(find_blender_executable(args.blender_bin)),
            "-b",
            "--python",
            str(ROOT / "tools/fisheye/render_equirectangular_source.py"),
            "--",
            "--geometry",
            str(args.geometry),
            "--clothing-geometry",
            str(args.clothing_geometry),
            "--cameras",
            str(args.cameras),
            "--output-dir",
            str(source),
            "--people-config",
            str(args.people_config),
            "--width",
            str(args.width),
            "--height",
            str(args.height),
            "--samples",
            str(args.samples),
            "--body-texture",
            appearance["body_diffuse_texture_path"],
            "--body-uv",
            appearance["body_uv_path"],
            "--hdri",
            appearance["hdri_path"],
        ]
        run_logged(command, cwd=ROOT, log_path=args.output / "render.log")
    results = []
    for camera_item in camera_config["cameras"]:
        camera_id = camera_item["id"]
        camera = np.asarray(camera_item["camera_to_world"], dtype=np.float64)
        rgb, depth, index, _ = read_combined_depth_index(source / f"{camera_id}.exr")
        h, w = depth.shape
        erp_xy, distance = project_erp(points.reshape(-1, 3), camera, w, h)
        erp_valid, erp_visible = erp_visibility(erp_xy, distance, depth, index, ids)
        cam_dir = args.output / camera_id
        cam_dir.mkdir(parents=True, exist_ok=True)
        for person_id in ids:
            mask = (np.rint(index) == person_id).astype(np.uint8) * 255
            Image.fromarray(mask).save(cam_dir / f"erp_person_{person_id}_mask.png")
        np.savez(
            cam_dir / "erp_annotations.npz",
            person_ids=np.asarray(ids),
            xy=erp_xy.reshape(len(people), 138, 2).astype(np.float32),
            valid=erp_valid.reshape(len(people), 138),
            visible=erp_visible.reshape(len(people), 138),
        )
        fish_dir = cam_dir / "fisheye"
        fish_report = warp_case(
            source / f"{camera_id}.exr",
            fish_dir,
            points.reshape(-1, 3),
            camera,
            intrinsics,
            person_ids=ids,
            point_person_ids=np.repeat(ids, 138),
        )
        np.savez(
            fish_dir / "person_annotations.npz",
            person_ids=np.asarray(ids),
            xy=np.load(fish_dir / "densekp_fisheye_2d.npy").reshape(len(people), 138, 2),
            valid=np.load(fish_dir / "densekp_valid.npy").reshape(len(people), 138),
            visible=np.load(fish_dir / "densekp_visible.npy").reshape(len(people), 138),
        )
        results.append(
            {
                "camera": camera_id,
                "erp_visible": erp_visible.reshape(len(people), 138).sum(axis=1).tolist(),
                "fisheye_visible": np.load(fish_dir / "densekp_visible.npy")
                .reshape(len(people), 138)
                .sum(axis=1)
                .tolist(),
                "fisheye_projection_error_max_px": fish_report["projection_error_px"]["max"],
            }
        )
    report = {
        "people": people,
        "camera_model": camera_config["camera_model"],
        "erp_size": [args.width, args.height],
        "fisheye_size": [intrinsics.width, intrinsics.height],
        "results": results,
        "note": "pilot repeats one motion sequence across people; not a validation set",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
