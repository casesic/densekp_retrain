#!/usr/bin/env python3
"""Small same-world, opposite-facing camera pilot in the two selected scenes.

20260320 intrinsics; same-height 180-degree opposing optical axes as specified
by the user. Baseline distance is an explicit synthetic pilot parameter, not a
claim about measured hardware. Produces diagnostic annotations, not train/val NPZ.
"""

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [
    str(ROOT),
    str(ROOT / "src"),
    str(ROOT / "tools"),
    str(ROOT / "assets_external/python_deps"),
]
import cv2
import numpy as np
from PIL import Image, ImageDraw
from densekp_retrain.camera.double_sphere import (
    DS_20260320,
    CAMERA_EXTRINSIC,
    project_double_sphere,
    unproject_double_sphere,
    build_erp_to_fisheye_lut,
    build_fisheye_to_source_erp_lut,
)
from densekp_retrain.io import read_combined_depth_index
from densekp_retrain.runtime import load_appearance_baseline
from densekp_retrain.imaging.remap import remap_periodic_erp
from densekp_retrain.imaging.remap import lens_mask_from_reference
from densekp_retrain.body.transforms import world_points
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.camera.visibility import erp_visibility

SCENES = {
    "cycling_forest": (
        "configs/fisheye_scene_cycling_forest.json",
        [13.676, -4.25, 0.0],
        -17.2,
    ),
    "urban_street_assets": (
        "configs/fisheye_scene_urban_street_assets.json",
        [0.0, 0.0, 0.0],
        0.0,
    ),
}


def add_lens_mask_arguments(parser):
    parser.add_argument(
        "--fisheye-reference-root",
        type=Path,
        required=True,
        help="Real 20260320 fisheye directory containing cam2 and cam4; only the binary footprint is used",
    )
    parser.add_argument(
        "--reference-frame", default="02214", help="Reference JPG stem, shared by both cameras"
    )


def load_lens_masks(reference_root, reference_frame):
    masks, metadata = {}, {}
    for name, intrinsics in DS_20260320.items():
        path = reference_root.resolve() / name / f"{reference_frame}.jpg"
        masks[name] = lens_mask_from_reference(path, (intrinsics.height, intrinsics.width))
        metadata[name] = dict(
            reference_image=str(path),
            method="gray > 5; elliptical close 51x51; fill largest external contour",
            erode_px=0,
            valid_pixels=int(masks[name].sum()),
            valid_fraction=float(masks[name].mean()),
        )
    return masks, metadata


from densekp_retrain.camera.rig import rig_poses  # historical export


from densekp_retrain.camera.domain import point_in_view  # historical export


def select_people(origin, yaw_deg, yaw, poses, vertices, dense, masks):
    people = []
    for person_id, sign, frame in [(1, 1, 0), (2, -1, 8)]:
        choices = []
        for distance in [0.9, 1.2, 1.5, 1.8, 2.2, 2.6]:
            for angle in np.linspace(77, 103, 27) * sign:
                radians = math.radians(angle)
                local = np.array(
                    [distance * math.sin(radians), distance * math.cos(radians), 0.0]
                )
                person = dict(
                    id=person_id,
                    frame=frame,
                    location=(np.asarray(origin) + yaw @ local).tolist(),
                    yaw_deg=yaw_deg + 90,
                    category="edge_partial",
                )
                points = world_points(dense[frame], vertices[frame], person)
                view = [
                    point_in_view(points, poses[n], n, masks[n])[-1] for n in ("cam2", "cam4")
                ]
                counts = [int(v.sum()) for v in view]
                common = int((view[0] & view[1]).sum())
                if min(counts) >= 20 and max(counts) <= 110 and common >= 8:
                    score = common - 1.5 * sum(abs(c - 90) for c in counts)
                    choices.append((score, person, counts, common))
        if not choices:
            raise RuntimeError("No paired edge-truncated layout found under supplied masks")
        _, person, counts, common = max(choices, key=lambda row: row[0])
        person.update(projected_in_view=counts, shared_in_view=common)
        people.append(person)
    # A farther person partially overlaps the right-side target in the image.
    farther = dict(people[0])
    farther.update(id=3, frame=16, category="overlap_behind_edge_person")
    relative = np.asarray(people[0]["location"]) - origin
    farther["location"] = (
        np.asarray(origin) + relative * 2.0 + yaw @ [0.0, 0.30, 0.0]
    ).tolist()
    farther.pop("projected_in_view")
    farther.pop("shared_in_view")
    people.append(farther)
    people.append(
        dict(
            id=4,
            frame=24,
            location=(np.asarray(origin) + yaw @ [0.0, 3.8, 0.0]).tolist(),
            yaw_deg=yaw_deg,
            category="complete_single_view_control",
        )
    )
    # Conservative spacing check: disjoint body AABBs guarantee no body intersection.
    boxes = []
    for person in people:
        body = world_points(vertices[person["frame"]], vertices[person["frame"]], person)
        boxes.append((body.min(0), body.max(0)))
    for i, (low, high) in enumerate(boxes):
        for other_low, other_high in boxes[i + 1 :]:
            if not ((high + 0.05 < other_low) | (other_high + 0.05 < low)).any():
                raise ValueError("Pilot body bounding boxes overlap or lack 5 cm clearance")
    return people


from densekp_retrain.visualization.imaging import draw_overlay  # historical export


from densekp_retrain.imaging.ds import process_scene  # historical export


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--baseline-m",
        type=float,
        default=0.18,
        help="Synthetic trial distance, not measured calibration",
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=Path("/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender"),
    )
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--device", choices=["CPU", "CUDA"], default="CPU")
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--scene", choices=list(SCENES), nargs="+", default=list(SCENES))
    add_lens_mask_arguments(parser)
    args = parser.parse_args()
    lens_masks, lens_metadata = load_lens_masks(
        args.fisheye_reference_root, args.reference_frame
    )
    geometry = ROOT / "outputs/baselines/temporal_case_aaron_1095/geometry"
    clothing = ROOT / "outputs/baselines/temporal_case_aaron_1095/clothing_geometry"
    vertices = np.load(geometry / "vertices.npy", mmap_mode="r")
    dense = np.load(geometry / "densekp_3d.npy", mmap_mode="r")
    appearance, _ = load_appearance_baseline()
    with np.load(ROOT / "assets_external/real_erp_valid_masks.npz") as z:
        masks = {n: z[f"valid_masks_{n}"] for n in ("cam2", "cam4")}
    for name in args.scene:
        config_path, origin, yaw_deg = SCENES[name]
        scene_config = json.loads((ROOT / config_path).read_text())
        output = args.output.resolve() / name
        output.mkdir(parents=True, exist_ok=True)
        poses, lenses, yaw = rig_poses(origin, yaw_deg, args.baseline_m)
        if args.skip_render:
            people = json.loads((output / "people.json").read_text())["people"]
            saved = json.loads((output / "rig.json").read_text())
            if saved["baseline_m"] != args.baseline_m:
                raise ValueError("Saved/rendered baseline differs")
            if saved["intrinsics"] != {n: asdict(DS_20260320[n]) for n in poses}:
                raise ValueError(
                    "Saved/rendered intrinsics differ from active projection parameters"
                )
            poses = {
                camera["id"]: np.asarray(camera["camera_to_world"])
                for camera in saved["cameras"]
            }
        else:
            if (output / "source/source_manifest.json").exists():
                raise FileExistsError(
                    "Existing rendered pilot; use --skip-render or a fresh output directory"
                )
            people = select_people(origin, yaw_deg, yaw, poses, vertices, dense, masks)
            rig = dict(
                camera_model="Double Sphere",
                calibration_group="20260320",
                pose_status="user-specified same-height 180-degree orientation; synthetic baseline distance",
                baseline_m=args.baseline_m,
                intrinsics={n: asdict(DS_20260320[n]) for n in poses},
                cameras=[
                    dict(
                        id=n,
                        camera_to_world=poses[n].tolist(),
                        physical_ds_camera_to_world=lenses[n].tolist(),
                    )
                    for n in poses
                ],
            )
            (output / "rig.json").write_text(json.dumps(rig, indent=2) + "\n")
            (output / "people.json").write_text(
                json.dumps(dict(people=people), indent=2) + "\n"
            )
            command = [
                str(args.blender_bin),
                "-b",
                "--disable-autoexec",
                "--python-exit-code",
                "1",
                "--python",
                str(ROOT / "tools/fisheye/render_equirectangular_source.py"),
                "--",
                "--geometry",
                str(geometry),
                "--clothing-geometry",
                str(clothing),
                "--cameras",
                str(output / "rig.json"),
                "--people-config",
                str(output / "people.json"),
                "--scene-blend",
                str(ROOT / scene_config["blend_path"]),
                "--output-dir",
                str(output / "source"),
                "--width",
                str(args.width),
                "--height",
                str(args.width // 2),
                "--samples",
                str(args.samples),
                "--device",
                args.device,
                "--body-texture",
                appearance["body_diffuse_texture_path"],
                "--body-uv",
                appearance["body_uv_path"],
            ]
            print("Rendering " + name, flush=True)
            with (output / "render.log").open("w") as log:
                subprocess.run(
                    command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True
                )
        rendered = json.loads((output / "source/source_manifest.json").read_text())
        if rendered["people"] != people:
            raise ValueError("Rendered people do not match annotation layout")
        for camera in rendered["sources"]:
            if not np.allclose(
                camera["camera_to_world"], poses[camera["id"]], atol=1e-12, rtol=0
            ):
                raise ValueError("Rendered camera differs from annotation camera")
        report = process_scene(
            output,
            people,
            poses,
            masks,
            vertices,
            dense,
            lens_masks=lens_masks,
            lens_metadata=lens_metadata,
        )
        report.update(
            scene=name,
            baseline_m=args.baseline_m,
            baseline_status="synthetic distance; not measured",
            orientation="same-height opposite optical axes",
            people=people,
            limitation="one repeated motion/identity, oracle boxes, low-sample rendering; not a train/val corpus",
            visibility_rule="matching instance and ray-range within 4 cm in a 5x5 source-ERP neighborhood",
            valid_mask_source="assets_external/real_erp_valid_masks.npz",
            regressor_source="assets/densekp_regressor_R.npy",
        )
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
