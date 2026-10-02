#!/usr/bin/env python3
"""Run independent Omni-Radtan center and dual-camera overlap validation."""

import argparse, hashlib, json, sys, time
from pathlib import Path
import cv2, numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from densekp_retrain.camera import (
    OmniRadtanIntrinsics,
    pose,
    project_points_omni_radtan,
    unproject_pixels_omni_radtan,
)
from densekp_retrain.io import linear_to_srgb, read_combined_depth_index
from densekp_retrain.rendering.process import run_logged
from densekp_retrain.runtime import find_blender_executable, load_appearance_baseline

from densekp_retrain.imaging.omni import scalar_reference  # historical export

from densekp_retrain.imaging.omni import warp_case  # historical export


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--geometry",
        type=Path,
        default=Path("outputs/baselines/temporal_case_aaron_1095/geometry"),
    )
    p.add_argument(
        "--clothing-geometry",
        type=Path,
        default=Path("outputs/baselines/temporal_case_aaron_1095/clothing_geometry"),
    )
    p.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=Path,
        default=Path("outputs/baselines/fisheye_camera_p0"),
    )
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--pano-width", type=int, default=2048)
    p.add_argument("--pano-height", type=int, default=1024)
    p.add_argument("--skip-render", action="store_true")
    p.add_argument("--blender-bin", type=Path)
    p.add_argument("--appearance-manifest", type=Path)
    a = p.parse_args()
    root = ROOT
    out = a.output_dir
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    vertices_all = np.load(a.geometry / "vertices.npy")
    dense_all = np.load(a.geometry / "densekp_3d.npy")
    vertices = vertices_all[a.frame]
    dense = dense_all[a.frame]
    source_fingerprint = hashlib.blake2b(dense[None].tobytes(), digest_size=16).hexdigest()
    np.save(out / "densekp_3d_source.npy", dense[None])
    if (
        hashlib.blake2b(
            np.load(out / "densekp_3d_source.npy").tobytes(), digest_size=16
        ).hexdigest()
        != source_fingerprint
    ):
        raise RuntimeError("densekp_3d_source.npy fingerprint mismatch after save")
    lift = -float(vertices[:, 1].min())
    world = np.column_stack((dense[:, 0], -dense[:, 2], dense[:, 1] + lift))
    vw = np.column_stack((vertices[:, 0], -vertices[:, 2], vertices[:, 1] + lift))
    target = (vw.min(0) + vw.max(0)) / 2
    origin = target - np.array([0, 3.2, 0])
    cams = [
        {"id": "fisheye_center", "camera_to_world": pose(target, origin, 0).tolist()},
        {"id": "cam_left", "camera_to_world": pose(target, origin, 75).tolist()},
        {"id": "cam_right", "camera_to_world": pose(target, origin, -75).tolist()},
    ]
    camera_file = out / "camera_config.json"
    intr = OmniRadtanIntrinsics()
    camera_file.write_text(
        json.dumps(
            {
                "camera_model": "Omni-Radtan",
                "intrinsics": intr.as_dict(),
                "coordinate_system": "CV +X right, +Y down, +Z forward; camera_to_world",
                "cameras": cams,
            },
            indent=2,
        )
        + "\n"
    )
    source = out / "equirectangular_source"
    source.mkdir(exist_ok=True)
    if not a.skip_render:
        baseline, _ = load_appearance_baseline(a.appearance_manifest)
        cmd = [
            str(find_blender_executable(a.blender_bin)),
            "-b",
            "--python",
            str(root / "tools/fisheye/render_equirectangular_source.py"),
            "--",
            "--geometry",
            str(a.geometry.resolve()),
            "--clothing-geometry",
            str(a.clothing_geometry.resolve()),
            "--cameras",
            str(camera_file.resolve()),
            "--output-dir",
            str(source.resolve()),
            "--frame",
            str(a.frame),
            "--body-texture",
            baseline["body_diffuse_texture_path"],
            "--body-uv",
            baseline["body_uv_path"],
            "--hdri",
            baseline["hdri_path"],
            "--width",
            str(a.pano_width),
            "--height",
            str(a.pano_height),
        ]
        run_logged(cmd, cwd=root, log_path=out / "render.log")
        if not (source / "source_manifest.json").is_file():
            raise RuntimeError(f'Blender source render failed; inspect {out/"render.log"}')
    reports = {}
    reports["center"] = warp_case(
        source / "fisheye_center.exr",
        out / "center",
        world,
        np.asarray(cams[0]["camera_to_world"]),
        intr,
        dense[None],
        source_fingerprint,
    )
    for i, name in enumerate(["cam_left", "cam_right"], 1):
        reports[name] = warp_case(
            source / f"{name}.exr",
            out / "seam" / name,
            world,
            np.asarray(cams[i]["camera_to_world"]),
            intr,
            dense[None],
            source_fingerprint,
        )
    fingerprint_consistent = all(
        x["densekp_3d_fingerprint"]["matches_projection_source"]
        and x["densekp_3d_fingerprint"]["value"] == source_fingerprint
        for x in reports.values()
    )
    fingerprint_consistent &= (
        hashlib.blake2b(
            np.load(out / "densekp_3d_source.npy").tobytes(), digest_size=16
        ).hexdigest()
        == source_fingerprint
    )
    report = {
        "baseline_commit": "1ad425978882ac96fbdaacec6fa15d1e21ec2a1b",
        "geometry": str(a.geometry.resolve()),
        "geometry_frame": a.frame,
        "densekp_3d_shape": [1, len(world), 3],
        "densekp_3d_fingerprint": {
            "algorithm": "blake2b-128",
            "value": source_fingerprint,
            "consistent_across_cases": fingerprint_consistent,
        },
        "intrinsics": intr.as_dict(),
        "case_center_pass": reports["center"]["pass"],
        "case_seam_pass": reports["cam_left"]["pass"] and reports["cam_right"]["pass"],
        "densekp_3d_fingerprint_consistent": fingerprint_consistent,
        "fisheye_camera_pass": all(x["pass"] for x in reports.values())
        and fingerprint_consistent,
        "cases": reports,
        "elapsed_seconds": time.perf_counter() - started,
    }
    (out / "regression_report.json").write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "phase": "FISHEYE_CAMERA_P0",
        "camera_model": "Omni-Radtan",
        "intrinsics": intr.as_dict(),
        "baseline_commit": report["baseline_commit"],
        "geometry": report["geometry"],
        "geometry_frame": a.frame,
        "densekp_3d_fingerprint": report["densekp_3d_fingerprint"],
        "annotation_format": {
            "densekp_fisheye_2d": "[T,138,2] float32, top-left-origin pixels",
            "densekp_valid": "[T,138] bool, projected inside image",
            "densekp_visible": "[T,138] bool, depth-visibility within 4 cm",
            "densekp_depth_error": "[T,138] float32, |surface_depth - keypoint_range| in meters",
        },
        "outputs": {
            "center": {
                "rgb": "center/rgb.png",
                "depth": "center/depth.npy",
                "mask": "center/mask.png",
                "densekp_2d": "center/densekp_fisheye_2d.npy",
                "densekp_valid": "center/densekp_valid.npy",
                "densekp_visible": "center/densekp_visible.npy",
                "densekp_3d_source": "densekp_3d_source.npy",
                "overlay": "center/densekp_overlay.png",
                "case_report": "center/case_report.json",
            },
            "seam": {
                "cam_left": {
                    "rgb": "seam/cam_left/rgb.png",
                    "depth": "seam/cam_left/depth.npy",
                    "mask": "seam/cam_left/mask.png",
                    "densekp_2d": "seam/cam_left/densekp_fisheye_2d.npy",
                    "densekp_valid": "seam/cam_left/densekp_valid.npy",
                    "densekp_visible": "seam/cam_left/densekp_visible.npy",
                    "overlay": "seam/cam_left/densekp_overlay.png",
                    "case_report": "seam/cam_left/case_report.json",
                },
                "cam_right": {
                    "rgb": "seam/cam_right/rgb.png",
                    "depth": "seam/cam_right/depth.npy",
                    "mask": "seam/cam_right/mask.png",
                    "densekp_2d": "seam/cam_right/densekp_fisheye_2d.npy",
                    "densekp_valid": "seam/cam_right/densekp_valid.npy",
                    "densekp_visible": "seam/cam_right/densekp_visible.npy",
                    "overlay": "seam/cam_right/densekp_overlay.png",
                    "case_report": "seam/cam_right/case_report.json",
                },
            },
            "equirectangular_source": "equirectangular_source/source_manifest.json",
            "camera_config": "camera_config.json",
            "regression_report": "regression_report.json",
        },
        "metrics": {
            "finite_keypoints": {n: int(r["finite_keypoints"]) for n, r in reports.items()},
            "valid_keypoints": {n: int(r["valid_keypoints"]) for n, r in reports.items()},
            "visible_keypoints": {n: int(r["visible_keypoints"]) for n, r in reports.items()},
            "max_projection_error_px": {
                n: float(r["projection_error_px"]["max"]) for n, r in reports.items()
            },
            "max_roundtrip_angular_error_rad": {
                n: float(r["roundtrip_angular_error_rad"]["max"]) for n, r in reports.items()
            },
        },
        "status": {
            "FISHEYE_CAMERA_P0": "PASS" if report["fisheye_camera_pass"] else "FAIL",
            "CENTER_CASE": "PASS" if report["case_center_pass"] else "FAIL",
            "SEAM_CASE": "PASS" if report["case_seam_pass"] else "FAIL",
        },
    }
    (out / "report.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
