#!/usr/bin/env python3
"""Render the frozen heavy DS scene at three source ERP resolutions.

This bounded experiment keeps the physical 2880-square DS raster and writes a
5760x2880 normal ERP for every source resolution.  It does not train a model.
"""

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools")]

import numpy as np

from densekp_retrain.camera.double_sphere import DS_20260320
from densekp_retrain.imaging.ds import process_scene
from densekp_retrain.runtime import load_appearance_baseline
from tools.fisheye.run_ds_scene_pair_pilot import load_lens_masks


GROUPS = (("A", 1536, 768), ("B", 3072, 1536), ("C", 5760, 2880))
CAMERAS = ("cam2", "cam4")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def gpu_used_mib(index):
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        capture_output=True,
        check=True,
    )
    rows = [row.split(",") for row in result.stdout.splitlines()]
    values = {int(gpu.strip()): int(used.strip()) for gpu, used in rows}
    if index not in values:
        raise ValueError(f"GPU index {index} is not reported by nvidia-smi")
    return values[index]


def run_render(command, log_path, gpu_index):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    before = gpu_used_mib(gpu_index)
    peak = before
    started = time.monotonic()
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        while process.poll() is None:
            try:
                peak = max(peak, gpu_used_mib(gpu_index))
            except (OSError, subprocess.SubprocessError, ValueError):
                pass
            time.sleep(0.5)
    if process.returncode:
        raise subprocess.CalledProcessError(process.returncode, command)
    return dict(
        wall_seconds=time.monotonic() - started,
        gpu_index=gpu_index,
        gpu_memory_before_mib=before,
        sampled_gpu_memory_peak_mib=peak,
        sampled_gpu_memory_increment_mib=max(0, peak - before),
        sampling_interval_seconds=0.5,
    )


def directory_bytes(path):
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs/experiments/ds_source_resolution_20261008",
    )
    parser.add_argument(
        "--historical-heavy",
        type=Path,
        default=ROOT / "outputs/experiments/ds_occlusion_control_20260928/heavy",
    )
    parser.add_argument(
        "--blender-bin",
        type=Path,
        default=Path("/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender"),
    )
    parser.add_argument(
        "--fisheye-reference-root",
        type=Path,
        default=Path(
            "/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/data_process/Output/"
            "20260320_riding_dataset3/image/fisheye"
        ),
    )
    parser.add_argument("--reference-frame", default="02214")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--final-width", type=int, default=5760)
    parser.add_argument("--final-height", type=int, default=2880)
    args = parser.parse_args()

    output = args.output.resolve()
    resumed_execution = None
    if output.exists():
        execution_path = output / "execution.json"
        if execution_path.is_file():
            resumed_execution = json.loads(execution_path.read_text())
        rendered = list(output.glob("*/source/source_manifest.json"))
        if not (
            resumed_execution
            and resumed_execution.get("status") == "failed"
            and not resumed_execution.get("attempts")
            and not rendered
        ):
            raise FileExistsError(f"Use a fresh output directory: {output}")
    else:
        output.mkdir(parents=True)

    historical = args.historical_heavy.resolve()
    people = json.loads((historical / "people.json").read_text())["people"]
    rig = json.loads((historical / "rig.json").read_text())
    if rig["calibration_group"] != "20260320":
        raise ValueError("Expected the 20260320 DS calibration")
    if rig["intrinsics"] != {name: asdict(DS_20260320[name]) for name in CAMERAS}:
        raise ValueError("Frozen rig does not match active 20260320 intrinsics")
    poses = {
        camera["id"]: np.asarray(camera["camera_to_world"])
        for camera in rig["cameras"]
    }

    geometry = ROOT / "outputs/baselines/temporal_case_aaron_1095/geometry"
    clothing = geometry.parent / "clothing_geometry"
    scene = ROOT / "assets/scenes/photoreal/urban_street/urban_street_assets.blend"
    appearance, appearance_manifest = load_appearance_baseline()
    assets = [
        args.blender_bin,
        geometry / "vertices.npy",
        geometry / "densekp_3d.npy",
        clothing / "clothing_vertices.npy",
        scene,
        Path(appearance["body_diffuse_texture_path"]),
        Path(appearance["body_uv_path"]),
        ROOT / "assets_external/real_erp_valid_masks.npz",
    ]
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        write_json(output / "failure.json", dict(stage="asset_preflight", missing=missing))
        raise FileNotFoundError(missing)

    lens_masks, lens_metadata = load_lens_masks(
        args.fisheye_reference_root, args.reference_frame
    )
    with np.load(ROOT / "assets_external/real_erp_valid_masks.npz") as archive:
        erp_masks = {
            name: archive[f"valid_masks_{name}"].copy() for name in CAMERAS
        }
    vertices = np.load(geometry / "vertices.npy", mmap_mode="r")
    dense = np.load(geometry / "densekp_3d.npy", mmap_mode="r")

    fixed_source = ROOT / "outputs/experiments/ds_occlusion_amodal_20260929/fixed_boxes.json"
    old_boxes = json.loads(fixed_source.read_text())
    scale_x = args.final_width / 1536
    scale_y = args.final_height / 768
    fixed_rows = []
    for row in old_boxes:
        x1, y1, x2, y2 = row["bbox_xyxy_exclusive"]
        box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
        center = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
        side = max(box[2] - box[0], box[3] - box[1]) * 1.2
        fixed_rows.append(
            dict(
                camera=row["camera"],
                person_id=row["person_id"],
                bbox_xyxy_exclusive=box,
                center=center,
                scale=side / 200,
            )
        )

    plan = dict(
        status="frozen_before_render",
        historical_heavy=str(historical),
        historical_reproduction_limit=(
            "old source did not freeze every external texture/scene byte and predates the "
            "current complete lens-mask path; all A/B/C groups are freshly rendered"
        ),
        groups=[dict(id=name, source_size=[width, height]) for name, width, height in GROUPS],
        cameras=list(CAMERAS),
        render_count=6,
        final_erp_size=[args.final_width, args.final_height],
        ds_size=[2880, 2880],
        samples=args.samples,
        seed=args.seed,
        animated_seed=False,
        engine="Cycles",
        denoising=False,
        scene_blend=str(scene.resolve()),
        people=people,
        rig=rig,
        fixed_box_source=str(fixed_source.resolve()),
        fixed_boxes=fixed_rows,
        box_mapping=[scale_x, scale_y],
        geometry=str(geometry.resolve()),
        clothing_geometry=str(clothing.resolve()),
        appearance_manifest=str(appearance_manifest),
        body_texture=appearance["body_diffuse_texture_path"],
        body_uv=appearance["body_uv_path"],
        lens_masks=lens_metadata,
        visibility=(
            "periodic source-raster 5x5 instance/ray-range neighborhood; fixed pixel "
            "neighborhood therefore changes angular support with source resolution; "
            "strict 0.04m physical range tolerance unchanged"
        ),
        interpolation=dict(
            source_rgb_to_fisheye="periodic cv2.INTER_LINEAR",
            source_instance_to_fisheye="periodic cv2.INTER_NEAREST after rint",
            fisheye_rgb_to_final_erp="cv2.INTER_LINEAR",
            fisheye_instance_to_final_erp="cv2.INTER_NEAREST",
            masks="cv2.INTER_NEAREST",
        ),
        gpu_index=args.gpu_index,
        gpu_visibility="CUDA_VISIBLE_DEVICES restricts Blender to the selected physical GPU",
    )
    write_json(output / "experiment_plan.json", plan)

    execution = resumed_execution or dict(status="running", attempts=[], failures=[])
    execution["status"] = "running"
    if resumed_execution:
        execution.setdefault("resume_events", []).append(
            "resumed after pre-render resource probe failure; zero renders existed"
        )
    write_json(output / "execution.json", execution)
    try:
        for group, width, height in GROUPS:
            folder = output / group
            source = folder / "source"
            source.mkdir(parents=True, exist_ok=True)
            write_json(folder / "people.json", dict(people=people))
            write_json(folder / "rig.json", rig)
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
                str(folder / "rig.json"),
                "--people-config",
                str(folder / "people.json"),
                "--scene-blend",
                str(scene),
                "--output-dir",
                str(source),
                "--width",
                str(width),
                "--height",
                str(height),
                "--samples",
                str(args.samples),
                "--seed",
                str(args.seed),
                "--device",
                "CUDA",
                "--body-texture",
                appearance["body_diffuse_texture_path"],
                "--body-uv",
                appearance["body_uv_path"],
            ]
            print(f"Rendering group {group}: {width}x{height}", flush=True)
            render_metrics = run_render(command, folder / "render.log", args.gpu_index)
            manifest = json.loads((source / "source_manifest.json").read_text())
            if manifest["resolution"] != [width, height]:
                raise ValueError(f"{group}: source resolution mismatch")
            if manifest["samples"] != args.samples or manifest["seed"] != args.seed:
                raise ValueError(f"{group}: render settings mismatch")
            if manifest["people"] != people or manifest["animated_seed"]:
                raise ValueError(f"{group}: people or animated seed changed")
            for camera in manifest["sources"]:
                np.testing.assert_allclose(
                    camera["camera_to_world"], poses[camera["id"]], atol=1e-12, rtol=0
                )

            print(f"Post-processing group {group}", flush=True)
            post_started = time.monotonic()
            rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result = process_scene(
                folder,
                people,
                poses,
                erp_masks,
                vertices,
                dense,
                lens_masks=lens_masks,
                lens_metadata=lens_metadata,
                output_size=(args.final_width, args.final_height),
                output_mode="full",
            )
            post_metrics = dict(
                wall_seconds=time.monotonic() - post_started,
                process_peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                peak_rss_before_stage_mib=rss_before / 1024,
            )
            result.update(
                group=group,
                source_erp_size=[width, height],
                final_erp_size=[args.final_width, args.final_height],
                fisheye_size=[2880, 2880],
                people=people,
                samples=args.samples,
                seed=args.seed,
                render_metrics=render_metrics,
                postprocess_metrics=post_metrics,
            )
            write_json(folder / "report.json", result)
            execution["attempts"].append(
                dict(
                    group=group,
                    source_size=[width, height],
                    status="completed",
                    render=render_metrics,
                    postprocess=post_metrics,
                    output_bytes=directory_bytes(folder),
                )
            )
            write_json(output / "execution.json", execution)
    except Exception as error:
        execution["status"] = "failed"
        execution["failures"].append(
            dict(type=type(error).__name__, message=str(error), time=time.time())
        )
        write_json(output / "execution.json", execution)
        raise
    execution["status"] = "render_and_postprocess_completed"
    execution["output_bytes"] = directory_bytes(output)
    write_json(output / "execution.json", execution)


if __name__ == "__main__":
    main()
