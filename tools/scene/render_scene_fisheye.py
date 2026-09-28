"""Render an environment-only Blender scene through the frozen Omni-Radtan model."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender-bin", type=Path)
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--camera", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pano-width", type=int, default=4096)
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--source-only", action="store_true", help=argparse.SUPPRESS)
    values = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    values.scene = values.scene.resolve()
    values.output_dir = values.output_dir.resolve()
    return values


def blender_source(args):
    import bpy

    bpy.ops.wm.open_mainfile(filepath=str(args.scene))
    scene = bpy.context.scene
    reference = bpy.data.objects.get(args.camera)
    if reference is None or reference.type != "CAMERA":
        available = sorted(obj.name for obj in scene.objects if obj.type == "CAMERA")
        raise RuntimeError(f"Camera {args.camera!r} not found; available={available}")
    data = bpy.data.cameras.new("OmniRadtan_ERP_Source")
    data.type = "PANO"
    data.panorama_type = "EQUIRECTANGULAR"
    data.longitude_min = -np.pi
    data.longitude_max = np.pi
    data.latitude_min = -np.pi / 2
    data.latitude_max = np.pi / 2
    camera = bpy.data.objects.new("OmniRadtan_ERP_Source", data)
    scene.collection.objects.link(camera)
    camera.matrix_world = reference.matrix_world.copy()
    scene.camera = camera
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = args.samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x = args.pano_width
    scene.render.resolution_y = args.pano_width // 2
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.filepath = str(args.output_dir / "equirectangular.png")
    bpy.ops.render.render(write_still=True)
    pose = [[float(value) for value in row] for row in reference.matrix_world]
    (args.output_dir / "source.json").write_text(json.dumps({
        "scene": str(args.scene), "reference_camera": args.camera,
        "camera_matrix_world_blender": pose,
        "projection": "equirectangular", "resolution": [args.pano_width, args.pano_width // 2],
        "samples": args.samples, "engine": "CYCLES", "device": "CPU",
    }, indent=2) + "\n")


def warp(args):
    from densekp_retrain.camera import OmniRadtanIntrinsics, unproject_pixels_omni_radtan

    intrinsics = OmniRadtanIntrinsics(
        xi=3.485643, f=4495.392, k1=7.048e-01, k2=7.548,
        cx=1500.0, cy=1500.0, width=3000, height=3000,
    )
    source_path = args.output_dir / "equirectangular.png"
    source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Cannot read source panorama: {source_path}")
    height, width = source.shape[:2]
    yy, xx = np.indices((intrinsics.height, intrinsics.width), dtype=np.float64)
    pixels = np.column_stack((xx.ravel(), yy.ravel()))
    rays, domain = unproject_pixels_omni_radtan(pixels, intrinsics, return_valid=True)
    longitude = np.arctan2(rays[:, 0], rays[:, 2])
    latitude = np.arcsin(np.clip(-rays[:, 1], -1, 1))
    map_x = ((longitude / (2 * np.pi) + 0.5) * width).reshape(intrinsics.height, intrinsics.width).astype(np.float32)
    map_y = ((0.5 - latitude / np.pi) * height).reshape(intrinsics.height, intrinsics.width).astype(np.float32)
    valid = domain.reshape(intrinsics.height, intrinsics.width)
    map_x[~valid] = -1
    map_y[~valid] = -1
    fisheye = cv2.remap(source, map_x, map_y, cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
    mask = valid.astype(np.uint8) * 255
    cv2.imwrite(str(args.output_dir / "fisheye.png"), fisheye)
    cv2.imwrite(str(args.output_dir / "valid_mask.png"), mask)
    report = {
        "camera_model": "Omni-Radtan",
        "formula": "m=sin(theta)/(cos(theta)+xi); r=f*(m+k1*m^3+k2*m^5)",
        "intrinsics": intrinsics.as_dict(),
        "nominal_fov_deg": 200.0,
        "source": str(source_path),
        "output": str((args.output_dir / "fisheye.png").resolve()),
        "valid_pixels": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
    }
    (args.output_dir / "fisheye_report.json").write_text(json.dumps(report, indent=2) + "\n")


def main():
    args = arguments()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import bpy  # noqa: F401
    except ImportError:
        from densekp_retrain.runtime import find_blender_executable
        command = [
            str(find_blender_executable(args.blender_bin)), "-b", "--disable-autoexec",
            "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--",
            "--scene", str(args.scene), "--camera", args.camera,
            "--output-dir", str(args.output_dir), "--pano-width", str(args.pano_width),
            "--samples", str(args.samples), "--source-only",
        ]
        with (args.output_dir / "render.log").open("w") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        warp(args)
    else:
        if not args.source_only:
            raise RuntimeError("Blender invocation requires --source-only")
        blender_source(args)


if __name__ == "__main__":
    main()
