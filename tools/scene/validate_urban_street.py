"""Reopen Urban Street and check scale, assets, and sampled standing clearance."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def validate(args):
    import bpy
    from mathutils import Vector

    bpy.ops.wm.open_mainfile(filepath=str(args.scene_file))
    scene = bpy.context.scene
    depsgraph = bpy.context.evaluated_depsgraph_get()
    errors = []

    def check(condition, message):
        if not condition:
            errors.append(message)

    required = ("ENV_Road", "ENV_Sidewalk", "ENV_Buildings", "ENV_StreetProps", "ENV_Vegetation", "SPAWN_MARKERS", "CAMERA_MARKERS", "LIGHTS", "CAMERAS")
    for name in required:
        coll = bpy.data.collections.get(name)
        check(coll is not None and len(coll.all_objects) > 0, f"Missing or empty collection: {name}")
    check(scene.unit_settings.system == "METRIC" and abs(scene.unit_settings.scale_length - 1) < 1e-8, "Expected metric scale 1 unit = 1 m")
    road = bpy.data.objects.get("Road_surface_7m")
    check(road is not None and abs(road.dimensions.x - 7) < 1e-5 and road.dimensions.y >= 140, "Road width/length invalid")
    sidewalks = {}
    for side in ("LEFT", "RIGHT"):
        obj = bpy.data.objects.get(f"Sidewalk_{side}_3.5m")
        check(obj is not None and abs(obj.dimensions.x - 3.5) < 1e-5 and obj.dimensions.y >= 140, f"Sidewalk dimensions invalid: {side}")
        sidewalks[side] = list(obj.dimensions) if obj else None
    buildings = [obj for obj in scene.objects if obj.get("role") == "building_shell"]
    for side in ("LEFT", "RIGHT"):
        check(sum(side in obj.name for obj in buildings) >= 5, f"Insufficient buildings on {side}")
    check(all(8 <= obj.dimensions.z <= 25 and abs(obj["floor_height_m"] - 3.1) < 1e-5 for obj in buildings), "Building scale outside design range")
    missing = []
    for image in bpy.data.images:
        if image.source == "FILE" and not image.packed_file and not Path(bpy.path.abspath(image.filepath)).is_file():
            missing.append(image.filepath)
    check(not missing, f"Missing texture dependencies: {missing}")
    check(not any(obj.type == "ARMATURE" for obj in scene.objects), "Unexpected rig")
    check(not any(any(word in obj.name.lower() for word in ("person", "pedestrian", "bicycle", "cyclist", "smpl")) for obj in scene.objects), "Unexpected person/bicycle object")
    check(all(obj.pass_index == 0 for obj in scene.objects), "Environment pass indices must remain zero")
    asset_facades = [obj for obj in scene.objects if obj.get("asset_source")]
    if scene.get("facade_asset"):
        check(len(asset_facades) >= 26 * 4 * 3, "Asset-backed scene has insufficient facade modules")
        check(sum(bool(image.packed_file) for image in bpy.data.images if image.source == "FILE") >= 19, "Asset-backed textures are not packed")
        check(any(obj.name.startswith("PH_detail_") for obj in scene.objects), "Missing reused street details")

    # Unlike the forest's scatter-specific audit, this street has explicit meshes.
    # Probe a footprint from head height down: the first hit must be its support
    # plane. Also test the footprint against obstacle AABBs to catch enclosure.
    obstacles = []
    for group in ("ENV_Buildings", "ENV_StreetProps", "ENV_Vegetation"):
        coll = bpy.data.collections.get(group)
        if coll:
            for obj in coll.all_objects:
                if obj.type != "MESH" or obj.hide_render:
                    continue
                bounds = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
                obstacles.append((obj.name, [min(p[i] for p in bounds) for i in range(3)], [max(p[i] for p in bounds) for i in range(3)]))

    samples = 0
    intrusions = []

    def support(x, y, expected_z, context):
        nonlocal samples
        samples += 1
        hit, loc, normal, face, obj, matrix = scene.ray_cast(depsgraph, Vector((x, y, expected_z + 2.0)), Vector((0, 0, -1)), distance=2.1)
        if not hit or abs(loc.z - expected_z) > 0.012:
            intrusions.append({"context": context, "xy": [x, y], "object": obj.name if obj else None, "z": loc.z if hit else None})

    markers = {}
    for side, sign in (("LEFT", -1), ("RIGHT", 1)):
        expected = [f"SPAWN_SIDEWALK_{side}_{distance}_001" for distance in ("NEAR", "MID", "FAR")]
        expected += [f"SPAWN_ROADSIDE_{side}_001", f"SPAWN_EDGE_{side}_001"]
        for name in expected:
            obj = bpy.data.objects.get(name)
            check(obj is not None and obj.type == "EMPTY", f"Missing spawn empty: {name}")
            if obj is None:
                continue
            x, y, z = obj.location
            markers[name] = [x, y, z]
            expected_z = 0 if "ROADSIDE" in name else 0.15
            check(abs(z - expected_z) < 1e-5, f"Wrong ground height: {name}")
            for dx in (-0.35, 0, 0.35):
                for dy in (-0.35, 0, 0.35):
                    support(x + dx, y + dy, expected_z, name)
            for obstacle, lo, hi in obstacles:
                if hi[2] > z + 0.025 and lo[2] < z + 1.95 and lo[0] < x + 0.35 and hi[0] > x - 0.35 and lo[1] < y + 0.35 and hi[1] > y - 0.35:
                    errors.append(f"Standing footprint overlaps {obstacle}: {name}")
        for y in range(-70, 71, 2):
            for x in (4.75, 5.25, 5.75, 6.25, 6.65):
                support(sign * x, y, 0.15, f"{side} continuous walking corridor")
    for y in range(-70, 71, 2):
        for x in (-3.1, -1.5, 0, 1.5, 3.1):
            support(x, y, 0, "road")
    camera_markers = {}
    for name in ("CAMERA_STREET_NEAR", "CAMERA_STREET_MID", "CAMERA_STREET_LONG", "CAMERA_SIDEWALK_LEFT", "CAMERA_SIDEWALK_RIGHT"):
        obj = bpy.data.objects.get(name)
        check(obj is not None and obj.type == "EMPTY", f"Missing camera empty: {name}")
        if obj:
            camera_markers[name] = list(obj.location)
            ground = 0.15 if "SIDEWALK" in name else 0
            check(abs(obj.location.z - ground - 1.65) < 1e-5, f"Camera not at 1.65 m above support: {name}")
            support(obj.location.x, obj.location.y, ground, name)
    check(not intrusions, f"{len(intrusions)} sampled support/clearance failures")
    previews = {}
    if args.require_previews:
        for name in ("street_level.png", "sidewalk.png", "long_street.png", "panorama.png"):
            path = args.output_dir / name
            check(path.is_file(), f"Missing preview: {name}")
            if path.is_file():
                image = bpy.data.images.load(str(path), check_existing=False)
                width, height = image.size
                check(width >= 64 and height > 0, f"Invalid preview image: {name}")
                previews[name] = [width, height]
                bpy.data.images.remove(image)
    report = {"passed": not errors, "scene": str(args.scene_file), "units": "meters; Z up", "road_dimensions": list(road.dimensions) if road else None, "sidewalk_dimensions": sidewalks, "building_count": len(buildings), "building_height_range": [min(obj.dimensions.z for obj in buildings), max(obj.dimensions.z for obj in buildings)] if buildings else [], "spawn_markers": markers, "camera_markers": camera_markers, "clear_sidewalk_width_m": 1.9, "sampled_corridor_length_m": 140, "ray_samples": samples, "missing_textures": missing, "intrusions": intrusions, "errors": errors, "previews": previews, "limits": "Discrete 0.7 m footprint and 2 m headroom tests plus conservative obstacle AABBs; not a continuous collision certificate or a human sampler."}
    report["facade_asset"] = scene.get("facade_asset")
    report["asset_facade_modules"] = len(asset_facades)
    (args.output_dir / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("URBAN_STREET_VALIDATION", json.dumps(report), flush=True)
    if errors:
        raise RuntimeError("Urban street validation failed; see validation.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender-bin", type=Path)
    parser.add_argument("--scene-file", type=Path, default=ROOT / "assets/scenes/photoreal/urban_street/urban_street.blend")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/scenes/urban_street")
    parser.add_argument("--require-previews", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    args.scene_file = args.scene_file.resolve()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.scene_file.is_file():
        parser.error(f"Scene file not found: {args.scene_file}")
    try:
        import bpy
    except ImportError:
        from densekp_retrain.runtime import find_blender_executable
        from densekp_retrain.rendering.process import run_logged

        command = [str(find_blender_executable(args.blender_bin)), "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--", "--scene-file", str(args.scene_file), "--output-dir", str(args.output_dir)]
        if args.require_previews:
            command.append("--require-previews")
        log = args.output_dir / "validate.log"
        print(f"Blender log: {log}", flush=True)
        run_logged(command, cwd=ROOT, log_path=log)
        print((args.output_dir / "validation.json").read_text())
    else:
        validate(args)


if __name__ == "__main__":
    main()
