"""Replace Urban Street facades with the CC0 Poly Haven apartment kit.

The procedural baseline is read-only. Output is a separate asset-backed scene.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
ASSET = "modular_urban_apartments_facade"
SOURCE = ROOT / "assets/scenes/photoreal/urban_street/source" / ASSET


def upgrade(args):
    import bpy
    from mathutils import Matrix, Vector
    from build_urban_street import blender_main

    bpy.ops.wm.open_mainfile(filepath=str(args.baseline))
    scene = bpy.context.scene
    group = bpy.data.collections["ENV_Buildings"]
    shells = sorted((o for o in group.objects if o.get("role") == "building_shell"), key=lambda o: o.name)
    # Preserve only volumes and the distant horizon; replace all near facade details.
    for obj in list(group.objects):
        if obj not in shells and not obj.name.startswith("Distant_"):
            bpy.data.objects.remove(obj, do_unlink=True)
    before_images = set(bpy.data.images)
    with bpy.data.libraries.load(str(SOURCE / f"{ASSET}_2k.blend"), link=False) as (src, dst):
        dst.objects = src.objects
    templates = {o.name: o for o in dst.objects if o is not None}
    for image in set(bpy.data.images) - before_images:
        if image.source == "FILE":
            path = SOURCE / "textures" / Path(image.filepath).name
            if not path.is_file():
                raise FileNotFoundError(path)
            image.filepath = str(path)
            image.pack()
    # Evaluate the original modifiers once, then share the resulting meshes.
    for obj in templates.values():
        group.objects.link(obj)
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    meshes = {}
    plaster = bpy.data.materials[ASSET + "_plaster"]
    finishes = []
    for index, tint in enumerate(((0.84, 0.82, 0.73), (0.65, 0.74, 0.77), (0.88, 0.76, 0.59), (0.73, 0.78, 0.68), (0.78, 0.65, 0.59))):
        mat = plaster.copy()
        mat.name = f"PH painted plaster {index}"
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = next(n for n in nodes if n.type == "BSDF_PRINCIPLED")
        original = bsdf.inputs["Base Color"].links[0].from_socket
        desaturate = nodes.new("ShaderNodeHueSaturation")
        desaturate.inputs["Saturation"].default_value = 0.12
        links.new(original, desaturate.inputs["Color"])
        multiply = nodes.new("ShaderNodeMixRGB")
        multiply.blend_type = "MULTIPLY"
        multiply.inputs[0].default_value = 1
        multiply.inputs[2].default_value = (*tint, 1)
        links.new(desaturate.outputs[0], multiply.inputs[1])
        links.new(multiply.outputs[0], bsdf.inputs["Base Color"])
        finishes.append(mat)

    def place(name, side, front, y, z, tag, zscale=3.1 / 3):
        if name not in meshes:
            meshes[name] = bpy.data.meshes.new_from_object(templates[name].evaluated_get(depsgraph), preserve_all_data_layers=True, depsgraph=depsgraph)
        obj = bpy.data.objects.new(f"PH_{tag}_{name}", meshes[name])
        group.objects.link(obj)
        # Kit fronts face local -Y; local X spans [-3, 0].
        obj.matrix_world = Matrix.Translation(Vector((front, y, z))) @ Matrix.Rotation(-side * math.pi / 2, 4, "Z") @ Matrix.Diagonal((1, 1, zscale, 1))
        for slot in obj.material_slots:
            if slot.material == plaster:
                slot.link = "OBJECT"
                slot.material = finishes[index % len(finishes)]
        obj["asset_source"] = ASSET
        obj.pass_index = 0
        return obj

    for index, shell in enumerate(shells):
        side = -1 if "LEFT" in shell.name else 1
        front = side * (7.25 + (index % 3) * 0.12)
        center = shell.location.y
        floors = round((shell.dimensions.z - 0.4) / 3.1)
        # Set the solid backing 0.6 m behind the facade, retaining window recesses.
        shell.location.x = front + side * 5.6
        shell.dimensions.y = 12
        style = ("large", "small", "double")[index % 3]
        variant = (1, 2, 3)[(index // 3) % 3]
        for bay in range(4):
            y = center - 6 + bay * 3 + (3 if side == -1 else 0)
            tag = f"{shell.name}_{bay}"
            if bay == index % 4:
                door = "door_centered_large_01"
                place("wall_" + door, side, front, y, 0.15, tag)
                place(door, side, front, y, 0.15, tag)
                place("dado_" + door, side, front, y, 0.15, tag)
            else:
                window = "window_centered_large_03"
                place("wall_" + window, side, front, y, 0.80, tag, 2.45 / 3)
                place(window, side, front, y, 0.80, tag, 2.45 / 3)
                place("base_standard_01", side, front, y, 0.15, tag, 0.65 / 0.75)
            for floor in range(1, floors):
                window = f"window_centered_{style}_{variant:02d}"
                z = 0.15 + floor * 3.1
                place("wall_" + window, side, front, y, z, tag)
                place(window, side, front, y, z, tag)
                place("cornice_standard_standard_01", side, front, y, z, tag)
            place("crown_standard_standard_01", side, front, y, 0.15 + floors * 3.1, tag)
    for obj in templates.values():
        bpy.data.objects.remove(obj, do_unlink=True)
    # Reuse a small, documented subset of the already-local CC0 Hidden Alley.
    hidden = ROOT / "assets/scenes/photoreal/hidden_alley"
    prop_names = ["exterior_aircon_unit_rusted.001", "street_lamp_01.002", "potted_plant_01_pot.001", "potted_plant_01_leaves.001", "potted_plant_01_stem.001"]
    before_images = set(bpy.data.images)
    with bpy.data.libraries.load(str(hidden / "ph_hidden_alley.blend"), link=False) as (src, dst):
        dst.objects = list(prop_names)
    props = dict(zip(prop_names, dst.objects))
    reused_maps = []
    for image in set(bpy.data.images) - before_images:
        if image.source == "FILE":
            path = hidden / "textures" / Path(image.filepath).name
            if not path.is_file():
                raise FileNotFoundError(path)
            image.filepath = str(path)
            image.pack()
            reused_maps.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    furniture = bpy.data.collections["ENV_StreetProps"]
    for obj in list(furniture.objects):
        if obj.name.startswith(("Lamp_post", "Lamp_arm", "Lamp_head")):
            bpy.data.objects.remove(obj, do_unlink=True)
    def prop(name, xyz, angle=0):
        obj = props[name].copy()
        obj.name = "PH_detail_" + name
        obj.parent = None
        obj.animation_data_clear()
        obj.location = xyz
        obj.rotation_euler = (0, 0, angle)
        obj.scale = (1, 1, 1)
        obj.pass_index = 0
        furniture.objects.link(obj)
    for side in (-1, 1):
        for y in (-60, -34, -8, 18, 44, 68):
            prop("street_lamp_01.002", (side * 3.95, y, 0.15))
        for y in (-43, -19, 17, 41):
            for name in prop_names[2:]:
                prop(name, (side * 7.0, y, 0.15))
        for y in (-47, -23, 13, 37):
            prop("exterior_aircon_unit_rusted.001", (side * 7.02, y, 3.1), -side * math.pi / 2)
    for obj in props.values():
        bpy.data.objects.remove(obj, do_unlink=True)
    scene["facade_asset"] = ASSET
    scene["facade_license"] = "CC0-1.0"
    bpy.context.view_layer.update()
    args.scene_file.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(args.scene_file))
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    manifest["reused_hidden_alley"] = {"source": "https://blog.polyhaven.com/hidden-alley/", "license": "CC0-1.0", "objects": prop_names, "textures": reused_maps}
    (args.output_dir / "asset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    blender_main(SimpleNamespace(preview_only=True, scene_file=args.scene_file, output_dir=args.output_dir, skip_render=False, samples=args.samples, width=args.width, seed=73))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender-bin", type=Path)
    parser.add_argument("--baseline", type=Path, default=ROOT / "assets/scenes/photoreal/urban_street/urban_street.blend")
    parser.add_argument("--scene-file", type=Path, default=ROOT / "assets/scenes/photoreal/urban_street/urban_street_assets.blend")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/scenes/urban_street_assets")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--samples", type=int, default=48)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    for key in ("baseline", "scene_file", "output_dir"):
        setattr(args, key, getattr(args, key).resolve())
    if args.baseline == args.scene_file:
        parser.error("Asset output must not overwrite the baseline")
    if args.width < 64 or args.samples < 1:
        parser.error("Invalid render settings")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import bpy
    except ImportError:
        from densekp_retrain.runtime import find_blender_executable
        from densekp_retrain.rendering.process import run_logged
        command = [str(find_blender_executable(args.blender_bin)), "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--", "--baseline", str(args.baseline), "--scene-file", str(args.scene_file), "--output-dir", str(args.output_dir), "--width", str(args.width), "--samples", str(args.samples)]
        run_logged(command, cwd=ROOT, log_path=args.output_dir / "upgrade.log")
    else:
        upgrade(args)


if __name__ == "__main__":
    main()
