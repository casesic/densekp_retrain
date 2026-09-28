"""Build a deterministic environment-only urban street, in Blender or via Python."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blender-bin", type=Path)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/scenes/urban_street")
    parser.add_argument("--scene-file", type=Path, default=ROOT / "assets/scenes/photoreal/urban_street/urban_street.blend")
    parser.add_argument("--samples", type=int, default=32)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--skip-render", action="store_true")
    parser.add_argument("--preview-only", action="store_true", help="Reopen the saved scene and render without rebuilding")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else None)
    if args.width < 64 or args.samples < 1:
        parser.error("width must be >=64 and samples must be positive")
    return args


def build(args):
    import bpy
    from mathutils import Vector

    rng = random.Random(args.seed)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    scene["purpose"] = "Environment only; modern urban street; no people or bicycles"
    scene["seed"] = args.seed
    collections = {}
    for name in ("ENV_Road", "ENV_Sidewalk", "ENV_Buildings", "ENV_StreetProps", "ENV_Vegetation", "SPAWN_MARKERS", "CAMERA_MARKERS", "LIGHTS", "CAMERAS"):
        coll = bpy.data.collections.new(name)
        scene.collection.children.link(coll)
        collections[name] = coll

    def move(obj, group):
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        collections[group].objects.link(obj)
        obj.pass_index = 0
        return obj

    def material(name, color, roughness=0.7, noise=0.12, metallic=0.0):
        mat = bpy.data.materials.new(name)
        mat.diffuse_color = (*color, 1)
        mat.use_nodes = True
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if noise:
            position = nodes.new("ShaderNodeNewGeometry")
            tex = nodes.new("ShaderNodeTexNoise")
            tex.inputs["Scale"].default_value = 18
            links.new(position.outputs["Position"], tex.inputs["Vector"])
            ramp = nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = (*(c * (1 - noise) for c in color), 1)
            ramp.color_ramp.elements[1].color = (*(min(1, c * (1 + noise)) for c in color), 1)
            links.new(tex.outputs["Fac"], ramp.inputs[0])
            links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
            bump = nodes.new("ShaderNodeBump")
            bump.inputs["Distance"].default_value = 0.009
            bump.inputs["Strength"].default_value = 0.18
            links.new(tex.outputs["Fac"], bump.inputs["Height"])
            links.new(bump.outputs[0], bsdf.inputs["Normal"])
        return mat

    texture_sources = []

    def scanned(mat, files, tile_size):
        nodes, links = mat.node_tree.nodes, mat.node_tree.links
        bsdf = nodes.get("Principled BSDF")
        texcoord = nodes.new("ShaderNodeNewGeometry")
        scale = nodes.new("ShaderNodeVectorMath")
        scale.operation = "SCALE"
        scale.inputs[3].default_value = 1 / tile_size
        links.new(texcoord.outputs["Position"], scale.inputs[0])
        for path, kind in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            texture_sources.append({"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            tex = nodes.new("ShaderNodeTexImage")
            tex.image = bpy.data.images.load(str(path), check_existing=True)
            tex.image.pack()
            tex.projection = "BOX"
            tex.projection_blend = 0.15
            links.new(scale.outputs[0], tex.inputs["Vector"])
            if kind != "Base Color":
                tex.image.colorspace_settings.name = "Non-Color"
            if kind == "Bump":
                bump = nodes.new("ShaderNodeBump")
                bump.inputs["Strength"].default_value = 0.16
                bump.inputs["Distance"].default_value = 0.018
                links.new(tex.outputs["Color"], bump.inputs["Height"])
                links.new(bump.outputs[0], bsdf.inputs["Normal"])
            else:
                links.new(tex.outputs["Color"], bsdf.inputs[kind])

    def box(name, loc, dimensions, mat, group, bevel=0):
        x, y, z = (value / 2 for value in dimensions)
        mesh = bpy.data.meshes.new(name)
        mesh.from_pydata(
            [(-x, -y, -z), (x, -y, -z), (x, y, -z), (-x, y, -z),
             (-x, -y, z), (x, -y, z), (x, y, z), (-x, y, z)],
            [],
            [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
             (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)],
        )
        obj = bpy.data.objects.new(name, mesh)
        obj.location = loc
        collections[group].objects.link(obj)
        mesh.materials.append(mat)
        if bevel:
            mod = obj.modifiers.new("Small physical edge bevel", "BEVEL")
            mod.width = bevel
            mod.segments = 2
        return obj

    def cylinder(name, start, end, radius, mat, group, vertices=12):
        direction = Vector(end) - Vector(start)
        bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=direction.length, location=(Vector(start) + Vector(end)) / 2)
        obj = bpy.context.object
        obj.name = name
        obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()
        obj.data.materials.append(mat)
        return move(obj, group)

    def label(text, loc, side, size=0.24):
        data = bpy.data.curves.new("Shop lettering", "FONT")
        data.body = text
        data.align_x = "CENTER"
        data.size = size
        data.extrude = 0.001
        obj = bpy.data.objects.new("Sign_" + text, data)
        collections["ENV_Buildings"].objects.link(obj)
        obj.location = loc
        # Text local +Z faces the street; text local +Y is vertical.
        obj.rotation_euler = (math.pi / 2, 0, side * math.pi / 2)
        data.materials.append(white)

    asphalt = material("Scanned asphalt 02 CC0", (0.10, 0.11, 0.12))
    source = ROOT / "assets/scenes/photoreal/cycling_forest/textures"
    scanned(asphalt, [(source / "asphalt_02_diff_2k.jpg", "Base Color"), (source / "asphalt_02_rough_2k.jpg", "Roughness")], 3)
    stone = material("Concrete sidewalk", (0.43, 0.43, 0.40), noise=0.25)
    curb = material("Granite curb", (0.35, 0.36, 0.35), noise=0.24)
    white = material("Worn warm white", (0.76, 0.75, 0.67), noise=0.18)
    metal = material("Dark powder coated steel", (0.055, 0.065, 0.069), 0.36, metallic=0.6)
    trim = material("Window aluminum", (0.31, 0.33, 0.33), 0.3, metallic=0.65)
    timber = material("Bench timber", (0.22, 0.11, 0.049), noise=0.4)
    soil = material("Tree pit soil", (0.055, 0.039, 0.021), noise=0.6)
    bark = material("Bark", (0.12, 0.076, 0.040), noise=0.5)
    leaves = [material(f"Leaf {i}", c, noise=0.25) for i, c in enumerate([(0.075, 0.15, 0.026), (0.15, 0.23, 0.045), (0.035, 0.09, 0.013)])]
    walls = [material(f"Facade finish {i}", c, noise=0.13) for i, c in enumerate([(0.62, 0.58, 0.48), (0.37, 0.39, 0.38), (0.57, 0.56, 0.52), (0.38, 0.22, 0.14), (0.72, 0.69, 0.60)])]
    hidden = ROOT / "assets/scenes/photoreal/hidden_alley/textures"
    scanned(walls[2], [(hidden / "grey_plaster_diff_1k.jpg", "Base Color"), (hidden / "grey_plaster_rough_1k.jpg", "Roughness")], 3)
    scanned(walls[3], [(hidden / "brick_wall_006_diff_2k.jpg", "Base Color")], 3)
    glasses = [material(f"Glazing {i}", c, 0.24, noise=0, metallic=0.75) for i, c in enumerate([(0.045, 0.065, 0.08), (0.085, 0.12, 0.14), (0.025, 0.032, 0.039), (0.13, 0.11, 0.08)])]
    shopcolors = [material(f"Shop fascia {i}", c, noise=0.06) for i, c in enumerate([(0.06, 0.15, 0.14), (0.12, 0.17, 0.23), (0.25, 0.11, 0.066), (0.12, 0.12, 0.11)])]

    box("Road_surface_7m", (0, 0, -0.15), (7, 160, 0.3), asphalt, "ENV_Road")
    box("Surrounding_ground", (0, 0, -0.36), (260, 260, 0.3), stone, "ENV_Road")
    for y in range(-74, 76, 6):
        box("Center_dash", (0, y, 0.004), (0.12, 3, 0.008), white, "ENV_Road")
    for side in (-1, 1):
        tag = "LEFT" if side == -1 else "RIGHT"
        box(f"Sidewalk_{tag}_3.5m", (side * 5.25, 0, -0.025), (3.5, 160, 0.35), stone, "ENV_Sidewalk")
        for y in range(-79, 80):
            box("Curb_block", (side * 3.55, y + 0.5, 0.065), (0.16, 0.987, 0.17), curb, "ENV_Sidewalk", 0.009)
        # Paving joints are flush shallow inlays; leave the walking plane continuous.
        for y in range(-79, 80, 2):
            box("Paving_joint", (side * 5.3, y, 0.151), (3.35, 0.012, 0.002), curb, "ENV_Sidewalk")
        box("Paving_long_joint", (side * 5.5, 0, 0.151), (0.012, 160, 0.002), curb, "ENV_Sidewalk")

        for index, y in enumerate(range(-72, 73, 12)):
            floors = rng.choice([3, 4, 4, 5, 6])
            height = floors * 3.1 + 0.4
            frontage = 11.6
            front = side * (7.10 + rng.choice([0, 0, 0.25, 0.5]))
            wall = walls[(index + (2 if side > 0 else 0)) % len(walls)]
            name = f"Building_{tag}_{index:02d}"
            shell = box(name, (front + side * 5, y, 0.15 + height / 2), (10, frontage, height), wall, "ENV_Buildings", 0.045)
            shell["role"] = "building_shell"
            shell["floor_height_m"] = 3.1
            box("Roof_coping", (front + side * 5, y, height + 0.18), (10.3, frontage + 0.15, 0.18), curb, "ENV_Buildings", 0.025)
            for floor in range(1, floors):
                z = 0.15 + 3.1 * floor + 1.45
                offsets = (-4.2, -1.4, 1.4, 4.2) if index % 3 != 1 else (-3.9, 0, 3.9)
                window_width = 1.61 if index % 3 != 1 else 2.35
                window_height = 1.72 if index % 3 != 2 else 2.1
                for offset in offsets:
                    yy = y + offset
                    box("Window_reveal", (front - side * 0.035, yy, z), (0.13, window_width + 0.23, window_height + 0.24), trim, "ENV_Buildings")
                    box("Window_glass", (front - side * 0.115, yy, z), (0.024, window_width, window_height), rng.choice(glasses), "ENV_Buildings")
                    box("Window_mullion", (front - side * 0.14, yy, z), (0.035, 0.05, window_height), trim, "ENV_Buildings")
                    box("Window_sill", (front - side * 0.17, yy, z - window_height / 2 - 0.14), (0.29, window_width + 0.43, 0.10), curb, "ENV_Buildings", 0.01)
                    if index % 4 == 2:
                        # Shallow Juliet railings vary silhouette without occupying pavement.
                        box("Juliet_toprail", (front - side * 0.38, yy, z - 0.1), (0.04, window_width + 0.2, 0.045), metal, "ENV_Buildings")
                        for rail in range(7):
                            box("Juliet_vertical", (front - side * 0.38, yy + (rail / 6 - 0.5) * window_width, z - 0.52), (0.022, 0.022, 0.83), metal, "ENV_Buildings")
                if index % 3 == 0:
                    box("Floor_band", (front - side * 0.035, y, 0.15 + floor * 3.1), (0.15, frontage, 0.16), stone, "ENV_Buildings")
            for offset in (-4.1, -1.4, 1.4, 4.1):
                box("Shop_frame", (front - side * 0.055, y + offset, 1.47), (0.15, 2.5, 2.58), metal, "ENV_Buildings")
                box("Shop_glass", (front - side * 0.15, y + offset, 1.47), (0.05, 2.29, 2.37), glasses[index % 4], "ENV_Buildings")
                box("Shop_handle", (front - side * 0.2, y + offset + 0.7, 1.25), (0.05, 0.04, 0.45), trim, "ENV_Buildings")
                box("Shop_transom", (front - side * 0.2, y + offset, 2.25), (0.04, 2.29, 0.055), metal, "ENV_Buildings")
                if index % 3 == 0:
                    box("Shop_lower_panel", (front - side * 0.19, y + offset, 0.47), (0.045, 2.29, 0.37), shopcolors[index % 4], "ENV_Buildings")
            box("Shop_fascia", (front - side * 0.18, y, 2.98), (0.2, 10.95, 0.46), shopcolors[index % 4], "ENV_Buildings", 0.02)
            label(["MARKET", "STUDIO", "COFFEE", "BOOKS", "PHARMACY", "OFFICES"][index % 6], (front - side * 0.30, y, 2.86), -side)
            if index % 4 == 1:
                box("Entrance_canopy", (front - side * 0.55, y, 2.69), (1.1, 3.0, 0.09), metal, "ENV_Buildings", 0.015)

        for y in (-60, -34, -8, 18, 44, 68):
            x = side * 3.95
            cylinder("Lamp_post", (x, y, 0.15), (x, y, 5.3), 0.065, metal, "ENV_StreetProps")
            cylinder("Lamp_arm", (x, y, 5.3), (x - side * 0.8, y, 5.3), 0.043, metal, "ENV_StreetProps")
            box("Lamp_head", (x - side * 0.85, y, 5.25), (0.65, 0.22, 0.12), metal, "ENV_StreetProps", 0.04)
        for y in (-50, 0, 50):
            x = side * 4.0
            box("Tree_pit", (x, y, 0.155), (0.8, 1.7, 0.01), soil, "ENV_Vegetation")
            cylinder("Tree_trunk", (x, y, 0.15), (x, y, 3.9), 0.105, bark, "ENV_Vegetation")
            for branch in range(6):
                angle = branch * math.tau / 6
                cylinder("Tree_branch", (x, y, 2.5), (x + 0.85 * math.cos(angle), y + 0.85 * math.sin(angle), 4.1), 0.035, bark, "ENV_Vegetation", 8)
            verts, faces = [], []
            for leaf in range(2500):
                azimuth = rng.uniform(0, math.tau)
                z = rng.uniform(-1, 1)
                radial = math.sqrt(1 - z * z) * rng.random() ** (1 / 3)
                center = Vector((x + 1.35 * radial * math.cos(azimuth), y + 1.35 * radial * math.sin(azimuth), 4.25 + 1.5 * z))
                u = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))).normalized() * rng.uniform(0.065, 0.13)
                v = u.cross(Vector((0, 0, 1))).normalized() * 0.045
                start = len(verts)
                verts.extend([center - u, center + v, center + u, center - v])
                faces.append((start, start + 1, start + 2, start + 3))
            mesh = bpy.data.meshes.new("Leaf cards")
            mesh.from_pydata(verts, [], faces)
            for mat in leaves:
                mesh.materials.append(mat)
            for poly in mesh.polygons:
                poly.material_index = rng.randrange(3)
            obj = bpy.data.objects.new("Urban_tree_canopy", mesh)
            collections["ENV_Vegetation"].objects.link(obj)
        for y in (-44, 36):
            for yy in (y - 0.7, y + 0.7):
                box("Bench_leg", (side * 4.05, yy, 0.34), (0.42, 0.07, 0.38), metal, "ENV_StreetProps")
            for xx in (3.83, 3.97, 4.11, 4.25):
                box("Bench_slat", (side * xx, y, 0.59), (0.11, 1.8, 0.065), timber, "ENV_StreetProps", 0.015)
            box("Bench_back", (side * 3.78, y, 0.85), (0.055, 1.8, 0.38), timber, "ENV_StreetProps", 0.015)
            cylinder("Trash_bin", (side * 4.0, y + 2, 0.15), (side * 4.0, y + 2, 0.98), 0.22, metal, "ENV_StreetProps")
        for y in (-24, 24):
            cylinder("Sign_post", (side * 3.92, y, 0.15), (side * 3.92, y, 2.9), 0.032, metal, "ENV_StreetProps")
            box("Traffic_sign", (side * 3.92, y, 2.65), (0.42, 0.045, 0.52), shopcolors[1], "ENV_StreetProps", 0.025)
            for yy in (y - 2, y + 2):
                cylinder("Bollard", (side * 3.85, yy, 0.15), (side * 3.85, yy, 0.95), 0.065, metal, "ENV_StreetProps")

    # A cross street at either end prevents a road stopping at a blank horizon.
    for y in (-84, 84):
        box("Cross_street", (0, y, -0.15), (160, 8, 0.3), asphalt, "ENV_Road")
        box("End_sidewalk", (0, y + math.copysign(6, y), 0.025), (160, 4, 0.25), stone, "ENV_Sidewalk")
        for x in range(-65, 66, 13):
            h = rng.choice([12, 15, 18])
            box("Distant_block", (x, y + math.copysign(13, y), h / 2), (12.6, 10, h), walls[(x // 13) % 5], "ENV_Buildings")
            for zz in range(4, h - 1, 3):
                for xx in range(-4, 5, 2):
                    box("Distant_window", (x + xx, y + math.copysign(7.98, y), zz), (1.2, 0.05, 1.6), glasses[1], "ENV_Buildings")

    def empty(name, xyz, group):
        obj = bpy.data.objects.new(name, None)
        collections[group].objects.link(obj)
        obj.location = xyz
        obj.empty_display_type = "ARROWS"
        obj.empty_display_size = 0.35
        return obj

    for side, tag in ((-1, "LEFT"), (1, "RIGHT")):
        for depth, y in (("NEAR", -28), ("MID", 2), ("FAR", 38)):
            empty(f"SPAWN_SIDEWALK_{tag}_{depth}_001", (side * 5.5, y, 0.15), "SPAWN_MARKERS")
        empty(f"SPAWN_ROADSIDE_{tag}_001", (side * 2.9, -18, 0), "SPAWN_MARKERS")
        empty(f"SPAWN_EDGE_{tag}_001", (side * 5.5, -33, 0.15), "SPAWN_MARKERS")

    views = [
        ("CAMERA_STREET_NEAR", (-1.8, -36, 1.65), (1, 15, 4.0), 29),
        ("CAMERA_SIDEWALK_LEFT", (-5.45, -30, 1.80), (-5.0, 18, 3.1), 26),
        ("CAMERA_STREET_LONG", (1.5, -65, 1.65), (-0.5, 28, 4.0), 40),
        ("CAMERA_STREET_MID", (0, 0, 1.65), (0, 40, 3), 32),
        ("CAMERA_SIDEWALK_RIGHT", (5.45, -30, 1.80), (5.0, 18, 3.1), 26),
    ]
    for name, xyz, target, lens in views:
        marker = empty(name, xyz, "CAMERA_MARKERS")
        marker.rotation_euler = (Vector(target) - Vector(xyz)).to_track_quat("-Z", "Y").to_euler()
        data = bpy.data.cameras.new(name + "_preview")
        data.lens = lens
        data.clip_end = 500
        obj = bpy.data.objects.new(name + "_preview", data)
        collections["CAMERAS"].objects.link(obj)
        obj.location = xyz
        obj.rotation_euler = marker.rotation_euler
    pano = bpy.data.cameras.new("ERP_preview")
    pano.type = "PANO"
    pano.panorama_type = "EQUIRECTANGULAR"
    obj = bpy.data.objects.new("ERP_preview", pano)
    collections["CAMERAS"].objects.link(obj)
    obj.location = (0, -20, 1.65)
    obj.rotation_euler = (math.pi / 2, 0, 0)
    world = bpy.data.worlds.new("Daylight sky")
    world.use_nodes = True
    scene.world = world
    sky = world.node_tree.nodes.new("ShaderNodeTexSky")
    sky_types = sky.bl_rna.properties["sky_type"].enum_items.keys()
    sky.sky_type = "MULTIPLE_SCATTERING" if "MULTIPLE_SCATTERING" in sky_types else "NISHITA"
    sky.sun_elevation = math.radians(42)
    sky.sun_rotation = math.radians(135)
    world.node_tree.links.new(sky.outputs[0], world.node_tree.nodes["Background"].inputs[0])
    world.node_tree.nodes["Background"].inputs[1].default_value = 0.35
    # Sky supplies illumination; the collection remains explicit and nonempty.
    light = bpy.data.lights.new("Soft daylight fill", "AREA")
    light.energy = 150
    light.shape = "DISK"
    light.size = 15
    obj = bpy.data.objects.new("Soft daylight fill", light)
    collections["LIGHTS"].objects.link(obj)
    obj.location = (0, 0, 25)
    scene.camera = bpy.data.objects["CAMERA_STREET_NEAR_preview"]
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.seed = args.seed
    scene.cycles.use_animated_seed = False
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 6
    scene.render.threads_mode = "FIXED"
    scene.render.threads = 12
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.exposure = -0.65
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.film_transparent = False
    scene.render.use_compositing = False
    scene.render.resolution_percentage = 100
    scene.cycles.samples = args.samples
    scene.render.resolution_x = args.width
    scene.render.resolution_y = round(args.width * 9 / 16)
    bpy.context.view_layer.update()
    args.scene_file.parent.mkdir(parents=True, exist_ok=True)
    bpy.context.preferences.filepaths.save_version = 0
    bpy.ops.wm.save_as_mainfile(filepath=str(args.scene_file))
    return texture_sources


def blender_main(args):
    import bpy
    import resource

    started = time.perf_counter()
    sources = []
    if args.preview_only:
        bpy.ops.wm.open_mainfile(filepath=str(args.scene_file))
    else:
        sources = build(args)
    scene = bpy.context.scene
    render_times = {}
    if not args.skip_render:
        for name, filename in [("CAMERA_STREET_NEAR_preview", "street_level.png"), ("CAMERA_SIDEWALK_LEFT_preview", "sidewalk.png"), ("CAMERA_STREET_LONG_preview", "long_street.png"), ("ERP_preview", "panorama.png")]:
            scene.camera = bpy.data.objects[name]
            scene.cycles.samples = args.samples
            scene.render.resolution_x = args.width
            scene.render.resolution_y = args.width // 2 if name == "ERP_preview" else round(args.width * 9 / 16)
            scene.render.filepath = str(args.output_dir / filename)
            tick = time.perf_counter()
            print("RENDER_START", filename, flush=True)
            bpy.ops.render.render(write_still=True)
            render_times[filename] = time.perf_counter() - tick
    triangles = 0
    depsgraph = bpy.context.evaluated_depsgraph_get()
    for obj in scene.objects:
        if obj.type == "MESH":
            evaluated = obj.evaluated_get(depsgraph)
            mesh = evaluated.to_mesh()
            mesh.calc_loop_triangles()
            triangles += len(mesh.loop_triangles)
            evaluated.to_mesh_clear()
    report = {"scene_file": str(args.scene_file), "blender": bpy.app.version_string, "seed": args.seed, "scene_bytes": args.scene_file.stat().st_size, "objects": len(scene.objects), "triangles_evaluated": triangles, "file_textures": len([i for i in bpy.data.images if i.source == "FILE"]), "texture_sources": sources, "device": "CPU", "samples": args.samples, "width": args.width, "render_seconds": render_times, "elapsed_seconds": time.perf_counter() - started, "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, "vram": "not used: CPU render"}
    filename = "preview_report.json" if args.preview_only else "build_report.json"
    (args.output_dir / filename).write_text(json.dumps(report, indent=2) + "\n")
    print("URBAN_STREET_COMPLETE", json.dumps(report), flush=True)


def main():
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.scene_file = args.scene_file.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import bpy
    except ImportError:
        from densekp_retrain.runtime import find_blender_executable
        from densekp_retrain.rendering.process import run_logged

        command = [str(find_blender_executable(args.blender_bin)), "-b", "--factory-startup", "--disable-autoexec", "--python-exit-code", "1", "--python", str(Path(__file__).resolve()), "--", "--scene-file", str(args.scene_file), "--output-dir", str(args.output_dir), "--width", str(args.width), "--samples", str(args.samples), "--seed", str(args.seed)]
        for flag in ("skip_render", "preview_only"):
            if getattr(args, flag):
                command.append("--" + flag.replace("_", "-"))
        log = args.output_dir / ("preview.log" if args.preview_only else "build.log")
        print(f"Blender log: {log}", flush=True)
        run_logged(command, cwd=ROOT, log_path=log)
    else:
        blender_main(args)


if __name__ == "__main__":
    main()
