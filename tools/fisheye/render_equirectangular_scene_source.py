"""Blender helper: equirectangular source buffers for Omni-Radtan warping with a
static .blend scene instead of an HDRI-only background. Phase 18.

Mirrors tools/fisheye/render_equirectangular_source.py pass setup exactly
(Combined/Depth/IndexOB, person pass_index 1/2) so the Phase 17 warp pipeline
works unchanged; scene objects carry pass_index=0.
"""

import argparse, json, math, sys
from pathlib import Path
import bpy, numpy as np
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from densekp_retrain.rendering.blender import scene_bridge as bridge
from densekp_retrain.rendering.blender.assembly import (
    configure_render,
    configure_compositor,
    create_environment,
    create_erp_camera,
    render_cameras,
    create_mesh_object,
    add_body_material,
    add_surface_material,
)


def arguments():
    p = argparse.ArgumentParser()
    q = sys.argv[sys.argv.index("--") + 1 :]
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--clothing-geometry", type=Path)
    p.add_argument("--cameras", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--frame", type=int, default=0)
    p.add_argument("--scene-blend", type=Path, required=True)
    p.add_argument("--scene-collection", type=str)
    p.add_argument("--hide-objects", type=str, default="")
    p.add_argument("--ground-mode", type=str, default="scene_native")
    p.add_argument("--body-location", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    p.add_argument("--body-yaw-deg", type=float, default=0.0)
    p.add_argument("--body-texture", type=Path)
    p.add_argument("--body-uv", type=Path)
    p.add_argument("--hdri", type=Path)
    p.add_argument("--sun-energy", type=float, default=3.0)
    p.add_argument("--width", type=int, default=2048)
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--samples", type=int, default=8)
    return p.parse_args(q)


def main():
    a = arguments()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    vertices = np.load(a.geometry / "vertices.npy")
    faces = np.load(a.geometry / "faces.npy")
    f = a.frame
    bpy.ops.wm.read_factory_settings(use_empty=True)
    s = bpy.context.scene
    configure_render(s, a.width, a.height, a.samples)
    s.view_layers[0].use_pass_z = True
    s.view_layers[0].use_pass_object_index = True
    s.view_settings.look = "None"
    file_node = configure_compositor(s, a.output_dir, "FisheyeSceneCompositor")
    create_environment(s, "FisheyeSceneWorld", a.hdri)
    sun = bpy.data.lights.new("SceneSun", "SUN")
    sun.energy = a.sun_energy
    sun_ob = bpy.data.objects.new("SceneSun", sun)
    s.collection.objects.link(sun_ob)
    sun_ob.rotation_euler = (math.radians(50), 0, math.radians(30))
    scene_names = bridge.link_blend_scene(a.scene_blend, a.scene_collection)
    hidden = bridge.set_visibility(
        scene_names, [x for x in a.hide_objects.split(",") if x], a.ground_mode
    )
    bounds = bridge.scene_bounds(scene_names)
    if a.ground_mode == "add_procedural_ground":
        gm = bpy.data.meshes.new("Ground")
        gm.from_pydata(
            [(-12, -12, -0.01), (12, -12, -0.01), (12, 12, -0.01), (-12, 12, -0.01)],
            [],
            [(0, 1, 2), (0, 2, 3)],
        )
        ground = bpy.data.objects.new("Ground", gm)
        s.collection.objects.link(ground)
        ground.pass_index = 0
        gmat = bpy.data.materials.new("GroundMaterial")
        gmat.use_nodes = True
        gb = gmat.node_tree.nodes.get("Principled BSDF")
        gb.inputs["Base Color"].default_value = (0.14, 0.16, 0.13, 1)
        gb.inputs["Roughness"].default_value = 0.85
        gm.materials.append(gmat)
        scene_names.append("Ground")
    body, mesh = create_mesh_object(s, "SMPLX_Fisheye", vertices[f], faces, 1)
    mat, bs = add_body_material(mesh, "Body")
    if a.body_texture and a.body_uv:
        uv = np.load(a.body_uv)
        layer = mesh.uv_layers.new(name="SMPLX_UV")
        for polygon, indices in zip(mesh.polygons, uv["faces_uv"]):
            for loop, index in zip(polygon.loop_indices, indices):
                layer.data[loop].uv = uv["uv"][index]
        tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
        tex.image = bpy.data.images.load(str(a.body_texture.resolve()))
        tex.image.colorspace_settings.name = "sRGB"
        mat.node_tree.links.new(tex.outputs["Color"], bs.inputs["Base Color"])
    conv = Matrix(((1, 0, 0, 0), (0, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)))
    body.matrix_world = (
        Matrix.Translation(Vector(a.body_location))
        @ Matrix.Rotation(math.radians(a.body_yaw_deg), 4, "Z")
        @ (Matrix.Translation((0, 0, -float(vertices[f, :, 1].min()))) @ conv)
    )
    if a.clothing_geometry:
        cg = a.clothing_geometry
        cv = np.load(cg / "clothing_vertices.npy")
        cf = np.load(cg / "clothing_faces.npy")
        co, cm = create_mesh_object(s, "Clothing", cv[f], cf, 2)
        co.matrix_world = body.matrix_world
        add_surface_material(cm, "ClothingMaterial", (0.04, 0.35, 0.5, 1), 0.6)
    cam = create_erp_camera(s)
    records = render_cameras(s, cam, file_node, a.cameras, a.output_dir)
    (a.output_dir / "source_manifest.json").write_text(
        json.dumps(
            {
                "projection": "equirectangular",
                "resolution": [a.width, a.height],
                "frame": f,
                "scene_blend": str(a.scene_blend.resolve()),
                "scene_objects": scene_names,
                "scene_hidden": hidden,
                "scene_bounds": bounds,
                "ground_mode": a.ground_mode,
                "body_placement": {
                    "location": list(a.body_location),
                    "yaw_deg": a.body_yaw_deg,
                },
                "sources": records,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
