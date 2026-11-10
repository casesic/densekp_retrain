"""Blender readback of pre-refactor and shared garment assemblers, no rendering."""
import argparse
import json
import re
import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.rendering.blender.garment_materials import add_garment_materials
from tools.validate.garment_material_before_a2 import add_before_materials


def make_mesh(label, vertices, faces):
    mesh = bpy.data.meshes.new(label)
    mesh.from_pydata(vertices.tolist(), [], faces.tolist())
    obj = bpy.data.objects.new(label, mesh)
    bpy.context.scene.collection.objects.link(obj)
    obj.pass_index = 17
    obj.location = (0.1, 0.2, 0.3)
    return obj


def snapshot(obj):
    mesh = obj.data
    uv = np.empty(len(mesh.loops) * 2, np.float32) if mesh.uv_layers else np.empty(0, np.float32)
    if mesh.uv_layers:
        mesh.uv_layers.active.data.foreach_get("uv", uv)
    coordinates = np.empty(len(mesh.vertices) * 3, np.float32)
    mesh.vertices.foreach_get("co", coordinates)
    materials = []
    for mat in mesh.materials:
        nodes = mat.node_tree.nodes
        def default(socket):
            if not hasattr(socket, "default_value"):
                return None
            value = socket.default_value
            try:
                return [float(v) for v in value]
            except TypeError:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    return str(value)
        materials.append(dict(name=re.sub(r"\.\d{3}$", "", mat.name),
            roughness=float(nodes.get("Principled BSDF").inputs["Roughness"].default_value),
            images=sorted((Path(node.image.filepath).resolve().as_posix(), node.image.colorspace_settings.name)
                          for node in nodes if node.type == "TEX_IMAGE"),
            node_parameters=sorted((node.bl_idname, node.name,
                getattr(node, "operation", None), getattr(node, "attribute_name", None),
                [(socket.name, default(socket)) for socket in node.inputs if not socket.is_linked])
                for node in nodes),
            links=sorted((link.from_node.bl_idname, link.from_socket.name,
                          link.to_node.bl_idname, link.to_socket.name)
                         for link in mat.node_tree.links)))
    return dict(vertices=coordinates.tolist(), faces=[list(poly.vertices) for poly in mesh.polygons],
                uv=uv.tolist(), uv_layers=[layer.name for layer in mesh.uv_layers],
                slots=len(mesh.materials), face_material=[p.material_index for p in mesh.polygons],
                materials=materials, matrix=[list(row) for row in obj.matrix_world],
                instance_id=obj.pass_index,
                rest_attribute="GarmentRest" in mesh.attributes)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--simple", type=Path, required=True)
    ap.add_argument("--fixed", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(sys.argv[sys.argv.index("--") + 1:])
    bpy.ops.wm.read_factory_settings(use_empty=True)
    result = {}
    for label, path in (("simple", args.simple), ("fixed", args.fixed)):
        d = json.loads(path.read_text())
        g = d["geometry"]
        vertices = np.load(g["cloth_vertices"], mmap_mode="r")[0]
        faces = np.load(g["cloth_faces"])
        before = make_mesh(label+"_before", vertices, faces)
        after = make_mesh(label+"_after", vertices, faces)
        if label == "simple":
            add_before_materials(before.data, Path(d["material"]["reference"]), None)
        else:
            add_before_materials(before.data, None, d["material"])
        add_garment_materials(after.data, d)
        a, b = snapshot(before), snapshot(after)
        differences = [key for key in a if a[key] != b[key]]
        if differences:
            if "materials" in differences:
                print("MATERIAL_DETAIL", label, json.dumps(dict(before=a["materials"],after=b["materials"])))
            raise ValueError(label+"_before_after_mismatch:"+",".join(differences))
        add_garment_materials(after.data, d)
        if snapshot(after) != a:
            raise ValueError(label+"_repeated_assembly_changed_mesh")
        result[label] = dict(status="exact_before_after_and_repeat", vertices=len(vertices),
            faces=len(faces), uv_loops=len(a["uv"])//2, slots=a["slots"],
            face_material_unique=sorted(set(a["face_material"])),
            images=a["materials"][0]["images"], matrix_and_instance_unchanged=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({key: value["status"] for key,value in result.items()}))


if __name__ == "__main__":
    main()
