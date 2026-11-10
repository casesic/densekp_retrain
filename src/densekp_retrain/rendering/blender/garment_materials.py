"""One Blender garment-material assembly path for studio and future consumers."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from densekp_retrain.rendering.blender.materials import add_clothing_materials


def legacy_native_description(value):
    """Translate existing preview recipes without changing their shading defaults."""
    if not all(value.get(k) for k in ("vertices_uv", "faces_uv", "diffuse", "normal")):
        raise ValueError("incomplete_native_material_recipe")
    return dict(material_kind="native_face_corner_uv_diffuse_normal_v1", material=dict(value,
        roughness=0.58, base_color=[0.23, 0.35, 0.48, 1.0],
        uv_layer="OriginalBEDLAM_UV", material_name="TransferredBEDLAMGarment"))


def add_garment_materials(mesh, description):
    import bpy

    def image_for(path, colorspace):
        wanted = Path(path).resolve()
        for candidate in bpy.data.images:
            if (candidate.filepath and Path(bpy.path.abspath(candidate.filepath)).resolve() == wanted and
                    candidate.colorspace_settings.name == colorspace):
                return candidate
        image = bpy.data.images.load(str(wanted), check_existing=False)
        image.colorspace_settings.name = colorspace
        return image

    kind = description.get("material_kind")
    spec = description.get("material", {})
    if kind == "GarmentRest_continuous_shell":
        ref = Path(spec["reference"])
        for name in ("materials.json", "garment_material.json", "clothing_rest_vertices.npy"):
            if not (ref / name).is_file():
                raise FileNotFoundError(ref / name)
        for layer in list(mesh.uv_layers):
            mesh.uv_layers.remove(layer)
        mesh.materials.clear()
        if "GarmentRest" in mesh.attributes:
            mesh.attributes.remove(mesh.attributes["GarmentRest"])
        add_clothing_materials(mesh, ref)
        return mesh
    if kind != "native_face_corner_uv_diffuse_normal_v1":
        raise ValueError("unsupported_garment_material_kind")
    required = ("vertices_uv", "faces_uv", "diffuse", "normal", "roughness", "base_color", "uv_layer", "material_name")
    if any(key not in spec or spec[key] is None for key in required):
        raise ValueError("incomplete_native_material_description")
    for key in ("vertices_uv", "faces_uv", "diffuse", "normal"):
        if not Path(spec[key]).is_file():
            raise FileNotFoundError(spec[key])
    uv = np.load(spec["vertices_uv"], mmap_mode="r")
    face_uv = np.load(spec["faces_uv"], mmap_mode="r")
    if (uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all() or
            face_uv.shape != (len(mesh.polygons), 3) or face_uv.size == 0 or
            face_uv.min() < 0 or face_uv.max() >= len(uv) or
            len(mesh.loops) != face_uv.size or any(len(p.loop_indices) != 3 for p in mesh.polygons)):
        raise ValueError("native_face_corner_uv_mismatch")
    if not 0 <= spec["roughness"] <= 1 or len(spec["base_color"]) != 4:
        raise ValueError("native_material_parameter_invalid")
    if "geometry" in description and "cloth_faces" in description["geometry"]:
        expected_faces = np.load(description["geometry"]["cloth_faces"], mmap_mode="r")
        actual_faces = np.empty((len(mesh.polygons), 3), np.int32)
        mesh.polygons.foreach_get("vertices", actual_faces.reshape(-1))
        if not np.array_equal(actual_faces, expected_faces):
            raise ValueError("native_mesh_face_order_conflict")
    # Replace, never accumulate, face-corner UV or material slots.
    for layer in list(mesh.uv_layers):
        mesh.uv_layers.remove(layer)
    mesh.materials.clear()
    for polygon in mesh.polygons:
        polygon.material_index = 0
    layer = mesh.uv_layers.new(name=spec["uv_layer"])
    layer.data.foreach_set("uv", uv[face_uv].reshape(-1))
    material = bpy.data.materials.new(spec["material_name"])
    material.use_nodes = True
    nodes, links = material.node_tree.nodes, material.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    bsdf.inputs["Roughness"].default_value = spec["roughness"]
    bsdf.inputs["Base Color"].default_value = spec["base_color"]
    diffuse = nodes.new("ShaderNodeTexImage")
    diffuse.image = image_for(spec["diffuse"], "sRGB")
    links.new(diffuse.outputs["Color"], bsdf.inputs["Base Color"])
    normal_image = nodes.new("ShaderNodeTexImage")
    normal_image.image = image_for(spec["normal"], "Non-Color")
    normal_map = nodes.new("ShaderNodeNormalMap")
    links.new(normal_image.outputs["Color"], normal_map.inputs["Color"])
    links.new(normal_map.outputs["Normal"], bsdf.inputs["Normal"])
    mesh.materials.append(material)
    return mesh
