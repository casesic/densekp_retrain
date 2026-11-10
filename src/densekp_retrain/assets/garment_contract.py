"""Read-only garment delivery and exact source-frame window contracts.

This module deliberately has no bpy, model construction, or rendering import.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SCHEMA = "densekp_garment_asset_v1"
VIEW_SCHEMA = "densekp_garment_window_view_v1"
SIMPLE = "simple_barycentric_shell"
FIXED = "fixed_vertex_cache_face_corner_uv"
WALK_ROOT = "source_root_relative_to_start_plus_fixed_scene_matrix"
FIXED_ROOT = "fixed_pelvis_scene_owns_progression"


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def describe_simple(authority_path):
    authority_path = Path(authority_path).resolve()
    a = read_json(authority_path)
    return dict(schema=SCHEMA, asset_id=a["asset_id"], authority_ref=str(authority_path),
        geometry_kind=SIMPLE, material_kind="GarmentRest_continuous_shell",
        material=dict(reference=a["clothing_reference"]),
        geometry=dict(body_vertices=a["body_vertices"], body_faces=a["body_faces"],
            cloth_vertices=a["cloth_vertices"], cloth_faces=a["cloth_faces"],
            head_transforms=a["head_transforms"], densekp=a["body_R138"],
            parameters=a["parameters"], groom_reference=a["groom_reference"]),
        timeline_ref=a["timeline"], frame_count=a["frame_count"],
        original_window=[0, a["frame_count"]], time_base=a["time_base"],
        root_mode=a["root_mode"], coordinates="standard_SMPLX_model_Y_up_metres",
        character_family=a["character_family"], motion_source_id=a["motion_source_id"],
        source_group=a["source_group"], compatibility=dict(mode="composable_simple_binding"),
        technical_status="numeric_passed_existing_authority", human_status="see_original_acceptance_record",
        scene_status="see_original_A1_evidence", source_license="see_original_motion_and_character_manifest",
        integration_only=True)


def describe_fixed(authority_path, simple_authority_path):
    authority_path = Path(authority_path).resolve()
    authority = read_json(authority_path)
    base_path = Path(simple_authority_path).resolve()
    base = read_json(base_path)
    lock = authority["clothing_compatibility"]
    if lock["mode"] != "exact_precomputed_cache_only" or authority["target_body_manifest"] != str(base_path):
        raise ValueError("fixed_garment_authority_lock_conflict")
    return dict(schema=SCHEMA, asset_id=authority["asset_id"], authority_ref=str(authority_path),
        parent_body_authority=str(base_path), geometry_kind=FIXED,
        material_kind="native_face_corner_uv_diffuse_normal_v1",
        material=dict(vertices_uv=authority["cloth_uv_vertices"], faces_uv=authority["cloth_uv_faces"],
            diffuse=authority["garment_texture_01"]["diffuse"],
            normal=authority["garment_texture_01"]["normal"],
            roughness=0.58, base_color=[0.23, 0.35, 0.48, 1.0],
            uv_layer="OriginalBEDLAM_UV", material_name="TransferredBEDLAMGarment"),
        geometry=dict(body_vertices=authority["target_body_vertices"],
            body_faces=authority["target_body_faces"], cloth_vertices=authority["cloth_vertices"],
            cloth_faces=authority["cloth_faces"], head_transforms=authority["target_head_transforms"],
            densekp=authority["target_R138"], parameters=base["parameters"],
            groom_reference=authority["target_groom"]),
        timeline_ref=base["timeline"], frame_count=authority["frame_count"],
        original_window=[0, authority["frame_count"]], time_base=authority["time_base"],
        root_mode=base["root_mode"], coordinates="standard_SMPLX_model_Y_up_metres",
        character_family=authority["character_family"], motion_source_id=authority["motion_source_id"],
        source_group=authority["source_group"], compatibility=dict(lock, mode="exact_precomputed_cache_only"),
        technical_status="numeric_passed_transfer_attempt_1",
        human_status=authority["dressed_human_status"], scene_status=authority["public_scene_status"],
        source_license=authority["license"], original_body_motion=authority["original_body_motion"],
        original_outfit=authority["original_outfit"], no_new_simulation=True, integration_only=True)


def validate_description(d):
    if d.get("schema") != SCHEMA or d.get("geometry_kind") not in (SIMPLE, FIXED):
        raise ValueError("unknown_garment_description")
    expected = "GarmentRest_continuous_shell" if d["geometry_kind"] == SIMPLE else "native_face_corner_uv_diffuse_normal_v1"
    if d.get("material_kind") != expected or d.get("frame_count", 0) < 1:
        raise ValueError("garment_type_or_material_mismatch")
    if d.get("root_mode") not in (WALK_ROOT, FIXED_ROOT):
        raise ValueError("unsupported_root_contract_for_this_delivery")
    if d.get("original_window") != [0, d["frame_count"]]:
        raise ValueError("original_window_mismatch")
    for path in (d["authority_ref"], d["timeline_ref"], *[d["geometry"][k] for k in
            ("body_vertices", "body_faces", "cloth_vertices", "cloth_faces", "head_transforms", "densekp", "parameters")]):
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    for key in ("diffuse", "normal", "vertices_uv", "faces_uv") if d["geometry_kind"] == FIXED else ():
        if not Path(d["material"][key]).is_file():
            raise FileNotFoundError(d["material"][key])
    if d["geometry_kind"] == SIMPLE:
        ref = Path(d["material"]["reference"])
        for file in ("materials.json", "garment_material.json", "clothing_rest_vertices.npy"):
            if not (ref / file).is_file():
                raise FileNotFoundError(ref / file)
    else:
        if (d["compatibility"].get("mode") != "exact_precomputed_cache_only" or
                d["compatibility"].get("window") != [0, d["frame_count"]]):
            raise ValueError("fixed_garment_lock_mismatch")
        lock = d["compatibility"]
        if (lock.get("character_family") != d["character_family"] or
                lock.get("motion_id") != d["motion_source_id"] or
                not lock.get("body_id") or not lock.get("hair_id")):
            raise ValueError("fixed_garment_identity_mismatch")
        uv = np.load(d["material"]["vertices_uv"], mmap_mode="r")
        uf = np.load(d["material"]["faces_uv"], mmap_mode="r")
        faces = np.load(d["geometry"]["cloth_faces"], mmap_mode="r")
        if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all() or uf.shape != faces.shape or uf.shape[1] != 3 or uf.size == 0 or uf.min() < 0 or uf.max() >= len(uv):
            raise ValueError("native_face_corner_uv_invalid")
    n = d["frame_count"]
    geometry = d["geometry"]
    body = np.load(geometry["body_vertices"], mmap_mode="r")
    cloth = np.load(geometry["cloth_vertices"], mmap_mode="r")
    head = np.load(geometry["head_transforms"], mmap_mode="r")
    dense = np.load(geometry["densekp"], mmap_mode="r")
    body_faces = np.load(geometry["body_faces"], mmap_mode="r")
    cloth_faces = np.load(geometry["cloth_faces"], mmap_mode="r")
    if (body.shape != (n, 10475, 3) or cloth.ndim != 3 or cloth.shape[0] != n or
            cloth.shape[2] != 3 or head.shape != (n, 4, 4) or dense.shape != (n, 138, 3)):
        raise ValueError("garment_component_time_or_shape_mismatch")
    if (body_faces.ndim != 2 or body_faces.shape[1] != 3 or
            cloth_faces.ndim != 2 or cloth_faces.shape[1] != 3 or
            body_faces.min() < 0 or body_faces.max() >= body.shape[1] or
            cloth_faces.min() < 0 or cloth_faces.max() >= cloth.shape[1]):
        raise ValueError("garment_topology_index_mismatch")
    if d["geometry_kind"] == FIXED:
        authority = read_json(d["authority_ref"])
        if (authority["asset_id"] != d["asset_id"] or
                Path(authority["cloth_vertices"]).resolve() != Path(geometry["cloth_vertices"]).resolve() or
                Path(authority["target_body_vertices"]).resolve() != Path(geometry["body_vertices"]).resolve()):
            raise ValueError("fixed_garment_authority_path_mismatch")
    rows = read_json(d["timeline_ref"])
    if len(rows) != n or any(row["asset_frame"] != i or row["time_base"] != d["time_base"] for i, row in enumerate(rows)) or any(rows[i]["source_time_tick"] >= rows[i+1]["source_time_tick"] for i in range(n-1)):
        raise ValueError("garment_timeline_mismatch")
    return d


def select_asset_window(description, window):
    d = validate_description(description)
    if (not isinstance(window, (list, tuple)) or len(window) != 2 or
            any(type(x) is not int for x in window)):
        raise ValueError("invalid_contiguous_window")
    first, end = window
    if first < 0 or end <= first or end > d["frame_count"]:
        raise ValueError("invalid_contiguous_window")
    rows = read_json(d["timeline_ref"])[first:end]
    with np.load(d["geometry"]["parameters"], allow_pickle=False) as source_root:
        if d["root_mode"] == WALK_ROOT:
            if "source_root_relative" not in source_root or source_root["source_root_relative"].shape[0] != d["frame_count"]:
                raise ValueError("source_root_reference_missing")
        elif "source_root_relative" in source_root:
            raise ValueError("fixed_pelvis_has_source_root_conflict")
    origin_tick = rows[0]["source_time_tick"]
    mapped = [dict(view_frame=i, source_asset_frame=first+i,
                   source_frame=row["source_frame"], source_time_tick=row["source_time_tick"],
                   source_time_base=row["time_base"], view_episode_time_tick=row["source_time_tick"]-origin_tick)
              for i, row in enumerate(rows)]
    return dict(schema=VIEW_SCHEMA, source_asset_id=d["asset_id"], authority_ref=d["authority_ref"],
        description_ref=None, selection=[first, end], source_asset_frame_indices=list(range(first, end)),
        frame_count=end-first, time_base=d["time_base"],
        source_root_reference_asset_frame=0 if d["root_mode"] == WALK_ROOT else None,
        root_mode=d["root_mode"], geometry=d["geometry"], material=d["material"],
        geometry_kind=d["geometry_kind"], material_kind=d["material_kind"],
        timeline=mapped, source_timeline_ref=d["timeline_ref"],
        inherited_human_status=d["human_status"], inherited_acceptance_scope=[0,d["frame_count"]],
        scene_status="not_consumed", integration_only=True)


def read_view_frame(view, local_frame):
    if view.get("schema") != VIEW_SCHEMA or type(local_frame) is not int or not 0 <= local_frame < view["frame_count"]:
        raise ValueError("invalid_view_frame")
    original = view["source_asset_frame_indices"][local_frame]
    if original != view["selection"][0] + local_frame or view["timeline"][local_frame]["source_asset_frame"] != original:
        raise ValueError("view_original_index_conflict")
    return {key: np.asarray(np.load(view["geometry"][key], mmap_mode="r")[original])
            for key in ("body_vertices", "cloth_vertices", "head_transforms", "densekp")}
