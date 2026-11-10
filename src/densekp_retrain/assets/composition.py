"""Compose existing standard SMPL-X motion, simple garments and static grooms.

The inputs remain authoritative. Cache directories are append-only: a changed
dependency or recipe at an occupied name is an error, never a silent cache hit.
"""
from __future__ import annotations

import json
import re
import shutil
import time
from pathlib import Path

import numpy as np

from densekp_retrain.body.geometry import compute_densekp_3d, load_smplx_model
from densekp_retrain.body.sequence_binding import bound_garment, head_transform

POSE = ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose", "expression")
DIMS = (3, 63, 3, 3, 3, 45, 45, 10)
ROOT_POLICY = "fixed_pelvis_scene_owns_progression"
WALK_ROOT_POLICY = "source_root_relative_to_start_plus_fixed_scene_matrix"
ROOT_POLICIES = (ROOT_POLICY, WALK_ROOT_POLICY)


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def safe(name):
    if not re.fullmatch(r"[A-Za-z0-9_]+", name):
        raise ValueError("unsafe_component_id")
    return name


def resolve(root, path):
    p = Path(path)
    return (p if p.is_absolute() else root / p).resolve()


def stamp(path):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    s = p.stat()
    return {"path": str(p.resolve()), "size": s.st_size, "mtime_ns": s.st_mtime_ns}


def cached_folder(path, frozen):
    """Return hit or reserve a new folder; never overwrite an occupied mismatch."""
    path = Path(path)
    marker = path / "cache_key.json"
    if path.exists():
        if not marker.is_file() or read(marker) != frozen:
            raise RuntimeError("cache_collision_or_dependency_change:" + str(path))
        return True
    path.mkdir(parents=True)
    write_new(marker, frozen)
    return False


def _locked_clothing_plan(root, selected, recipe):
    """Resolve an accepted, precomputed garment only for its exact asset tuple."""
    b, m, c, h = (selected[k] for k in ("body", "motion", "clothing", "hair"))
    if (b["id"], m["id"], h["id"], recipe["window"]) != (
        c["locked_body"], c["locked_motion"], c["locked_hair"], c["locked_window"]
    ):
        raise ValueError("clothing_locked_body_motion_hair_window_mismatch")
    asset_path = resolve(root, c["asset_manifest"])
    asset = read(asset_path)
    character = read(resolve(root, b["manifest"]))
    if (asset["asset_id"] != c["locked_asset_id"] or
            asset["character_family"] != character["identity_family_id"] or
            asset["motion_source_id"] != m["id"] or
            asset["source_group"] != m["source_group"] or
            asset["frame_count"] != c["locked_window"][1] - c["locked_window"][0] or
            Path(asset["target_body_vertices"]).resolve() != resolve(root, c["locked_body_vertices"]) or
            Path(asset["target_body_manifest"]).resolve() != resolve(root, c["locked_body_manifest"]) or
            Path(asset["target_groom"]).resolve() != resolve(root, h["reference"])):
        raise ValueError("locked_clothing_asset_authority_mismatch")
    dependencies = [asset_path, resolve(root, b["manifest"]), resolve(root, m["parameters"]),
                    resolve(root, m["timeline"])]
    dependencies += [Path(asset[key]) for key in (
        "target_body_vertices", "target_body_faces", "target_R138", "target_head_transforms",
        "cloth_vertices", "cloth_faces", "cloth_uv_vertices", "cloth_uv_faces")]
    dependencies += [Path(asset["garment_texture_01"][key]) for key in ("diffuse", "normal")]
    dependencies = [stamp(path) for path in dependencies]
    if any(item["size"] == 0 for item in dependencies):
        raise ValueError("empty_locked_clothing_dependency")
    timeline = read(resolve(root, m["timeline"]))
    if (len(timeline) != asset["frame_count"] or
            any(row["asset_frame"] != i or row["time_base"] != asset["time_base"]
                for i, row in enumerate(timeline)) or
            any(timeline[i]["source_time_tick"] >= timeline[i + 1]["source_time_tick"]
                for i in range(len(timeline) - 1))):
        raise ValueError("locked_clothing_timeline_mismatch")
    return {"schema": "densekp_asset_composition_resolved_v1", "selection_mode": "reuse_precomputed_locked_asset",
            "components": selected, "window": c["locked_window"], "frame_count": asset["frame_count"],
            "root_policy": recipe["root_policy"], "asset_manifest": str(asset_path),
            "output_dir": str(resolve(root, recipe["output_dir"])), "dependencies": dependencies,
            "compatibility": {"clothing": "locked_exact_body_motion_hair_window"},
            "source_group": m["source_group"], "character_family": b["character_family"],
            "integration_only": True}


def plan(root, registry, recipe):
    if registry.get("schema") != "densekp_asset_components_v1":
        raise ValueError("unsupported_component_schema")
    selected = {}
    for name, plural, key in (("body", "bodies", "body_id"), ("motion", "motions", "motion_id"),
                              ("clothing", "clothing", "clothing_id"), ("hair", "hair", "hair_id")):
        ident = safe(recipe[key])
        selected[name] = dict(id=ident, **registry[plural][ident])
    b, m, c, h = (selected[k] for k in ("body", "motion", "clothing", "hair"))
    policy = recipe.get("root_policy")
    if policy not in ROOT_POLICIES or m.get("root_mode") != policy:
        raise ValueError("unsupported_root_ownership")
    if c["kind"] == "retargeted_native_cache_locked_body_motion_time":
        return _locked_clothing_plan(root, selected, recipe)
    if c["kind"] != "simple_fixed_barycentric_normal_offset":
        raise ValueError("native_cache_locked_to_original_body_motion_time")
    if h["kind"] != "static_curves_head_rigid":
        raise ValueError("unsupported_hair_type")
    character = read(resolve(root, b["manifest"]))
    if character.get("character_id") != b["version"] or character.get("identity_family_id") != b["character_family"]:
        raise ValueError("body_manifest_identity_conflict")
    if character.get("num_betas") != 30 or character.get("gender") not in ("male", "female"):
        raise ValueError("unsupported_model_or_shape_basis")
    if len(character.get("betas", [])) != 30:
        raise ValueError("beta_dimension_conflict")
    model = Path(character["model_dir"]) / ("SMPLX_" + character["gender"].upper() + ".npz")
    target_faces_file = Path(character["reference"]["geometry"]) / "faces.npy"
    # Verify the registered semantic template before allowing cross-identity
    # barycentric IDs. Matching vertex counts alone cannot establish this.
    if m["fitted_body"] not in registry["bodies"] or c["reference_body"] not in registry["bodies"] or h["reference_body"] not in registry["bodies"]:
        raise ValueError("unregistered_reference_body")
    target_faces = np.load(target_faces_file)
    for source_body in (m["fitted_body"], c["reference_body"], h["reference_body"]):
        source_character = read(resolve(root, registry["bodies"][source_body]["manifest"]))
        if Path(source_character["model_dir"]).resolve() != Path(character["model_dir"]).resolve():
            raise ValueError("different_smplx_model_release_or_pose_basis:" + source_body)
        source_faces_file = Path(source_character["reference"]["geometry"]) / "faces.npy"
        if not np.array_equal(target_faces, np.load(source_faces_file)):
            raise ValueError("semantic_topology_face_order_conflict:" + source_body)
    dependencies = [resolve(root, b["manifest"]), model, resolve(root, m["parameters"]), resolve(root, m["timeline"])]
    ref = Path(character["reference"]["geometry"])
    dependencies += [ref / "vertices.npy", ref / "faces.npy", root / "assets/densekp_regressor_R.npy"]
    cloth = resolve(root, c["reference"])
    cloth_owner = read(cloth / "clothing_manifest.json")["body_geometry"]
    expected_cloth_owner = read(resolve(root, registry["bodies"][c["reference_body"]]["manifest"]))["reference"]["geometry"]
    if Path(cloth_owner).resolve() != Path(expected_cloth_owner).resolve():
        raise ValueError("clothing_reference_body_conflict")
    dependencies += [cloth / f for f in ("body_barycentric_indices.npy", "body_barycentric_weights.npy", "clothing_faces.npy", "materials.json", "garment_material.json", "clothing_manifest.json")]
    groom = resolve(root, h["reference"])
    dependencies += [groom / f for f in ("points.npy", "radius.npy", "curve_offsets.npy", "active_curve_mask.npy", "groom_manifest.json", "collision_correction.npy")]
    gm = read(groom / "groom_manifest.json")
    expected_hair_owner = read(resolve(root, registry["bodies"][h["reference_body"]]["manifest"]))["reference"]["geometry"]
    if Path(gm["body_geometry"]).resolve() != Path(expected_hair_owner).resolve():
        raise ValueError("groom_reference_body_conflict")
    dependencies.append(Path(gm["binding"]) / "binding.npz")
    dependencies.append(Path(gm["source_abc"]))
    dependencies = [stamp(p) for p in dependencies]
    rows = read(resolve(root, m["timeline"]))
    with np.load(resolve(root, m["parameters"]), allow_pickle=False) as z:
        n = len(z["global_orient"])
        for key, dim in zip(POSE, DIMS):
            if key == "expression" and key not in z and m.get("expression_policy") == "neutral_zero_missing":
                continue
            if key not in z or z[key].shape != (n, dim):
                raise ValueError("motion_parameter_dimension:" + key)
        if z["betas"].shape not in ((30,), (10,)) or z["transl"].shape != (n, 3):
            raise ValueError("motion_shape_or_translation_dimension")
        if not np.isfinite(np.concatenate([z[k].reshape(n, -1) for k in POSE if k in z], axis=1)).all():
            raise ValueError("nonfinite_source_motion")
        if policy == ROOT_POLICY:
            if not np.allclose(z["transl"], z["transl"][0], atol=1e-7) or "source_root_relative" in z:
                raise ValueError("fixed_root_motion_conflict")
        else:
            if "source_frame_ids" not in z or z["source_frame_ids"].shape != (n,):
                raise ValueError("missing_source_frame_ids_for_root_reference")
            if "source_root_relative" not in z or z["source_root_relative"].shape != (n, 3):
                raise ValueError("missing_source_root_relative")
            relative = z["source_root_relative"]
            if not np.isfinite(relative).all() or not np.allclose(relative[0], 0, atol=1e-7) or not np.allclose(z["transl"] - z["transl"][0], relative, atol=2e-6, rtol=0):
                raise ValueError("source_root_translation_conflict")
            if m.get("root_reference_source_frame") != int(z["source_frame_ids"][0]):
                raise ValueError("source_root_reference_conflict")
        source_frame_key = "source_frame_ids" if "source_frame_ids" in z else "source_decoded_frames" if "source_decoded_frames" in z else None
        if source_frame_key and not np.array_equal(z[source_frame_key], [x["source_frame"] for x in read(resolve(root, m["timeline"]))]):
            raise ValueError("source_frame_mapping_conflict")
    if len(rows) != n or any(row.get("asset_frame") != i for i, row in enumerate(rows)):
        raise ValueError("timeline_count_or_order_conflict")
    if any(rows[i]["source_time_tick"] >= rows[i+1]["source_time_tick"] for i in range(n-1)):
        raise ValueError("nonmonotonic_source_time")
    if len({tuple(x["time_base"]) for x in rows}) != 1:
        raise ValueError("mixed_time_bases")
    first, end = recipe["window"]
    if not (type(first) is type(end) is int and 0 <= first < end <= n):
        raise ValueError("invalid_window")
    for p in dependencies:
        if p["size"] == 0:
            raise ValueError("empty_dependency:" + p["path"])
    result = {"schema": "densekp_asset_composition_resolved_v1", "components": selected,
              "output_dir": str(resolve(root, recipe["output_dir"])),
              "body_manifest": str(resolve(root, b["manifest"])), "model": str(model),
              "reference_geometry": str(ref), "clothing_reference": str(cloth), "hair_reference": str(groom),
              "parameters": str(resolve(root, m["parameters"])), "source_timeline": str(resolve(root, m["timeline"])),
              "window": [first, end], "frame_count": end-first, "root_policy": policy,
              "root_reference_source_frame": m.get("root_reference_source_frame") if policy == WALK_ROOT_POLICY else None,
              "pose_basis": {"hand": "full_45D_axis_angle_flat_hand_mean", "body": "21_joints_axis_angle", "expression": "10D_existing_model_basis", "jaw_eye": "3D_axis_angle_each"},
              "compatibility": {"body_motion": "direct_same_identity" if b["id"] == m["fitted_body"] else "existing_standard_smplx_pose_transfer",
                                "clothing": "direct" if b["id"] == c["reference_body"] else "existing_standard_topology_barycentric_transfer",
                                "hair": "direct" if b["id"] == h["reference_body"] else "rest_scalp_triangle_frame_transfer"},
              "source_group": m["source_group"], "character_family": b["character_family"],
              "license": {k: selected[k].get("license", "see source character manifest") for k in ("body", "motion", "clothing", "hair")},
              "dependencies": dependencies, "integration_only": True}
    return result


def _model_and_rest(resolved):
    import torch
    character = read(resolved["body_manifest"])
    model = load_smplx_model(Path(resolved["model"]).parent, gender=character["gender"], num_betas=30, batch_size=1).eval()
    model.requires_grad_(False)
    if model.use_pca or not model.flat_hand_mean or tuple(model.shapedirs.shape) != (10475, 3, 30) or tuple(model.expr_dirs.shape) != (10475, 3, 10):
        raise ValueError("unexpected_smplx_hand_or_expression_basis")
    beta = np.asarray(character["betas"], dtype=np.float32)
    with torch.no_grad():
        unplaced = model(betas=torch.from_numpy(beta[None]))
        translation = -unplaced.joints[0, 0].numpy().astype(np.float32)
        rest = model(betas=torch.from_numpy(beta[None]), transl=torch.from_numpy(translation[None]))
    faces = np.asarray(model.faces)
    reference = Path(resolved["reference_geometry"])
    np.testing.assert_array_equal(faces, np.load(reference / "faces.npy"))
    rv, rj = rest.vertices[0].numpy(), rest.joints[0].numpy()
    if np.max(np.abs(rv - np.load(reference / "vertices.npy")[0])) > 2e-6:
        raise ValueError("target_rest_reference_or_topology_mismatch")
    return model, beta, translation, rv, rj, faces


def build(root, resolved, output_root, name):
    if resolved.get("selection_mode") == "reuse_precomputed_locked_asset":
        if [stamp(item["path"]) for item in resolved["dependencies"]] != resolved["dependencies"]:
            raise ValueError("locked_clothing_dependency_changed_since_plan")
        manifest = read(resolved["asset_manifest"])
        if manifest.get("dressed_human_status") != "accepted_user_visual_20261010":
            raise ValueError("locked_clothing_acceptance_changed_since_plan")
        return Path(resolved["asset_manifest"]).parent, {
            "status": "reused_precomputed_locked_asset", "asset_id": manifest["asset_id"],
            "dressed_human_status": manifest["dressed_human_status"],
            "public_scene_status": manifest["public_scene_status"], "no_new_geometry": True}
    import torch
    from smplx.lbs import batch_rodrigues
    from densekp_retrain.clothing.groom_transfer import transfer_groom
    from densekp_retrain.data.dynamic_scene import SCHEMA as A1_ASSET_SCHEMA, adapt_proposal, geometry_paths, person_matrices, select_timeline

    began = time.monotonic()
    torch.set_num_threads(2)
    name = safe(name)
    output_root = Path(output_root).resolve()
    out = output_root / "builds" / name
    if out.exists():
        raise FileExistsError("existing_output_never_overwritten:" + str(out))
    out.mkdir(parents=True)
    write_new(out / "frozen_recipe.json", resolved)
    model, beta, translation, rv, rj, faces = _model_and_rest(resolved)
    b, m, c, h = (resolved["components"][k] for k in ("body", "motion", "clothing", "hair"))
    first, end = resolved["window"]
    n = end-first
    source = np.load(resolved["parameters"], allow_pickle=False)
    source_rows = read(resolved["source_timeline"])[first:end]
    rows = [dict(row, asset_frame=i, episode_time_tick=row["source_time_tick"]-source_rows[0]["source_time_tick"]) for i, row in enumerate(source_rows)]
    write_new(out / "timeline.json", rows)
    policy = resolved["root_policy"]
    body_key = {"body": b, "motion": m, "window": [first, end], "root_policy": policy,
                "dependencies": [x for x in resolved["dependencies"] if x["path"] in (resolved["body_manifest"], resolved["model"], resolved["parameters"], resolved["source_timeline"], str(root / "assets/densekp_regressor_R.npy"))]}
    body_dir = output_root / "cache/body" / f'{b["id"]}__{m["id"]}__{first}_{end}'
    body_hit = cached_folder(body_dir, body_key)
    if not body_hit:
        p = {k: np.asarray(source[k][first:end], np.float32).copy() if k in source else np.zeros((n, 10), np.float32) for k in POSE}
        p["betas"] = beta
        relative = np.asarray(source["source_root_relative"][first:end], np.float32) if policy == WALK_ROOT_POLICY else np.zeros((n, 3), np.float32)
        p["transl"] = relative + translation[None]
        if policy == WALK_ROOT_POLICY:
            p["source_root_relative"] = np.asarray(source["source_root_relative"][first:end]).copy()
        p["time_base"] = np.asarray(rows[0]["time_base"], np.int64)
        p["source_frame_ids"] = np.asarray([x["source_frame"] for x in rows], np.int64)
        p["source_time_ticks"] = np.asarray([x["source_time_tick"] for x in rows], np.int64)
        p["asset_time_ticks"] = np.asarray([x["episode_time_tick"] for x in rows], np.int64)
        vertices = np.empty((n, 10475, 3), np.float32)
        joints = np.empty((n, 22, 3), np.float32)
        heads = np.empty((n, 4, 4), np.float64)
        with torch.no_grad():
            for i in range(n):
                kwargs = {k: torch.from_numpy(p[k][i:i+1]) for k in POSE}
                result = model(**kwargs, betas=torch.from_numpy(beta[None]), transl=torch.from_numpy(p["transl"][i:i+1]))
                vertices[i] = result.vertices[0].numpy()
                joints[i] = result.joints[0, :22].numpy()
                local = batch_rodrigues(torch.from_numpy(np.concatenate((p["global_orient"][i], p["body_pose"][i])).reshape(-1, 3))).numpy()
                heads[i] = head_transform(local, model.parents.numpy(), joints[i, 15], rj[15])
        R = np.load(root / "assets/densekp_regressor_R.npy")
        dense = compute_densekp_3d(R, vertices)
        if not all(np.isfinite(x).all() for x in (vertices, joints, heads, dense)) or np.max(np.linalg.norm(joints[:, 0] - relative, axis=1)) > 2e-5:
            raise ValueError("invalid_body_or_pelvis")
        np.savez(body_dir / "parameters.npz", **p)
        for key, value in {"vertices": vertices, "joints": joints, "head_transforms": heads, "densekp_3d": dense, "faces": faces}.items():
            np.save(body_dir / (key + ".npy"), value)
    else:
        for key in ("parameters.npz", "vertices.npy", "joints.npy", "head_transforms.npy", "densekp_3d.npy", "faces.npy"):
            if not (body_dir / key).is_file():
                raise ValueError("incomplete_body_cache:" + key)
        with np.load(body_dir / "parameters.npz", allow_pickle=False) as cached:
            expected_relative = np.asarray(source["source_root_relative"][first:end], np.float32) if policy == WALK_ROOT_POLICY else np.zeros((n, 3), np.float32)
            if not np.array_equal(cached["betas"], beta) or not np.array_equal(cached["transl"], expected_relative + translation[None]):
                raise ValueError("body_cache_shape_or_root_conflict")
            for key in POSE:
                expected = np.asarray(source[key][first:end], np.float32) if key in source else np.zeros((n, 10), np.float32)
                if not np.array_equal(cached[key], expected):
                    raise ValueError("body_cache_motion_conflict:" + key)
    vertices = np.load(body_dir / "vertices.npy", mmap_mode="r")
    heads = np.load(body_dir / "head_transforms.npy", mmap_mode="r")
    np.testing.assert_array_equal(faces, np.load(body_dir / "faces.npy"))
    if vertices.shape != (n, 10475, 3) or heads.shape != (n, 4, 4):
        raise ValueError("stale_body_cache_shape")
    cloth_source = Path(resolved["clothing_reference"])
    ids = np.load(cloth_source / "body_barycentric_indices.npy")
    weights = np.load(cloth_source / "body_barycentric_weights.npy")
    cloth_faces = np.load(cloth_source / "clothing_faces.npy")
    if ids.shape != weights.shape or ids.ndim != 2 or ids.shape[1] != 3 or ids.min() < 0 or ids.max() >= 10475 or not np.allclose(weights.sum(1), 1, atol=1e-6):
        raise ValueError("invalid_cloth_binding")
    binding_key = {"body": b, "cloth": c, "model": stamp(resolved["model"]), "target_ref": stamp(Path(resolved["reference_geometry"]) / "vertices.npy"),
                   "source": [stamp(cloth_source / x) for x in ("body_barycentric_indices.npy", "body_barycentric_weights.npy", "clothing_faces.npy", "materials.json", "garment_material.json")]}
    cloth_ref = output_root / "cache/cloth_rest" / f'{b["id"]}__{c["id"]}'
    cloth_rest_hit = cached_folder(cloth_ref, binding_key)
    if not cloth_rest_hit:
        np.save(cloth_ref / "clothing_rest_vertices.npy", bound_garment(rv, faces, ids, weights, offset_m=c["offset_m"]))
        np.save(cloth_ref / "clothing_faces.npy", cloth_faces)
        np.save(cloth_ref / "body_barycentric_indices.npy", ids)
        np.save(cloth_ref / "body_barycentric_weights.npy", weights)
        for filename in ("materials.json", "garment_material.json"):
            shutil.copyfile(cloth_source / filename, cloth_ref / filename)
    for filename in ("clothing_rest_vertices.npy", "clothing_faces.npy", "materials.json", "garment_material.json"):
        if not (cloth_ref / filename).is_file():
            raise ValueError("incomplete_cloth_rest_cache:" + filename)
    np.testing.assert_array_equal(np.load(cloth_ref / "clothing_faces.npy"), cloth_faces)
    if np.max(np.abs(np.load(cloth_ref / "clothing_rest_vertices.npy") - bound_garment(rv, faces, ids, weights, offset_m=c["offset_m"]))) > 2e-6:
        raise ValueError("cloth_rest_cache_binding_conflict")
    cloth_key = {"body_cache": str(body_dir), "body_key": body_key, "binding_key": binding_key}
    cloth_dir = output_root / "cache/cloth_motion" / f'{b["id"]}__{m["id"]}__{first}_{end}__{c["id"]}'
    cloth_hit = cached_folder(cloth_dir, cloth_key)
    if not cloth_hit:
        clothing = np.empty((n, len(ids), 3), np.float32)
        for i in range(n):
            clothing[i] = bound_garment(vertices[i], faces, ids, weights, offset_m=c["offset_m"])
        if not np.isfinite(clothing).all():
            raise ValueError("nonfinite_clothing")
        np.save(cloth_dir / "clothing_vertices.npy", clothing)
    if np.load(cloth_dir / "clothing_vertices.npy", mmap_mode="r").shape != (n, len(ids), 3):
        raise ValueError("stale_cloth_cache_shape")
    hair_source = Path(resolved["hair_reference"])
    hair_key = {"body": b, "hair": h, "model": stamp(resolved["model"]), "target_ref": stamp(Path(resolved["reference_geometry"]) / "vertices.npy"),
                "source": [stamp(hair_source / x) for x in ("points.npy", "radius.npy", "curve_offsets.npy", "active_curve_mask.npy", "groom_manifest.json", "collision_correction.npy")],
                "binding": stamp(Path(read(hair_source / "groom_manifest.json")["binding"]) / "binding.npz"),
                "alembic_topology": stamp(Path(read(hair_source / "groom_manifest.json")["source_abc"]))}
    hair_dir = output_root / "cache/hair_rest" / f'{b["id"]}__{h["id"]}'
    hair_hit = cached_folder(hair_dir, hair_key)
    if not hair_hit:
        if b["id"] == h["reference_body"]:
            # Source groom remains the authority: no copy of millions of points.
            write_new(hair_dir / "reference.json", {"path": str(hair_source), "mode": "direct"})
        else:
            target_rest = hair_dir / "target_rest"
            target_rest.mkdir()
            np.save(target_rest / "vertices.npy", rv[None])
            np.save(target_rest / "faces.npy", faces)
            source_ref = Path(read(hair_source / "groom_manifest.json")["body_geometry"])
            transfer_groom({"reference": {"groom_geometry": str(hair_source), "geometry": str(source_ref)}}, target_rest, hair_dir / "transferred")
            write_new(hair_dir / "reference.json", {"path": str(hair_dir / "transferred"), "mode": "rest_scalp_triangle_frame_transfer", "source": str(hair_source)})
    hair_ref = Path(read(hair_dir / "reference.json")["path"])
    for filename in ("points.npy", "radius.npy", "curve_offsets.npy", "active_curve_mask.npy"):
        if not (hair_ref / filename).is_file():
            raise ValueError("incomplete_hair_rest_cache:" + filename)
    # The proposal is an A2-owned descriptor; A1 reads but is not modified.
    asset_id = "compose_" + name + "_" + b["id"] + "_" + m["id"]
    intervals = np.diff([x["source_time_tick"] for x in rows])
    nominal_rate = [rows[0]["time_base"][1], rows[0]["time_base"][0] * int(intervals[0])] if len(intervals) and np.all(intervals == intervals[0]) else None
    manifest = {"schema": "densekp_composed_asset_a2_v1", "asset_id": asset_id, "integration_only": True,
                "components": resolved["components"], "resolved_recipe": str(out / "frozen_recipe.json"),
                "character_family": b["character_family"], "source_group": m["source_group"], "motion_source_id": m["id"],
                "original_motion_fitted_body": m["fitted_body"], "source_ref": resolved["parameters"],
                "nominal_source_rate": nominal_rate, "parameters": str(body_dir / "parameters.npz"),
                "model": resolved["model"], "model_standard": "standard_smplx_10475_y_up_metres", "betas": beta.tolist(),
                "body_vertices": str(body_dir / "vertices.npy"), "body_joints": str(body_dir / "joints.npy"),
                "body_faces": str(body_dir / "faces.npy"), "body_R138": str(body_dir / "densekp_3d.npy"),
                "cloth_vertices": str(cloth_dir / "clothing_vertices.npy"), "cloth_faces": str(cloth_ref / "clothing_faces.npy"),
                "clothing_reference": str(cloth_ref), "clothing_source_reference": str(cloth_source),
                "groom_reference": str(hair_ref), "groom_source_reference": str(hair_source),
                "head_transforms": str(body_dir / "head_transforms.npy"), "timeline": str(out / "timeline.json"),
                "time_base": rows[0]["time_base"], "frame_count": n, "source_frames": [x["source_frame"] for x in rows],
                "root_mode": policy, "root_baked": ("source root relative to registered reference plus target shaped-rest negative pelvis; source global orientation once" if policy == WALK_ROOT_POLICY else "constant target shaped-rest negative pelvis; source global orientation once"),
                "root_reference_source_frame": resolved["root_reference_source_frame"],
                "support_policy": ("pedestrian_source_root_constant_lift" if policy == WALK_ROOT_POLICY else "fixed_scene_placement_height_no_pedestrian_foot_contact"),
                "scene_contract": "body/cloth/R138 scene @ model; groom scene @ head @ target-rest reference",
                "body_human_status": "accepted_in_source_shape_only" if b["id"] != m["fitted_body"] else "accepted_in_existing_combination_only",
                "dressed_human_status": "pending_user", "public_scene_status": "not_consumed",
                "license": resolved["license"], "source_identity_status": "see parent motion; derived appearance is not source actor",
                "dependencies": resolved["dependencies"]}
    write_new(out / "delivery_manifest.json", manifest)
    proposal = {"schema": "densekp_explicit_asset_combination_a2_proposal_v1", "asset_id": asset_id,
                "integration_only": True, "A1_schema_effective": False, "public_consumer_tested": False,
                "units": "metres", "coordinates": "SMPL-X model Y-up", "regressor": str(root / "assets/densekp_regressor_R.npy"),
                "densekp_label_ready": True, "densekp_status": "compatible_standard_smplx", "scene_observations_produced": False,
                "independent_validation": False, "source_actor_identity": "unknown_or_parent_record", "persistent_dependencies_only": True,
                "character_family": b["character_family"], "character_manifest": resolved["body_manifest"],
                "authority_manifest": str(out / "delivery_manifest.json"), "parameters": manifest["parameters"],
                "source_group": m["source_group"], "motion_source_id": m["id"], "frame_count": n,
                "timeline": manifest["timeline"], "time_base": manifest["time_base"],
                "time_unit_status": "published source integer timeline; actual capture PTS only where parent source provides it",
                "geometry": {"vertices": manifest["body_vertices"], "faces": manifest["body_faces"], "densekp": manifest["body_R138"]},
                "garment": {"representation": "simple_fixed_barycentric_normal_offset", "vertices": manifest["cloth_vertices"],
                            "faces": manifest["cloth_faces"], "binding_indices": str(cloth_ref / "body_barycentric_indices.npy"),
                            "binding_weights": str(cloth_ref / "body_barycentric_weights.npy"), "offset_m": c["offset_m"],
                            "material_kind": "GarmentRest_continuous_shell", "materials": str(cloth_ref / "materials.json"),
                            "material_reference": str(cloth_ref / "garment_material.json"),
                            "rest_coordinates": str(cloth_ref / "clothing_rest_vertices.npy"), "uv": None},
                "groom": {"representation": "fixed_reference_curves_rigid_head", "reference": str(hair_ref),
                          "points": str(hair_ref / "points.npy"), "radius": str(hair_ref / "radius.npy"),
                          "curve_offsets": str(hair_ref / "curve_offsets.npy"), "active_mask": str(hair_ref / "active_curve_mask.npy"),
                          "head_transforms": manifest["head_transforms"], "reference_space": "target shaped-rest pelvis-centered model",
                          "root_in_head": True, "physics": False},
                "appearance": {"kind": "existing_character_skin_and_native_eye_shader", "character_manifest": resolved["body_manifest"]},
                "root_mode": policy, "root_baked": manifest["root_baked"], "support_policy": manifest["support_policy"],
                "human_acceptance": "pending_new_combination_review", "technical_evidence": str(out / "validation.json")}
    write_new(out / "a1_proposal.json", proposal)
    R = np.load(root / "assets/densekp_regressor_R.npy")
    dense = np.load(body_dir / "densekp_3d.npy", mmap_mode="r")
    max_r = max(float(np.max(np.abs(compute_densekp_3d(R, vertices[i:i+1])[0] - dense[i]))) for i in range(n))
    if max_r > 2e-6:
        raise ValueError("R138_readback_failure")
    if policy == ROOT_POLICY:
        adapted = adapt_proposal(out / "a1_proposal.json")
        paths = geometry_paths(adapted)
        adapter_status = "passed_read_only"
    else:
        # A1's explicit proposal adapter is cycling-only. Its existing legacy
        # walking path consumes this descriptor and geometry without rewriting
        # the source-root policy as cycling.
        with np.load(body_dir / "parameters.npz", allow_pickle=False) as params:
            root_relative = params["source_root_relative"]
            if not np.allclose(params["transl"] - params["transl"][0], root_relative - root_relative[0], atol=2e-6, rtol=0):
                raise ValueError("walking_root_readback_failure")
        walking_view = {"schema": A1_ASSET_SCHEMA, "asset_id": asset_id, "character_manifest": resolved["body_manifest"],
                        "character_family": b["character_family"], "clothing_reference": str(cloth_ref),
                        "groom_reference": str(hair_ref), "parameters": str(body_dir / "parameters.npz"), "source_ref": resolved["parameters"],
                        "motion_source_id": m["id"], "root_mode": policy, "source_start_reference": resolved["root_reference_source_frame"],
                        "nominal_source_rate": nominal_rate, "timeline_ref": str(out / "timeline.json"), "time_base": rows[0]["time_base"],
                        "available_combinations_verified": True, "integration_only": True, "independent_validation": False}
        walking_view.update(body_vertices=manifest["body_vertices"], body_faces=manifest["body_faces"],
                            cloth_vertices=manifest["cloth_vertices"], cloth_faces=manifest["cloth_faces"],
                            head_transforms=manifest["head_transforms"], densekp=manifest["body_R138"])
        write_new(out / "a1_walking_view.json", walking_view)
        paths = geometry_paths(read(out / "a1_walking_view.json"))
        identity = np.eye(4).tolist()
        for start in (0, min(3, n-2)):
            scene = dict(root_mode=policy, model_to_world=identity)
            matrices = person_matrices(scene, 2)
            if not np.array_equal(matrices[0], matrices[1]):
                raise ValueError("walking_scene_matrix_not_fixed")
            with np.load(body_dir / "parameters.npz", allow_pickle=False) as params:
                local_root = params["source_root_relative"][start:start+2]
                if np.max(np.abs((np.load(body_dir / "joints.npy", mmap_mode="r")[start:start+2, 0] - local_root))) > 2e-5:
                    raise ValueError("walking_subwindow_root_twice_or_missing")
        adapter_status = "passed_A1_geometry_paths_timeline_person_matrices_read_only"
    selected = select_timeline(rows, 0, n, n)
    if len(selected) != n or any(not v.is_file() for v in paths.values()):
        raise ValueError("A1_readonly_adapter_failure")
    write_new(out / "a1_adapter_check.json", {"status": adapter_status, "frame_count": n, "paths": {k: str(v) for k, v in paths.items()}, "scene_consumed": False})
    validation = {"status": "numeric_passed", "integration_only": True, "frames": n,
                  "body_cache_hit": body_hit, "cloth_rest_cache_hit": cloth_rest_hit, "cloth_motion_cache_hit": cloth_hit,
                  "hair_rest_cache_hit": hair_hit, "R138_readback_max_m": max_r,
                  "body_finite": bool(np.isfinite(vertices).all()), "cloth_finite": bool(np.isfinite(np.load(cloth_dir / "clothing_vertices.npy", mmap_mode="r")).all()),
                  "head_finite": bool(np.isfinite(heads).all()), "fixed_shape": True,
                  "fixed_body_topology": int(len(faces)), "fixed_cloth_topology": int(len(cloth_faces)),
                  "source_time_monotonic": True, "no_resampling": True, "no_per_frame_grounding": True,
                  "hair_reference_mode": read(hair_dir / "reference.json")["mode"],
                  "body_model_bounds_m": [np.min(vertices, axis=(0, 1)).tolist(), np.max(vertices, axis=(0, 1)).tolist()],
                  "compute_seconds": time.monotonic() - began, "dressed_human_status": "pending_user", "public_scene_status": "not_consumed"}
    write_new(out / "validation.json", validation)
    return out, validation


def prepare_preview(output_root, name, gpu_uuid, frames=60):
    """Freeze a native-time short studio preview for an already built combination."""
    name = safe(name)
    out = Path(output_root).resolve() / "builds" / name
    manifest = read(out / "delivery_manifest.json")
    resolved = read(out / "frozen_recipe.json")
    if manifest["frame_count"] < frames or frames < 3:
        raise ValueError("invalid_preview_window")
    groom = Path(manifest["groom_reference"])
    active = np.load(groom / "active_curve_mask.npy")
    offsets = np.load(groom / "curve_offsets.npy")
    counts = {"curves": int(active.sum()), "points": int(np.diff(offsets)[active].sum())}
    chosen = [0, frames // 2, frames - 1]
    cloth = np.load(manifest["cloth_vertices"], mmap_mode="r")
    heads = np.load(manifest["head_transforms"], mmap_mode="r")
    camera_target, ortho_scale = [0.0, 0.0, -0.10], 2.3
    if resolved["root_policy"] == WALK_ROOT_POLICY:
        bounds = np.asarray(read(out / "validation.json")["body_model_bounds_m"], float)
        blender_bounds = np.stack((bounds[:, 0], -bounds[::-1, 2], bounds[:, 1]), axis=1)
        camera_target = blender_bounds.mean(axis=0).tolist()
        ortho_scale = max(2.3, float(np.max(blender_bounds[1] - blender_bounds[0]) * 1.25))
    for stage, indices in (("probe", chosen), ("full", list(range(frames)))):
        target = out / stage
        if target.exists():
            raise FileExistsError(target)
        target.mkdir()
        np.save(target / "clothing_vertices.npy", cloth[indices])
        np.save(target / "head_transforms.npy", heads[indices])
        np.save(target / "asset_frames.npy", np.asarray(indices, np.int32))
    recipe = {"character_manifest": resolved["body_manifest"], "body_cache": manifest["body_vertices"],
              "faces": manifest["body_faces"], "source_clothing": manifest["clothing_reference"],
              "source_groom": manifest["groom_reference"], "timeline": manifest["timeline"],
              "probe_asset_frames": chosen, "groom_render_counts": counts,
              "preview_window": [0, frames], "budget": {"gpu_uuid": gpu_uuid, "resolution_each": [512, 512],
              "samples": 64, "denoising": False, "views": ["front", "side"], "max_render_seconds": 2700},
              "camera_target_blender_m": camera_target, "camera_ortho_scale_m": ortho_scale,
              "time_base": manifest["time_base"], "original_source_frames": manifest["source_frames"][:frames]}
    write_new(out / "preview_recipe.json", recipe)
    return out / "preview_recipe.json"
