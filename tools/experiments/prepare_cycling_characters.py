#!/usr/bin/env python3
"""Instantiate fixed local characters and independent cycling derivatives."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.body.geometry import load_smplx_model, compute_densekp_3d
from densekp_retrain.clothing.shells import outfit_regions, shell

POSES = ["coco_292226_person_1330527", "coco_306800_person_225160"]
from densekp_retrain.body.parameters import PARAMETERS  # historical export

MATERIALS = [
    dict(name="cycling_jersey", color=[0.10, 0.23, 0.42, 1]),
    dict(name="cycling_shorts", color=[0.022, 0.03, 0.045, 1]),
    dict(name="jersey_band", color=[0.85, 0.88, 0.9, 1]),
]


from densekp_retrain.io.json import write_json  # historical export


from densekp_retrain.body.parameters import evaluate  # historical export


from densekp_retrain.body.parameters import zero_pose  # historical export


def save_body_and_clothing(out, model, params, rest, rest_joints, regressor, manifest):
    geometry = out / "geometry"
    geometry.mkdir(parents=True, exist_ok=False)
    clothing_dir = out / "clothing"
    clothing_dir.mkdir()
    body, joints = evaluate(model, params)
    if not np.isfinite(body).all():
        raise ValueError("Nonfinite body")
    faces = model.faces.astype(np.int32)
    np.save(geometry / "vertices.npy", body)
    np.save(geometry / "faces.npy", faces)
    dense = compute_densekp_3d(regressor, body)
    np.save(geometry / "densekp_3d.npy", dense)
    np.savez(
        geometry / "motion_params.npz",
        **params,
        gender=np.array(manifest["gender"]),
        fps=np.float32(0),
        sequence=np.array(False),
        source_frame_ids=np.array([0]),
    )
    with np.load(geometry / "motion_params.npz") as z:
        restored, _ = evaluate(model, {k: z[k] for k in PARAMETERS})
    restore_error = float(np.abs(restored - body).max())
    if restore_error > 2e-5:
        raise ValueError("Saved parameters do not restore mesh within existing 2e-5 m gate")
    if np.abs(joints[0, 0]).max() > 2e-6:
        raise ValueError("Recomputed pelvis should be at canonical origin")
    operator_error = float(np.abs(regressor @ body[0] - dense[0]).max())
    if operator_error > 1e-6:
        raise ValueError("Existing 1e-6 m operator gate failed")
    selected, material_ids = outfit_regions(rest, rest_joints, faces)
    clothing, clothing_faces, body_ids, _ = shell(body[0], faces, selected, 0.004)
    triangles = body[0, faces[selected]]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    separation = ((clothing[clothing_faces].mean(1) - triangles.mean(1)) * normals).sum(1)
    if separation.min() < 0.0004 - 2e-7:
        raise ValueError("Clothing failed existing local outward clearance")
    offset = float(np.linalg.norm(clothing - body[0, body_ids], axis=1).max())
    if offset > 0.01:
        raise ValueError("Clothing exceeds existing 1cm offset gate")
    for name, array in dict(
        clothing_vertices=clothing[None],
        clothing_faces=clothing_faces,
        body_vertex_ids=body_ids,
        body_face_ids=np.flatnonzero(selected),
        face_material_ids=material_ids,
        reference_body_vertices=body[:, body_ids],
    ).items():
        np.save(clothing_dir / f"{name}.npy", array)
    write_json(clothing_dir / "materials.json", MATERIALS)
    clothing_info = dict(
        body_geometry=str(geometry.resolve()),
        character_id=manifest["character_id"],
        model_file=manifest["model_file"],
        gender=manifest["gender"],
        betas=params["betas"].tolist(),
        split="integration_only",
        method="4mm normal shell, shaped rest jersey/shorts regions, local clearance correction",
        max_vertex_offset_m=offset,
        minimum_face_normal_separation_m=float(separation.min()),
        placement="body object transform including BODY minimum-Y lift",
        limitation="tight procedural clothing; no cloth simulation or exhaustive collision proof",
    )
    write_json(clothing_dir / "clothing_manifest.json", clothing_info)
    info = dict(
        manifest,
        split="integration_only",
        sequence=False,
        frames=1,
        parameters=str((geometry / "motion_params.npz").resolve()),
        pelvis_after_translation_m=joints[0, 0].tolist(),
        root_policy="target shaped pelvis at canonical origin; source translation discarded",
        export_max_abs_error_m=restore_error,
        operator_max_error_m=operator_error,
        regressor=str((ROOT / "assets/densekp_regressor_R.npy").resolve()),
    )
    write_json(geometry / "geometry_manifest.json", info)
    return dict(
        geometry=str(geometry.resolve()),
        clothing_geometry=str(clothing_dir.resolve()),
        export_max_abs_error_m=restore_error,
        operator_max_error_m=operator_error,
        clothing_max_offset_m=offset,
        clothing_min_clearance_m=float(separation.min()),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--source", type=Path, default=ROOT / "outputs/experiments/cycling_smplx_20260929"
    )
    p.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/smpl_models/smplx"),
    )
    p.add_argument(
        "--texture-archive",
        type=Path,
        default=Path("/home/yy/datasets/BEDLAM/body_textures/bedlam_body_textures_meshcapade.zip"),
    )
    a = p.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    regressor = np.load(ROOT / "assets/densekp_regressor_R.npy")
    uv_path = ROOT / "outputs/body_texture_audit/body_uv.npz"
    labels = np.load(uv_path)["component_labels"]
    aaron_path = Path(
        "/home/yy/datasets/BEDLAM/motion_data/tiny/rp_aaron_posed_009/1095/motion_seq.npz"
    )
    with np.load(aaron_path) as z:
        aaron = z["betas"].copy()
    definitions = [
        ("F01", "female", [0] * 10, "skin_f_asian_01_ALB", "synthetic zero-shape preset"),
        (
            "F02",
            "female",
            [0.75, -0.5, 0.25] + [0] * 7,
            "skin_f_african_01_ALB",
            "synthetic preset from feasibility report",
        ),
        ("M01", "male", aaron, "skin_m_white_01_ALB", str(aaron_path)),
        ("M02", "male", [0] * 10, "skin_m_african_01_ALB", "synthetic zero-shape preset"),
    ]
    characters = []
    derivatives = []
    rest_bodies = []
    for short, gender, betas, texture_id, shape_source in definitions:
        char_id = f"cycling_{short}_v1"
        out = a.output / "characters" / char_id
        out.mkdir(parents=True)
        model = load_smplx_model(a.model_dir, gender=gender, num_betas=10, batch_size=1).eval()
        beta = np.asarray(betas, np.float32)
        with torch.no_grad():
            shaped = model(betas=torch.from_numpy(beta[None]))
        pelvis = shaped.joints[0, 0].numpy()
        rest = shaped.vertices[0].numpy() - pelvis
        rest_joints = shaped.joints[0].numpy() - pelvis
        rest_bodies.append(rest)
        reference_faces = np.load(a.source / POSES[0] / "geometry/faces.npy")
        np.testing.assert_array_equal(model.faces, reference_faces)
        member = f"bedlam_body_textures_meshcapade/smpl/MC_texture_skintones/{gender}/skin/{texture_id}.png"
        texture_path = out / f"{texture_id}.png"
        with zipfile.ZipFile(a.texture_archive) as z:
            texture_path.write_bytes(z.read(member))
        coordinates = np.zeros_like(rest)
        eye_components = []
        for component in np.unique(labels):
            ids = np.flatnonzero(labels == component)
            joint = int(model.lbs_weights.numpy()[ids].mean(0).argmax())
            if joint not in (23, 24):
                continue
            center = rest[ids].mean(0)
            coordinates[ids] = rest[ids] - center
            eye_components.append(
                dict(component=int(component), joint=joint, center_m=center.tolist())
            )
        if len(eye_components) != 2:
            raise ValueError("Expected both native eye components")
        mapping = out / "eye_mapping.npz"
        np.savez(mapping, coordinates=coordinates, faces=model.faces, component_labels=labels)
        appearance = dict(
            body_texture=str(texture_path),
            body_uv=str(uv_path.resolve()),
            eye_mapping=str(mapping),
            eye_patch=dict(
                centers=[[0.239, 0.821], [0.354, 0.821]],
                radii=[0.037, 0.023],
                sample_offset=[0, 0.075],
                feather_inner=0.65,
            ),
            iris_color=[0.12, 0.075, 0.035, 1],
            hair_mode="body albedo only; no separate 3D hair",
            character_id=char_id,
            material_version=1,
        )
        manifest = dict(
            character_id=char_id,
            status="candidate_pending_visual_review",
            version=1,
            split="integration_only",
            gender=gender,
            model_dir=str(a.model_dir),
            model_file=str(a.model_dir / f"SMPLX_{gender.upper()}.npz"),
            num_betas=10,
            betas=beta.tolist(),
            shape_source=shape_source,
            shaped_pelvis_m=pelvis.tolist(),
            model_version="local standard SMPL-X, first 10 shape directions verified against archived model",
            texture_archive=str(a.texture_archive),
            texture_member=member,
            face_source="head shape plus Meshcapade albedo; not the COCO source identity",
            texture_license="Meshcapade CC BY-NC 4.0, version 20220207",
            model_license="https://smpl-x.is.tue.mpg.de/modellicense.html",
            shape_license=(
                "https://bedlam.is.tuebingen.mpg.de/license.html"
                if short == "M01"
                else "SMPL-X derivative"
            ),
            appearance=appearance,
            eye_components=eye_components,
            clothing=dict(
                source="tools/experiments/prepare_cycling_clothing.py", materials=MATERIALS
            ),
            proportions=dict(
                rest_height_m=float(np.ptp(rest[:, 1])),
                shoulder_joint_width_m=float(np.linalg.norm(rest_joints[16] - rest_joints[17])),
                hip_joint_width_m=float(np.linalg.norm(rest_joints[1] - rest_joints[2])),
            ),
            history=[
                dict(version=1, reason="instantiate feasibility candidate; review before freezing")
            ],
        )
        params = zero_pose(beta, pelvis)
        reference = save_body_and_clothing(
            out / "reference", model, params, rest, rest_joints, regressor, manifest
        )
        manifest["reference"] = reference
        write_json(out / "manifest.json", manifest)
        characters.append(dict(character_id=char_id, manifest=str(out / "manifest.json")))
        for pose_id in POSES:
            source_path = a.source / pose_id / "canonical_pose.npz"
            with np.load(source_path) as z:
                source_params = {k: z[k].copy() for k in PARAMETERS}
            params = {k: source_params[k] for k in PARAMETERS}
            params["betas"] = beta
            params["transl"] = -pelvis[None].copy()
            derived = a.output / "derivatives" / char_id / pose_id
            info = dict(
                manifest,
                pose_source_id=pose_id,
                source_parameters=str(source_path.resolve()),
                source_character_shape_used=False,
                motion=False,
                source_translation_used=False,
            )
            record = save_body_and_clothing(
                derived, model, params, rest, rest_joints, regressor, info
            )
            record.update(
                character_id=char_id,
                pose_source_id=pose_id,
                appearance=appearance,
                split="integration_only",
            )
            derivatives.append(record)
    pairwise = []
    head_ids = np.flatnonzero(model.lbs_weights.numpy()[:, 15] > 0.3)
    for i, first in enumerate(rest_bodies):
        for j in range(i + 1, len(rest_bodies)):
            second = rest_bodies[j]
            ha = first[head_ids]
            hb = second[head_ids]
            pairwise.append(
                dict(
                    a=characters[i]["character_id"],
                    b=characters[j]["character_id"],
                    body_rms_m=float(np.sqrt(np.mean((first - second) ** 2))),
                    centered_head_rms_m=float(
                        np.sqrt(np.mean(((ha - ha.mean(0)) - (hb - hb.mean(0))) ** 2))
                    ),
                )
            )
    report = dict(
        status="geometry_ready_visual_review_pending",
        split="integration_only",
        created_utc=datetime.now(timezone.utc).isoformat(),
        seconds=time.monotonic() - start,
        characters=characters,
        derivatives=derivatives,
        shape_differences=pairwise,
        thresholds=dict(
            export_m=2e-5,
            operator_m=1e-6,
            world_mesh_m=5e-6,
            clothing_world_m=5e-6,
            projection_crop_px=2e-4,
            local_clothing_clearance_m=0.0004,
            clothing_max_offset_m=0.01,
        ),
        pending=[
            "uniform appearance review before DS rendering",
            "DS rendering and actual loader validation",
        ],
    )
    write_json(a.output / "report.json", report)
    print(
        json.dumps(dict(output=str(a.output), seconds=report["seconds"], shapes=pairwise), indent=2)
    )


if __name__ == "__main__":
    main()
