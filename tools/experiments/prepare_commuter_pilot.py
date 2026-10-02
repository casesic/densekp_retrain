#!/usr/bin/env python3
"""Two preserved identities: full-space regression, bounded shape fit, four poses."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import torch
from scipy.optimize import lsq_linear

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.body.geometry import load_smplx_model, compute_densekp_3d
from densekp_retrain.body.parameters import PARAMETERS, evaluate, zero_pose
from tools.experiments.prepare_cycling_characters import POSES
from densekp_retrain.io.json import write_json


from densekp_retrain.body.parameters import rest_body  # historical export


from densekp_retrain.body.measurements import measurement_rows  # historical export


from densekp_retrain.body.measurements import measure  # historical export


def choose_shape(model, parent, labels, gender):
    base = np.pad(np.asarray(parent, np.float32), (0, model.num_betas - len(parent)))
    v, joints, _ = rest_body(model, base)
    rows = measurement_rows(v, joints, labels, model.lbs_weights[:, 15].numpy())
    original = measure(v, rows)
    # Modest physical design targets; coefficient bounds are engineering guards,
    # not a claim about a learned population distribution or beta standard deviations.
    ratios = np.array(
        [
            1.0,
            1.035 if gender == "female" else 1.06,
            1.025 if gender == "female" else 1.055,
            1.04,
            1.055,
            0.995,
        ]
    )
    target = original * ratios
    jac = []
    for i in range(len(base)):
        delta = base.copy()
        delta[i] += 0.1
        changed, _, _ = rest_body(model, delta)
        jac.append((measure(changed, rows) - original) / 0.1)
    jac = np.array(jac).T
    tolerance = np.array([0.006, 0.006, 0.006, 0.0015, 0.0015, 0.003])
    matrix = np.vstack((jac / tolerance[:, None], np.eye(len(base)) * 0.7))
    rhs = np.r_[(target - original) / tolerance, np.zeros(len(base))]
    fitted = lsq_linear(matrix, rhs, bounds=(-1.0, 1.0), tol=1e-9)
    beta = (base + fitted.x).astype(np.float32)
    final, _, _ = rest_body(model, beta)
    if np.max(np.linalg.norm(final - v, axis=1)) > 0.06:
        raise ValueError("Candidate exceeds the declared 6cm rest displacement guard")
    measurements = dict(
        names=[r[0] for r in rows],
        parent_m=original.tolist(),
        target_m=target.tolist(),
        achieved_m=measure(final, rows).tolist(),
        method="fixed rest anatomical-band spans; not clinical anthropometry",
        optimization="regularized least squares in complete standard shape space; expression fixed at zero",
        coefficient_change_limit=1.0,
        coefficient_limit_meaning="engineering guard, not population standard deviation",
        max_vertex_displacement_m=float(np.linalg.norm(final - v, axis=1).max()),
        status="candidate_requires_uniform_visual_review",
    )
    return beta, measurements


from densekp_retrain.io.geometry import save_geometry  # historical export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-betas", type=int, default=30)
    a = p.parse_args()
    start = time.monotonic()
    torch.set_num_threads(2)
    a.output = a.output.resolve()
    a.output.mkdir(parents=True, exist_ok=False)
    old = ROOT / "outputs/experiments/cycling_four_characters_20260929"
    archive = Path(
        "/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools/PEAR/assets/models_smplx_v1_1.zip"
    )
    models = a.output / "models"
    models.mkdir()
    with zipfile.ZipFile(archive) as z:
        for gender in ["FEMALE", "MALE"]:
            name = f"SMPLX_{gender}.npz"
            (models / name).write_bytes(z.read("models/smplx/" + name))
    regressor = np.load(ROOT / "assets/densekp_regressor_R.npy")
    labels = np.load(ROOT / "outputs/body_texture_audit/body_uv.npz")["component_labels"]
    characters = []
    derivatives = []
    regression = []
    for short, gender in [("F02", "female"), ("M02", "male")]:
        parent_path = old / "characters" / f"cycling_{short}_v1" / "manifest.json"
        parent = json.loads(parent_path.read_text())
        model = load_smplx_model(models, gender=gender, num_betas=a.num_betas, batch_size=1).eval()
        small = load_smplx_model(
            Path(parent["model_dir"]), gender=gender, num_betas=10, batch_size=1
        ).eval()
        for key in [
            "v_template",
            "faces_tensor",
            "posedirs",
            "lbs_weights",
            "J_regressor",
            "parents",
        ]:
            np.testing.assert_array_equal(getattr(model, key).numpy(), getattr(small, key).numpy())
        np.testing.assert_array_equal(model.shapedirs[:, :, :10].numpy(), small.shapedirs.numpy())
        np.testing.assert_array_equal(model.expr_dirs.numpy(), small.expr_dirs.numpy())
        for pose in ["reference"] + POSES:
            folder = (
                Path(parent["reference"]["geometry"])
                if pose == "reference"
                else old / "derivatives" / parent["character_id"] / pose / "geometry"
            )
            z = np.load(folder / "motion_params.npz")
            params = {k: z[k] for k in PARAMETERS}
            params["betas"] = np.pad(params["betas"], (0, a.num_betas - 10))
            body, _ = evaluate(model, params)
            ve = float(np.abs(body - np.load(folder / "vertices.npy")).max())
            ke = float(np.abs(regressor @ body[0] - np.load(folder / "densekp_3d.npy")[0]).max())
            if ve > 2e-5 or ke > 1e-6:
                raise ValueError("Full-space legacy regression failed")
            regression.append(
                dict(
                    character_id=parent["character_id"],
                    pose=pose,
                    vertex_max_m=ve,
                    densekp_max_m=ke,
                )
            )
        beta, measurements = choose_shape(model, parent["betas"], labels, gender)
        rest, joints, pelvis = rest_body(model, beta)
        cid = f"cycling_{short}_v2"
        folder = a.output / "characters" / cid
        folder.mkdir(parents=True)
        coords = np.zeros_like(rest)
        for label in np.unique(labels):
            ids = np.flatnonzero(labels == label)
            if label:
                coords[ids] = rest[ids] - rest[ids].mean(0)
        mapping = folder / "eye_mapping.npz"
        np.savez(mapping, coordinates=coords, faces=model.faces, component_labels=labels)
        appearance = deepcopy(parent["appearance"])
        appearance.update(eye_mapping=str(mapping), character_id=cid)
        manifest = dict(
            character_id=cid,
            identity_family_id=f"cycling_{short}",
            parent_asset_id=parent["character_id"],
            parent_manifest=str(parent_path),
            version=2,
            shape_version=2,
            status="candidate_not_frozen",
            gender=gender,
            model_dir=str(models),
            model_file=str(models / f"SMPLX_{gender.upper()}.npz"),
            model_archive=str(archive),
            num_betas=a.num_betas,
            betas=beta.tolist(),
            expression_policy="zero, separate full-file directions 300:310; not identity",
            shape_source="regularized whole-body shape-space fit derived from parent",
            measurements=measurements,
            appearance=appearance,
            split="integration_only",
            hair_status="blocked_missing_authenticated_groom_and_authoring_reference",
            model_license=parent["model_license"],
            texture_license=parent["texture_license"],
        )
        reference = save_geometry(
            folder / "reference" / "geometry", model, zero_pose(beta, pelvis), manifest, regressor
        )
        manifest["reference"] = reference
        np.save(folder / "rest_joints.npy", joints)
        write_json(folder / "manifest.json", manifest)
        characters.append(dict(character_id=cid, manifest=str(folder / "manifest.json")))
        for pose in POSES:
            source = (
                ROOT / "outputs/experiments/cycling_smplx_20260929" / pose / "canonical_pose.npz"
            )
            z = np.load(source)
            params = {k: z[k] for k in PARAMETERS}
            params.update(betas=beta, transl=-pelvis[None])
            info = dict(
                manifest, pose_source_id=pose, source_parameters=str(source), sequence=False
            )
            row = save_geometry(
                a.output / "derivatives" / cid / pose / "geometry", model, params, info, regressor
            )
            row.update(
                character_id=cid,
                pose_source_id=pose,
                appearance=appearance,
                split="integration_only",
            )
            derivatives.append(row)
    write_json(
        a.output / "model_regression.json",
        dict(
            status="passed",
            cases=regression,
            shape_indices=[0, 300],
            expression_indices=[300, 310],
            legacy_compact_expression_indices=[10, 20],
            arrays_compared=[
                "template",
                "faces",
                "posedirs",
                "weights",
                "J_regressor",
                "parents",
                "first10_shape",
                "expression10",
            ],
        ),
    )
    write_json(
        a.output / "report.json",
        dict(
            status="shape_candidates_ready_clothing_pending_hair_blocked",
            created_utc=datetime.now(timezone.utc).isoformat(),
            split="integration_only",
            characters=characters,
            derivatives=derivatives,
            compute_seconds=time.monotonic() - start,
        ),
    )
    print(
        json.dumps(
            dict(output=str(a.output), seconds=time.monotonic() - start, regression=regression)
        )
    )


if __name__ == "__main__":
    main()
