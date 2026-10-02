"""Write/reload the existing CPU geometry format with its original gates.

This writer retains the commuter/source-trial contract. The older cycling
writer adds separate clothing/material records and remains at its entry point.
"""

import numpy as np
from densekp_retrain.body.parameters import PARAMETERS, evaluate
from densekp_retrain.body.geometry import compute_densekp_3d
from densekp_retrain.io.json import write_json


def save_geometry(out, model, params, info, regressor):
    """Write body [T,V,3], faces, 138 points and motion params into a new directory.

    Re-evaluate serialized parameters using existing 2e-5 m export, 1e-6 m
    regressor and 2e-6 m pelvis gates; retain errors and partial-output order."""
    out.mkdir(parents=True)
    body, joints = evaluate(model, params)
    dense = compute_densekp_3d(regressor, body)
    np.save(out / "vertices.npy", body)
    np.save(out / "faces.npy", model.faces)
    np.save(out / "densekp_3d.npy", dense)
    np.savez(out / "motion_params.npz", **params)
    loaded = np.load(out / "motion_params.npz")
    restored, _ = evaluate(model, {k: loaded[k] for k in PARAMETERS})
    error = float(np.abs(restored - body).max())
    operator = float(np.abs(regressor @ body[0] - dense[0]).max())
    if error > 2e-5 or operator > 1e-6 or np.abs(joints[0, 0]).max() > 2e-6:
        raise ValueError("Existing geometry gate failed")
    write_json(
        out / "geometry_manifest.json",
        dict(
            info,
            export_max_abs_error_m=error,
            operator_max_error_m=operator,
            pelvis_after_translation_m=joints[0, 0].tolist(),
            root_policy="recomputed target shaped pelvis; no source translation",
            split="integration_only",
        ),
    )
    return dict(geometry=str(out), export_max_abs_error_m=error, operator_max_error_m=operator)
