"""CPU SMPL-X parameters and shaped-rest geometry, in model-axis meters.

PARAMETERS preserves argument order and excludes expression, exactly as before.
No fitted identity, pose recipe or rendering asset is selected here.
"""

import numpy as np
import torch

PARAMETERS = [
    "global_orient",
    "body_pose",
    "jaw_pose",
    "leye_pose",
    "reye_pose",
    "left_hand_pose",
    "right_hand_pose",
    "betas",
    "transl",
]


def evaluate(model, params):
    """Evaluate a CPU model using ordered float32 parameters.

    Returns vertices [T,V,3] and joints [T,J,3] in model-axis meters.
    Betas are reshaped to [1,B]; pose angles are axis-angle radians."""
    tensors = {k: torch.from_numpy(np.asarray(params[k], np.float32)) for k in PARAMETERS}
    tensors["betas"] = tensors["betas"].reshape(1, -1)
    with torch.no_grad():
        result = model(**tensors)
    return result.vertices.numpy(), result.joints.numpy()


def zero_pose(betas, pelvis):
    """Zero axis angles (radians), supplied shape coefficients and -pelvis translation.

    Translation is [1,3] in meters; the historical betas shape is retained."""
    params = {
        k: np.zeros((1, n), np.float32)
        for k, n in [
            ("global_orient", 3),
            ("body_pose", 63),
            ("jaw_pose", 3),
            ("leye_pose", 3),
            ("reye_pose", 3),
            ("left_hand_pose", 45),
            ("right_hand_pose", 45),
        ]
    }
    params.update(betas=np.asarray(betas, np.float32), transl=-pelvis.reshape(1, 3))
    return params


def rest_body(model, beta):
    """Return pelvis-centered rest vertices [V,3], joints [J,3], and pelvis [3], in meters."""
    with torch.no_grad():
        result = model(betas=torch.tensor(beta, dtype=torch.float32)[None])
    pelvis = result.joints[0, 0].numpy()
    return result.vertices[0].numpy() - pelvis, result.joints[0].numpy() - pelvis, pelvis
