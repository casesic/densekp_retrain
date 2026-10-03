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


def parameters_from_pose(pose, beta, pelvis):
    """Split one SMPL-X axis-angle pose into ordered, batched float32 blocks.

    A complete pose has 165 values in radians. Preserve ordinary slice behavior
    for shorter inputs. ``beta`` is returned unchanged, including its identity;
    translation is ``-pelvis[None]`` in model-axis meters, without reshaping.
    """
    limits = [
        ("global_orient", 0, 3),
        ("body_pose", 3, 66),
        ("jaw_pose", 66, 69),
        ("leye_pose", 69, 72),
        ("reye_pose", 72, 75),
        ("left_hand_pose", 75, 120),
        ("right_hand_pose", 120, 165),
    ]
    return dict(
        {key: np.asarray(pose[a:b], np.float32)[None] for key, a, b in limits},
        betas=beta,
        transl=-pelvis[None],
    )


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
