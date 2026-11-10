"""Fixed surface clothing and rigid-head transforms for manifest-backed sequences.

No character, action source or scene is selected here. Vertices are in SMPL-X
model metres, including the caller-owned root translation. Apply the same scene
matrix once to body/garment/landmarks, and scene @ head to reference groom.
"""
import numpy as np
from densekp_retrain.clothing.shells import vertex_normals


def bound_garment(vertices, faces, indices, weights, *, offset_m):
    surface = (vertices[indices] * weights[..., None]).sum(1)
    normals = (vertex_normals(vertices, faces)[indices] * weights[..., None]).sum(1)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    result = (surface + offset_m * normals).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite bound garment")
    return result


def head_transform(local_rotations, parents, posed_joint, rest_joint, *, joint_id=15):
    rotations = np.asarray(local_rotations).copy()
    for k in range(1, joint_id + 1):
        rotations[k] = rotations[int(parents[k])] @ rotations[k]
    matrix = np.eye(4)
    matrix[:3, :3] = rotations[joint_id]
    matrix[:3, 3] = posed_joint - matrix[:3, :3] @ rest_joint
    if not np.isfinite(matrix).all() or not np.allclose(matrix[:3, :3].T @ matrix[:3, :3], np.eye(3), atol=2e-6):
        raise ValueError("Invalid rigid head transform")
    return matrix
