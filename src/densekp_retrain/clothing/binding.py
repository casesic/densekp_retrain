"""Stable triangle clipping, surface binding and pose-frame computations.

Inputs use SMPL-X axes/meters unless callers explicitly apply CONV. Candidate
surface search uses 24 nearest face centroids, not exhaustive mesh distance.
The 8-digit clipping key here deliberately differs from clipped_shell's 9.
"""

import numpy as np
import torch
import trimesh
from scipy.spatial import cKDTree
from smplx.lbs import batch_rodrigues, batch_rigid_transform


def clip_surface(rest, faces, planes):
    """Clip triangles by axis-aligned half spaces in model coordinates.

    planes holds (axis, sign, bias): sign * coordinate + bias >= 0.
    Return body vertex IDs [N,3], float32 barycentric weights [N,3], local faces [F,3]."""
    lookup = {}
    indices = []
    weights = []
    output = []
    for tri in faces:
        rv = rest[tri]
        if any(np.max(s * rv[:, axis] + b) < 0 for axis, s, b in planes):
            continue
        polygon = list(np.eye(3))
        for axis, sign, bias in planes:
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da = sign * (a @ rv)[axis] + bias
                db = sign * (b @ rv)[axis] + bias
                if da >= 0:
                    clipped.append(a)
                if (da >= 0) != (db >= 0):
                    clipped.append(a + (b - a) * da / (da - db))
            polygon = clipped
            if not polygon:
                break
        ids = []
        for w in polygon:
            pairs = sorted((int(i), float(x)) for i, x in zip(tri, w) if x > 1e-9)
            key = tuple((i, round(x, 8)) for i, x in pairs)
            if key not in lookup:
                lookup[key] = len(indices)
                indices.append([i for i, _ in pairs] + [pairs[0][0]] * (3 - len(pairs)))
                weights.append([x for _, x in pairs] + [0.0] * (3 - len(pairs)))
            ids.append(lookup[key])
        output.extend(
            (ids[0], ids[i], ids[i + 1])
            for i in range(1, len(ids) - 1)
            if len({ids[0], ids[i], ids[i + 1]}) == 3
        )
    w = np.asarray(weights, np.float32)
    w /= w.sum(1, keepdims=True)
    return np.asarray(indices, np.int32), w, np.asarray(output, np.int32)


def closest_surface(points, body, faces, tree=None):
    """Return near points, unit face normals, signed normal distances and body-face IDs.

    All positions/distances are meters. The 24-centroid candidate search is
    approximate; signed normal samples do not prove inside/outside for a mesh."""
    triangles = body[faces]
    tree = tree or cKDTree(triangles.mean(1))
    _, ids = tree.query(points, k=min(24, len(faces)))
    candidates = triangles[ids]
    q = trimesh.triangles.closest_point(
        candidates.reshape(-1, 3, 3), np.repeat(points, ids.shape[1], axis=0)
    ).reshape(candidates.shape[:2] + (3,))
    best = np.linalg.norm(q - points[:, None], axis=2).argmin(1)
    chosen = ids[np.arange(len(points)), best]
    near = q[np.arange(len(points)), best]
    normal = np.cross(
        triangles[chosen, 1] - triangles[chosen, 0], triangles[chosen, 2] - triangles[chosen, 0]
    )
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-12)
    signed = ((points - near) * normal).sum(1)
    return near, normal, signed, chosen


def bind_new_points(points, rest, faces):
    """Project points to rest triangles; return body IDs and clipped normalized weights."""
    near, _, _, ids = closest_surface(points, rest, faces)
    weights = trimesh.triangles.points_to_barycentric(rest[faces[ids]], near).astype(np.float32)
    weights = np.clip(weights, 0, 1)
    weights /= weights.sum(1, keepdims=True)
    return faces[ids].astype(np.int32), weights


def pose_rotations(model, params, joints):
    """Return global joint rotations [55,3,3] from ordered axis-angle radians.

    joints is pelvis-centered shaped rest geometry in meters. Only rotations
    are returned; the existing callers combine them with posed surface positions."""
    full = torch.cat(
        [
            torch.tensor(params[k], dtype=torch.float32).reshape(1, -1)
            for k in [
                "global_orient",
                "body_pose",
                "jaw_pose",
                "leye_pose",
                "reye_pose",
                "left_hand_pose",
                "right_hand_pose",
            ]
        ],
        dim=1,
    )
    rot = batch_rodrigues(full.reshape(-1, 3)).reshape(1, -1, 3, 3)
    _, transforms = batch_rigid_transform(
        rot, torch.tensor(joints, dtype=torch.float32)[None, :55], model.parents
    )
    return transforms[0, :, :3, :3].numpy()


CONV = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)


def frame(normal, axis):
    """Build column frames [N,3,3] = tangent, normal-cross-tangent, normal."""
    normal = normal / np.linalg.norm(normal, axis=1, keepdims=True)
    tangent = axis - normal * (axis * normal).sum(1, keepdims=True)
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-10)
    return np.stack((tangent, np.cross(normal, tangent), normal), axis=2)


def triangle_frame(vertices, faces):
    """Build column frames for indexed triangles, retaining degenerate epsilon behavior."""
    triangles = vertices[faces]
    tangent = triangles[:, 1] - triangles[:, 0]
    tangent /= np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1e-10)
    normal = np.cross(tangent, triangles[:, 2] - triangles[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-10)
    return np.stack((tangent, np.cross(normal, tangent), normal), axis=2)
