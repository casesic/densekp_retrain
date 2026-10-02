"""Two historical close-fitting garment algorithms, kept separate.

All vertices/joints are SMPL-X model-axis meters. Faces index their respective
vertex arrays. Whole-face shells and barycentrically clipped shells have
separate selection, rounding and clearance rules; neither is cloth simulation.
"""

import numpy as np


def vertex_normals(vertices, faces):
    """Area-weighted vertex normals [V,3]; retain float64 accumulation and epsilon."""
    triangles = vertices[faces]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    summed = np.zeros_like(vertices, dtype=np.float64)
    for corner in range(3):
        np.add.at(summed, faces[:, corner], normals)
    lengths = np.linalg.norm(summed, axis=1, keepdims=True)
    return summed / np.maximum(lengths, 1e-12)


def outfit_regions(rest, joints, faces):
    """Use SMPL-X anatomical joints, never DenseKP semantic row indices."""
    from smplx.joint_names import JOINT_NAMES

    joint = {name: joints[index] for index, name in enumerate(JOINT_NAMES[: len(joints)])}
    centers = rest[faces].mean(axis=1)
    waist = joint["pelvis"][1] + 0.10
    neck = joint["neck"][1] + 0.015
    sleeve_x = {}
    for side in ("left", "right"):
        shoulder = joint[f"{side}_shoulder"][0]
        elbow = joint[f"{side}_elbow"][0]
        sleeve_x[side] = shoulder + 0.65 * (elbow - shoulder)
    upper = (
        (centers[:, 1] >= waist)
        & (centers[:, 1] < neck)
        & (centers[:, 0] >= sleeve_x["right"])
        & (centers[:, 0] <= sleeve_x["left"])
    )
    hip_y = (joint["left_hip"][1] + joint["right_hip"][1]) / 2
    knee_y = (joint["left_knee"][1] + joint["right_knee"][1]) / 2
    hem_y = knee_y + 0.35 * (hip_y - knee_y)
    lower = (centers[:, 1] >= hem_y) & (centers[:, 1] < waist)
    selected = upper | lower
    if upper.sum() < 100 or lower.sum() < 100:
        raise ValueError("Rest-pose garment regions are unexpectedly small")
    material = np.where(upper, 0, 1)
    stripe = upper & (centers[:, 1] > waist + 0.13) & (centers[:, 1] < waist + 0.19)
    material[stripe] = 2
    return selected, material[selected].astype(np.int32)


def shell(vertices, faces, selected, offset):
    """Return float32 cloth vertices, int32 local faces, body vertex IDs and clearance.

    selected indexes whole body faces; offset and clearance are in meters.
    Preserve the historical correction iteration order and 1cm displacement gate."""
    if not np.isfinite(offset) or not 0 < offset <= 0.01:
        raise ValueError("Expected a shell offset in (0, 1cm]")
    source_faces = faces[selected]
    ids, inverse = np.unique(source_faces, return_inverse=True)
    normals = vertex_normals(vertices, faces)
    if np.any(np.linalg.norm(normals[ids], axis=1) < 0.99):
        raise ValueError("Undefined body normal under garment")
    clothing_faces = inverse.reshape(-1, 3).astype(np.int32)
    original = vertices[source_faces]
    face_normals = np.cross(original[:, 1] - original[:, 0], original[:, 2] - original[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    displacement = offset * normals[ids]
    # At tight hip/armpit folds, averaged vertex normals can point behind an
    # adjacent face. Project those local offsets onto outward-clearance constraints.
    clearance = offset * 0.1
    for _ in range(100):
        separation = (displacement[clothing_faces].mean(1) * face_normals).sum(1)
        bad = np.flatnonzero(separation < clearance - 1e-8)
        if not len(bad):
            break
        for face_id in bad:
            face = clothing_faces[face_id]
            normal = face_normals[face_id]
            current = displacement[face].mean(0) @ normal
            displacement[face] += max(clearance - current, 0) * normal
    else:
        raise ValueError("Unable to find a locally outward garment shell")
    if np.linalg.norm(displacement, axis=1).max() > 0.01:
        raise ValueError("Local shell correction exceeded 1cm")
    clothing = vertices[ids] + displacement
    return clothing.astype(np.float32), clothing_faces, ids, separation


def clipped_shell(rest, joints, body, faces, profile=None):
    """Clip rest-pose triangles and carry barycentric correspondence to the posed body.

    Returns named geometry arrays and scalar boundary/clearance metadata.
    Barycentric IDs index body vertices, not DenseKP rows; weights remain float32.
    profile controls coverage only. Defaults and 9-digit deduplication are retained."""
    waist = float(joints[0, 1] + 0.10)
    hem = float(
        (joints[4, 1] + joints[5, 1]) * 0.5
        + 0.35 * ((joints[1, 1] + joints[2, 1] - joints[4, 1] - joints[5, 1]) * 0.5)
    )
    collar = float(joints[12, 1] + 0.045)
    left = float(joints[16, 0] + 0.65 * (joints[18, 0] - joints[16, 0]))
    right = float(joints[17, 0] + 0.65 * (joints[19, 0] - joints[17, 0]))
    if profile:
        # Anatomical interpolation changes garment coverage, not body geometry.
        sleeve = float(profile["sleeve"])
        left = float(joints[16, 0] + sleeve * (joints[20, 0] - joints[16, 0]))
        right = float(joints[17, 0] + sleeve * (joints[21, 0] - joints[17, 0]))
        hip = float(joints[1:3, 1].mean())
        ankle = float(joints[7:9, 1].mean())
        hem = hip + float(profile["leg"]) * (ankle - hip)
        collar = float(joints[12, 1] + profile.get("collar", 0.045))
    planes = [(0, 1, -right), (0, -1, left), (1, 1, -hem), (1, -1, collar)]
    lookup = {}
    indices = []
    weights = []
    cloth_faces = []
    for triangle in faces:
        rv = rest[triangle]
        if any(np.max(sign * rv[:, axis] + offset) < 0 for axis, sign, offset in planes):
            continue
        polygon = list(np.eye(3))
        for axis, sign, offset in planes:
            if not polygon:
                break
            clipped = []
            for a, b in zip(polygon, polygon[1:] + polygon[:1]):
                da = float(sign * (a @ rv)[axis] + offset)
                db = float(sign * (b @ rv)[axis] + offset)
                if da >= 0:
                    clipped.append(a)
                if (da >= 0) != (db >= 0):
                    clipped.append(a + (b - a) * (da / (da - db)))
            polygon = clipped
        ids = []
        for bary in polygon:
            pairs = sorted(
                (int(vertex), float(weight))
                for vertex, weight in zip(triangle, bary)
                if weight > 1e-9
            )
            key = tuple((v, round(w, 9)) for v, w in pairs)
            if key not in lookup:
                lookup[key] = len(indices)
                indices.append([p[0] for p in pairs] + [pairs[0][0]] * (3 - len(pairs)))
                weights.append([p[1] for p in pairs] + [0.0] * (3 - len(pairs)))
            ids.append(lookup[key])
        for i in range(1, len(ids) - 1):
            if len({ids[0], ids[i], ids[i + 1]}) == 3:
                cloth_faces.append([ids[0], ids[i], ids[i + 1]])
    indices = np.asarray(indices, np.int32)
    weights = np.asarray(weights, np.float32)
    faces_out = np.asarray(cloth_faces, np.int32)
    weights /= weights.sum(1, keepdims=True)
    reference = (body[indices] * weights[:, :, None]).sum(1)
    rest_clothing = (rest[indices] * weights[:, :, None]).sum(1)
    original = reference[faces_out]
    normals = np.cross(original[:, 1] - original[:, 0], original[:, 2] - original[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    faces_out = faces_out[lengths > 1e-12]
    normals = normals[lengths > 1e-12] / lengths[lengths > 1e-12, None]
    body_normals = vertex_normals(body, faces)
    displacement = (body_normals[indices] * weights[:, :, None]).sum(1)
    displacement *= 0.004 / np.maximum(np.linalg.norm(displacement, axis=1, keepdims=True), 1e-12)
    for _ in range(100):
        separation = (displacement[faces_out].mean(1) * normals).sum(1)
        bad = np.flatnonzero(separation < 0.0004 - 1e-8)
        if not len(bad):
            break
        for i in bad:
            f = faces_out[i]
            n = normals[i]
            displacement[f] += max(0.0004 - displacement[f].mean(0) @ n, 0) * n
    else:
        raise ValueError("Clipped cloth cannot meet existing local clearance")
    clothing = (reference + displacement).astype(np.float32)
    separation = ((clothing[faces_out].mean(1) - reference[faces_out].mean(1)) * normals).sum(1)
    offset = float(np.linalg.norm(clothing - reference, axis=1).max())
    if separation.min() < 0.0004 - 2e-7 or offset > 0.01:
        raise ValueError("Existing shell gates failed")
    return dict(
        clothing_vertices=clothing[None],
        clothing_faces=faces_out,
        body_barycentric_indices=indices,
        body_barycentric_weights=weights,
        reference_body_vertices=reference[None],
        clothing_rest_vertices=rest_clothing,
    ), dict(
        waist_y=waist,
        stripe_min_y=waist + 0.13,
        stripe_max_y=waist + 0.19,
        hem_y=hem,
        collar_y=collar,
        sleeve_left_x=left,
        sleeve_right_x=right,
        max_vertex_offset_m=offset,
        minimum_face_normal_separation_m=float(separation.min()),
    )
