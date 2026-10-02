#!/usr/bin/env python3
"""Rest-shaped separates with straight trouser legs, fitted tops and folded polo collar.

The body is immutable. Garment rest offsets follow interpolated SMPL-X rotations
plus the actual pose-corrected body surface. Collision diagnostics are separate
from the old 4mm sports-shell contract.
"""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from scipy.spatial import cKDTree
import trimesh

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.body.geometry import load_smplx_model
from densekp_retrain.io.json import write_json
from densekp_retrain.clothing.shells import vertex_normals
from smplx.lbs import batch_rodrigues, batch_rigid_transform


from densekp_retrain.clothing.binding import clip_surface  # historical export


from densekp_retrain.clothing.binding import closest_surface  # historical export


from densekp_retrain.clothing.binding import bind_new_points  # historical export


def neck_ring(rest, faces, center, height, clearance, count=48):
    """Intersect the actual shaped neck/shoulder surface with a horizontal plane."""
    segments = []
    for tri in rest[faces]:
        points = []
        for a, b in zip(tri, np.roll(tri, -1, axis=0)):
            if (a[1] > height) != (b[1] > height):
                points.append((a + (b - a) * (height - a[1]) / (b[1] - a[1]))[[0, 2]])
        if len(points) == 2:
            segments.append(points)
    segments = np.asarray(segments) - center[[0, 2]]
    ring = []
    for theta in np.linspace(0, 2 * np.pi, count, endpoint=False):
        direction = np.array([np.sin(theta), np.cos(theta)])
        radii = []
        for a, b in segments:
            matrix = np.column_stack((direction, a - b))
            if abs(np.linalg.det(matrix)) < 1e-10:
                continue
            radius, fraction = np.linalg.solve(matrix, a)
            if radius > 0 and 0 <= fraction <= 1:
                radii.append(radius)
        if not radii:
            raise ValueError("Neck cross section is not closed")
        radial = direction * (min(radii) + clearance) + center[[0, 2]]
        ring.append([radial[0], height, radial[1]])
    return np.asarray(ring, np.float32)


def rest_garment(rest, joints, faces, gender):
    parts = []
    cuff_left = (
        joints[20, 0] - 0.018
        if gender == "female"
        else joints[16, 0] + 0.58 * (joints[18, 0] - joints[16, 0])
    )
    cuff_right = (
        joints[21, 0] + 0.018
        if gender == "female"
        else joints[17, 0] + 0.58 * (joints[19, 0] - joints[17, 0])
    )
    # A lower global plane accidentally removes the raised deltoid caps.
    neck = joints[12, 1] + 0.075
    definitions = [
        ("top", [(1, 1, -0.045), (1, -1, neck), (0, 1, -cuff_right), (0, -1, cuff_left)]),
        ("trousers", [(1, 1, -float(joints[[7, 8], 1].mean() + 0.03)), (1, -1, 0.13)]),
    ]
    normals = vertex_normals(rest, faces)
    for name, planes in definitions:
        idx, w, cf = clip_surface(rest, faces, planes)
        reference = (rest[idx] * w[:, :, None]).sum(1)
        n = (normals[idx] * w[:, :, None]).sum(1)
        n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
        clearance = np.full(len(reference), 0.018 if name == "top" else 0.015)
        if name == "trousers":
            crotch = (
                (np.abs(reference[:, 0]) < 0.055)
                & (reference[:, 1] > -0.28)
                & (reference[:, 1] < -0.10)
            )
            clearance[crotch] = 0.004
        else:
            armpit = (
                (np.abs(reference[:, 0]) > 0.13)
                & (np.abs(reference[:, 0]) < 0.32)
                & (reference[:, 1] < joints[16:18, 1].mean() - 0.04)
            )
            clearance[armpit] = 0.004
        vertices = reference + n * clearance[:, None]
        if name == "top":
            # Lower torso follows a shaped, nearly straight hem rather than the waist indentation.
            torso = (np.abs(reference[:, 0]) < 0.23) & (reference[:, 1] < 0.32)
            radial = vertices[torso][:, [0, 2]] - np.array([0.0, -0.015])
            radii = np.array(
                [0.177 if gender == "female" else 0.209, 0.12 if gender == "female" else 0.145]
            )
            extent = np.sqrt(((radial / radii) ** 2).sum(1))
            radial *= np.maximum(1.0, 1 / np.maximum(extent, 1e-8))[:, None]
            vertices[np.flatnonzero(torso), 0] = radial[:, 0]
            vertices[np.flatnonzero(torso), 2] = radial[:, 1] - 0.015
            # Span the chest indentation with a continuous fabric front.
            front = (
                (np.abs(reference[:, 0]) < 0.16)
                & (reference[:, 1] > 0.18)
                & (reference[:, 1] < 0.43)
                & (reference[:, 2] > 0.015)
            )
            ids = np.flatnonzero(front)
            for i in ids:
                band = (np.abs(reference[:, 1] - reference[i, 1]) < 0.02) & (
                    np.abs(reference[:, 0]) < 0.17
                )
                radius_z = max(0.12, float(reference[band, 2].max() + 0.018))
                ellipse = radius_z * np.sqrt(max(0, 1 - (reference[i, 0] / 0.19) ** 2))
                vertices[i, 2] = max(vertices[i, 2], ellipse)
        else:
            # Below the knee, constant elliptical cross-section creates actual straight-leg ease.
            for sign, hip, knee, ankle in [(1, 1, 4, 7), (-1, 2, 5, 8)]:
                use = (reference[:, 0] * sign > 0) & (reference[:, 1] < joints[hip, 1] - 0.10)
                y = reference[use, 1]
                center = np.stack(
                    [
                        np.interp(
                            y, joints[[ankle, knee, hip], 1], joints[[ankle, knee, hip], axis]
                        )
                        for axis in [0, 2]
                    ],
                    axis=1,
                )
                radial = vertices[use][:, [0, 2]] - center
                radius_x = 0.080 if gender == "female" else 0.090
                radius_z = 0.079 if gender == "female" else 0.089
                scale = np.sqrt((radial[:, 0] / radius_x) ** 2 + (radial[:, 1] / radius_z) ** 2)
                # Preserve the crotch; ease begins at mid-thigh, away from the other leg.
                blend = np.clip((joints[hip, 1] - 0.18 - y) / 0.15, 0, 1)
                radial *= (1 + blend * np.maximum(0.0, 1 / np.maximum(scale, 1e-8) - 1))[:, None]
                vertices[np.flatnonzero(use), 0] = center[:, 0] + radial[:, 0]
                vertices[np.flatnonzero(use), 2] = center[:, 1] + radial[:, 1]
        parts.append(
            (vertices, cf, idx, w, np.full(len(cf), 0 if name == "top" else 1, np.int32), name)
        )
    if gender == "male":
        # Folded collar annulus: a separate surface, not a painted neckline.
        n = 48
        center = joints[12].copy()
        collar = np.concatenate(
            [
                neck_ring(rest, faces, center, center[1] + 0.078, 0.014, n),
                neck_ring(rest, faces, center, center[1] + 0.020, 0.024, n),
            ]
        )
        # Open front and two lowered tips distinguish the fold from a flat annulus.
        collar[n + 2, 1] -= 0.025
        collar[2 * n - 2, 1] -= 0.025
        cf = np.array(
            [(i, i + 1, n + i + 1) for i in range(2, n - 2)]
            + [(i, n + i + 1, n + i) for i in range(2, n - 2)],
            np.int32,
        )
        idx, w = bind_new_points(collar, rest, faces)
        parts.append((collar, cf, idx, w, np.full(len(cf), 2, np.int32), "folded_collar"))
        # Raised narrow placket anchored to the actual front torso surface.
        ys = np.linspace(neck - 0.015, neck - 0.14, 10)
        placket = []
        for y in ys:
            band = rest[(np.abs(rest[:, 0]) < 0.045) & (np.abs(rest[:, 1] - y) < 0.03)]
            z = np.max(band[:, 2]) + 0.025
            placket.extend([[-0.016, y, z], [0.016, y, z]])
        placket = np.asarray(placket, np.float32)
        cf = np.array(
            [(2 * i, 2 * i + 2, 2 * i + 3) for i in range(9)]
            + [(2 * i, 2 * i + 3, 2 * i + 1) for i in range(9)],
            np.int32,
        )
        idx, w = bind_new_points(placket, rest, faces)
        parts.append((placket, cf, idx, w, np.full(len(cf), 2, np.int32), "placket"))
    vertices = []
    out_faces = []
    indices = []
    weights = []
    materials = []
    components = []
    offset = 0
    for v, f, i, w, m, name in parts:
        components.append(dict(name=name, vertex_start=offset, vertex_count=len(v)))
        vertices.append(v)
        out_faces.append(f + offset)
        indices.append(i)
        weights.append(w)
        materials.append(m)
        offset += len(v)
    return (
        np.concatenate(vertices).astype(np.float32),
        np.concatenate(out_faces),
        np.concatenate(indices),
        np.concatenate(weights),
        np.concatenate(materials),
        components,
    )


from densekp_retrain.clothing.binding import pose_rotations  # historical export


def correct_collisions(vertices, cf, body, faces, initial=None):
    initial = vertices.copy() if initial is None else initial
    tree = cKDTree(body[faces].mean(1))
    saturated = False
    for iteration in range(64):
        _, normal, d, _ = closest_surface(vertices, body, faces, tree)
        vertices += np.maximum(0.003 - d, 0)[:, None] * normal
        centers = vertices[cf].mean(1)
        _, normal, d, _ = closest_surface(centers, body, faces, tree)
        bad = np.flatnonzero(d < 0.001)
        if not len(bad):
            _, _, vertex_distance, _ = closest_surface(vertices, body, faces, tree)
            if vertex_distance.min() >= 0.001:
                break
        delta = np.zeros_like(vertices)
        count = np.zeros(len(vertices))
        for corner in range(3):
            np.add.at(delta, cf[bad, corner], np.maximum(0.002 - d[bad], 0)[:, None] * normal[bad])
            np.add.at(count, cf[bad, corner], 1)
        vertices += delta / np.maximum(count[:, None], 1)
        displacement = vertices - initial
        length = np.linalg.norm(displacement, axis=1)
        excessive = length >= 0.06
        if excessive.any():
            saturated = True
            vertices[excessive] = initial[excessive] + displacement[excessive] * (
                0.06 / length[excessive, None]
            )
            break
    _, _, dv, _ = closest_surface(vertices, body, faces, tree)
    _, _, df, _ = closest_surface(vertices[cf].mean(1), body, faces, tree)
    return vertices, dict(
        method="closest point among 24 nearest triangle centroids; vertex and face-center signed normal samples",
        min_vertex_signed_m=float(dv.min()),
        min_face_center_signed_m=float(df.min()),
        negative_vertex_count=int((dv < -0.001).sum()),
        negative_face_center_count=int((df < -0.001).sum()),
        max_correction_m=float(np.linalg.norm(vertices - initial, axis=1).max()),
        iterations=iteration + 1,
        correction_budget_exhausted=saturated,
        limitation="local sampled collision diagnostic, not exhaustive triangle-triangle/self-intersection proof",
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("assets", type=Path)
    p.add_argument("--version", default="v2")
    a = p.parse_args()
    if not a.version.isalnum():
        raise ValueError("Version must be alphanumeric")
    start = time.monotonic()
    torch.set_num_threads(2)
    report = json.loads((a.assets / "report.json").read_text())
    results = []
    build_path = a.assets / f"clothing_build_{a.version}.json"
    if build_path.exists():
        raise FileExistsError(build_path)
    for c in report["characters"]:
        path = Path(c["manifest"])
        manifest = json.loads(path.read_text())
        model = load_smplx_model(
            Path(manifest["model_dir"]),
            gender=manifest["gender"],
            num_betas=manifest["num_betas"],
            batch_size=1,
        )
        rest = np.load(Path(manifest["reference"]["geometry"]) / "vertices.npy")[0]
        joints = np.load(path.parent / "rest_joints.npy")
        faces = model.faces
        rv, cf, idx, w, material_ids, components = rest_garment(
            rest, joints, faces, manifest["gender"]
        )
        reference = (rest[idx] * w[:, :, None]).sum(1)
        offsets = rv - reference
        skin_weights = (model.lbs_weights.numpy()[idx] * w[:, :, None]).sum(1)
        jobs = [manifest["reference"]] + [
            d for d in report["derivatives"] if d["character_id"] == c["character_id"]
        ]
        for job in jobs:
            geom = Path(job["geometry"])
            body = np.load(geom / "vertices.npy")[0]
            z = np.load(geom / "motion_params.npz")
            rotations = pose_rotations(model, z, joints)
            blended = np.einsum("nj,jab->nab", skin_weights, rotations)
            ref = (body[idx] * w[:, :, None]).sum(1)
            posed = ref + np.einsum("nab,nb->na", blended, offsets)
            # Pose correctives can rotate the local surface away from skeletal LBS.
            # Resolve against the attachment normal before the nearest-surface pass.
            body_normal = vertex_normals(body, faces)
            anchor_normal = (body_normal[idx] * w[:, :, None]).sum(1)
            anchor_normal /= np.maximum(np.linalg.norm(anchor_normal, axis=1, keepdims=True), 1e-12)
            projection = ((posed - ref) * anchor_normal).sum(1)
            seeded = posed + np.maximum(0.003 - projection, 0)[:, None] * anchor_normal
            final, collision = correct_collisions(seeded, cf, body, faces, initial=posed)
            collision["max_correction_m"] = float(np.linalg.norm(final - posed, axis=1).max())
            out = geom.parent / f"commuter_clothing_{a.version}"
            out.mkdir()
            arrays = dict(
                clothing_vertices=final[None].astype(np.float32),
                clothing_faces=cf,
                body_barycentric_indices=idx,
                body_barycentric_weights=w,
                reference_body_vertices=ref[None],
                clothing_rest_vertices=rv,
                rest_offsets=offsets,
                skin_weights=skin_weights,
                binding_rotations=blended,
                collision_correction=(final - posed).astype(np.float32),
                face_material_ids=material_ids,
            )
            for name, value in arrays.items():
                np.save(out / f"{name}.npy", value)
            top = [0.48, 0.39, 0.27, 1] if manifest["gender"] == "female" else [0.10, 0.22, 0.31, 1]
            write_json(
                out / "materials.json",
                [
                    dict(name="commuter_top", color=top),
                    dict(name="straight_trousers", color=[0.045, 0.065, 0.095, 1]),
                    dict(name="polo_collar_placket", color=[0.16, 0.31, 0.40, 1]),
                ],
            )
            passed = (
                collision["negative_vertex_count"] == 0
                and collision["negative_face_center_count"] == 0
                and collision["max_correction_m"] < 0.06
                and not collision["correction_budget_exhausted"]
            )
            info = dict(
                body_geometry=str(geom),
                character_id=c["character_id"],
                split="integration_only",
                contract="commuter_surface_binding_v1",
                components=components,
                body_changed=False,
                binding="barycentric posed-body reference plus interpolated LBS rotation of rest offset plus recorded collision correction",
                max_rest_ease_m=float(np.linalg.norm(offsets, axis=1).max()),
                collision=collision,
                thresholds=dict(max_collision_correction_m=0.06, min_sampled_signed_m=-0.001),
                status="numeric_candidate_visual_pending" if passed else "failed_collision",
            )
            write_json(out / "clothing_manifest.json", info)
            job["clothing_geometry"] = str(out)
            results.append(dict(geometry=str(geom), clothing=str(out), **info))
        manifest["clothing_status"] = (
            "failed_collision"
            if any(
                r["character_id"] == c["character_id"] and r["status"] == "failed_collision"
                for r in results
            )
            else "candidate_visual_pending"
        )
        write_json(path, manifest)
    report["status"] = "clothing_candidates_pending_review_hair_blocked"
    write_json(a.assets / "report.json", report)
    write_json(build_path, dict(compute_seconds=time.monotonic() - start, cases=results))
    print(
        json.dumps(
            dict(
                seconds=time.monotonic() - start,
                cases=[
                    dict(geometry=r["geometry"], status=r["status"], collision=r["collision"])
                    for r in results
                ],
            )
        )
    )


if __name__ == "__main__":
    main()
