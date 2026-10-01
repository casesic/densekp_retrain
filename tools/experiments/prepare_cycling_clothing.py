#!/usr/bin/env python3
"""Build fitted jersey/shorts shells for each independent cycling body.

Garment regions are selected in the shaped rest pose; the shell follows the
saved posed body with a small normal offset. This is a procedural close-fitting
outfit, not cloth simulation or recovery of the source photograph's clothes.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))


def vertex_normals(vertices, faces):
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

    joint = {name: joints[index] for index, name in enumerate(JOINT_NAMES[:len(joints)])}
    centers = rest[faces].mean(axis=1)
    waist = joint['pelvis'][1] + .10
    neck = joint['neck'][1] + .015
    sleeve_x = {}
    for side in ('left', 'right'):
        shoulder = joint[f'{side}_shoulder'][0]
        elbow = joint[f'{side}_elbow'][0]
        sleeve_x[side] = shoulder + .65 * (elbow - shoulder)
    upper = ((centers[:, 1] >= waist) & (centers[:, 1] < neck)
             & (centers[:, 0] >= sleeve_x['right']) & (centers[:, 0] <= sleeve_x['left']))
    hip_y = (joint['left_hip'][1] + joint['right_hip'][1]) / 2
    knee_y = (joint['left_knee'][1] + joint['right_knee'][1]) / 2
    hem_y = knee_y + .35 * (hip_y - knee_y)
    lower = (centers[:, 1] >= hem_y) & (centers[:, 1] < waist)
    selected = upper | lower
    if upper.sum() < 100 or lower.sum() < 100:
        raise ValueError('Rest-pose garment regions are unexpectedly small')
    material = np.where(upper, 0, 1)
    stripe = upper & (centers[:, 1] > waist + .13) & (centers[:, 1] < waist + .19)
    material[stripe] = 2
    return selected, material[selected].astype(np.int32)


def shell(vertices, faces, selected, offset):
    if not np.isfinite(offset) or not 0 < offset <= .01:
        raise ValueError('Expected a shell offset in (0, 1cm]')
    source_faces = faces[selected]
    ids, inverse = np.unique(source_faces, return_inverse=True)
    normals = vertex_normals(vertices, faces)
    if np.any(np.linalg.norm(normals[ids], axis=1) < .99):
        raise ValueError('Undefined body normal under garment')
    clothing_faces = inverse.reshape(-1, 3).astype(np.int32)
    original = vertices[source_faces]
    face_normals = np.cross(original[:, 1] - original[:, 0], original[:, 2] - original[:, 0])
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    displacement = offset * normals[ids]
    # At tight hip/armpit folds, averaged vertex normals can point behind an
    # adjacent face. Project those local offsets onto outward-clearance constraints.
    clearance = offset * .1
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
        raise ValueError('Unable to find a locally outward garment shell')
    if np.linalg.norm(displacement, axis=1).max() > .01:
        raise ValueError('Local shell correction exceeded 1cm')
    clothing = vertices[ids] + displacement
    return clothing.astype(np.float32), clothing_faces, ids, separation


def review(path, body, body_faces, clothing, clothing_faces, material_ids, materials):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    # y-up canonical geometry -> z-up display, with equal scale in all views.
    body = body[:, [0, 2, 1]] * [1, -1, 1]
    clothing = clothing[:, [0, 2, 1]] * [1, -1, 1]
    center = (body.max(0) + body.min(0)) / 2
    radius = np.ptp(body, axis=0).max() * .53
    fig = plt.figure(figsize=(12, 4))
    for column, azimuth in enumerate((-90, 0, 90, 180)):
        ax = fig.add_subplot(1, 4, column + 1, projection='3d')
        all_triangles = np.concatenate((body[body_faces], clothing[clothing_faces]))
        colors = np.concatenate((np.tile([.65, .44, .30], (len(body_faces), 1)),
                                 np.array([m['color'][:3] for m in materials])[material_ids]))
        normals = np.cross(all_triangles[:, 1] - all_triangles[:, 0],
                           all_triangles[:, 2] - all_triangles[:, 0])
        normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
        light = np.array([1., -1., 2.]) / np.sqrt(6)
        colors *= (.4 + .6 * np.maximum(normals @ light, 0))[:, None]
        ax.add_collection3d(Poly3DCollection(all_triangles, facecolors=colors, linewidths=0))
        ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[1]-radius, center[1]+radius),
               zlim=(center[2]-radius, center[2]+radius), title=f'{azimuth} degrees')
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=5, azim=azimuth)
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--converted', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--offset-m', type=float, default=.004)
    args = parser.parse_args()
    import torch
    from densekp_retrain.body.geometry import load_smplx_model

    torch.set_num_threads(2)
    conversion = json.loads((args.converted / 'report.json').read_text())
    model = load_smplx_model(Path(conversion['model_dir']), gender='neutral', num_betas=10, batch_size=1)
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    for index, case in enumerate(conversion['cases']):
        if not case['accepted_for_synthetic_pilot']:
            raise ValueError('Clothing requires an accepted body conversion')
        sample = case['sample_id']
        geometry = (args.converted / sample / 'geometry').resolve()
        body = np.load(geometry / 'vertices.npy')
        faces = np.load(geometry / 'faces.npy')
        np.testing.assert_array_equal(faces, model.faces)
        if body.shape != (1, 10475, 3):
            raise ValueError('Expected a single independent SMPL-X pose')
        with np.load(geometry / 'motion_params.npz') as z:
            betas = torch.from_numpy(z['betas'][None])
        with torch.no_grad():
            rest = model(betas=betas)
        selected, material_ids = outfit_regions(rest.vertices[0].numpy(), rest.joints[0].numpy(), faces)
        clothing, clothing_faces, body_ids, separation = shell(body[0], faces, selected, args.offset_m)
        jersey = [.65, .035, .025, 1.] if index % 2 == 0 else [.025, .18, .65, 1.]
        materials = [dict(name='cycling_jersey', color=jersey),
                     dict(name='cycling_shorts', color=[.022, .03, .045, 1.]),
                     dict(name='jersey_band', color=[.85, .88, .9, 1.])]
        out = args.output / sample
        out.mkdir()
        np.save(out / 'clothing_vertices.npy', clothing[None])
        np.save(out / 'clothing_faces.npy', clothing_faces)
        np.save(out / 'face_material_ids.npy', material_ids)
        np.save(out / 'body_vertex_ids.npy', body_ids)
        np.save(out / 'reference_body_vertices.npy', body[:, body_ids])
        np.save(out / 'body_face_ids.npy', np.flatnonzero(selected))
        (out / 'materials.json').write_text(json.dumps(materials, indent=2) + '\n')
        review(out / 'clothing_review.jpg', body[0], faces, clothing, clothing_faces, material_ids, materials)
        report = dict(sample_id=sample, body_geometry=str(geometry), frames=1, sequence=False,
                      method='shaped rest-pose jersey/shorts regions; posed-body normal shell with local fold clearance',
                      offset_m=args.offset_m, vertices=len(clothing), faces=len(clothing_faces),
                      max_vertex_offset_m=float(np.linalg.norm(clothing-body[0,body_ids],axis=1).max()),
                      minimum_face_normal_separation_m=float(separation.min()),
                      body_vertices_unchanged=True, densekp_source='original body, not clothing surface',
                      placement='apply the BODY lift/yaw/translation; never lift clothing independently',
                      limitation='procedural tight outfit, no cloth dynamics or exhaustive collision test')
        np.testing.assert_array_equal(body, np.load(geometry / 'vertices.npy'))
        (out / 'clothing_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        reports.append(report)
        print(json.dumps(report), flush=True)
    (args.output / 'report.json').write_text(json.dumps(dict(cases=reports), indent=2) + '\n')


if __name__ == '__main__':
    main()
