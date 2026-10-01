#!/usr/bin/env python3
"""Fit independent SMPL-X bodies to saved HMR2 meshes via surface correspondences.

This is inverse fitting through the local SMPL-X -> SMPL barycentric operator,
not the official deformation-transfer implementation or a 3D accuracy benchmark.
No image-camera translation enters scene geometry. Hands/face stay neutral.
"""
import argparse
import json
import pickle
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from smplx import SMPL

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from densekp_retrain.body.geometry import load_smplx_model, compute_densekp_3d


def correspondence(path, faces):
    with path.open('rb') as stream:
        data = pickle.load(stream, encoding='latin1')
    matrix = np.asarray(data['matrix'])
    indices = np.asarray(data['closest_faces'], dtype=np.int64)
    weights = np.asarray(data['bc'])
    valid = np.asarray(data['valid_vertices'], dtype=np.int64)
    if matrix.shape != (6890, 10475) or indices.shape != (6890, 3):
        raise ValueError('Unexpected correspondence topology')
    if not np.isfinite(matrix).all() or weights.min() < -1e-10:
        raise ValueError('Invalid barycentric weights')
    np.testing.assert_allclose(weights.sum(1), 1, atol=1e-10)
    rebuilt = np.zeros_like(matrix)
    np.add.at(rebuilt, (np.arange(6890)[:, None], indices), weights)
    np.testing.assert_allclose(rebuilt, matrix, atol=1e-10, rtol=0)
    triangles = {tuple(sorted(face)) for face in faces.tolist()}
    if any(tuple(sorted(face)) not in triangles for face in indices.tolist()):
        raise ValueError('Correspondence triangle is absent from this SMPL-X model')
    if len(np.unique(valid)) != len(valid) or valid.min() < 0 or valid.max() >= 6890:
        raise ValueError('Invalid fitting subset')
    return indices, weights.astype(np.float32), valid


def stats(errors):
    return dict(mean_mm=float(errors.mean()*1000), p95_mm=float(np.percentile(errors, 95)*1000),
                max_mm=float(errors.max()*1000), count=len(errors))


def mesh_review(path, source, fitted, source_faces, fitted_faces):
    # Orthographic views use one joint framing per view; no independent alignment.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    fig = plt.figure(figsize=(12, 6))
    center = np.concatenate((source, fitted)).mean(0)
    radius = np.ptp(np.concatenate((source, fitted)), axis=0).max() * .55
    for row, (vertices, faces, title) in enumerate(((source, source_faces, 'source SMPL'),
                                                   (fitted, fitted_faces, 'fitted SMPL-X'))):
        for col, azimuth in enumerate((-90, 0, 90, 180)):
            ax = fig.add_subplot(2, 4, row*4+col+1, projection='3d')
            # Model camera coordinates -> upright plotting coordinates.
            v = vertices[:, [0, 2, 1]] * [1, 1, -1]
            c = center[[0, 2, 1]] * [1, 1, -1]
            mesh = Poly3DCollection(v[faces], linewidths=0, facecolor=(.35, .65, .8))
            ax.add_collection3d(mesh)
            ax.set(xlim=(c[0]-radius,c[0]+radius), ylim=(c[1]-radius,c[1]+radius),
                   zlim=(c[2]-radius,c[2]+radius), title=f'{title} / {azimuth} deg')
            ax.set_box_aspect((1,1,1)); ax.view_init(elev=5, azim=azimuth); ax.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--smpl-model', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--transfer', type=Path, required=True)
    p.add_argument('--samples', nargs='+', default=['coco_292226_person_1330527', 'coco_306800_person_225160'])
    p.add_argument('--iterations', type=int, default=200)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    torch.set_num_threads(4); torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_smplx_model(args.model_dir, gender='neutral', num_betas=10, batch_size=1).to(args.device).eval()
    model.requires_grad_(False)
    smpl = SMPL(str(args.smpl_model), gender='neutral').eval()
    indices, weights, valid = correspondence(args.transfer, model.faces)
    index = torch.as_tensor(indices, device=args.device)
    weight = torch.as_tensor(weights, device=args.device)
    regressor = np.load(ROOT/'assets/densekp_regressor_R.npy')
    if regressor.shape != (138, 10475) or not np.isfinite(regressor).all():
        raise ValueError('Invalid DenseKP operator')
    np.testing.assert_allclose(regressor.sum(1), 1, atol=1e-6)
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(method='inverse surface fitting through SMPL-X to SMPL barycentric operator',
                  transfer=str(args.transfer.resolve()), model_dir=str(args.model_dir.resolve()),
                  source=str(args.input.resolve()), sequence=False, device=args.device,
                  torch=torch.__version__, fitting_vertices=len(valid), cases=[],
                  acceptance='pilot gate: fitting mean <= 15 mm and p95 <= 30 mm; export <= 2e-5 m; not real 3D accuracy',
                  unobserved_detail='neutral flat hands, neutral jaw/eyes/expression; no recovered finger pose')
    for sample in args.samples:
        folder = args.output/sample; folder.mkdir()
        with np.load(args.input/sample/'smpl_prediction.npz') as z:
            raw = {k:z[k] for k in z.files}
        with torch.no_grad():
            restored = smpl(**{k:torch.from_numpy(raw[k]) for k in ('global_orient','body_pose','betas')}).vertices[0].numpy()
        np.testing.assert_allclose(restored, raw['vertices_model_m'], atol=2e-5, rtol=0)
        np.testing.assert_array_equal(smpl.faces, raw['faces'])
        # Shared body rotations initialize the optimizer only; shape starts at zero.
        params = {k:torch.tensor(v, device=args.device, requires_grad=True) for k,v in dict(
            global_orient=raw['global_orient'], body_pose=raw['body_pose'][:,:63],
            betas=np.zeros((1,10),np.float32), transl=np.zeros((1,3),np.float32)).items()}
        target = torch.as_tensor(restored, device=args.device)
        initial_pose = params['body_pose'].detach().clone()
        def predict():
            verts = model(**params).vertices[0]
            return verts, (verts[index]*weight[...,None]).sum(1)
        with torch.no_grad():
            initial = stats(torch.linalg.vector_norm(predict()[1]-target, dim=1).cpu().numpy()[valid])
        optimizer = torch.optim.LBFGS(list(params.values()), lr=1., max_iter=args.iterations,
                                     tolerance_grad=1e-8, tolerance_change=1e-11, line_search_fn='strong_wolfe')
        calls = []
        def closure():
            optimizer.zero_grad()
            _, mapped = predict()
            loss = (mapped[valid]-target[valid]).square().mean()
            loss = loss + 1e-7*params['betas'].square().mean() + 1e-7*(params['body_pose']-initial_pose).square().mean()
            if not torch.isfinite(loss): raise ValueError('Nonfinite fit')
            loss.backward(); calls.append(float(loss.detach()))
            return loss
        optimizer.step(closure)
        with torch.no_grad():
            vertices, mapped = (v.cpu().numpy() for v in predict())
        errors = np.linalg.norm(mapped-restored, axis=1)
        fitted = {k:v.detach().cpu().numpy() for k,v in params.items()}
        np.savez(folder/'fit_parameters.npz', **fitted, vertices=vertices,
                 mapped_smpl_vertices=mapped, residual_m=errors, fitting_vertex_ids=valid)
        mesh_review(folder/'conversion_review.jpg', restored, vertices, smpl.faces, model.faces)
        final = stats(errors[valid])
        accepted = final['mean_mm'] <= 15 and final['p95_mm'] <= 30
        # HMR2 image camera: x right, y down, z forward. Canonical: y up.
        rotation = np.diag([1., -1., -1.])
        root_matrix = cv2.Rodrigues(fitted['global_orient'][0].astype(float))[0]
        canonical = dict(fitted)
        canonical['global_orient'] = cv2.Rodrigues(rotation@root_matrix)[0].reshape(1,3).astype(np.float32)
        # SMPL-X root rotates about shaped pelvis, not the coordinate origin.
        with torch.no_grad():
            neutral = model(betas=torch.as_tensor(fitted['betas'],device=args.device))
        pelvis = neutral.joints[0,0].cpu().numpy()
        canonical['transl'] = ((fitted['transl']+pelvis)@rotation.T-pelvis).astype(np.float32)
        canonical.update({k:np.zeros((1,n),np.float32) for k,n in (
            ('jaw_pose',3),('leye_pose',3),('reye_pose',3),('left_hand_pose',45),('right_hand_pose',45))})
        with torch.no_grad():
            scene_vertices = model(**{k:torch.as_tensor(v, device=args.device) for k,v in canonical.items()}).vertices.cpu().numpy()
        export_error = float(np.abs(scene_vertices[0]-vertices@rotation.T).max())
        np.testing.assert_allclose(scene_vertices[0], vertices@rotation.T, atol=2e-5, rtol=0)
        canonical.update(betas=fitted['betas'][0], gender=np.array('neutral'),
                         source_frame_ids=np.array([0]), source_total_frames=np.int64(1),
                         source_translation_used=np.array(False), source_transl=np.zeros((1,3),np.float32),
                         fps=np.float32(0), sequence=np.array(False), sample_id=np.array(sample))
        np.savez(folder/'canonical_pose.npz', **canonical)
        if accepted:
            geom = folder/'geometry'; geom.mkdir()
            np.save(geom/'vertices.npy', scene_vertices)
            np.save(geom/'faces.npy', model.faces.astype(np.int32))
            np.save(geom/'densekp_3d.npy', compute_densekp_3d(regressor, scene_vertices))
            np.savez(geom/'motion_params.npz', **canonical)
            (geom/'geometry_manifest.json').write_text(json.dumps(dict(
                artifact='independent_image_pose_smplx', sample_id=sample, sequence=False,
                source_prediction=str((args.input/sample/'smpl_prediction.npz').resolve()),
                parameters=str((folder/'canonical_pose.npz').resolve()), frames=1,
                model_dir=str(args.model_dir.resolve()), regressor=str((ROOT/'assets/densekp_regressor_R.npy').resolve()),
                camera_translation_used=False, axis_rotation=rotation.tolist(),
                root_pivot_correction_m=(canonical['transl']-fitted['transl']@rotation.T).tolist(),
                placement='renderer lifts min vertex Y to ground, then scene yaw and explicit location',
                fitting_residual=final, training_data=False), indent=2)+'\n')
        case = dict(sample_id=sample, accepted_for_synthetic_pilot=accepted, initial=initial,
                    fitting_subset=final, all_vertices=stats(errors), evaluations=len(calls),
                    export_max_abs_error_m=export_error, betas=fitted['betas'].tolist())
        report['cases'].append(case)
        (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(json.dumps(case), flush=True)
    if not all(c['accepted_for_synthetic_pilot'] for c in report['cases']):
        raise RuntimeError('At least one conversion failed the pilot gate; inspect residuals')


if __name__ == '__main__':
    main()
