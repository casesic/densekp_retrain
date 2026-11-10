"""Attach one existing BEDLAM groom and native eye materials to accepted caches."""
import json
from pathlib import Path
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/experiments/bedlam_native_walk_eyes_hair_a2_20261009'


def write(p, x):
    p.write_text(json.dumps(x, indent=2) + '\n')


def rigid(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    u, _, vt = np.linalg.svd((x-x.mean(0)).T @ (y-y.mean(0)))
    q = u @ vt
    if np.linalg.det(q) < 0:
        u[:, -1] *= -1
        q = u @ vt
    t = y.mean(0) - x.mean(0) @ q
    h = np.eye(4)
    h[:3, :3] = q.T
    h[:3, 3] = t
    return h


def main():
    start = time.monotonic()
    recipe = json.loads((OUT/'frozen_recipe.json').read_text())
    base = Path(recipe['accepted_asset'])
    body = np.load(base/'geometry/vertices.npy')
    faces = np.load(base/'geometry/faces.npy')
    ref = np.load(recipe['groom_reference'])
    np.testing.assert_array_equal(ref['faces'], faces)
    C = np.array([[1,0,0],[0,0,-1],[0,1,0]], float)
    rv = ref['vertices'] @ C
    names = list(ref['group_names'])
    weight = ref['weights'][:, names.index('head')]
    anchors = np.flatnonzero(weight > .999)
    scalp = (weight > .5) & (ref['vertices'][:, 2] > .32)
    # Single shape/alignment adaptation to accepted frame0, not framewise scaling.
    affine = np.linalg.lstsq(np.c_[rv[scalp], np.ones(scalp.sum())], body[0,scalp], rcond=None)[0]
    singular = np.linalg.svd(affine[:3], compute_uv=False)
    assert np.linalg.det(affine[:3]) > 0 and singular.min() > .9 and singular.max() < 1.1
    transforms = np.array([rigid(body[0,anchors], x[anchors]) for x in body])
    residuals = []
    for i, h in enumerate(transforms):
        predicted = body[0] @ h[:3,:3].T + h[:3,3]
        e = np.linalg.norm(predicted-body[i], axis=1)
        residuals.append(dict(asset_frame=i, anchor_max_m=float(e[anchors].max()), scalp_p95_m=float(np.quantile(e[scalp],.95)), scalp_max_m=float(e[scalp].max())))
        assert e[anchors].max() < .001 and e[scalp].max() < .005
        np.testing.assert_allclose(h[:3,:3].T @ h[:3,:3], np.eye(3), atol=1e-10)
    np.save(OUT/'head_transforms.npy', transforms)
    np.save(OUT/'head_anchor_indices.npy', anchors)
    raw = np.load(recipe['groom_export'])
    points = raw['points'] * .01 @ C
    points = (points @ affine[:3] + affine[3]).astype(np.float32)
    radius = (raw['imported_radius']*.01).astype(np.float32)
    offsets = raw['offsets']
    assert np.isfinite(points).all() and np.isfinite(radius).all() and radius.min() >= 0
    assert offsets[0] == 0 and offsets[-1] == len(points) and (np.diff(offsets)>0).all()
    folder = OUT/'groom';folder.mkdir(exist_ok=False)
    np.save(folder/'points.npy', points)
    np.save(folder/'radius.npy', radius)
    np.save(folder/'curve_offsets.npy', offsets)
    cases = json.loads((ROOT/'outputs/experiments/cycling_commuter_pilot_20260930/groom_import_audit_v2/report.json').read_text())['cases']
    source = next(x for x in cases if x['source'] == recipe['groom_source'])
    write(folder/'groom_manifest.json', dict(source_abc=source['source'], source_objects=source['objects'],
          units='metres, accepted asset model coordinates at frame0', source_unit_scale=.01,
          binding='single affine reference-to-frame0, then rigid frame0-to-frame head attachment',
          affine_reference_y_up_to_frame0=affine.tolist(), affine_singular_values=singular.tolist(),
          source_reference=recipe['groom_reference'], curves=len(offsets)-1, points=len(points),
          active_set='all original curves, fixed; no per-frame filtering', physics=False,
          frame_transform='scene @ head_transforms[asset_frame], source root included in head once',
          material=dict(consumer='src/densekp_retrain/rendering/blender/groom.py:add_groom',
                        shader='Principled Hair BSDF', parametrization='COLOR',
                        color_linear=[.022,.009,.004,1],roughness=.4,
                        origin='existing project preview shader, not official actor color'),
          license='https://bedlam2.is.tuebingen.mpg.de/license.html', pairing='local hairstyle selection, not original actor hair identity', integration_only=True))
    labels=np.zeros(len(rv),np.int32);coords=np.zeros_like(rv,dtype=np.float32);eyes=[]
    for label, name in enumerate(['left_eye_smplhf','right_eye_smplhf'],1):
        ix=np.flatnonzero(ref['weights'][:,names.index(name)]>.999)
        assert len(ix)==546
        labels[ix]=label;center=rv[ix].mean(0);coords[ix]=rv[ix]-center
        eyes.append(dict(name=name,vertices=len(ix),center_reference_yup=center.tolist(),rest_gaze_axis=[0,0,1]))
    eye_faces=np.all(labels[faces]>0,axis=1)
    assert not np.any(np.any(labels[faces]>0,axis=1) != eye_faces)
    np.savez(OUT/'eye_mapping.npz', coordinates=coords, component_labels=labels, faces=faces)
    write(OUT/'appearance.json', dict(eye_mapping=str(OUT/'eye_mapping.npz'),eyes=eyes,
          iris_color=[.09,.035,.012,1],sclera_color=[.62,.64,.65,1],iris_radius_m=.0051,pupil_radius_m=.0022,
          method='native eye material only; fixed rest-coordinate attribute; existing eye geometry/pose unchanged',
          groom=str(folder),head_transforms=str(OUT/'head_transforms.npy')))
    write(OUT/'binding_validation.json', dict(frames=91,finite=True,head_residuals=residuals,
          max_anchor_residual_m=max(x['anchor_max_m'] for x in residuals),max_scalp_residual_m=max(x['scalp_max_m'] for x in residuals),
          affine_once=True,head_transforms_rigid=True,body_changed=False,cloth_changed=False,
          eye_vertices=1092,eye_faces=int(eye_faces.sum()),groom_curves=len(offsets)-1,groom_points=len(points),
          no_pruning=True,no_per_frame_grounding=True,R138_status='unchanged pending template semantics',
          seconds=time.monotonic()-start,human_dynamic_review='pending_user'))
    print('APPEARANCE_PREPARED',time.monotonic()-start,'seconds',max(x['scalp_max_m'] for x in residuals),'scalp residual')


if __name__ == '__main__':
    main()
