#!/usr/bin/env python3
"""Bounded, frozen-body clothes and rigid-head groom pilot; no pose optimization."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import csv
import json
from pathlib import Path
import resource
import subprocess
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
BASE = ROOT / 'outputs/experiments/apd_6526_dressed_pilot_20261008'
BODY = ROOT / 'outputs/experiments/apd_6526_smplx_pilot_20261008'
CHAR = ROOT / 'outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_M01_v2/manifest.json'
GPU = 'GPU-d481acc7-106f-29e5-1298-8c0d398261b2'
PROBE = [0, 6, 16, 38, 75]  # source30,36,46,68,105; includes accepted rapid arm motion
KEYS = ['global_orient', 'body_pose', 'transl', 'jaw_pose', 'leye_pose', 'reye_pose', 'left_hand_pose', 'right_hand_pose', 'expression']


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def snapshot(paths):
    return {str(p): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for p in sorted(set(paths)) if p.is_file()}


def freeze():
    BASE.mkdir(parents=True, exist_ok=False)
    m = json.loads(CHAR.read_text())
    paths = [ROOT/'docs/TODO.md', ROOT/'docs/README.md', CHAR, ROOT/'assets/densekp_regressor_R.npy', Path(m['model_file'])]
    for folder in [ROOT/'src', ROOT/'tests', ROOT/'docs', BODY, CHAR.parent]:
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in str(p))
    for key in ['geometry', 'clothing_geometry', 'groom_geometry']:
        paths.extend(Path(m['reference'][key]).glob('*'))
    paths += [Path(m['groom_source']), Path(m['appearance']['body_texture']), Path(m['appearance']['body_uv'])]
    write(BASE/'protected_before.json', snapshot(paths))
    (BASE/'git_before.txt').write_text(subprocess.check_output(['git', 'status', '--porcelain=v1'], cwd=ROOT, text=True))
    (BASE/'time_mapping.csv').write_bytes((BODY/'time_mapping.csv').read_bytes())
    write(BASE/'frozen_recipe.json', dict(integration_only=True, character='cycling_M01_v2', character_manifest=str(CHAR),
        authoritative_body=str(BODY/'fit_attempt_2/parameters.npz'), body_cache=str(BODY/'fit_attempt_2/vertices.npy'),
        body_user_review='accepted by latest user; rapid left-arm swing remains accepted; no body edits',
        source_frames_half_open=[30,106], frame_count=76, span_seconds=3.0, probe_asset_frames=PROBE,
        source_group='APD_5724486_v1_identity_session_unknown', identity_family='cycling_M01', independent_split=False,
        clothing='fixed source barycentric IDs/weights/faces; 4mm normalized barycentric posed-body vertex normals; no topology filtering',
        groom='existing corrected reference points and fixed active mask; rigid SMPL-X head15 transform H=[Q, J15(t)-Q*J15(rest)]',
        transforms='body in frozen pelvis-centered model metres; hair H already includes global orientation/root; display C(x,y,z)=(x,-z,y) once; no floor',
        source_clothing=m['reference']['clothing_geometry'], source_groom=m['reference']['groom_geometry'],
        budget=dict(cpu_threads=2, probe_frames=5, full_frames=76, max_binding_corrections=1,
                    render_engine='Cycles CUDA', gpu_uuid=GPU, views_degrees=[0,90], resolution_each=[512,512],
                    samples=8, denoising=False, fixed_seed=29, max_full_preview_seconds=1800,
                    fallback='lower to384x384 and4 samples only if measured prediction exceeds1800; otherwise stop',
                    expected_seconds_before_probe=None, estimated_cost_status='not measured yet', storage_ceiling_mb=250,
                    groom_cache_policy='one reference in source +76 small head matrices, no per-frame groom copies'),
        human_dressed_dynamic_review='pending_user'))
    print('Frozen recipe saved before geometry computation.')


def garment(body, faces, ids, weights):
    from densekp_retrain.clothing.shells import vertex_normals
    surface = (body[ids] * weights[..., None]).sum(1)
    n = (vertex_normals(body, faces)[ids] * weights[..., None]).sum(1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    return (surface + .004*n).astype(np.float32), surface


def prepare(stage):
    import torch
    from densekp_retrain.body.geometry import load_smplx_model, compute_densekp_3d
    from densekp_retrain.clothing.binding import closest_surface
    from smplx.lbs import batch_rodrigues
    began = time.perf_counter(); torch.set_num_threads(2)
    recipe = json.loads((BASE/'frozen_recipe.json').read_text())
    out = BASE/stage; out.mkdir(exist_ok=False)
    chosen = PROBE if stage == 'probe' else list(range(76))
    if stage == 'full':
        decision = json.loads((BASE/'probe_decision.json').read_text())
        assert decision['proceed'] and decision['estimated_full_preview_seconds'] < 1800
    m = json.loads(CHAR.read_text()); ref=Path(m['reference']['geometry']); cloth=Path(m['reference']['clothing_geometry']); hair=Path(m['reference']['groom_geometry'])
    p = dict(np.load(recipe['authoritative_body'], allow_pickle=False))
    assert np.array_equal(p['source_frame_ids'], np.arange(30,106))
    assert np.array_equal(p['betas'], np.array(m['betas'], np.float32))
    assert np.all(p['transl'] == p['transl'][0])
    rows=list(csv.DictReader((BASE/'time_mapping.csv').open()))
    assert len(rows)==76
    for i,row in enumerate(rows):
        assert int(row['asset_frame'])==i and int(row['apd_source_frame'])==int(p['source_frame_ids'][i])
        assert int(row['timestamp_original'])==int(p['timestamps'][i]) and float(row['clip_seconds'])==p['clip_seconds'][i]
    model=load_smplx_model(Path(m['model_dir']),gender='male',num_betas=30,batch_size=1).eval(); model.requires_grad_(False)
    faces=np.load(BODY/'faces.npy'); np.testing.assert_array_equal(faces, model.faces); np.testing.assert_array_equal(faces,np.load(ref/'faces.npy'))
    cached=np.load(recipe['body_cache'],mmap_mode='r'); dense=np.load(BODY/'fit_attempt_2/densekp_3d.npy'); R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    assert R.shape==(138,10475)
    rest_ref=np.load(ref/'vertices.npy')[0]
    ids=np.load(cloth/'body_barycentric_indices.npy'); w=np.load(cloth/'body_barycentric_weights.npy'); cf=np.load(cloth/'clothing_faces.npy')
    assert np.all(ids>=0) and np.all(ids<10475) and np.allclose(w.sum(1),1,atol=1e-6) and (w>=0).all()
    gi=json.loads((hair/'groom_manifest.json').read_text()); bind=np.load(Path(gi['binding'])/'binding.npz')
    active=np.load(hair/'active_curve_mask.npy'); offsets=np.load(hair/'curve_offsets.npy'); hp=np.load(hair/'points.npy',mmap_mode='r')
    radius=np.load(hair/'radius.npy'); assert np.isfinite(hp).all() and np.isfinite(radius).all()
    assert np.array_equal(bind['curve_offsets'],offsets) and offsets[-1]==len(hp) and len(active)==len(offsets)-1
    gtri=bind['body_faces'][active]; gw=bind['weights'][active]
    cloth_frames=[]; transforms=[]; records=[]; bodyerr=0.; rerr=0.
    with torch.no_grad():
        rest=model(betas=torch.from_numpy(p['betas'][None]),transl=torch.from_numpy(p['transl'][:1]))
        restv=rest.vertices[0].numpy(); restj=rest.joints[0].numpy()
        resterr=float(np.abs(restv-rest_ref).max()); assert resterr<2e-6
        assert float(np.abs((rest_ref[ids]*w[...,None]).sum(1)-np.load(cloth/'reference_body_vertices.npy')[0]).max())<2e-6
        cr,_=garment(rest_ref,faces,ids,w)
        restclotherr=float(np.abs(cr-np.load(cloth/'clothing_vertices.npy')[0]).max()); assert restclotherr<2e-6
        restroots=(rest_ref[gtri]*gw[...,None]).sum(1)
        for n in chosen:
            ts=time.perf_counter()
            kw={k:torch.from_numpy(p[k][n:n+1]) for k in KEYS};kw['betas']=torch.from_numpy(p['betas'][None])
            evaluated=model(**kw); v=evaluated.vertices[0].numpy(); j=evaluated.joints[0].numpy()
            bodyerr=max(bodyerr,float(np.abs(v-cached[n]).max())); rerr=max(rerr,float(np.abs(compute_densekp_3d(R,v[None])[0]-dense[n]).max()))
            cv,base=garment(v,faces,ids,w); assert np.isfinite(cv).all()
            tri=base[cf]; normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); lengths=np.linalg.norm(normals,axis=1); normals/=np.maximum(lengths[:,None],1e-12)
            ct=cv[cf]; cn=np.cross(ct[:,1]-ct[:,0],ct[:,2]-ct[:,0]); area=np.linalg.norm(cn,axis=1)
            cosine=(cn*normals).sum(1)/np.maximum(area,1e-12)
            _,_,signed,_=closest_surface(cv,v,faces)
            axis=np.concatenate([p['global_orient'][n],p['body_pose'][n]])
            local=batch_rodrigues(torch.tensor(axis.reshape(-1,3))).numpy(); glob=local.copy()
            for k in range(1,22): glob[k]=glob[int(model.parents[k])]@local[k]
            Q=glob[15]; H=np.eye(4); H[:3,:3]=Q; H[:3,3]=j[15]-Q@restj[15]
            assert np.allclose(Q.T@Q,np.eye(3),atol=2e-6) and np.linalg.det(Q)>.99999
            # Stream all points through numerical finite check, never save 76 groom arrays.
            transformed=hp@Q.T+H[:3,3]; assert np.isfinite(transformed).all()
            rootdiff=np.linalg.norm(restroots@Q.T+H[:3,3]-(v[gtri]*gw[...,None]).sum(1),axis=1)
            assert np.max(rootdiff)<.01, 'Rigid head diverges >1cm from frozen scalp anchors'
            records.append(dict(asset_frame=n,source_frame=int(p['source_frame_ids'][n]),clip_seconds=float(p['clip_seconds'][n]),
                garment_min_offset_m=float(np.linalg.norm(cv-base,axis=1).min()), garment_max_offset_m=float(np.linalg.norm(cv-base,axis=1).max()),
                local_face_negative_alignment_count=int((cosine<0).sum()), degenerate_faces=int((area<1e-12).sum()),
                garment_signed_near_surface_min_m=float(signed.min()), garment_vertices_below_minus1mm=int((signed<-.001).sum()),
                cloth_area_ratio_max=float((area/np.maximum(lengths,1e-12)).max()),
                groom_scalp_anchor_error_max_m=float(rootdiff.max()),groom_scalp_anchor_error_p95_m=float(np.quantile(rootdiff,.95)),
                cpu_seconds=time.perf_counter()-ts))
            cloth_frames.append(cv);transforms.append(H)
    assert bodyerr<2e-5 and rerr<2e-6
    np.save(out/'clothing_vertices.npy',np.array(cloth_frames));np.save(out/'head_transforms.npy',np.array(transforms));np.save(out/'asset_frames.npy',chosen)
    write(out/'per_frame.json',records)
    summary=dict(integration_only=True,stage=stage,frames=chosen,parameter_readback_max_error_m=bodyerr,R138_max_error_m=rerr,
        reference_body_max_error_m=resterr,reference_clothing_recipe_max_error_m=restclotherr,
        fixed_shape=True,body_unchanged=True,clothing_vertices=2612,clothing_faces=5018,
        garment_topology_source=str(cloth/'clothing_faces.npy'),garment_uv='no UV asset; existing continuous GarmentRest shader coordinates unchanged',
        groom_points_source=len(hp),groom_active_curves=int(active.sum()),groom_active_points=int(np.diff(offsets)[active].sum()),
        max_groom_scalp_anchor_error_m=max(x['groom_scalp_anchor_error_max_m'] for x in records),
        garment_flipped_faces_max=max(x['local_face_negative_alignment_count'] for x in records),
        garment_degenerate_faces_max=max(x['degenerate_faces'] for x in records),
        garment_vertices_below_minus1mm_max=max(x['garment_vertices_below_minus1mm'] for x in records),
        no_face_deletion=True,no_dynamic_pruning=True,head_transform_applications=1,display_transform_applications=1,
        no_minY_grounding=True,elapsed_seconds=time.perf_counter()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        collision_diagnostic='nearest of24 face centroids, signed samples; not exhaustive intersection proof',human_dynamic_review='pending_user')
    write(out/'validation.json',summary);print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['freeze','probe','full']);a=ap.parse_args()
    freeze() if a.stage=='freeze' else prepare(a.stage)
