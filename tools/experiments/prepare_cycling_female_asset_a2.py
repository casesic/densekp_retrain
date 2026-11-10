"""Reuse one accepted APD rotation sequence with the existing female character."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','2')
import json
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import torch
from smplx.lbs import batch_rodrigues

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.body.geometry import load_smplx_model,compute_densekp_3d
from densekp_retrain.body.sequence_binding import bound_garment,head_transform
OUT=ROOT/'outputs/experiments/cycling_female_asset_a2_20261009'
POSE_KEYS=['global_orient','body_pose','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']


def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--asset-root',type=Path);args=parser.parse_args()
    if args.asset_root:OUT=args.asset_root.resolve()
    began=time.monotonic();torch.set_num_threads(2)
    recipe=json.loads((OUT/'frozen_recipe.json').read_text());m=json.loads(Path(recipe['character_manifest']).read_text())
    ref=Path(m['reference']['geometry']);cloth=Path(recipe['source_clothing']);hair=Path(recipe['source_groom'])
    assert m['gender']=='female' and m['num_betas']==30
    source=dict(np.load(recipe['source_authoritative_parameters'],allow_pickle=False))
    n=recipe.get('frame_count',76)
    timeline=json.loads(Path(recipe['timeline']).read_text()) if 'timeline' in recipe else None
    if timeline is None:assert np.array_equal(source['source_frame_ids'],np.arange(30,106))
    else:
        assert len(timeline)==n
        np.testing.assert_array_equal(source['source_decoded_frames'],[x['source_frame'] for x in timeline])
        np.testing.assert_array_equal(source['source_pts'],[x['source_time_tick'] for x in timeline])
        np.testing.assert_array_equal(source['time_base'],timeline[0]['time_base'])
    model=load_smplx_model(Path(m['model_dir']),gender='female',num_betas=30,batch_size=1).eval();model.requires_grad_(False)
    betas=np.array(m['betas'],np.float32);p={k:v.copy() for k,v in source.items()}
    p.update(betas=betas,gender=np.array('female'),root_mode=np.array('fixed_pelvis_model_origin_scene_trajectory_owned_by_scene'))
    faces=np.asarray(model.faces);np.testing.assert_array_equal(faces,np.load(ref/'faces.npy'))
    np.testing.assert_array_equal(faces,np.load(ROOT/'outputs/experiments/apd_6526_smplx_pilot_20261008/faces.npy'))
    R=np.load(ROOT/'assets/densekp_regressor_R.npy');ids=np.load(cloth/'body_barycentric_indices.npy');w=np.load(cloth/'body_barycentric_weights.npy')
    cf=np.load(cloth/'clothing_faces.npy');assert w.shape==ids.shape and np.allclose(w.sum(1),1,atol=1e-6)
    hi=json.loads((hair/'groom_manifest.json').read_text());binding=np.load(Path(hi['binding'])/'binding.npz')
    hp=np.load(hair/'points.npy');radius=np.load(hair/'radius.npy');offsets=np.load(hair/'curve_offsets.npy');active=np.load(hair/'active_curve_mask.npy')
    np.testing.assert_array_equal(binding['curve_offsets'],offsets)
    assert np.isfinite(hp).all() and np.isfinite(radius).all() and offsets[-1]==len(hp)
    tri=binding['body_faces'][active];weights=binding['weights'][active]
    full=OUT/'full';full.mkdir(exist_ok=False);probe=OUT/'probe';probe.mkdir(exist_ok=False)
    vertices=[];joints=[];garments=[];heads=[];records=[]
    with torch.no_grad():
        rest0=model(betas=torch.from_numpy(betas[None]))
        pelvis=rest0.joints[0,0].numpy().copy();transl=-pelvis
        p['transl']=np.broadcast_to(transl,(n,3)).copy()
        rest=model(betas=torch.from_numpy(betas[None]),transl=torch.from_numpy(transl[None]))
        rv=rest.vertices[0].numpy();rj=rest.joints[0].numpy()
        rest_error=float(np.max(np.abs(rv-np.load(ref/'vertices.npy')[0])))
        assert rest_error<2e-6
        reference_cloth=bound_garment(rv,faces,ids,w,offset_m=.004)
        cloth_error=float(np.max(np.abs(reference_cloth-np.load(cloth/'clothing_vertices.npy')[0])))
        assert cloth_error<2e-6
        reference_roots=(rv[tri]*weights[...,None]).sum(1)
        for i in range(n):
            start=time.monotonic();kw={k:torch.from_numpy(p[k][i:i+1]) for k in POSE_KEYS}
            kw.update(betas=torch.from_numpy(betas[None]),transl=torch.from_numpy(p['transl'][i:i+1]))
            q=model(**kw);v=q.vertices[0].numpy();j=q.joints[0].numpy()
            assert np.isfinite(v).all() and np.linalg.norm(j[0])<2e-6
            cv=bound_garment(v,faces,ids,w,offset_m=.004)
            axis=np.concatenate([p['global_orient'][i],p['body_pose'][i]])
            local=batch_rodrigues(torch.from_numpy(axis.reshape(-1,3))).numpy()
            if 'local_rotation_matrices' in source:np.testing.assert_allclose(local,source['local_rotation_matrices'][i],atol=2e-6,rtol=0)
            H=head_transform(local,model.parents.numpy(),j[15],rj[15])
            assert np.isfinite(hp@H[:3,:3].T+H[:3,3]).all()
            root_error=np.linalg.norm(reference_roots@H[:3,:3].T+H[:3,3]-(v[tri]*weights[...,None]).sum(1),axis=1)
            shell=(v[ids]*w[...,None]).sum(1);ct=cv[cf];bt=shell[cf]
            normal=np.cross(ct[:,1]-ct[:,0],ct[:,2]-ct[:,0]);bn=np.cross(bt[:,1]-bt[:,0],bt[:,2]-bt[:,0])
            area=np.linalg.norm(normal,axis=1)/2;flipped=(normal*bn).sum(1)<0
            records.append(dict(asset_frame=i,source_frame=int(timeline[i]['source_frame'] if timeline else p['source_frame_ids'][i]),timestamp=int(timeline[i]['source_time_tick'] if timeline else p['timestamps'][i]),
                pelvis_norm_m=float(np.linalg.norm(j[0])),groom_anchor_max_m=float(root_error.max()),groom_anchor_p95_m=float(np.quantile(root_error,.95)),
                garment_negative_local_normal_faces=int(flipped.sum()),negative_local_face_total_area_m2=float(area[flipped].sum()),
                degenerate_cloth_faces=int((area<1e-12).sum()),seconds=time.monotonic()-start))
            vertices.append(v);joints.append(j);garments.append(cv);heads.append(H)
    vertices=np.array(vertices);garments=np.array(garments);heads=np.array(heads);joints=np.array(joints)
    dense=compute_densekp_3d(R,vertices)
    inherited=[k for k in source if k not in ['betas','transl','gender','root_mode']]
    for k in inherited:np.testing.assert_array_equal(p[k],source[k])
    np.savez(OUT/'parameters.npz',**p)
    for k,x in dict(vertices=vertices,joints=joints,clothing_vertices=garments,head_transforms=heads,densekp_3d=dense,asset_frames=np.arange(n)).items():np.save(full/(k+'.npy'),x)
    np.save(OUT/'faces.npy',faces)
    chosen=recipe['probe_asset_frames']
    for k,x in dict(clothing_vertices=garments[chosen],head_transforms=heads[chosen],asset_frames=np.array(chosen)).items():np.save(probe/(k+'.npy'),x)
    reread=dict(np.load(OUT/'parameters.npz',allow_pickle=False))
    for k in p:np.testing.assert_array_equal(p[k],reread[k])
    err=float(np.max(np.abs(compute_densekp_3d(R,np.load(full/'vertices.npy'))-np.load(full/'densekp_3d.npy'))));assert err<2e-6
    summary=dict(integration_only=True,character='cycling_F02_v3',gender='female',frames=n,rest_pelvis_untranslated=pelvis.tolist(),fixed_target_transl=transl.tolist(),
        source_male_transl=source['transl'][0].tolist(),transl_recomputed_not_copied=True,all_source_pose_and_time_arrays_exact=True,
        source_actor_sex='unknown; target female does not identify original source actor',body_shape_fixed=True,body_vertices=10475,body_faces=len(faces),
        female_rest_readback_error_m=rest_error,clothing_reference_readback_error_m=cloth_error,R_numeric_readback_error_m=err,
        clothing_vertices=garments.shape[1],clothing_faces=len(cf),groom_curves_source=len(active),groom_curves_active=int(active.sum()),
        groom_points_active=int(np.diff(offsets)[active].sum()),groom_points_source=len(hp),head_matrices_from_female=True,
        max_groom_anchor_error_m=max(x['groom_anchor_max_m'] for x in records),
        max_negative_local_normal_faces=max(x['garment_negative_local_normal_faces'] for x in records),
        no_per_frame_filter=True,no_per_frame_grounding=True,scene_owns_progression=True,
        elapsed_seconds=time.monotonic()-began,human_dynamic_review='pending_user',per_frame=records)
    write(OUT/'validation.json',summary)
    write(OUT/'parameter_semantics.json',dict(authority=str(OUT/'parameters.npz'),source=recipe['source_authoritative_parameters'],estimated_target_parameters=['constant transl from female rest pelvis'],fixed_target=['female model','30 betas'],inherited_source=inherited,root='pelvis fixed at model origin; source progress removed; scene owns progression',groom_reference_space='female shaped rest body with transl=-female rest pelvis; stored groom_v5 points already in this space',head='Q from unchanged pose chain; H translation=J_head(t)-Q J_head(rest); root/global_orient included once'))
    print(json.dumps({k:v for k,v in summary.items() if k!='per_frame'}))


if __name__=='__main__':main()
