#!/usr/bin/env python3
"""Derive M01 simple clothing and rigid groom matrices from accepted BVH body."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='2'
import argparse,csv,json,resource,sys,time
from pathlib import Path
import numpy as np
import torch
from smplx.lbs import batch_rodrigues

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.body.geometry import load_smplx_model,compute_densekp_3d
from densekp_retrain.body.sequence_binding import bound_garment,head_transform
BASE=ROOT/'outputs/experiments/bvh_p2_dressed_asset_a2_20261010'
KEYS=['global_orient','body_pose','transl','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']

def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def main():
    global BASE
    ap=argparse.ArgumentParser();ap.add_argument('--asset-root',type=Path);a=ap.parse_args()
    if a.asset_root:BASE=a.asset_root.resolve()
    began=time.monotonic();torch.set_num_threads(2)
    recipe=json.loads((BASE/'frozen_recipe.json').read_text());m=json.loads(Path(recipe['character_manifest']).read_text())
    assert m['character_id']==recipe['character_family']+'_v3' if recipe['character_family']=='cycling_M02' else m['character_id']=='cycling_M01_v2'
    assert m['gender']=='male' and m['num_betas']==30
    p=dict(np.load(recipe['source_authoritative_parameters'],allow_pickle=False));n=recipe['asset_frames'];lo,hi=recipe['source_frame_half_open'];tick=recipe['ticks_per_frame'];offset=recipe.get('garment_offset_m',.004)
    assert hi-lo==n and np.array_equal(p['source_frame_ids'],np.arange(lo,hi)) and np.array_equal(p['asset_frame_ids'],np.arange(n))
    assert np.array_equal(p['nominal_asset_time_ticks'],np.arange(n)*tick)
    assert np.array_equal(p['nominal_source_time_ticks'],np.arange(lo,hi)*tick)
    assert np.array_equal(p['time_base'],[1,10000000]) and np.array_equal(p['betas'],np.asarray(m['betas'],np.float32))
    assert np.array_equal(p['transl'],np.broadcast_to(p['transl'][0],p['transl'].shape))
    rows=list(csv.DictReader(Path(recipe['source_time_mapping']).open()));assert len(rows)==n
    for i,row in enumerate(rows):
        assert int(row['asset_frame'])==i and int(row['source_frame'])==lo+i
        assert int(row['nominal_asset_time_ticks'])==int(p['nominal_asset_time_ticks'][i])
        assert int(row['nominal_source_time_ticks'])==int(p['nominal_source_time_ticks'][i])
        assert [int(row['time_base_num']),int(row['time_base_den'])]==[1,10000000]
    model=load_smplx_model(Path(m['model_dir']),gender='male',num_betas=30,batch_size=1).eval();model.requires_grad_(False)
    faces=np.load(recipe['faces']);ref=Path(m['reference']['geometry']);cloth=Path(recipe['source_clothing']);groom=Path(recipe['source_groom'])
    np.testing.assert_array_equal(faces,model.faces);np.testing.assert_array_equal(faces,np.load(ref/'faces.npy'))
    ids=np.load(cloth/'body_barycentric_indices.npy');w=np.load(cloth/'body_barycentric_weights.npy');cf=np.load(cloth/'clothing_faces.npy')
    assert ids.shape==w.shape and ids.min()>=0 and ids.max()<10475 and np.allclose(w.sum(1),1,atol=1e-6) and (w>=0).all()
    hp=np.load(groom/'points.npy',mmap_mode='r');rad=np.load(groom/'radius.npy',mmap_mode='r');offsets=np.load(groom/'curve_offsets.npy');active=np.load(groom/'active_curve_mask.npy')
    gi=json.loads((groom/'groom_manifest.json').read_text());bind=np.load(Path(gi['binding'])/'binding.npz')
    np.testing.assert_array_equal(bind['curve_offsets'],offsets)
    assert len(hp)==offsets[-1] and len(rad)==len(hp) and active.shape==(len(offsets)-1,) and np.isfinite(hp).all() and np.isfinite(rad).all()
    gtri=bind['body_faces'][active];gw=bind['weights'][active]
    vcache=np.load(recipe['body_cache'],mmap_mode='r');jcache=np.load(recipe['body_joints'],mmap_mode='r');dcache=np.load(recipe['densekp_cache'],mmap_mode='r');R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    assert vcache.shape==(n,10475,3) and jcache.shape==(n,22,3) and dcache.shape==(n,138,3) and R.shape==(138,10475)
    with torch.no_grad():
        rest=model(betas=torch.from_numpy(p['betas'][None]),transl=torch.from_numpy(p['transl'][:1]))
    rv=rest.vertices[0].numpy();rj=rest.joints[0].numpy()
    rest_body_error=float(np.max(np.abs(rv-np.load(ref/'vertices.npy')[0])));assert rest_body_error<2e-6
    rest_cloth=bound_garment(rv,faces,ids,w,offset_m=offset)
    rest_cloth_error=float(np.max(np.abs(rest_cloth-np.load(cloth/'clothing_vertices.npy')[0])));assert rest_cloth_error<2e-6
    restroots=(rv[gtri]*gw[...,None]).sum(1)
    clothing=np.empty((n,len(ids),3),np.float32);heads=np.empty((n,4,4),np.float64);records=[]
    max_body=0.;max_joint=0.;max_R=0.;max_rot=0.
    with torch.no_grad():
        for i in range(n):
            t=time.monotonic();kw={k:torch.from_numpy(p[k][i:i+1]) for k in KEYS};kw['betas']=torch.from_numpy(p['betas'][None]);ev=model(**kw)
            v=ev.vertices[0].numpy();j=ev.joints[0,:22].numpy()
            max_body=max(max_body,float(np.max(np.abs(v-vcache[i]))));max_joint=max(max_joint,float(np.max(np.abs(j-jcache[i]))))
            max_R=max(max_R,float(np.max(np.abs(compute_densekp_3d(R,v[None])[0]-dcache[i]))))
            cv=bound_garment(v,faces,ids,w,offset_m=offset);clothing[i]=cv
            axis=np.concatenate([p['global_orient'][i],p['body_pose'][i]]).reshape(-1,3)
            local=batch_rodrigues(torch.from_numpy(axis)).numpy()
            max_rot=max(max_rot,float(np.max(np.abs(local-p['local_rotation_matrices'][i]))))
            H=head_transform(local,model.parents.numpy(),j[15],rj[15]);heads[i]=H
            rootdiff=np.linalg.norm(restroots@H[:3,:3].T+H[:3,3]-(v[gtri]*gw[...,None]).sum(1),axis=1)
            base=(v[ids]*w[...,None]).sum(1);ct=cv[cf];bt=base[cf]
            cn=np.cross(ct[:,1]-ct[:,0],ct[:,2]-ct[:,0]);bn=np.cross(bt[:,1]-bt[:,0],bt[:,2]-bt[:,0])
            flipped=(cn*bn).sum(1)<0;area=.5*np.linalg.norm(cn,axis=1)
            records.append(dict(asset_frame=i,source_frame=int(p['source_frame_ids'][i]),nominal_asset_time_ticks=int(p['nominal_asset_time_ticks'][i]),
                scalp_anchor_max_m=float(rootdiff.max()),scalp_anchor_p95_m=float(np.quantile(rootdiff,.95)),
                garment_local_negative_normal_faces=int(flipped.sum()),garment_negative_normal_area_m2=float(area[flipped].sum()),
                garment_degenerate_faces=int((area<1e-12).sum()),cpu_seconds=time.monotonic()-t))
    assert max_body<2e-5 and max_joint<2e-5 and max_R<2e-6 and max_rot<2e-6
    assert np.isfinite(clothing).all() and np.isfinite(heads).all() and np.allclose(heads[:,3,:],[0,0,0,1])
    full=BASE/'full';full.mkdir(exist_ok=False);probe=BASE/'probe';probe.mkdir(exist_ok=False)
    np.save(full/'clothing_vertices.npy',clothing);np.save(full/'head_transforms.npy',heads);np.save(full/'asset_frames.npy',np.arange(n))
    chosen=np.asarray(recipe['probe_asset_frames'],int)
    np.save(probe/'clothing_vertices.npy',clothing[chosen]);np.save(probe/'head_transforms.npy',heads[chosen]);np.save(probe/'asset_frames.npy',chosen)
    # One arbitrary rigid scene transform checks ownership algebra without making a road scene.
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float);M=np.eye(4);M[:3,:3]=C;M[:3,3]=[.2,-.3,1.1]
    k=n//2;h=heads[k];sample=np.asarray(hp[::1200])
    lhs=(sample@h[:3,:3].T+h[:3,3])@M[:3,:3].T+M[:3,3]
    combined=M@h;rhs=sample@combined[:3,:3].T+combined[:3,3]
    scene_head_error=float(np.max(np.abs(lhs-rhs)));assert scene_head_error<2e-6
    summary=dict(integration_only=True,frames=n,all_source_parameters_unchanged=True,source_parameters=recipe['source_authoritative_parameters'],
        body_cache_readback_max_m=max_body,joint_cache_readback_max_m=max_joint,R138_readback_max_m=max_R,
        local_rotation_readback_max=max_rot,reference_body_readback_max_m=rest_body_error,reference_clothing_readback_max_m=rest_cloth_error,
        body_vertices=10475,body_faces=len(faces),clothing_vertices=len(ids),clothing_faces=len(cf),fixed_clothing_topology=True,
        garment_offset_m=offset,garment_material='existing '+m['character_id']+' GarmentRest continuous shell',groom_reference_points=len(hp),
        groom_source_curves=len(active),groom_active_curves=int(active.sum()),groom_active_points=int(np.diff(offsets)[active].sum()),
        groom_fixed_active_set=True,groom_reference_space=m['character_id']+' shaped-rest pelvis-centered model',
        head_matrix_count=n,head_already_includes_root_global_orient=True,scene_head_composition_error_m=scene_head_error,
        scalp_anchor_error_max_m=max(x['scalp_anchor_max_m'] for x in records),
        garment_negative_local_faces_max=max(x['garment_local_negative_normal_faces'] for x in records),
        garment_degenerate_faces_max=max(x['garment_degenerate_faces'] for x in records),
        body_bounds_model_m=[np.asarray(vcache).min((0,1)).tolist(),np.asarray(vcache).max((0,1)).tolist()],
        clothing_bounds_model_m=[clothing.min((0,1)).tolist(),clothing.max((0,1)).tolist()],
        body_lowpoint_y_range_m=[float(np.min(vcache[:,:,1],axis=1).min()),float(np.min(vcache[:,:,1],axis=1).max())],
        no_per_frame_grounding=True,no_face_deletion=True,no_groom_rescreen=True,all_finite=True,
        dressed_dynamic_human_review='pending_user',numeric_intersection_status='only local normal diagnostic; no full mesh collision test',
        cpu_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(BASE/'validation.json',summary);write(BASE/'per_frame.json',records)
    print(json.dumps({k:summary[k] for k in ['cpu_seconds','body_cache_readback_max_m','R138_readback_max_m','scalp_anchor_error_max_m','garment_negative_local_faces_max']}),flush=True)

if __name__=='__main__':main()
