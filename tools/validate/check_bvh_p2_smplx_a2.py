#!/usr/bin/env python3
"""Independent readback of the frozen 120-frame BVH cycling body pilot."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,csv,json,sys,time
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.body.geometry import load_smplx_model
BASE=ROOT/'outputs/experiments/bvh_p2_smplx_pilot_a2_20261010'
def angle(a,b,c):
    x=a-b;y=c-b
    return np.rad2deg(np.arccos(np.clip(np.sum(x*y,axis=-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(y,axis=-1)),-1,1)))
def main():
    start=time.perf_counter();torch.set_num_threads(2)
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=BASE);args=ap.parse_args();base=args.root
    freeze=json.loads((base/'frozen_manifest.json').read_text());lo,hi=freeze['source_frame_half_open'];n=freeze['asset_frames'];tick=freeze['ticks_per_frame']
    p=np.load(base/'fit_attempt_1/parameters.npz');obs=np.load(base/'observations.npz');v=np.load(base/'fit_attempt_1/vertices.npy',mmap_mode='r')
    j=np.load(base/'fit_attempt_1/joints.npy',mmap_mode='r');R=np.load(ROOT/'assets/densekp_regressor_R.npy')
    dense=np.load(base/'fit_attempt_1/densekp_3d.npy',mmap_mode='r');faces=np.load(base/'faces.npy')
    assert v.shape==(n,10475,3) and dense.shape==(n,138,3) and R.shape==(138,10475)
    np.testing.assert_array_equal(p['source_frame_ids'],np.arange(lo,hi));np.testing.assert_array_equal(p['asset_frame_ids'],np.arange(n))
    np.testing.assert_array_equal(p['nominal_asset_time_ticks'],np.arange(n,dtype=np.int64)*tick)
    np.testing.assert_array_equal(p['nominal_source_time_ticks'],np.arange(lo,hi,dtype=np.int64)*tick)
    np.testing.assert_array_equal(p['time_base'],np.array([1,10000000]));assert np.all(np.diff(p['nominal_asset_time_ticks'])==tick)
    assert np.max(np.abs(p['betas'][None]-p['betas']))==0 and np.max(np.ptp(p['transl'],axis=0))==0
    model=load_smplx_model(ROOT/'outputs/experiments/cycling_commuter_pilot_20260930/models',gender='male',num_betas=30,batch_size=1).eval()
    charpath=Path(freeze['character_manifest']);charpath=charpath if charpath.is_absolute() else ROOT/charpath
    char=json.loads(charpath.read_text());assert char['character_id']==freeze['target_character']
    assert np.array_equal(p['betas'],np.asarray(char['betas'],np.float32))
    ref=Path(char['reference']['geometry']);np.testing.assert_array_equal(faces,np.load(ref/'faces.npy'))
    assert np.array_equal(faces,model.faces)
    max_body=0.;max_joint=0.;max_r=0.
    keys=['global_orient','body_pose','transl','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']
    with torch.no_grad():
        for a in range(0,n,4):
            b=min(a+4,n);kw={k:torch.tensor(p[k][a:b]) for k in keys};kw['betas']=torch.tensor(p['betas'][None]).expand(b-a,-1)
            res=model(**kw)
            max_body=max(max_body,float(np.max(np.abs(res.vertices.numpy()-v[a:b]))))
            max_joint=max(max_joint,float(np.max(np.abs(res.joints[:,:22].numpy()-j[a:b]))))
            for k in range(a,b):max_r=max(max_r,float(np.max(np.abs(R@v[k]-dense[k]))))
    assert max_body<2e-6 and max_joint<2e-6 and max_r<2e-6
    assert np.isfinite(v).all() and np.isfinite(dense).all() and np.max(np.abs(j[:,0]))<2e-5
    m=json.loads((base/'joint_mapping.json').read_text())['correspondences'];lookup={x['source']:i for i,x in enumerate(m)}
    src=obs['target_joints_m']
    raw_elbow={side:np.quantile(angle(src[:,lookup[side+'Arm']],src[:,lookup[side+'ForeArm']],src[:,lookup[side+'Hand']]),[.05,.5,.95]).tolist() for side in ('Left','Right')}
    target_elbow={'Left':np.quantile(angle(j[:,16],j[:,18],j[:,20]),[.05,.5,.95]).tolist(),
                  'Right':np.quantile(angle(j[:,17],j[:,19],j[:,21]),[.05,.5,.95]).tolist()}
    foot_y_corr=float(np.corrcoef(j[:,7,1],j[:,8,1])[0,1])
    foot_y_range=[float(np.ptp(j[:,7,1])),float(np.ptp(j[:,8,1]))]
    with (base/'time_mapping.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['asset_frame','source_frame','nominal_asset_time_ticks','nominal_source_time_ticks','time_base_num','time_base_den','timing_status'])
        for i in range(n):w.writerow([i,int(p['source_frame_ids'][i]),int(p['nominal_asset_time_ticks'][i]),int(p['nominal_source_time_ticks'][i]),1,10000000,'BVH_nominal_not_exposure_PTS'])
    out=dict(integration_only=True,frames=n,source_frame_half_open=[lo,hi],time_exact_nominal=True,source_samples_uninterpolated=True,
        fixed_shape=True,model_standard_male=True,reference_character=char['character_id'],faces_equal_character_reference=True,full_parameter_body_readback_max_abs_m=max_body,
        full_parameter_joint_readback_max_abs_m=max_joint,R138_full_readback_max_abs_m=max_r,all_finite=True,
        pelvis_center_max_abs_m=float(np.max(np.abs(j[:,0]))),
        raw_elbow_internal_angle_deg_q05_q50_q95=raw_elbow,target_elbow_internal_angle_deg_q05_q50_q95=target_elbow,
        target_foot_y_correlation=foot_y_corr,target_foot_y_range_m=foot_y_range,
        source_foot_y_range_after_one_scale_m=[float(np.ptp(src[:,lookup['LeftFoot'],1])),float(np.ptp(src[:,lookup['RightFoot'],1]))],
        source_root_native_displacement=float(np.linalg.norm(obs['native_root'][-1]-obs['native_root'][0])),
        source_root_disposition='retained in observations; not exported as road trajectory',
        body_human_dynamic_review='pending_user',compute_seconds=time.perf_counter()-start)
    (base/'readback_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
