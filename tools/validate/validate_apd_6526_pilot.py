#!/usr/bin/env python3
"""Independent parameter readback and all-frame temporal diagnostics, no fitting."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES','')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('MPLCONFIGDIR','/tmp/densekp_apd_pilot_mpl')
os.environ.setdefault('MPLBACKEND','Agg')
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.body.geometry import load_smplx_model

def main():
    base=ROOT/'outputs/experiments/apd_6526_smplx_pilot_20261008';out=base/'readback';out.mkdir(exist_ok=False)
    began=time.perf_counter();torch.set_num_threads(2)
    freeze=json.loads((base/'frozen_manifest.json').read_text());obs=dict(np.load(base/'observations.npz'))
    chosen=base/'fit_attempt_2';p=dict(np.load(chosen/'parameters.npz',allow_pickle=False))
    assert np.array_equal(p['betas'],np.array(freeze['betas'],np.float32))
    assert np.array_equal(p['source_frame_ids'],np.arange(30,106)) and len(p['body_pose'])==76
    assert np.array_equal(p['timestamps'],np.array(freeze['timestamps'],np.int64))
    assert np.array_equal(p['clip_seconds'],np.array(freeze['clip_seconds']))
    assert (np.diff(p['timestamps'])>0).all() and not np.isnan(obs['aligned_target_m']).any()
    model=load_smplx_model(Path(freeze['model_dir']),gender='male',num_betas=30,batch_size=1).eval();model.requires_grad_(False)
    cached=np.load(chosen/'vertices.npy',mmap_mode='r');dense=np.load(chosen/'densekp_3d.npy');R=np.load(freeze['regressor'])
    assert np.array_equal(np.load(base/'faces.npy'),model.faces)
    assert R.shape==(138,10475) and cached.shape==(76,10475,3) and dense.shape==(76,138,3)
    for k,v in p.items():
        if v.dtype.kind in 'fiu':assert np.isfinite(v).all(),k
    for k in ['left_hand_pose','right_hand_pose','jaw_pose','leye_pose','reye_pose','expression']:assert np.count_nonzero(p[k])==0
    assert np.all(p['transl']==p['transl'][0])
    maximum=0.;rerror=0.
    with torch.no_grad():
        for start in range(0,76,4):
            kwargs={k:torch.from_numpy(p[k][start:start+4]) for k in ['global_orient','body_pose','transl','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']}
            kwargs['betas']=torch.tensor(p['betas'][None]).expand(len(kwargs['body_pose']),-1)
            v=model(**kwargs).vertices.numpy()
            maximum=max(maximum,float(np.abs(v-cached[start:start+4]).max()))
            for i,row in enumerate(v):rerror=max(rerror,float(np.abs(R@row-dense[start+i]).max()))
    assert maximum<2e-5 and rerror<2e-6
    rot=p['local_rotation_matrices'];axis=np.concatenate([p['global_orient'][:,None],p['body_pose'].reshape(76,21,3)],1)
    restored=Rotation.from_rotvec(axis.reshape(-1,3)).as_matrix().reshape(rot.shape)
    re=float(np.abs(restored-rot).max());orth=float(np.abs(rot.transpose(0,1,3,2)@rot-np.eye(3)).max())
    assert re<2e-6 and orth<2e-6 and np.allclose(np.linalg.det(rot),1,atol=2e-6)
    t=p['clip_seconds'];src=obs['aligned_target_m'];fit=np.load(chosen/'mapped_joints.npy');joints=np.load(chosen/'joints.npy')
    def knees(x):
        a=x[:,[8,9]]-x[:,[10,11]];b=x[:,[12,13]]-x[:,[10,11]]
        return np.rad2deg(np.arccos(np.clip((a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)),-1,1)))
    sk=knees(src);fk=knees(fit);signals={}
    for i,name in enumerate(['left','right']):
        signals[name]=dict(source_knee_range_deg=float(np.ptp(sk[:,i])),fitted_knee_range_deg=float(np.ptp(fk[:,i])),
            source_knee_std_deg=float(np.std(sk[:,i])),fitted_knee_std_deg=float(np.std(fk[:,i])),
            knee_correlation=float(np.corrcoef(sk[:,i],fk[:,i])[0,1]),
            source_foot_vertical_range_m=float(np.ptp(src[:,12+i,1])),fitted_ankle_vertical_range_m=float(np.ptp(fit[:,12+i,1])),
            foot_vertical_correlation=float(np.corrcoef(src[:,12+i,1],fit[:,12+i,1])[0,1]))
    # Both numerical evidence and visual review are required; these are not a motion-acceptance gate.
    leg_lengths=np.stack([np.linalg.norm(joints[:,a]-joints[:,b],axis=1) for a,b in [(1,4),(4,7),(2,5),(5,8)]],1)
    dif=np.linalg.norm(np.diff(cached,axis=0),axis=2);maxframe=dif.max(1)
    i,k=np.unravel_index(np.argmax(dif),dif.shape)
    angles=np.rad2deg(np.arccos(np.clip((np.trace(rot[:-1].transpose(0,1,3,2)@rot[1:],axis1=2,axis2=3)-1)/2,-1,1)))
    residual=np.load(chosen/'residuals_m.npy');jstep=np.linalg.norm(np.diff(joints,axis=0),axis=2)
    with (out/'per_frame.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['source_frame','clip_s','mean_residual_m','max_residual_m','max_vertex_step_from_previous_m','left_knee_deg','right_knee_deg','left_wrist_step_from_previous_m','right_wrist_step_from_previous_m'])
        for n in range(76):w.writerow([int(p['source_frame_ids'][n]),t[n],residual[n].mean(),residual[n].max(),maxframe[n-1] if n else 0,*fk[n],jstep[n-1,20] if n else 0,jstep[n-1,21] if n else 0])
    fig,ax=plt.subplots(4,1,figsize=(11,11),sharex=True)
    for side,c in [(0,'#176bb3'),(1,'#d56a13')]:
        name=['left','right'][side];ax[0].plot(t,sk[:,side],':',color=c,label=name+' APD');ax[0].plot(t,fk[:,side],color=c,label=name+' target')
        ax[1].plot(t,src[:,12+side,1],':',color=c,label=name+' APD weak foot');ax[1].plot(t,fit[:,12+side,1],color=c,label=name+' target ankle')
    ax[0].set_ylabel('Knee angle (degrees)');ax[0].legend(ncol=4,fontsize=8)
    ax[1].set_ylabel('Aligned vertical (model m)');ax[1].legend(ncol=2,fontsize=8)
    for attempt,c in [('fit_attempt_1','#a33'),('fit_attempt_2','#075')]:
        v=np.load(base/attempt/'vertices.npy',mmap_mode='r');a=np.linalg.norm(np.diff(v,axis=0),axis=2)
        ax[2].plot(t[1:],a.max(1),color=c,label=attempt+' max vertex step')
    ax[2].set_ylabel('Adjacent displacement (m)');ax[2].legend(fontsize=8)
    ax[3].plot(t,residual.mean(1),label='mean14 joint residual');ax[3].plot(t,residual[:,[8,9]].mean(1),label='mean hip residual (proportion mismatch)')
    ax[3].set_ylabel('Residual (model m)');ax[3].legend(fontsize=8);ax[3].set_xlabel('Clip seconds; source offset1.200018s, no resampling')
    for a in ax:a.grid(alpha=.25)
    fig.suptitle('APD6526: motion retained vs isolated-jump suppression. Original observations are not ground truth.')
    fig.tight_layout();fig.savefig(out/'temporal_diagnostics.png',dpi=140);plt.close(fig)
    result=dict(integration_only=True,numerical_chain_passed=True,all76_parameter_frames_recomputed=True,
       parameter_to_cache_max_error_m=maximum,parameter_to_R138_max_error_m=rerror,
       axis_angle_to_matrix_max_error=re,rotation_orthogonality_max_error=orth,
       lower_limb_bone_length_cv_max=float(np.max(leg_lengths.std(0)/leg_lengths.mean(0))),
       pelvis_origin_max_error_m=float(np.abs(joints[:,0]).max()),translation_constant=True,minY_grounding_applied=False,
       motion_signals=signals,source_left_right_knee_correlation=float(np.corrcoef(sk.T)[0,1]),fitted_left_right_knee_correlation=float(np.corrcoef(fk.T)[0,1]),
       max_vertex_step=dict(source_pair=[int(p['source_frame_ids'][i]),int(p['source_frame_ids'][i+1])],vertex_id=int(k),metres=float(dif[i,k])),
       max_local_rotation_step_deg=float(angles.max()),left_wrist_step_max_m=float(jstep[:,20].max()),right_wrist_step_max_m=float(jstep[:,21].max()),
       frames_to_review=[35,36,37,44,45,46,47,48],local_knee_hinge_positive=True,
       dynamic_human_acceptance='pending_user',self_intersection='all76 front/side contact frames inspected, no obvious severe body penetration observed; no exhaustive triangle-intersection certificate',
       qualitative_limit='left-arm fast raise/lower remains; fingers/feet orientations fixed, no pedal/handlebar constraint',
       elapsed_seconds=time.perf_counter()-began)
    (out/'validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
