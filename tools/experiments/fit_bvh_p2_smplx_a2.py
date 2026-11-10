#!/usr/bin/env python3
"""Bounded native BVH -> fixed-shape SMPL-X cycling body experiment."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse, importlib.util, json, math, resource, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.body.geometry import load_smplx_model, compute_densekp_3d
spec=importlib.util.spec_from_file_location('native_bvh_audit',ROOT/'tools/validate/audit_cycling_motion_library_a2.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

DEFAULT=ROOT/'outputs/experiments/bvh_p2_smplx_pilot_a2_20261010'
CHAR=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_M01_v2/manifest.json'
SOURCE=Path('/mnt/data1/yy/home/yy/datasets/cycling_mocap/candidates/2_RB_M27_20230809_PA/2_RB_M27_20230809_PA.bvh')
CLIP=ROOT/'outputs/motion_inputs/cycling_fit_20260929/participant_2'
MODEL=ROOT/'outputs/experiments/cycling_commuter_pilot_20260930/models'
RPATH=ROOT/'assets/densekp_regressor_R.npy'
PAIRS=[('LeftUpLeg','LeftLeg',1,4),('RightUpLeg','RightLeg',2,5),('LeftLeg','LeftFoot',4,7),('RightLeg','RightFoot',5,8),
       ('LeftArm','LeftForeArm',16,18),('RightArm','RightForeArm',17,19),('LeftForeArm','LeftHand',18,20),('RightForeArm','RightHand',19,21)]
SMPLX_NAMES=['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3',
 'left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']
# Native joint, standard SMPL-X 22-joint index, relative importance, status.
CORR=[('Hips',0,.8,'root'),('Spine',3,.45,'approximate'),('Spine1',6,.65,'approximate'),('Spine2',9,.8,'approximate'),
 ('Neck',12,.5,'approximate'),('Head',15,.25,'weak_head'),
 ('LeftShoulder',13,.6,'collar_approximate'),('RightShoulder',14,.6,'collar_approximate'),
 ('LeftArm',16,.9,'shoulder_approximate'),('RightArm',17,.9,'shoulder_approximate'),
 ('LeftForeArm',18,1.,'elbow'),('RightForeArm',19,1.,'elbow'),('LeftHand',20,.7,'wrist'),('RightHand',21,.7,'wrist'),
 ('LeftUpLeg',1,1.,'hip'),('RightUpLeg',2,1.,'hip'),('LeftLeg',4,1.3,'knee'),('RightLeg',5,1.3,'knee'),
 ('LeftFoot',7,1.,'ankle_approximate'),('RightFoot',8,1.,'ankle_approximate'),
 ('LeftToeBase',10,.25,'toe_weak'),('RightToeBase',11,.25,'toe_weak')]
FREE=[0,1,2,3,6,9,12,13,14,15,16,17,18,19,20,21]

def savejson(path,obj):
    def conv(v):
        if isinstance(v,dict):return {str(k):conv(x) for k,x in v.items()}
        if isinstance(v,(list,tuple,np.ndarray)):return [conv(x) for x in v]
        if isinstance(v,np.generic):return conv(v.item())
        if isinstance(v,float) and not math.isfinite(v):return None
        return v
    path.write_text(json.dumps(conv(obj),ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def stats(a):
    x=np.asarray(a).ravel();return dict(mean=float(x.mean()),p95=float(np.quantile(x,.95)),max=float(x.max()))

def rot6(x):
    a=F.normalize(x[...,:3],dim=-1)
    b=F.normalize(x[...,3:]-(a*x[...,3:]).sum(-1,keepdim=True)*a,dim=-1)
    return torch.stack([a,b,torch.cross(a,b,dim=-1)],dim=-1)

def make_rot(six,knee):
    n=six.shape[0];r=torch.eye(3).repeat(n,22,1,1)
    r[:,FREE]=rot6(six)
    angle=.02+2.48*torch.sigmoid(knee)
    c=angle.cos();s=angle.sin();z=torch.zeros_like(c);o=torch.ones_like(c)
    r[:,[4,5]]=torch.stack([o,z,z,z,c,-s,z,s,c],-1).reshape(n,2,3,3)
    return r,angle

def initial(n):
    aa=np.zeros((22,3),np.float32)
    aa[[1,2],0]=-.45;aa[[4,5],0]=.9;aa[3,0]=.2
    aa[16,2]=-1.;aa[17,2]=1.;aa[18,1]=-.8;aa[19,1]=.8
    mat=Rotation.from_rotvec(aa).as_matrix().astype(np.float32)
    six=np.concatenate([mat[FREE,:,0],mat[FREE,:,1]],-1)
    return torch.tensor(np.repeat(six[None],n,0)),torch.full((n,2),float(np.log((.9-.02)/(2.5-.9))))

def resolved(value):
    path=Path(value)
    return path if path.is_absolute() else ROOT/path

def bvh_time_text(path):
    with open(path) as f:
        for line in f:
            if line.startswith('Frame Time:'):
                return line.split(':',1)[1].strip()
    raise ValueError('missing BVH Frame Time')

def estimate_alignment(native,names,rest_abs):
    lateral=np.median((native[:,names.index('LeftUpLeg')]-native[:,names.index('RightUpLeg')])+
                      (native[:,names.index('LeftArm')]-native[:,names.index('RightArm')]),axis=0)
    yaw=-math.atan2(float(lateral[2]),float(lateral[0]));cy,sy=math.cos(yaw),math.sin(yaw)
    basis=np.array([[cy,0,-sy],[0,1,0],[sy,0,cy]])
    assert abs(np.linalg.det(basis)-1)<1e-9
    ratios=[]
    for a,b,u,v in PAIRS:
        source_len=np.median(np.linalg.norm(native[:,names.index(a)]-native[:,names.index(b)],axis=1))
        ratios.append(float(np.linalg.norm(rest_abs[u]-rest_abs[v])/source_len))
    return basis,float(np.median(ratios)),yaw,ratios

def calibrate_p1(base):
    """Freeze one alignment from both participant_1 windows before fitting either."""
    assert not (base/'shared_alignment.json').exists()
    char_path=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_M02_v3/manifest.json'
    char=json.loads(char_path.read_text())
    model=load_smplx_model(resolved(char['model_dir']),gender=char['gender'],num_betas=30,batch_size=1).eval()
    with torch.no_grad():rest=model(betas=torch.tensor(char['betas'],dtype=torch.float32)[None]).joints[0,:22].numpy()
    source=Path('/mnt/data1/yy/home/yy/datasets/cycling_mocap/candidates/1_RB_M42_20230719_PK/1_RB_M42_20230719_PK.bvh')
    joints=json.loads((ROOT/'outputs/motion_inputs/cycling_fit_20260929/participant_1/manifest.json').read_text())['joints']
    windows=[[7560,7656],[9600,9696]];parts=[]
    for a,b in windows:
        rows,dt,total,header=audit.bvh_window(source,a,b-a)
        assert bvh_time_text(source)=='0.0416667' and total==20060 and len(header)>0
        parts.append(audit.native_fk(rows,joints))
    native=np.concatenate(parts);names=[j['name'] for j in joints]
    basis,scale,yaw,ratios=estimate_alignment(native,names,rest)
    savejson(base/'shared_alignment.json',dict(integration_only=True,source_bvh=str(source),source_windows=windows,
      target_character=char['character_id'],character_manifest=str(char_path),basis_native_to_model=basis,
      scale_model_m_per_native_unit=scale,yaw_radians=yaw,length_ratio_samples=ratios,
      length_pairs=[[a,b] for a,b,_,_ in PAIRS],unit_status='inferred, not official',up_axis_status='+Y inferred',
      method='one proper yaw and median of eight native/target limb length ratios over both frozen windows',
      no_per_frame_alignment=True))

def setup(base,freeze):
    char_path=resolved(freeze['character_manifest']);char=json.loads(char_path.read_text());betas=np.asarray(char['betas'],np.float32)
    assert len(betas)==30 and char['gender']=='male'
    model=load_smplx_model(resolved(char['model_dir']),gender=char['gender'],num_betas=30,batch_size=1).eval();model.requires_grad_(False)
    with torch.no_grad():rest_abs=model(betas=torch.tensor(betas)[None]).joints[0,:22].numpy()
    source=resolved(freeze['source_bvh']);a,b=freeze['source_frame_half_open'];n=b-a
    assert n==freeze['asset_frames'] and bvh_time_text(source)==freeze['frame_time_original_decimal']
    clip=resolved(freeze['local_clip'])
    if clip.is_file():clip=clip.parent
    cache_manifest=json.loads((clip/'manifest.json').read_text());joints=cache_manifest['joints']
    rows,dt,total,header=audit.bvh_window(source,a,n)
    assert total==cache_manifest['source_total_frames']
    assert all(j['channels']==['Xposition','Yposition','Zposition','Zrotation','Xrotation','Yrotation'] for j in joints if j['channels'])
    native=audit.native_fk(rows,joints)
    # The local cache covers only the first p1 window; never double-add its source offset.
    ca,cb=cache_manifest['source_frame_range'];fkerr=None
    if ca<=a and b<=cb:
        local_rows,local_dt,local_total,local_header=audit.bvh_window(clip/'source_clip.bvh',a-ca,n)
        assert dt==local_dt and header==local_header and local_total==cb-ca
        np.testing.assert_allclose(rows,local_rows,rtol=0,atol=1e-5)
        with np.load(clip/'clip_native.npz',allow_pickle=False) as z:
            np.testing.assert_array_equal(z['source_frame_ids'][a-ca:b-ca],np.arange(a,b))
            np.testing.assert_allclose(z['channels_native'][a-ca:b-ca],rows,rtol=0,atol=1e-5)
            cached=z['joint_positions_native'][a-ca:b-ca]
        fkerr=float(np.max(np.abs(native-cached)));assert fkerr<1e-7
    names=[j['name'] for j in joints];srcids=np.array([names.index(x[0]) for x in CORR]);dstids=np.array([x[1] for x in CORR]);weights=np.array([x[2] for x in CORR],np.float32)
    raw=native[:,srcids];pelvis=native[:,names.index('Hips')]
    local_basis,local_scale,local_yaw,local_ratios=estimate_alignment(native,names,rest_abs)
    if freeze.get('shared_alignment'):
        shared=json.loads(resolved(freeze['shared_alignment']).read_text())
        assert shared['source_windows']==[[7560,7656],[9600,9696]] and shared['target_character']==char['character_id']
        basis=np.asarray(shared['basis_native_to_model']);scale=float(shared['scale_model_m_per_native_unit']);yaw=float(shared['yaw_radians']);ratios=shared['length_ratio_samples']
    else:basis,scale,yaw,ratios=local_basis,local_scale,local_yaw,local_ratios
    target=(raw-pelvis[:,None])@basis.T*scale
    tick=int(freeze['ticks_per_frame']);timebase=np.asarray(freeze['time_base']);assert timebase.tolist()==[1,10000000]
    np.savez(base/'observations.npz',native_joint_positions=raw.astype(np.float32),native_root=pelvis.astype(np.float32),
             target_joints_m=target.astype(np.float32),source_frame_ids=np.arange(a,b),asset_frame_ids=np.arange(n),
             nominal_source_time_ticks=np.arange(a,b,dtype=np.int64)*tick,nominal_asset_time_ticks=np.arange(n,dtype=np.int64)*tick,
             time_base=timebase,raw_channels=rows.astype(np.float32),basis=basis,scale=np.array(scale))
    savejson(base/'alignment.json',dict(integration_only=True,native_up='+Y inferred',native_unit='unknown; centimetres plausible but unconfirmed',
        formula='(native_joints - native_Hips_each_frame) @ basis.T * one_scale',basis_native_to_model=basis,determinant=np.linalg.det(basis),
        yaw_radians=yaw,scale_model_m_per_native_unit=scale,length_ratio_samples=ratios,length_pairs=[(p[0],p[1]) for p in PAIRS],
        local_window_alignment_diagnostic=dict(yaw_radians=local_yaw,scale=local_scale,length_ratios=local_ratios),
        shared_alignment=freeze.get('shared_alignment'),
        source_root_saved='observations.npz/native_root',asset_root='pelvis-centered; scene owns travel and road yaw',
        fixed_transl=-rest_abs[0],no_per_frame_rotation_or_scale=True,no_foot_grounding=True,
        official_axis_unit_confirmation=False,real_world_accuracy='not established'))
    savejson(base/'joint_mapping.json',dict(integration_only=True,source_joint_names=names,
        correspondences=[dict(source=x[0],source_index=int(srcids[i]),target_joint_index=x[1],target_name=SMPLX_NAMES[x[1]],weight=x[2],status=x[3],constraint='position and selected adjacent segment direction') for i,x in enumerate(CORR)],
        source_channels='6 per non-End joint; local translation channels replace OFFSET; local Zrotation Xrotation Yrotation order',
        native_fk_cached_max_abs_error=fkerr))
    return model,betas,rest_abs,dict(np.load(base/'observations.npz')),srcids,dstids,weights,fkerr

def fit(base,stage,freeze):
    out=base/stage;out.mkdir(exist_ok=False);t0=time.perf_counter();torch.set_num_threads(2)
    model,betas,rest_abs,obs,srcids,dstids,weights,fkerr=setup(base,freeze)
    n=freeze['asset_frames'];ids=np.asarray(freeze['probe_asset_frames']) if stage=='probe' else np.arange(n)
    target=torch.tensor(obs['target_joints_m'][ids],dtype=torch.float32)
    rest=torch.tensor(rest_abs-rest_abs[0],dtype=torch.float32);parents=model.parents[:22]
    wt=torch.tensor(weights)[None];dest=torch.tensor(dstids)
    six,knee=initial(len(ids));prior=rot6(six).detach();six.requires_grad_();knee.requires_grad_()
    opt=torch.optim.Adam([six,knee],lr=.035)
    steps=freeze['budget']['probe_iterations'] if stage=='probe' else freeze['budget']['full_iterations']
    limit=freeze['budget']['probe_wall_seconds'] if stage=='probe' else freeze['budget']['full_wall_seconds']
    # Adjacent directions are derived from the named correspondences, not BVH index order.
    edge_names=[('LeftArm','LeftForeArm'),('RightArm','RightForeArm'),('LeftForeArm','LeftHand'),('RightForeArm','RightHand'),
                ('LeftUpLeg','LeftLeg'),('RightUpLeg','RightLeg'),('LeftLeg','LeftFoot'),('RightLeg','RightFoot'),
                ('Hips','Spine2'),('Spine2','Neck')]
    lookup={x[0]:i for i,x in enumerate(CORR)};edges=np.array([(lookup[a],lookup[b]) for a,b in edge_names])
    tdir=F.normalize(target[:,edges[:,1]]-target[:,edges[:,0]],dim=-1)
    logs=[];start=time.perf_counter()
    savejson(out/'run_config.json',dict(stage=stage,asset_frames=ids,iterations=steps,limit_seconds=limit,cpu_threads=2,device='cpu',
        position_loss='Huber-like with 0.05m scale',direction_edges=edge_names,temporal='SO3 matrix finite differences, no Euler filtering'))
    for it in range(steps):
        if time.perf_counter()-start>limit:raise RuntimeError('fit wall limit; failed version preserved')
        opt.zero_grad();rot,_=make_rot(six,knee)
        joints,_=batch_rigid_transform(rot,rest[None].expand(len(ids),-1,-1),parents)
        pred=joints[:,dest];d2=(pred-target).square().sum(-1)
        data=(.05**2*(torch.sqrt(1+d2/.05**2)-1)*wt).sum()/wt.sum()/len(ids)
        pdir=F.normalize(pred[:,edges[:,1]]-pred[:,edges[:,0]],dim=-1)
        direction=(pdir-tdir).square().sum(-1).mean()
        pose=(rot[:,FREE]-prior).square().mean()
        # Probe frames are spaced apart: regularize only the full consecutive segment.
        temporal=(rot[1:]-rot[:-1]).square().mean() if stage!='probe' else torch.zeros(())
        loss=data+.004*direction+.00035*pose+.02*temporal
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([six,knee],1);opt.step()
        if it==steps//2:
            for g in opt.param_groups:g['lr']=.012
        if it%50==0 or it==steps-1:
            row=dict(iteration=it+1,loss=float(loss.detach()),mean_residual_m=float(torch.sqrt(d2).mean().detach()),seconds=time.perf_counter()-start)
            logs.append(row);print(json.dumps(row),flush=True)
    savejson(out/'optimization_log.json',logs)
    with torch.no_grad():rot,_=make_rot(six,knee)
    mats=rot.numpy();aa=Rotation.from_matrix(mats.reshape(-1,3,3)).as_rotvec().reshape(len(ids),22,3).astype(np.float32)
    params=dict(betas=betas,global_orient=aa[:,0],body_pose=aa[:,1:].reshape(len(ids),63),transl=np.repeat(-rest_abs[0:1],len(ids),0).astype(np.float32),
      jaw_pose=np.zeros((len(ids),3),np.float32),leye_pose=np.zeros((len(ids),3),np.float32),reye_pose=np.zeros((len(ids),3),np.float32),
      left_hand_pose=np.zeros((len(ids),45),np.float32),right_hand_pose=np.zeros((len(ids),45),np.float32),expression=np.zeros((len(ids),10),np.float32),
      local_rotation_matrices=mats,asset_frame_ids=ids,source_frame_ids=obs['source_frame_ids'][ids],
      nominal_asset_time_ticks=obs['nominal_asset_time_ticks'][ids],nominal_source_time_ticks=obs['nominal_source_time_ticks'][ids],time_base=np.asarray(freeze['time_base']),
      root_mode=np.array('pelvis_centered_scene_owns_travel'),integration_only=np.array(True))
    np.savez(out/'parameters.npz',**params)
    verts=[];js=[]
    with torch.no_grad():
        for k in range(0,len(ids),4):
            end=min(k+4,len(ids));kw={key:torch.tensor(params[key][k:end]) for key in ['global_orient','body_pose','transl','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']}
            kw['betas']=torch.tensor(betas[None]).expand(end-k,-1)
            res=model(**kw);verts.append(res.vertices.numpy());js.append(res.joints[:,:22].numpy())
    verts=np.concatenate(verts);js=np.concatenate(js)
    R=np.load(RPATH);assert R.shape==(138,10475)
    dense=compute_densekp_3d(R,verts)
    np.save(out/'vertices.npy',verts);np.save(out/'joints.npy',js);np.save(out/'densekp_3d.npy',dense)
    if not (base/'faces.npy').exists():np.save(base/'faces.npy',model.faces.astype(np.int32))
    err=np.linalg.norm(js[:,dstids]-obs['target_joints_m'][ids],axis=-1)
    np.save(out/'residuals_m.npy',err)
    # Re-read params and regenerate one body exactly; R is a project surface regressor.
    again=np.load(out/'parameters.npz');withheld=0
    with torch.no_grad():
        kw={key:torch.tensor(again[key][withheld:withheld+1]) for key in ['global_orient','body_pose','transl','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']}
        kw['betas']=torch.tensor(again['betas'][None]);check=model(**kw).vertices.numpy()[0]
    body_readback=float(np.max(np.abs(check-verts[0])))
    rread=max(float(np.max(np.abs(R@verts[k]-dense[k]))) for k in range(len(ids)))
    adj=np.matmul(mats[:-1].transpose(0,1,3,2),mats[1:]);degrees=np.rad2deg(np.arccos(np.clip((np.trace(adj,axis1=2,axis2=3)-1)/2,-1,1)))
    move=np.linalg.norm(np.diff(verts,axis=0),axis=2) if len(ids)>1 else np.zeros((1,10475))
    limb_errors=[]
    for a,b in edges:
        pv=js[:,dstids[b]]-js[:,dstids[a]];tv=obs['target_joints_m'][ids,b]-obs['target_joints_m'][ids,a]
        ang=np.rad2deg(np.arccos(np.clip(np.sum(F.normalize(torch.tensor(pv),dim=-1).numpy()*F.normalize(torch.tensor(tv),dim=-1).numpy(),axis=-1),-1,1)))
        limb_errors.append(dict(edge=[CORR[a][0],CORR[b][0]],angle_deg=stats(ang)))
    result=dict(integration_only=True,frames=len(ids),source_frame_half_open=[int(params['source_frame_ids'][0]),int(params['source_frame_ids'][-1]+1)],
      finite=bool(np.isfinite(verts).all() and np.isfinite(dense).all() and all(np.isfinite(x).all() for x in params.values() if x.dtype.kind in 'fi')),
      fixed_shape=True,vertex_count=verts.shape[1],face_count=int(len(model.faces)),shape_count=len(betas),
      native_FK_cache_max_abs_native=fkerr,body_parameter_readback_max_m=body_readback,R138_readback_max_m=rread,
      pelvis_center_max_abs_m=float(np.max(np.abs(js[:,0]))),residual_m=stats(err),per_joint_residual_m={x[0]:stats(err[:,i]) for i,x in enumerate(CORR)},
      segment_direction_error=limb_errors,adjacent_rotation_degrees=stats(degrees),adjacent_vertex_move_m=stats(move),
      largest_rotation_steps=[dict(asset_from=int(ids[i]),asset_to=int(ids[i+1]),joint=int(j),degrees=float(degrees[i,j])) for i,j in np.argwhere(degrees>25)[:100]],
      per_frame_mean_residual_m=err.mean(1),body_min_y_m=[float(verts[:,:,1].min(1).min()),float(verts[:,:,1].min(1).max())],
      no_per_frame_grounding=True,actual_dynamic_human_review='pending_user',real_3D_accuracy='unknown; source BVH-to-target mapping residual only')
    savejson(out/'validation.json',result)
    elapsed=time.perf_counter()-t0
    estimate=elapsed*n/len(ids)*1.5 if stage=='probe' else None
    savejson(out/'resource.json',dict(cpu_only=True,gpu_uuid=None,fit_check_seconds=elapsed,optimizer_seconds=time.perf_counter()-start,
        iterations=steps,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,full_conservative_estimate_seconds=estimate,
        budget_seconds=1800))
    print(json.dumps(dict(stage=stage,validation=str(out/'validation.json'),elapsed=elapsed,estimate=estimate,mean_residual=result['residual_m']['mean'])),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['calibrate-p1','probe','fit'],required=True);p.add_argument('--root',type=Path,default=DEFAULT)
    a=p.parse_args();base=a.root
    if a.stage=='calibrate-p1':calibrate_p1(base);return
    freeze=json.loads((base/'frozen_manifest.json').read_text())
    start,end=freeze['source_frame_half_open'];assert end-start==freeze['asset_frames']
    assert int(freeze['ticks_per_frame'])>0 and len(freeze['probe_asset_frames'])<=5
    if a.stage=='fit':
        probe=json.loads((base/'probe/resource.json').read_text());assert probe['full_conservative_estimate_seconds']<1800
    fit(base,'probe' if a.stage=='probe' else 'fit_attempt_1',freeze)

if __name__=='__main__':main()
