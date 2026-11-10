#!/usr/bin/env python3
"""Isolated, budgeted APD 6526 pose adaptation. Never edits production interfaces."""
import os
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '')
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import json
import resource
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation
from smplx.lbs import batch_rigid_transform

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.body.geometry import load_smplx_model, compute_densekp_3d

NAMES=['head','thorax','lshoulder','rshoulder','lelbow','relbow','lwrist','rwrist','lhip','rhip','lknee','rknee','lfoot','rfoot']
MAP=[15,12,16,17,18,19,20,21,1,2,4,5,7,8]
WEIGHTS=[.3,.5,1,1,.8,.8,.45,.45,1,1,1,1,.3,.3]
EDGES=[(2,4),(4,6),(3,5),(5,7),(8,10),(10,12),(9,11),(11,13),(2,8),(3,9)]
FREE=[0,1,2,3,6,9,13,14,16,17,18,19]
DEFAULT=ROOT/'outputs/experiments/apd_6526_smplx_pilot_20261008'

def dump(path,data):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(a) for k,a in v.items()}
        if isinstance(v,(list,tuple)):return [clean(a) for a in v]
        if isinstance(v,np.ndarray):return clean(v.tolist())
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    path.write_text(json.dumps(clean(data),indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def stat(a):
    a=np.asarray(a);a=a[np.isfinite(a)]
    return dict(mean=float(a.mean()),p95=float(np.quantile(a,.95)),max=float(a.max()),min=float(a.min()))

def r6(x):
    a=F.normalize(x[...,:3],dim=-1)
    b=F.normalize(x[...,3:]-(a*x[...,3:]).sum(-1,keepdim=True)*a,dim=-1)
    return torch.stack([a,b,torch.cross(a,b,dim=-1)],-1)

def initial(n):
    aa=np.zeros((22,3),np.float32)
    aa[[1,2],0]=-.55;aa[[4,5],0]=1.0;aa[3,0]=.12
    aa[16,2]=-1.15;aa[17,2]=1.15;aa[18,1]=-1.;aa[19,1]=1.
    mat=Rotation.from_rotvec(aa).as_matrix().astype(np.float32)
    six=np.concatenate([mat[FREE,:,0],mat[FREE,:,1]],-1)
    return torch.tensor(np.repeat(six[None],n,0)),torch.full((n,2),float(np.log((1-.02)/(2.5-1))))

def rotations(six,knee):
    n=len(six);rot=torch.eye(3).repeat(n,22,1,1)
    rot[:,FREE]=r6(six)
    angle=.02+2.48*torch.sigmoid(knee)
    c=angle.cos();s=angle.sin();one=torch.ones_like(c);zero=torch.zeros_like(c)
    rx=torch.stack([one,zero,zero,zero,c,-s,zero,s,c],-1).reshape(n,2,3,3)
    rot[:,[4,5]]=rx
    return rot,angle

def fk(rot,rest,parents):
    joints,tf=batch_rigid_transform(rot,rest[None].expand(len(rot),-1,-1),parents)
    mapped=joints[:,MAP].clone()
    mapped[:,0]=mapped[:,0]+tf[:,15,:3,:3]@torch.tensor([0.,.08,0.])
    return mapped,joints

def prepare(base,model,freeze,rest_abs):
    # All decisions written before any optimizer step.
    dest=base/'alignment.json'
    if dest.exists():return
    d=json.loads(Path(freeze['source']).read_text());ids=np.arange(*freeze['source_frame_interval_half_open'])
    raw=np.stack([np.asarray(d['pose3d'][k],np.float64)[ids] for k in NAMES],1)
    hip=(raw[:,8]+raw[:,9])/2
    x=np.median((raw[:,8]-raw[:,9])+(raw[:,2]-raw[:,3]),axis=0);x[2]=0;x/=np.linalg.norm(x)
    up=np.array([0.,0.,1.]);front=np.cross(x,up);basis=np.stack([x,up,front])
    rest=rest_abs-rest_abs[0];pairs=[(2,4),(3,5),(8,10),(9,11),(2,8),(3,9)]
    ratios=[]
    for a,b in pairs:
        ratios.append(float(np.linalg.norm(rest[MAP[a]]-rest[MAP[b]])/np.median(np.linalg.norm(raw[:,a]-raw[:,b],axis=1))))
    scale=float(np.median(ratios));offset=(rest[1]+rest[2])/2
    target=(raw-hip[:,None])@basis.T*scale+offset
    np.savez(base/'observations.npz',raw_apd=raw,valid=np.isfinite(raw).all(2),source_hip_midpoint_removed=hip,
             aligned_target_m=target,source_frame_ids=ids,timestamps=np.array(freeze['timestamps'],np.int64),
             clip_seconds=np.array(freeze['clip_seconds']),basis=basis,scale=np.array(scale),constant_offset_m=offset)
    dump(dest,dict(integration_only=True,formula='target=(raw-source_hip_midpoint_per_frame) @ basis.T * scalar_scale + constant_rest_hip_offset',
         basis_raw_to_model=basis,basis_determinant=np.linalg.det(basis),scale_model_m_per_raw_unit=scale,
         scale_ratios=ratios,scale_pairs=[[NAMES[a],NAMES[b]] for a,b in pairs],constant_offset_m=offset,
         raw_up_assumption='+rawZ -> model+Y, anatomical named left->model+X; cross product->model+Z',
         official_coordinate_confirmation=False,scope='synthetic adaptation convention; no real world/camera/metric identity recovery',
         model_rest_pelvis_m=rest_abs[0],fixed_transl_m=-rest_abs[0],no_per_frame_similarity=True,
         root_residual='zero at pelvis; original source hip translation saved separately, no source trajectory applied',
         head_virtual_offset_local_m=[0,.08,0],head_mapping='approximate head landmark; head joint15+0.08 along its localY, weight0.3',
         foot_mapping='unknown APD foot approximated by SMPL-X ankle7/8, weak weight0.3, no ankle/toe GT claim'))
    dump(base/'joint_mapping.json',dict(integration_only=True,joints=[dict(apd=k,smplx_joint=j,weight=w,
          status='weak_approximation' if k in ['head','thorax','lfoot','rfoot'] else 'named_approximate_correspondence') for k,j,w in zip(NAMES,MAP,WEIGHTS)],
          pose2d_uncertainty_used=False,missing_policy='finite3D mask; no zero observations or confidence interpretation'))
    dump(base/'model_reference.json',dict(integration_only=True,path=freeze['model_file'],gender=freeze['gender'],num_betas=30,
         vertex_count=int(model.v_template.shape[0]),faces=int(len(model.faces)),model_buffers=dict(shapedirs=list(model.shapedirs.shape),J_regressor=list(model.J_regressor.shape)),
         fixed_betas=freeze['betas'],torch_version=torch.__version__,device='cpu',model_license='https://smpl-x.is.tue.mpg.de/modellicense.html',
         license_basis='existing licensed local research model expressly authorized by user; internal non-commercial integration experiment, no redistribution of model or textures',
         character_manifest=freeze['character_manifest'],source_person_reconstructed=False))

def export(base,out,model,freeze,rot,rest_abs,observation,chosen):
    aa=Rotation.from_matrix(rot.reshape(-1,3,3)).as_rotvec().reshape(len(rot),22,3).astype(np.float32)
    params=dict(betas=np.asarray(freeze['betas'],np.float32),global_orient=aa[:,0],body_pose=aa[:,1:].reshape(len(rot),63),
        transl=np.repeat(-rest_abs[0:1],len(rot),0).astype(np.float32),jaw_pose=np.zeros((len(rot),3),np.float32),
        leye_pose=np.zeros((len(rot),3),np.float32),reye_pose=np.zeros((len(rot),3),np.float32),
        left_hand_pose=np.zeros((len(rot),45),np.float32),right_hand_pose=np.zeros((len(rot),45),np.float32),
        expression=np.zeros((len(rot),10),np.float32),local_rotation_matrices=rot,
        source_frame_ids=observation['source_frame_ids'][chosen],timestamps=observation['timestamps'][chosen],
        clip_seconds=observation['clip_seconds'][chosen],gender=np.array('male'),integration_only=np.array(True),
        root_mode=np.array('fixed_pelvis_in_place'),shape_fixed=np.array(True))
    np.savez(out/'parameters.npz',**params)
    verts=[];joints=[]
    with torch.no_grad():
        for start in range(0,len(rot),4):
            keys=['global_orient','body_pose','transl','jaw_pose','leye_pose','reye_pose','left_hand_pose','right_hand_pose','expression']
            kw={k:torch.tensor(params[k][start:start+4]) for k in keys}
            kw['betas']=torch.tensor(params['betas'][None]).expand(len(kw['body_pose']),-1)
            res=model(**kw);verts.append(res.vertices.numpy());joints.append(res.joints[:,:22].numpy())
    verts=np.concatenate(verts);joints=np.concatenate(joints)
    mapped,fastj=fk(torch.tensor(rot),torch.tensor(rest_abs-rest_abs[0]),model.parents[:22])
    fkerror=float(np.max(np.abs(joints-fastj.numpy())))
    assert fkerror<2e-5, fkerror
    assert verts.shape==(len(rot),10475,3) and np.isfinite(verts).all()
    ref=Path(json.loads(Path(freeze['character_manifest']).read_text())['reference']['geometry'])
    assert np.array_equal(model.faces,np.load(ref/'faces.npy')), 'target topology differs from original character reference'
    R=np.load(freeze['regressor']);assert R.shape==(138,10475) and np.isfinite(R).all()
    dense=compute_densekp_3d(R,verts)
    np.save(out/'vertices.npy',verts);np.save(out/'densekp_3d.npy',dense)
    np.save(out/'joints.npy',joints);np.save(out/'mapped_joints.npy',mapped.numpy())
    if not (base/'faces.npy').exists():np.save(base/'faces.npy',model.faces.astype(np.int32))
    errors=np.linalg.norm(mapped.numpy()-observation['aligned_target_m'][chosen],axis=-1)
    valid=observation['valid'][chosen];np.save(out/'residuals_m.npy',errors)
    reloadverts=np.load(out/'vertices.npy',mmap_mode='r');reloaddense=np.load(out/'densekp_3d.npy')
    reread=max(float(np.max(np.abs(R@reloadverts[i]-reloaddense[i]))) for i in range(len(rot)))
    assert reread<2e-6,reread
    relative=np.matmul(rot[:-1].transpose(0,1,3,2),rot[1:])
    angle=np.rad2deg(np.arccos(np.clip((np.trace(relative,axis1=2,axis2=3)-1)/2,-1,1)))
    movement=np.linalg.norm(np.diff(verts,axis=0),axis=-1)
    result=dict(integration_only=True,frames=len(rot),parameter_dimensions={k:list(v.shape) for k,v in params.items()},
        all_parameters_finite=all(np.isfinite(v).all() for v in params.values() if v.dtype.kind in 'fiu'),
        fixed_shape=True,topology_equals_reference=True,vertex_order='unchanged standard SMPL-X model order',
        residual_m=stat(errors[valid]),per_joint_residual_m={k:stat(errors[:,i][valid[:,i]]) for i,k in enumerate(NAMES)},
        fk_native_max_error_m=fkerror,R_readback_max_error_m=reread,
        adjacent_rotation_degrees=stat(angle),adjacent_vertex_displacement_m=stat(movement),
        pelvis_fixed_origin_max_error_m=float(np.abs(joints[:,0]).max()),
        min_body_y_range_m=[float(verts[:,:,1].min(1).min()),float(verts[:,:,1].min(1).max())],
        suspicious_rotation_changes=[dict(source_from=int(params['source_frame_ids'][i]),source_to=int(params['source_frame_ids'][i+1]),joint=int(j),degrees=float(angle[i,j])) for i,j in np.argwhere(angle>25)],
        source_frames_high_residual=[int(x) for x in params['source_frame_ids'][errors.mean(1)>.10]],
        actual_dynamic_human_review='pending_user',self_intersection='not certified; inspect full body previews, no zero-intersection requirement')
    dump(out/'validation.json',result)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=DEFAULT);ap.add_argument('--stage',choices=['probe','full','correction'],required=True)
    args=ap.parse_args();base=args.root;freeze=json.loads((base/'frozen_manifest.json').read_text())
    out=base/('probe' if args.stage=='probe' else 'fit_attempt_2' if args.stage=='correction' else 'fit_attempt_1');out.mkdir(exist_ok=False)
    assert (base/'gpu_before.txt').exists()
    torch.set_num_threads(2);torch.manual_seed(6526);beg=time.perf_counter()
    model=load_smplx_model(Path(freeze['model_dir']),gender='male',num_betas=30,batch_size=1).eval();model.requires_grad_(False)
    with torch.no_grad():rest_abs=model(betas=torch.tensor(freeze['betas'],dtype=torch.float32)[None]).joints[0,:22].numpy()
    prepare(base,model,freeze,rest_abs)
    obs=dict(np.load(base/'observations.npz',allow_pickle=False))
    chosen=np.array([int(np.flatnonzero(obs['source_frame_ids']==v)[0]) for v in freeze['probe_source_frames']]) if args.stage=='probe' else np.arange(76)
    if args.stage!='probe':
        p=json.loads((base/'probe'/'resource.json').read_text())
        assert p['full_seconds_conservative_estimate']<1800
        assert (base/'probe'/'validation.json').exists()
    steps=freeze['budget']['probe_iterations' if args.stage=='probe' else 'full_iterations']
    limit=freeze['budget']['probe_wall_limit_seconds' if args.stage=='probe' else 'full_wall_limit_seconds']
    config=dict(freeze['optimization'])
    if args.stage=='correction':config.update(json.loads((base/'correction_config.json').read_text())['changes'])
    dump(out/'run_config.json',dict(integration_only=True,stage=args.stage,source_frames=obs['source_frame_ids'][chosen],iterations=steps,wall_limit_seconds=limit,device='cpu',gpu_uuid=None,threads=2,optimization=config))
    target=torch.tensor(obs['aligned_target_m'][chosen],dtype=torch.float32)
    mask=torch.tensor(obs['valid'][chosen]);target=torch.where(mask[...,None],target,torch.zeros_like(target))
    weights=torch.tensor(WEIGHTS)[None]*mask;rest=torch.tensor(rest_abs-rest_abs[0]);parents=model.parents[:22]
    six,knee=initial(len(chosen));prior=r6(six).detach()
    if args.stage=='correction':
        prev=np.load(base/'fit_attempt_1'/'optimizer_final.npz');six=torch.tensor(prev['six']);knee=torch.tensor(prev['knee'])
    six.requires_grad_();knee.requires_grad_()
    optim=torch.optim.Adam([six,knee],lr=.035)
    target_dir=torch.stack([target[:,b]-target[:,a] for a,b in EDGES],1)
    edge_valid=torch.stack([mask[:,a]&mask[:,b] for a,b in EDGES],1)
    # Source knee angle prior is a geometric observation, with fixed positive bend direction.
    ref_angles=[]
    for a,b,c in [(8,10,12),(9,11,13)]:
        u=F.normalize(target[:,a]-target[:,b],dim=-1);v=F.normalize(target[:,c]-target[:,b],dim=-1)
        ref_angles.append(torch.pi-torch.acos((u*v).sum(-1).clamp(-.9999,.9999)))
    knee_ref=torch.stack(ref_angles,-1).detach()
    times=torch.tensor(obs['clip_seconds'][chosen],dtype=torch.float32);interval=(times[1:]-times[:-1])/.04
    logs=[];start=time.perf_counter();initial_residual=None
    for it in range(steps):
        if time.perf_counter()-start>limit:raise RuntimeError('Optimization wall budget reached; preserve output and stop')
        optim.zero_grad();rot,ka=rotations(six,knee);pred,_=fk(rot,rest,parents)
        r2=(pred-target).square().sum(-1)
        data=(.05**2*(torch.sqrt(1+r2/.05**2)-1)*weights).sum()/weights.sum()
        pdir=torch.stack([pred[:,b]-pred[:,a] for a,b in EDGES],1)
        dl=((F.normalize(pdir,dim=-1)-F.normalize(target_dir,dim=-1)).square().sum(-1)*edge_valid).sum()/edge_valid.sum()
        pr=(rot[:,FREE]-prior).square().mean()
        velocity=(rot[1:]-rot[:-1])/interval[:,None,None,None]
        vl=velocity.square().mean()
        ac=((velocity[1:]-velocity[:-1])/((interval[1:]+interval[:-1])/2)[:,None,None,None]).square().mean()
        loss=data+.003*dl+.0003*pr+config['rotation_velocity_weight']*vl+config['rotation_acceleration_weight']*ac+.0001*(ka-knee_ref).square().mean()
        if args.stage=='correction':
            jvel=(pred[1:]-pred[:-1])/interval[:,None,None]
            jacc=(jvel[1:]-jvel[:-1])/((interval[1:]+interval[:-1])/2)[:,None,None]
            loss=loss+config['joint_velocity_weight']*jvel.square().mean()+config['joint_acceleration_weight']*jacc.square().mean()
        if initial_residual is None:initial_residual=float(torch.sqrt(r2).mean())
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([six,knee],1.)
        optim.step()
        if it==steps//2:
            for group in optim.param_groups:group['lr']=.012
        if it%25==0 or it==steps-1:
            row=dict(iteration=it+1,loss=float(loss.detach()),position=float(data.detach()),direction=float(dl.detach()),mean_joint_residual_m=float(torch.sqrt(r2).mean().detach()),seconds=time.perf_counter()-start)
            logs.append(row);print(json.dumps(row),flush=True)
    elapsed=time.perf_counter()-start
    rot,ka=rotations(six.detach(),knee.detach());rot=rot.numpy()
    np.savez(out/'optimizer_final.npz',six=six.detach().numpy(),knee=knee.detach().numpy())
    dump(out/'optimization_log.json',logs)
    result=export(base,out,model,freeze,rot,rest_abs,obs,chosen)
    cost=elapsed/steps*freeze['budget']['full_iterations']*(76/len(chosen))*2+60
    dump(out/'resource.json',dict(integration_only=True,stage=args.stage,iterations_executed=steps,optimizer='Adam pose parameters only; not DenseKP training',
         optimization_seconds=elapsed,total_seconds=time.perf_counter()-beg,cpu_threads=2,device='cpu',gpu_uuid=None,
         peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,full_seconds_conservative_estimate=cost,
         estimate_formula='measured seconds/iteration *500 *76/probe_frames *2 +60 export/import margin',initial_mean_residual_m=initial_residual,
         final_mean_residual_m=result['residual_m']['mean']))
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
