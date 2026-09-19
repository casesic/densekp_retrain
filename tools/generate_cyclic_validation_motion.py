#!/usr/bin/env python3
"""Generate a deterministic cyclic gait articulation for validation only."""
import argparse, json, time
from pathlib import Path
import numpy as np, smplx, torch

def main():
    p=argparse.ArgumentParser(); p.add_argument('--model-dir',type=Path,required=True); p.add_argument('--regressor',type=Path,default=Path('assets/densekp_regressor_R.npy')); p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--frames',type=int,default=32); a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True); st=time.perf_counter()
    t=a.frames; phase=np.arange(t,dtype=np.float32)*2*np.pi/16; s=np.sin(phase); c=np.cos(phase)
    global_orient=np.zeros((t,3),np.float32); body=np.zeros((t,21,3),np.float32)
    # SMPL-X body-pose indices after pelvis: hips 0/1, knees 3/4, ankles 6/7,
    # shoulders 15/16, elbows 17/18. This is a repeatable gait-like stress case.
    body[:,0,0]=np.deg2rad(24)*s; body[:,1,0]=-np.deg2rad(24)*s
    body[:,3,0]=np.deg2rad(38)*np.maximum(0,-s); body[:,4,0]=np.deg2rad(38)*np.maximum(0,s)
    body[:,6,0]=-np.deg2rad(12)*np.maximum(0,-s); body[:,7,0]=-np.deg2rad(12)*np.maximum(0,s)
    body[:,15,0]=-np.deg2rad(18)*s; body[:,16,0]=np.deg2rad(18)*s
    body[:,17,0]=np.deg2rad(12)*(1+c); body[:,18,0]=np.deg2rad(12)*(1-c)
    transl=np.zeros((t,3),np.float32); betas=np.zeros((t,10),np.float32)
    kwargs=dict(gender='neutral',ext='npz',num_betas=10,use_pca=False,flat_hand_mean=True,batch_size=t)
    model=smplx.SMPLX(str(a.model_dir),**kwargs) if (a.model_dir/'SMPLX_NEUTRAL.npz').is_file() else smplx.create(str(a.model_dir),model_type='smplx',**kwargs)
    with torch.no_grad(): vertices=model(global_orient=torch.from_numpy(global_orient),body_pose=torch.from_numpy(body.reshape(t,63)),betas=torch.from_numpy(betas),transl=torch.from_numpy(transl),return_verts=True).vertices.numpy().astype(np.float32)
    faces=np.asarray(model.faces,np.int32); reg=np.load(a.regressor); dense=np.einsum('kv,tvc->tkc',reg,vertices,optimize=True).astype(np.float32)
    if vertices.shape!=(t,10475,3) or dense.shape!=(t,138,3) or not np.isfinite(vertices).all() or not np.isfinite(dense).all(): raise RuntimeError('invalid generated geometry')
    np.save(a.output_dir/'vertices.npy',vertices); np.save(a.output_dir/'faces.npy',faces); np.save(a.output_dir/'densekp_3d.npy',dense)
    np.savez(a.output_dir/'motion_params.npz',global_orient=global_orient,body_pose=body.reshape(t,63),betas=betas[:1],transl=transl,fps=np.float32(16))
    report={'artifact':'p0_procedural_smplx_geometry','avatar_id':'smplx_neutral_zero_betas','motion_id':'procedural_cyclic_gait_v1','training_data':False,'deterministic':True,'fallback_reason':'No second local real locomotion motion was available','frames':t,'fps':16.0,'cycles':t/16,'vertices_shape':list(vertices.shape),'faces_shape':list(faces.shape),'densekp_3d_shape':list(dense.shape),'model_dir':str(a.model_dir.resolve()),'regressor':str(a.regressor.resolve()),'generation_seconds':time.perf_counter()-st}
    (a.output_dir/'geometry_manifest.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
