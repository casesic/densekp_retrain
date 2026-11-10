#!/usr/bin/env python3
"""Re-evaluate native SMPL-X, export pelvis-centered asset, R138 and temporal diagnostics."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','2')
import argparse,json,time,csv
from pathlib import Path
import numpy as np,torch,smplx
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/experiments/mtb_video_smplx_pilot_a2_20261009'
def evaluate(model,p,n):
 vv=[];jj=[]
 with torch.no_grad():
  for start in range(0,n,8):
   batch={k:torch.from_numpy(v[start:start+8].copy()).float() for k,v in p.items()}
   o=model(**batch);vv.append(o.vertices.numpy());jj.append(o.joints.numpy())
 return np.concatenate(vv),np.concatenate(jj)
def main():
 a=argparse.ArgumentParser();a.add_argument('--fit',default='fit_attempt_1');args=a.parse_args();fit=BASE/args.fit;dest=fit/'readback';dest.mkdir(exist_ok=False);started=time.monotonic();torch.set_num_threads(2)
 f=json.loads((BASE/'frozen_manifest.json').read_text());time_rows=json.loads((BASE/'decoded_timeline.json').read_text())['frames'];z=dict(np.load(fit/'parameters_camera.npz'));ids=z.pop('asset_frames');n=len(ids);assert n==120
 p={k:v.astype(np.float32) for k,v in z.items()};p.update({k:np.zeros((n,3),np.float32) for k in ['jaw_pose','leye_pose','reye_pose'] if k not in p});assert all(np.isfinite(v).all() for v in p.values())
 assert np.array_equal(p['betas'],np.broadcast_to(p['betas'][0],p['betas'].shape));frozen=np.load(BASE/'probe_smplx/parameters_camera.npz')['betas'][0];assert np.array_equal(p['betas'][0],frozen)
 m=smplx.SMPLX(f['smplx_model'],gender='male',num_betas=10,use_pca=False,flat_hand_mean=True,batch_size=1)
 cam_v,cam_j=evaluate(m,p,n);original=np.load(fit/'vertices_camera.npy');camera_readback_error=float(np.max(np.abs(cam_v-original)));assert camera_readback_error<2e-5
 A=np.diag([1.,-1.,-1.]);asset={k:v.copy() for k,v in p.items()};asset['global_orient']=Rotation.from_matrix(A@Rotation.from_rotvec(p['global_orient']).as_matrix()).as_rotvec().astype(np.float32)
 with torch.no_grad():rest=m(betas=torch.from_numpy(p['betas'][:1]),body_pose=torch.zeros(1,63),global_orient=torch.zeros(1,3),transl=torch.zeros(1,3));J=rest.joints[0,0].numpy()
 asset['transl']=np.broadcast_to(-J,(n,3)).copy();v,j=evaluate(m,asset,n);expected=(cam_v-cam_j[:,0:1])@A.T;asset_space_error=float(np.max(np.abs(v-expected)));assert asset_space_error<2e-5
 R=np.load(f['regressor']);assert R.shape==(138,10475);dense=np.einsum('kv,tvc->tkc',R,v,optimize=True).astype(np.float32)
 points=np.array([row['source_pts'] for row in time_rows],np.int64);sourceids=np.array([row['source_decoded_frame'] for row in time_rows]);assert np.all(np.diff(points)==1001) and np.all(np.diff(sourceids)==1)
 store={k:(vv[0] if k=='betas' else vv) for k,vv in asset.items()};np.savez_compressed(fit/'parameters_asset.npz',**store,asset_frames=ids,source_decoded_frames=sourceids,source_pts=points,time_base=np.array([1,30000]),camera_to_asset_rotation=A.astype(np.float32))
 np.save(fit/'vertices_asset.npy',v);np.save(fit/'joints_asset.npy',j);np.save(fit/'densekp138_asset.npy',dense)
 back=dict(np.load(fit/'parameters_asset.npz'));back={k:(np.broadcast_to(back[k],(n,10)).copy() if k=='betas' else back[k])for k in asset};vr,jr=evaluate(m,back,n);dense2=np.einsum('kv,tvc->tkc',R,vr,optimize=True).astype(np.float32);readback=float(np.max(np.abs(v-vr)));kp_error=float(np.max(np.abs(dense-dense2)));assert readback<2e-5 and kp_error<2e-6
 allrot=np.concatenate([asset['global_orient'][:,None,:],asset['body_pose'].reshape(n,21,3)],axis=1);rot=Rotation.from_rotvec(allrot.reshape(-1,3)).as_matrix().reshape(n,22,3,3);rel=np.swapaxes(rot[:-1],-1,-2)@rot[1:];ang=np.rad2deg(np.arccos(np.clip((np.trace(rel,axis1=-1,axis2=-2)-1)/2,-1,1)));pos=np.linalg.norm(np.diff(j[:,:22],axis=0),axis=-1);idx=np.unravel_index(ang.argmax(),ang.shape)
 def knee(j,a,b,c):
  u=j[:,a]-j[:,b];w=j[:,c]-j[:,b];return np.rad2deg(np.arccos(np.clip(np.sum(u*w,1)/np.maximum(np.linalg.norm(u,axis=1)*np.linalg.norm(w,axis=1),1e-8),-1,1)))
 kl=knee(j,1,4,7);kr=knee(j,2,5,8)
 diag={'left_knee_internal_angle_deg':kl.tolist(),'right_knee_internal_angle_deg':kr.tolist(),'adjacent_local_rotation_deg':ang.tolist(),'adjacent_joint_displacement_m':pos.tolist()};(dest/'temporal.json').write_text(json.dumps(diag))
 stats={'frames':n,'shape_fixed':True,'shape_equals_probe':True,'all_finite':bool(np.isfinite(v).all() and np.isfinite(dense).all()),'vertices':10475,'faces':len(m.faces),'faces_equal_existing_standard':bool(np.array_equal(m.faces,np.load(BASE/'smplx_faces.npy'))),'camera_readback_max_abs_m':camera_readback_error,'asset_transform_max_abs_m':asset_space_error,'asset_parameter_readback_max_abs_m':readback,'R138_readback_max_abs_m':kp_error,'pelvis_max_abs_m':float(np.abs(j[:,0]).max()),'max_adjacent_rotation_deg':float(ang.max()),'max_rotation_location':{'asset_transition':[int(idx[0]),int(idx[0]+1)],'source_transition':[int(sourceids[idx[0]]),int(sourceids[idx[0]+1])],'joint_smplx':int(idx[1])},'max_adjacent_joint_displacement_m':float(pos.max()),'rotation_events_gt20deg':np.argwhere(ang>20).tolist(),'knee_angle_ranges_deg':[float(np.ptp(kl)),float(np.ptp(kr))],'knee_correlation':float(np.corrcoef(kl,kr)[0,1]),'min_y_range_m':[float(v[:,:,1].min(1).min()),float(v[:,:,1].min(1).max())],'constant_translation_m':(-J).tolist(),'camera_to_asset_rotation':A.tolist(),'per_frame_alignment_grounding':False,'seconds':time.monotonic()-started,'dynamic_human_review':'pending','source_visibility_note':'head out of source frame and far side occluded; values are model estimates, not observed GT'}
 (dest/'validation.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2));print(json.dumps(stats,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
