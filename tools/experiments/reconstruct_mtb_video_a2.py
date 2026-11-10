#!/usr/bin/env python3
"""Bounded local MTB MHR reconstruction and fixed-shape SMPL-X adaptation; external code read-only."""
import os,sys,json,time,subprocess,argparse
from pathlib import Path
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/mtb_video_smplx_pilot_a2_20261009'
os.environ.setdefault('CUDA_VISIBLE_DEVICES','GPU-6f046d56-acbe-d49b-774a-9d463e768cac')
os.environ.setdefault('CUDA_CACHE_PATH',str(BASE/'cuda_cache'))
os.environ.setdefault('XDG_CACHE_HOME',str(BASE/'cache'))
EXT=Path('/mnt/hdd/yuyue/home/yuyue/Projects/pano-stitch/tools')
CPP=EXT/'SAM3DBody-cpp';CONV=EXT/'MHR/tools/mhr_smpl_conversion'
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2))
def reconstruct(probe):
 f=json.loads((BASE/'frozen_manifest.json').read_text());ids=f['probe_asset_frames'] if probe else list(range(f['frames']))
 dest=BASE/'probe' if probe else BASE/'raw_mhr';dest.mkdir(exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='GPU-6f046d56-acbe-d49b-774a-9d463e768cac',CUDA_CACHE_PATH=str(BASE/'cuda_cache'),XDG_CACHE_HOME=str(BASE/'cache'))
 env['LD_LIBRARY_PATH']='/home/yuyue/anaconda3/envs/sam3dbody/lib/python3.11/site-packages/nvidia/cudnn/lib:/home/yuyue/anaconda3/envs/sam3dbody/lib/python3.11/site-packages/nvidia/cublas/lib:/usr/local/cuda/lib64'
 runs=[]
 for i in ids:
  if (BASE/f'probe/{i:04d}/person_000/metadata.json').exists():continue
  assert not (dest/f'{i:04d}').exists()
  cmd=[str(CPP/'build/fast_sam_3dbody_run'),'--onnx-dir',str(CPP/'onnx'),'--gguf',str(CPP/'onnx/pipeline.gguf'),'--backbone','backbone.onnx','--cuda','0','--headless','--from',str(BASE/f'frames/{i:04d}.png'),'--bbox',','.join(map(str,f['bbox_xyxy'])),'--export-dir',str(dest)]
  t=time.monotonic()
  with (dest/f'frame_{i:04d}.log').open('w') as log:p=subprocess.run(cmd,cwd=BASE,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=120)
  runs.append({'asset_frame':i,'seconds':time.monotonic()-t,'exit_code':p.returncode});write(dest/'execution.json',runs)
  print(runs[-1],flush=True)
  if p.returncode:raise RuntimeError('MHR inference failed, preserved output')
  if sum(x['seconds'] for x in runs)>1800:raise RuntimeError('MHR inference subbudget exceeded')
def convert(probe):
 import numpy as np,torch,smplx
 from types import SimpleNamespace
 torch.set_num_threads(2);torch.manual_seed(20261009)
 sys.path.insert(0,str(CONV))
 # Vendor relative resource paths are redirected in memory, no chdir/write to vendor.
 import file_assets
 for k,v in vars(file_assets).copy().items():
  if isinstance(v,str) and v.startswith('./assets/'):setattr(file_assets,k,str(CONV/v))
 from conversion import Conversion
 from pytorch_fitting import PyTorchSMPLFitting
 f=json.loads((BASE/'frozen_manifest.json').read_text());ids=f['probe_asset_frames'] if probe else list(range(f['frames']));n=len(ids)
 dest=BASE/('probe_smplx' if probe else 'fit_attempt_1');dest.mkdir(exist_ok=False)
 device='cuda:0';model=smplx.SMPLX(f['smplx_model'],gender='male',num_betas=10,use_pca=False,flat_hand_mean=True,batch_size=1).to(device)
 class MeshAdapter:
  character=SimpleNamespace(mesh=SimpleNamespace(faces=np.load(BASE/'mhr_faces.npy').astype(np.int64)))
  def to(self,device):return self
 conv=Conversion(MeshAdapter(),model,method='pytorch',batch_size=10)
 verts=[];cameras=[];kp=[]
 for i in ids:
  dr=BASE/f'probe/{i:04d}/person_000'
  if not dr.exists():dr=BASE/f'raw_mhr/{i:04d}/person_000'
  m=json.loads((dr/'metadata.json').read_text());v=np.fromfile(dr/'vertices.f32',np.float32).reshape(-1,3);cam=np.array(m['pred_cam_t'],np.float32);verts.append(v+cam);cameras.append([m['fx'],m['fy'],m['cx'],m['cy'],*cam]);kp.append(np.fromfile(dr/'keypoints_3d.f32',np.float32).reshape(70,3)+cam)
 verts=np.stack(verts);assert np.isfinite(verts).all()
 target=conv._compute_target_vertices(torch.tensor(verts*100,device=device),'mhr2smpl')
 # Official mapping returns meters for mhr2smpl.
 np.save(dest/'mapped_target.npy',target.detach().cpu().numpy());np.save(dest/'source_camera.npy',np.array(cameras));np.save(dest/'mhr_keypoints_camera.npy',np.array(kp))
 solver=PyTorchSMPLFitting(model,conv._smpl_edges,'smplx',45,device=device,batch_size=10)
 t=time.monotonic();known={}
 if not probe:known['betas']=torch.tensor(np.load(BASE/'probe_smplx/parameters_camera.npz')['betas'][:1],device=device)
 params=solver.fit(target.clone(),single_identity=True,is_tracking=False,known_parameters=known)
 torch.cuda.synchronize();elapsed=time.monotonic()-t
 params={k:v.detach() for k,v in params.items()};arrays={k:v.cpu().numpy() for k,v in params.items()};np.savez_compressed(dest/'parameters_camera.npz',**arrays,asset_frames=ids)
 assert np.array_equal(arrays['betas'],np.broadcast_to(arrays['betas'][0],arrays['betas'].shape))
 body=[];joints=[]
 with torch.no_grad():
  for i in range(n):
   out=model(**{k:v[i:i+1] for k,v in params.items()});body.append(out.vertices[0].cpu().numpy());joints.append(out.joints[0].cpu().numpy())
 body=np.stack(body);np.save(dest/'vertices_camera.npy',body);np.save(dest/'joints_camera.npy',np.stack(joints));np.save(BASE/'smplx_faces.npy',model.faces)
 err=np.linalg.norm(body-target.detach().cpu().numpy(),axis=-1)
 write(dest/'conversion.json',{'seconds':elapsed,'frames':n,'single_identity':True,'tracking_flag':False,'known_fixed_beta_from_probe':not probe,'iterations_per_batch':{'coarse':[40,80,40],'fine':300},'batch_size':10,'error_mean_m':float(err.mean()),'error_p95_m':float(np.percentile(err,95)),'error_max_m':float(err.max()),'error_mean_per_frame_m':err.mean(1).tolist(),'not_gt_error':True,'shape_fixed':True,'model':f['smplx_model'],'device_uuid':os.environ.get('CUDA_VISIBLE_DEVICES')})
 print('conversion_done',elapsed,float(err.mean()),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['reconstruct','convert']);p.add_argument('--probe',action='store_true');a=p.parse_args();(reconstruct if a.stage=='reconstruct' else convert)(a.probe)
