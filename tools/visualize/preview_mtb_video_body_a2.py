#!/usr/bin/env python3
"""CPU two fixed view/source projection diagnostic for the frozen MTB body pilot."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,json,time,subprocess
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'outputs/experiments/mtb_video_smplx_pilot_a2_20261009'
EDGES=[(0,1),(0,2),(0,3),(1,4),(2,5),(4,7),(5,8),(7,10),(8,11),(3,6),(6,9),(9,12),(12,15),(9,13),(9,14),(13,16),(14,17),(16,18),(17,19),(18,20),(19,21)]
def main():
 p=argparse.ArgumentParser();p.add_argument('--fit',default='probe_smplx');a=p.parse_args();fit=BASE/a.fit;out=fit/'preview';out.mkdir(exist_ok=False);t=time.monotonic()
 body=np.load(fit/'vertices_camera.npy');joints=np.load(fit/'joints_camera.npy');params=np.load(fit/'parameters_camera.npz');ids=params['asset_frames'];faces=np.load(BASE/'smplx_faces.npy');cams=np.load(fit/'source_camera.npy');timeline=json.loads((BASE/'decoded_timeline.json').read_text())['frames'];A=np.array([1,-1,-1]);loc=(body-joints[:,0:1])*A;js=(joints-joints[:,0:1])*A
 # One global fixed scale and center across the whole segment and both views.
 scale=410/max(np.ptp(loc[:,:,1]),1.6);cy=(loc[:,:,1].min()+loc[:,:,1].max())/2;W,H=1440,600
 font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
 for k,i in enumerate(ids):
  im=Image.new('RGB',(W,H),'#f7f7fa');d=ImageDraw.Draw(im);row=timeline[int(i)];d.text((12,8),f'Bikefit MTB | asset {i} | source decoded frame {row["source_decoded_frame"]} | source {row["source_seconds"]:.6f}s',fill='#101010',font=font)
  d.text((12,30),'INTERNAL integration_only | estimated head/far side | fixed shape | no road root | source license unknown',fill='#333333',font=font)
  src=Image.open(row['path']).resize((480,270));im.paste(src,(0,70));d.text((10,345),'Original decoded source (unaltered)',fill='black',font=font)
  # Separate source camera projection, no independently aligned overlay.
  overlay=src.copy();od=ImageDraw.Draw(overlay);v=body[k];fx,fy,cx,cyp=cams[k,:4];q=v[:,:2]/v[:,2:]*[fx,fy]+[cx,cyp]
  for x,y in q[::20]*.25:
   if 0<=x<480 and 0<=y<270:od.ellipse((x-1,y-1,x+1,y+1),fill='#00ffff')
  overlay=overlay.resize((384,216));im.paste(overlay,(48,375));d.text((10,365),'SMPL-X projection (saved source camera)',fill='black',font=font)
  v=loc[k];tri=v[faces];norm=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm/=np.maximum(np.linalg.norm(norm,axis=1,keepdims=True),1e-9);shade=np.clip(.45+.48*np.abs(norm@np.array([.35,.55,.76])),.35,1);colors=(np.array([112,160,188])*shade[:,None]).astype(np.uint8)
  for col,yaw in enumerate([0,90]):
   theta=np.deg2rad(yaw);rot=np.array([[np.cos(theta),0,np.sin(theta)],[0,1,0],[-np.sin(theta),0,np.cos(theta)]])
   vv=v@rot.T;pt=vv[:,:2]*[scale,-scale]+[720+col*480,320+cy*scale]
   for fid in np.argsort(vv[faces,2].mean(1)):d.polygon([tuple(x) for x in pt[faces[fid]]],fill=tuple(colors[fid]))
   jj=js[k,:22]@rot.T;qp=jj[:,:2]*[scale,-scale]+[720+col*480,320+cy*scale]
   for b,e in EDGES:d.line([tuple(qp[b]),tuple(qp[e])],fill='#c56317' if e in [2,5,8,11,14,17,19,21] else '#175a9d',width=2)
   d.text((500+col*480,70),f'Fixed body view {yaw} degrees',fill='black',font=font)
  im.save(out/f'frame_{k:04d}.jpg',quality=92)
 if len(ids)<=5:
  c=Image.new('RGB',(W,H*len(ids)))
  for k in range(len(ids)):c.paste(Image.open(out/f'frame_{k:04d}.jpg'),(0,k*H))
  c.save(out/'contact.jpg',quality=92)
 else:
  cmd=['ffmpeg','-v','warning','-nostdin','-n','-framerate','30000/1001','-i',str(out/'frame_%04d.jpg'),'-c:v','libx264','-threads','2','-crf','20','-pix_fmt','yuv420p','-video_track_timescale','30000','-movflags','+faststart',str(out/'comparison.mp4')]
  with (out/'encode.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
  (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>MTB SMPL-X body review</h1><p>120 frames, 30000/1001fps, original speed. Source timestamps and decoded frame IDs shown. Head/far limbs are estimates. User review pending. All integration_only.</p><video style="width:100%" controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="bikefit_body_comparison.mp4">Download MP4</a></p>')
 (out/'execution.json').write_text(json.dumps({'wall_seconds':time.monotonic()-t,'frames':len(ids),'fixed_scale_pixels_per_m':scale,'fixed_vertical_center_m':cy,'gpu_used':False},indent=2));print(out,flush=True)
if __name__=='__main__':main()
