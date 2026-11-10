#!/usr/bin/env python3
"""Fixed-scale, CPU-only raw BVH versus target SMPL-X body review."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json, subprocess, time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/bvh_p2_smplx_pilot_a2_20261010'
NAMES=['Hips','Spine','Spine1','Spine2','Neck','Head','LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightShoulder','RightArm','RightForeArm','RightHand',
 'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase','RightUpLeg','RightLeg','RightFoot','RightToeBase']
EDGES=[('Hips','Spine'),('Spine','Spine1'),('Spine1','Spine2'),('Spine2','Neck'),('Neck','Head'),
 ('Spine2','LeftShoulder'),('LeftShoulder','LeftArm'),('LeftArm','LeftForeArm'),('LeftForeArm','LeftHand'),
 ('Spine2','RightShoulder'),('RightShoulder','RightArm'),('RightArm','RightForeArm'),('RightForeArm','RightHand'),
 ('Hips','LeftUpLeg'),('LeftUpLeg','LeftLeg'),('LeftLeg','LeftFoot'),('LeftFoot','LeftToeBase'),
 ('Hips','RightUpLeg'),('RightUpLeg','RightLeg'),('RightLeg','RightFoot'),('RightFoot','RightToeBase')]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=BASE);args=ap.parse_args();base=args.root
    freeze=json.loads((base/'frozen_manifest.json').read_text());lo,hi=freeze['source_frame_half_open'];n=hi-lo;tick=freeze['ticks_per_frame'];character_name=freeze['target_character']
    start=time.perf_counter();out=base/'preview';out.mkdir(exist_ok=True)
    obs=np.load(base/'observations.npz');fit=base/'fit_attempt_1';params=np.load(fit/'parameters.npz')
    verts=np.load(fit/'vertices.npy',mmap_mode='r');joints=np.load(fit/'joints.npy',mmap_mode='r');faces=np.load(base/'faces.npy')
    mapping=json.loads((base/'joint_mapping.json').read_text());indices={j['source']:i for i,j in enumerate(mapping['correspondences'])}
    # Same pelvis-centered coordinates and one global scale/center for raw and fitted.
    raw=(obs['native_joint_positions']-obs['native_root'][:,None])@obs['basis'].T*float(obs['scale'])
    ext=np.concatenate([raw.reshape(-1,3),np.asarray(verts).reshape(-1,3)])
    ylim=[float(ext[:,1].min()),float(ext[:,1].max())]
    hmax=float(max(np.abs(ext[:,[0,2]]).max(),.9));half_y=(ylim[1]-ylim[0])/2
    scale=min(320/(2*hmax),390/(2*half_y));ycenter=(ylim[0]+ylim[1])/2
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
    W,H=1600,535;tilew=400
    # Model faces are stable; draw all of them. No per-frame crop, root tracking, or smoothing.
    for t in range(n):
        if (out/f'frame_{t:04d}.jpg').exists():
            continue
        im=Image.new('RGB',(W,H),'#f4f6f9');d=ImageDraw.Draw(im)
        time_tick=int(params['nominal_asset_time_ticks'][t]);src=int(params['source_frame_ids'][t])
        d.text((10,7),f'BVH -> {character_name} | asset {t} | source {src} | nominal clip t={time_tick/10000000:.7f}s',font=font,fill='#101923')
        d.text((10,31),'integration_only | BVH units unknown | fixed inferred yaw+scale | root removed for in-place asset | SMPL-X body, no clothing',font=font,fill='#334154')
        for view,axis in enumerate((0,2)):
            for kind in range(2):
                col=view*2+kind;cx=col*tilew+tilew/2;cy=286+ycenter*scale
                d.text((col*tilew+10,61),('X-Y' if view==0 else 'Z-Y')+(' | original BVH FK' if kind==0 else ' | fitted SMPL-X'),font=font,fill='#15233a')
                d.line((col*tilew+20,475,(col+1)*tilew-20,475),fill='#cdd3df',width=1)
                def xy(v):return (cx+float(v[axis])*scale,cy-float(v[1])*scale)
                if kind==0:
                    for a,b in EDGES:
                        color='#dc6235' if b.startswith('Left') else '#296fb7' if b.startswith('Right') else '#29364b'
                        d.line((xy(raw[t,indices[a]]),xy(raw[t,indices[b]])),fill=color,width=4)
                    for name in NAMES:
                        x,y=xy(raw[t,indices[name]]);d.ellipse((x-3,y-3,x+3,y+3),fill='#15233a')
                else:
                    v=np.asarray(verts[t]);tri=v[faces]
                    normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);norm=np.linalg.norm(normal,axis=1)
                    shade=np.clip(.48+.42*np.abs(normal[:,2 if axis==0 else 0])/np.maximum(norm,1e-9),.35,1)
                    colors=(np.array([100,151,184])*shade[:,None]).astype(np.uint8)
                    # Exact same target-space geometry; depth sort depends only on fixed view axis.
                    depth_axis=2 if axis==0 else 0
                    for fid in np.argsort(tri[:,:,depth_axis].mean(1)):
                        points=[xy(q) for q in tri[fid]]
                        d.polygon(points,fill=tuple(int(z) for z in colors[fid]))
                    for item in mapping['correspondences']:
                        if item['source'] in ('Hips','LeftHand','RightHand','LeftFoot','RightFoot'):
                            x,y=xy(joints[t,item['target_joint_index']]);d.ellipse((x-3,y-3,x+3,y+3),fill='#d65a22')
        d.text((10,510),f'One world-space scale: {scale:.3f} px/model m. Source skeleton positions use fitted pelvis-removal/axis/scale, no individual view normalization.',font=font,fill='#354257')
        im.save(out/f'frame_{t:04d}.jpg',quality=88)
    rate=f'10000000/{tick}'
    cmd=['ffmpeg','-nostdin','-v','error','-n','-framerate',rate,'-i',str(out/'frame_%04d.jpg'),'-frames:v','120',
         '-vf','pad=iw:ih+1','-c:v','libx264','-preset','veryfast','-crf','22','-threads','2','-pix_fmt','yuv420p','-video_track_timescale','10000000','-movflags','+faststart',str(out/'comparison.mp4')]
    cmd[cmd.index('-frames:v')+1]=str(n)
    subprocess.run(cmd,check=True)
    html='''<!doctype html><meta charset="utf-8"><title>BVH body pilot</title><style>body{font-family:sans-serif;max-width:1600px;margin:auto;background:#f4f6f9;color:#1b2638}video,img{width:100%}input{width:100%}</style><h1>BVH → TARGET</h1><p>N native BVH samples [LO,HI), nominal Frame Time DT s. Left: original FK skeleton; right: fitted SMPL-X body, two fixed axes. Inferred axis/scale; source root removed for in-place asset. Internal review only; dynamic human review pending.</p><video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="BVH_body_comparison.mp4">下载本页对照视频 MP4</a></p><p>Frame: <span id="label">0</span> · source frame <span id="source">LO</span> · nominal clip seconds <span id="seconds">0</span></p><input id="slider" type="range" min="0" max="LAST" value="0"><img id="still" src="frame_0000.jpg"><script>const s=document.getElementById('slider');s.oninput=()=>{let i=+s.value;document.getElementById('label').textContent=i;document.getElementById('source').textContent=LO+i;document.getElementById('seconds').textContent=(i*DT).toFixed(7);document.getElementById('still').src='frame_'+String(i).padStart(4,'0')+'.jpg'};</script>'''
    for key,value in [('TARGET',character_name),('N',str(n)),('LO',str(lo)),('HI',str(hi)),('LAST',str(n-1)),('DT',freeze['frame_time_original_decimal'])]:
        html=html.replace(key,value)
    (out/'index.html').write_text(html)
    # Decoder timing is a container grid approximation to the exact nominal BVH interval.
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',str(out/'comparison.mp4')],capture_output=True,text=True,check=True)
    pts=np.array([float(x.strip()) for x in probe.stdout.splitlines() if x.strip()]);assert len(pts)==n
    err=float(np.max(np.abs(pts-np.arange(n)*tick/10000000)));assert err<1e-5
    report=dict(integration_only=True,frames=n,source_frame_half_open=[lo,hi],video_rate=rate,video_time_base='1/10000000',
        decoded_frames=len(pts),max_nominal_pts_error_seconds=err,nominal_sample_interval=f'{tick}/10000000 s; BVH header, not real exposure PTS',
        fixed_scale_px_per_model_m=scale,vertical_center_model_m=ycenter,source_and_body_same_alignment=True,
        root_display='pelvis removed each frame by frozen asset convention; original root retained in observations.npz',
        views=['X-Y','Z-Y'],dynamic_human_acceptance='pending_user',compute_seconds=time.perf_counter()-start)
    (out/'execution.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)

if __name__=='__main__':main()
