#!/usr/bin/env python3
"""CPU painter diagnostic of already generated SMPL-X body and APD skeleton."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import json
import math
import subprocess
import time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/apd_6526_smplx_pilot_20261008'
EDGES=[(0,1),(1,2),(1,3),(2,4),(4,6),(3,5),(5,7),(2,8),(3,9),(8,9),(8,10),(10,12),(9,11),(11,13)]
LEFT={2,4,6,8,10,12};RIGHT={3,5,7,9,11,13}

def projection(v,yaw):
    a=np.deg2rad(yaw);e=np.deg2rad(7.)
    x=v[...,0]*np.cos(a)+v[...,2]*np.sin(a)
    depth=-v[...,0]*np.sin(a)+v[...,2]*np.cos(a)
    y=v[...,1]*np.cos(e)-depth*np.sin(e)
    z=v[...,1]*np.sin(e)+depth*np.cos(e)
    return np.stack([x,y,z],-1)

def encode(dest,times):
    # Older local ffmpeg lacks concat per-file options. Assign exact source PTS
    # through a filter, retaining 25 Hz input only as an image-reader time base.
    ticks=np.rint(np.asarray(times)*1e6).astype(np.int64)
    expr='+'.join(f'{int(t)}*eq(N,{i})' for i,t in enumerate(ticks))
    cmd=['/usr/bin/ffmpeg','-v','warning','-framerate','25','-i',str(dest/'frame_%03d.png'),
         '-vf',"settb=expr=1/1000000,setpts='"+expr+"'",'-vsync','0','-enc_time_base','1:1000000',
         '-video_track_timescale','1000000','-c:v','libx264','-threads','2','-crf','23','-pix_fmt','yuv420p',
         '-movflags','+faststart',str(dest/'comparison.mp4')]
    assert not (dest/'comparison.mp4').exists(), 'Do not overwrite prior encoding'
    run=subprocess.run(cmd,capture_output=True,text=True)
    (dest/'encoding_v2.log').write_text(run.stderr);assert run.returncode==0,run.stderr
    metadata=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=width,height,time_base,nb_frames,duration:frame=best_effort_timestamp_time','-of','json',str(dest/'comparison.mp4')],text=True))
    pts=np.array([float(row['best_effort_timestamp_time']) for row in metadata['frames']])
    metadata['source_pts_max_error_seconds']=float(np.max(np.abs(pts-times))) if len(pts)==len(times) else None
    metadata['frame_count_matches']=len(pts)==len(times)
    (dest/'mp4_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    assert len(pts)==len(times) and np.max(np.abs(pts-times))<2e-6

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=BASE);ap.add_argument('--fit',default='fit_attempt_1');ap.add_argument('--encode-existing',action='store_true');args=ap.parse_args()
    base=args.root;fit=base/args.fit;dest=fit/'preview'
    if args.encode_existing:
        times=np.load(fit/'parameters.npz')['clip_seconds'];started=time.perf_counter();encode(dest,times)
        (dest/'encoding_repair.json').write_text(json.dumps(dict(integration_only=True,frames=len(times),elapsed_seconds=time.perf_counter()-started,prior_error='old encoder concat/fps options unsupported; original logs preserved',method='exact source microsecond PTS via settb/setpts',frames_regenerated=False),indent=2)+'\n')
        return
    dest.mkdir(exist_ok=False)
    started=time.perf_counter();f=json.loads((base/'frozen_manifest.json').read_text())
    obs=dict(np.load(base/'observations.npz'));params=dict(np.load(fit/'parameters.npz'))
    vertices=np.load(fit/'vertices.npy',mmap_mode='r');joints=np.load(fit/'mapped_joints.npy');faces=np.load(base/'faces.npy')
    ids=params['source_frame_ids'];chosen=[int(np.flatnonzero(obs['source_frame_ids']==i)[0]) for i in ids]
    source=obs['aligned_target_m'][chosen];times=params['clip_seconds'];n=len(ids)
    cloud=np.concatenate([vertices.reshape(-1,3),source.reshape(-1,3)])
    # One common scale and vertical center across ALL frames, views, source and body.
    height=max(float(np.ptp(cloud[:,1])),1.8);scale=min(270,560/(height*1.12));center_y=float((cloud[:,1].min()+cloud[:,1].max())/2)
    fontpath='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(fontpath,14) if Path(fontpath).exists() else ImageFont.load_default()
    small=ImageFont.truetype(fontpath,11) if Path(fontpath).exists() else ImageFont.load_default()
    paths=[]
    for idx in range(n):
        im=Image.new('RGB',(1280,760),'#f5f6f8');dr=ImageDraw.Draw(im)
        dr.text((15,8),f'APD 6526 -> cycling_M01_v2 | source frame {ids[idx]} | source {f["source_relative_seconds"][chosen[idx]]:.6f}s | clip {times[idx]:.6f}s',font=font,fill='#142033')
        dr.text((15,30),'INTEGRATION ONLY | one constant scale/axis transform; source hip translation removed; fixed target shape and pelvis. No floor alignment.',font=small,fill='#384458')
        v=vertices[idx];tri=v[faces];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]);normal/=np.maximum(np.linalg.norm(normal,axis=1,keepdims=True),1e-10)
        shade=np.clip(.48+.45*np.abs(normal@np.array([.35,.55,.76])),.35,1.)
        colors=(np.array([114,159,186])[None]*shade[:,None]).astype(np.uint8)
        for col,(yaw,isbody) in enumerate([(0,False),(0,True),(90,False),(90,True)]):
            ox=col*320;dr.rectangle((ox+2,64,ox+318,697),fill='white')
            dr.text((ox+12,73),('SMPL-X body + joints' if isbody else 'APD observed skeleton')+f' / {yaw} deg',font=small,fill='#172033')
            origin=np.array([ox+160,375+center_y*scale])
            def screen(arr):
                q=projection(arr,yaw);return q[:,:2]*[scale,-scale]+origin,q[:,2]
            if isbody:
                q,depth=screen(v)
                for faceid in np.argsort(depth[faces].mean(1)):
                    dr.polygon([tuple(p) for p in q[faces[faceid]]],fill=tuple(colors[faceid]))
            p= joints[idx] if isbody else source[idx];q,_=screen(p)
            for a,b in EDGES:
                if not np.isfinite(q[[a,b]]).all():continue
                color='#0757b0' if b in LEFT else '#d46108' if b in RIGHT else '#334155'
                dr.line([tuple(q[a]),tuple(q[b])],fill=color,width=2 if isbody else 3)
            for j,point in enumerate(q):
                if not np.isfinite(point).all():continue
                color='#0757b0' if j in LEFT else '#d46108' if j in RIGHT else '#334155'
                dr.ellipse((point[0]-3,point[1]-3,point[0]+3,point[1]+3),fill=color,outline='white')
            dr.text((ox+10,669),f'Shared scale {scale:.2f} px/model metre',font=small,fill='#374151')
        dr.text((15,708),'Blue: named left; orange: named right. APD foot weakly maps to ankle; head is a virtual landmark. Joints drawn through surface for diagnosis.',font=small,fill='#374151')
        dr.text((15,729),'Body is an adapted fixed-shape synthetic target, not recovered source identity. No cloth/groom/bike. Pose3D axes and units remain assumptions.',font=small,fill='#374151')
        name=f'frame_{idx:03d}.png';im.save(dest/name);paths.append(name)
    contacts=[]
    for first in range(0,n,8):
        batch=paths[first:first+8];sheet=Image.new('RGB',(1280,380*math.ceil(len(batch)/2)),'white')
        for k,name in enumerate(batch):
            with Image.open(dest/name) as im:sheet.paste(im.resize((640,380)),((k%2)*640,(k//2)*380))
        name=f'contact_{first:03d}.jpg';sheet.save(dest/name,quality=92);contacts.append(name)
    data=dict(files=paths,times=times.tolist(),ids=ids.tolist(),source_seconds=[f['source_relative_seconds'][i] for i in chosen])
    html=r'''<!doctype html><html><meta charset="utf-8"><title>APD6526 fixed-shape body pilot</title>
<style>body{font:16px system-ui;max-width:1300px;margin:20px auto;background:#eee}img{width:100%}input{width:55%}button{padding:8px}</style>
<h1>APD6526: raw observations / fitted SMPL-X body</h1><p>Two fixed views; common constant scale and framing across source, target and time. No per-frame similarity alignment. <b>Integration only — dynamic human review pending.</b></p>
<button id="play">Play / pause</button><button id="restart">Restart</button><input id="seek" type="range" min="0" max="__LAST__" value="0"> __DOWNLOAD__<p id="status"></p><img id="frame">
<p>Playback follows original timestamp differences, no interpolation, looping or speed change. Last source sample is displayed at its timestamp; no new observation is added. This is a body-only diagnostic, without Blender or scene rendering.</p>
<p>APD: Kress et al., <a href="https://doi.org/10.5281/zenodo.5724486">v1.0</a>, CC BY4.0. SMPL-X local research license applies separately. <a href="../../ATTRIBUTION.md">Attribution / changes</a>.</p>
<script>const D=__DATA__;D.files.forEach(f=>{const im=new Image();im.src=f;});let i=0,active=false,epoch=0;const slider=document.getElementById('seek');
function show(j){i=j;slider.value=j;document.getElementById('frame').src=D.files[j];document.getElementById('status').textContent='Source frame '+D.ids[j]+' | source '+D.source_seconds[j].toFixed(6)+' s | clip '+D.times[j].toFixed(6)+' s';}
function tick(now){if(!active)return;const t=(now-epoch)/1000;let j=0;while(j<D.times.length-1&&D.times[j+1]<=t)j++;show(j);if(t>=D.times.at(-1)){active=false;return;}requestAnimationFrame(tick);}
document.getElementById('play').onclick=()=>{active=!active;if(active){epoch=performance.now()-D.times[i]*1000;requestAnimationFrame(tick);}};document.getElementById('restart').onclick=()=>{active=false;show(0);};slider.oninput=()=>{active=false;show(+slider.value);};show(0);</script></html>'''
    download = '<a href="comparison.mp4" download="apd6526_body_comparison.mp4">下载本页对照视频 MP4</a>' if args.fit!='probe' else ''
    (dest/'index.html').write_text(html.replace('__LAST__',str(n-1)).replace('__DATA__',json.dumps(data)).replace('__DOWNLOAD__',download))
    video=None
    if args.fit!='probe':
        encode(dest,times);video='comparison.mp4'
    result=dict(integration_only=True,fit=args.fit,frames=n,source_frames=ids.tolist(),frame_files=paths,contacts=contacts,
         views_degrees=[0,90],elevation_degrees=7,common_scale_px_per_model_m=scale,common_y_center_m=center_y,
         independent_auto_scaling=False,body_surface_method='CPU flat shaded triangle painter, joints shown through mesh; diagnostic only',
         html='index.html',mp4=video,elapsed_seconds=time.perf_counter()-started,actual_browser_playback_review='pending_user')
    (dest/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
