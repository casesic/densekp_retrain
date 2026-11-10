#!/usr/bin/env python3
"""Bounded CPU-only APD diagnostics; no model fitting, downloading or production imports.

Consumes exactly three already extracted official JSON files. Original data remain
untouched. HTML playback uses the original timestamps (microseconds inferred from
25 Hz, not documented by the publisher), without interpolation or smoothing.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

JOINTS = ['head', 'thorax', 'lshoulder', 'rshoulder', 'lelbow', 'relbow',
          'lwrist', 'rwrist', 'lhip', 'rhip', 'lknee', 'rknee', 'lfoot', 'rfoot']
EDGE_NAMES = [('head','thorax'), ('thorax','lshoulder'), ('thorax','rshoulder'),
              ('lshoulder','lelbow'), ('lelbow','lwrist'),
              ('rshoulder','relbow'), ('relbow','rwrist'),
              ('lshoulder','lhip'), ('rshoulder','rhip'), ('lhip','rhip'),
              ('lhip','lknee'), ('lknee','lfoot'),
              ('rhip','rknee'), ('rknee','rfoot')]
EDGES = [(JOINTS.index(a), JOINTS.index(b)) for a,b in EDGE_NAMES]
IDS = (6526, 6900, 7500)


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k,v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if math.isfinite(x) else None
    if isinstance(x, np.bool_): return bool(x)
    return x


def dump(path, data):
    path.write_text(json.dumps(clean(data), ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def summary(a):
    a = np.asarray(a); a = a[np.isfinite(a)]
    if not len(a): return {'count': 0}
    return dict(count=len(a), min=a.min(), median=np.median(a), max=a.max(),
                mean=a.mean(), std=a.std(), p05=np.quantile(a,.05), p95=np.quantile(a,.95))


def runs(mask, t, nominal=.04):
    result=[]; start=None
    for i in range(len(mask)+1):
        good=i<len(mask) and mask[i]
        gap=i>0 and i<len(mask) and t[i]-t[i-1]>1.5*nominal
        if start is not None and (not good or gap):
            result.append(dict(start_index=start,end_index_exclusive=i,
                               frames=i-start,span_seconds=t[i-1]-t[start]))
            start=None
        if good and start is None: start=i
    return result


def project(points, az):
    # Explicit raw coordinate display; no claim of anatomical up or handedness.
    a=np.deg2rad(az); e=np.deg2rad(15)
    x=points[...,0]*np.cos(a)+points[...,1]*np.sin(a)
    y=points[...,2]*np.cos(e)-(points[...,0]*np.sin(a)-points[...,1]*np.cos(a))*np.sin(e)
    return np.stack([x,y],axis=-1)


HTML = r'''<!doctype html><html lang="en"><meta charset="utf-8">
<title>APD ID __ID__ raw skeleton audit</title>
<style>body{font:16px system-ui;max-width:1300px;margin:20px auto;background:#f5f6f8;color:#172033}canvas{background:white;width:48%;border:1px solid #bbb}button,input{margin:8px}pre{white-space:pre-wrap;font-size:14px}</style>
<h1>APD __ID__ — original 3D estimates, two fixed views</h1>
<p>Blue = named left; orange = named right. Hip midpoint subtracted per frame, translation only.
Raw axes/units for pose3d: full convention unconfirmed. No interpolation, filtering, rotation or gap filling.
Feet are APD foot endpoints, not verified anatomical ankle landmarks. No RGB or identity verification.</p>
<button id="play">Play / pause</button><button id="restart">Restart</button><input id="seek" type="range" min="0" max="__LAST__" value="0" style="width:50%"><br>
<canvas id="a" width="600" height="520"></canvas><canvas id="b" width="600" height="520"></canvas><pre id="status"></pre>
<p>Playback uses recorded timestamp differences × 10⁻⁶ seconds (unit inferred from 25 Hz).
Between observations the frame is displayed until the next timestamp, with a 1.5 nominal interval cap;
longer unobserved intervals are blank. The selected last frame has no invented extra source time.
Use slider to inspect every original frame. Playback stops at the end; it does not loop.</p>
<p>Source: Aschaffenburg Pose Dataset v1.0, Kress et al., <a href="https://doi.org/10.5281/zenodo.5724486">Zenodo</a>,
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Changes: bounded time selection, hip centering, diagnostic projections.
See <a href="../../ATTRIBUTION.md">full attribution</a>. No author endorsement.</p>
<script>
const D=__DATA__;const edges=__EDGES__;const names=__NAMES__;
let active=false,epoch=0,current=0;const seek=document.getElementById('seek');
function draw(cid,az,p,gap){const c=document.getElementById(cid),x=c.getContext('2d');x.clearRect(0,0,c.width,c.height);x.font='17px sans-serif';x.fillStyle='#172033';x.fillText('Fixed raw azimuth '+az+'°, elevation 15°',15,24);
if(gap||!p){x.fillStyle='#b3261e';x.fillText(gap?'MISSING TIME INTERVAL':'MISSING HIP CENTER: no relative geometry',15,80);return;}
let a=az*Math.PI/180,e=15*Math.PI/180;
let q=p.map(v=>v.some(z=>z===null)?null:[300+(v[0]*Math.cos(a)+v[1]*Math.sin(a))*D.scale,290-(v[2]*Math.cos(e)-(v[0]*Math.sin(a)-v[1]*Math.cos(a))*Math.sin(e))*D.scale]);
for(let [i,j] of edges){if(!q[i]||!q[j])continue;x.strokeStyle=names[j].startsWith('l')?'#176bb3':names[j].startsWith('r')?'#d56a13':'#586174';x.lineWidth=4;x.beginPath();x.moveTo(...q[i]);x.lineTo(...q[j]);x.stroke();}
q.forEach((v,i)=>{if(!v)return;x.fillStyle=names[i].startsWith('l')?'#176bb3':names[i].startsWith('r')?'#d56a13':'#586174';x.beginPath();x.arc(...v,5,0,7);x.fill();});
// Raw axis triad; this is a display convention, not a recovered floor normal.
for(let [v,col,name] of [[[.2,0,0],'#a33','raw x'],[[0,.2,0],'#393','raw y'],[[0,0,.2],'#33a','raw z']]){let u=v[0]*Math.cos(a)+v[1]*Math.sin(a),w=v[2]*Math.cos(e)-(v[0]*Math.sin(a)-v[1]*Math.cos(a))*Math.sin(e);x.strokeStyle=col;x.beginPath();x.moveTo(75,450);x.lineTo(75+u*240,450-w*240);x.stroke();x.fillStyle=col;x.fillText(name,75+u*240,450-w*240);}}
function show(i,gap=false,elapsed=null){current=i;seek.value=i;draw('a',25,D.frames[i],gap);draw('b',115,D.frames[i],gap);document.getElementById('status').textContent='Source frame '+D.source_indices[i]+'; timestamp '+D.timestamps[i]+' (original integer)\nsource relative time '+D.times[i].toFixed(6)+' s; playback '+(elapsed===null?D.times[i]:elapsed).toFixed(6)+' s; labels: '+D.labels[i].join(', ')+'\nMissing original 3D joints: '+(D.missing[i].join(', ')||'none')+(gap?'\nUNOBSERVED TIMESTAMP GAP':'');}
function tick(now){if(!active)return;let t=(now-epoch)/1000;let i=0;while(i<D.times.length-1&&D.times[i+1]<=t)i++;let gap=t-D.times[i]>.06&&i<D.times.length-1;show(i,gap,t);if(t>=D.times.at(-1)){active=false;show(D.times.length-1);return;}requestAnimationFrame(tick);}
document.getElementById('play').onclick=()=>{active=!active;if(active){epoch=performance.now()-D.times[current]*1000;requestAnimationFrame(tick);}};
document.getElementById('restart').onclick=()=>{active=false;show(0);};seek.oninput=()=>{active=false;show(+seek.value);};show(0);
</script></html>'''


def make_previews(out, ident, rel, valid, t, timestamps, labels, scale):
    payload=dict(frames=rel,times=t,timestamps=[str(v) for v in timestamps],
                 source_indices=list(range(len(t))),labels=labels,scale=scale,
                 missing=[[JOINTS[j] for j in range(14) if not row[j]] for row in valid])
    # A wholly missing hip center is explicit; individual missing joints remain null.
    payload=clean(payload)
    payload['frames']=[None if p[8][0] is None or p[9][0] is None else p for p in payload['frames']]
    html=HTML.replace('__ID__',str(ident)).replace('__LAST__',str(len(t)-1))
    html=html.replace('__DATA__',json.dumps(payload,allow_nan=False)).replace('__EDGES__',json.dumps(EDGES)).replace('__NAMES__',json.dumps(JOINTS))
    (out/'preview.html').write_text(html)
    # All original frames, two directions; four pages rather than only endpoints.
    pages=[]
    for first in range(0,len(t),70):
        last=min(first+70,len(t)); w=1500; h=70+math.ceil((last-first)/10)*185
        im=Image.new('RGB',(w,h),'white'); dr=ImageDraw.Draw(im)
        dr.text((10,8),f'APD {ident} | ALL source frames {first}..{last-1} | two fixed views 25/115 deg; left blue/right orange',fill='black')
        dr.text((10,26),'Hip-centered RAW estimates. No smoothing/interpolation. Raw axes; no confirmed pose3d units/up. See HTML for real-time playback.',fill='black')
        for idx in range(first,last):
            col=(idx-first)%10; row=(idx-first)//10; ox=col*150; oy=60+row*185
            dr.text((ox+4,oy),f'{idx:03d} | {t[idx]:.3f}s',fill='black')
            dr.text((ox+4,oy+13),'/'.join(labels[idx]),fill='#555555')
            if not valid[idx,8:10].all():
                dr.text((ox+4,oy+60),'MISSING HIP',fill='red');continue
            for view,az in enumerate([25,115]):
                q=project(rel[idx],az)*min(scale/4,85);q[:,1]*=-1;q+=np.array([ox+38+view*74,oy+103])
                for (i,j),(_,b) in zip(EDGES,EDGE_NAMES):
                    if not np.isfinite(q[[i,j]]).all():continue
                    color='#176bb3' if b.startswith('l') else '#d56a13' if b.startswith('r') else '#586174'
                    dr.line([tuple(q[i]),tuple(q[j])],fill=color,width=2)
                for point in q:
                    if np.isfinite(point).all():dr.ellipse((point[0]-1,point[1]-1,point[0]+1,point[1]+1),fill='black')
            missing=14-int(valid[idx].sum())
            if missing:dr.text((ox+3,oy+163),f'{missing} missing joints',fill='red')
        name=f'contact_{first:04d}_{last-1:04d}.png'; im.save(out/name);pages.append(name)
    return pages


def audit(base, ident):
    started=time.perf_counter();out=base/'samples'/str(ident)
    d=json.loads((out/'source.json').read_text())
    assert d['vru_type']=='bike'
    all_ts=np.asarray(d['timestamps'],dtype=np.int64)
    assert np.all(np.diff(all_ts)>0), 'Unexpected timestamps: review before selecting window'
    n=int(np.searchsorted(all_ts,all_ts[0]+10_000_000,side='right'))
    ts=all_ts[:n];t=(ts-ts[0])/1e6;dt=np.diff(t)
    p=np.stack([np.asarray(d['pose3d'][j],float)[:n] for j in JOINTS],axis=1)
    assert p.shape==(n,14,3)
    valid=np.isfinite(p).all(2);hip=(p[:,8]+p[:,9])/2;rel=p-hip[:,None,:]
    labels=[[k for k,v in d['motion_primitives'].items() if v[i]] for i in range(n)]
    head=np.asarray(d['head_smoothed'],float)[:n]
    gaps=np.flatnonzero(dt>.06)+1
    jointstats={}
    for j,name in enumerate(JOINTS):
        rr=runs(valid[:,j],t)
        jointstats[name]=dict(valid_frames=int(valid[:,j].sum()),valid_ratio=valid[:,j].mean(),
                             longest_valid_run=max(rr,key=lambda x:x['frames']) if rr else None,
                             missing_source_indices=np.flatnonzero(~valid[:,j]),valid_runs=rr)
    bones={};lens=[]
    for (a,b),(i,j) in zip(EDGE_NAMES,EDGES):
        length=np.linalg.norm(p[:,i]-p[:,j],axis=1);lens.append(length)
        s=summary(length);s['cv']=s.get('std',np.nan)/s.get('mean',np.nan)
        s['relative_adjacent_change_max']=np.nanmax(np.abs(np.diff(length)))/np.nanmedian(length)
        bones[a+'--'+b]=s
    step=np.linalg.norm(np.diff(rel,axis=0),axis=2)
    medleg=np.nanmedian(np.asarray(lens)[[10,11,12,13]])
    jump_pairs=np.argwhere(step>.25*medleg)
    # Heuristic evidence only: swaps may also reflect real motion or bad root estimates.
    left=[8,10,12];right=[9,11,13]
    same=np.sum((rel[1:,left]-rel[:-1,left])**2,axis=(1,2))+np.sum((rel[1:,right]-rel[:-1,right])**2,axis=(1,2))
    swap=np.sum((rel[1:,left]-rel[:-1,right])**2,axis=(1,2))+np.sum((rel[1:,right]-rel[:-1,left])**2,axis=(1,2))
    suspect=np.flatnonzero((swap<.5*same)&np.isfinite(swap))+1
    # Knee angles are observed geometry, not joint model parameters.
    knees=[]
    for i,j,k in [(8,10,12),(9,11,13)]:
        a=p[:,i]-p[:,j];b=p[:,k]-p[:,j]
        denom=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1)
        knees.append(np.rad2deg(np.arccos(np.clip(np.sum(a*b,axis=1)/denom,-1,1))))
    knees=np.array(knees)
    hs=np.linalg.norm(np.diff(head,axis=0),axis=1)/dt
    rs=np.linalg.norm(np.diff(p[:,0,:2],axis=0),axis=1)/dt
    states={k:dict(frames=sum(k in labs for labs in labels),
                   head_smoothed_xy_speed=summary(hs[np.array([k in labs for labs in labels[1:]])])) for k in d['motion_primitives']}
    # Original 2D third value is named uncertainty by publisher, NOT interpreted as confidence.
    two={}
    for k,v in d['pose2d'].items():
        a=np.asarray(v,float)[:n];two[k]=dict(xy_valid_ratio=np.isfinite(a[:,:2]).all(1).mean(),third_component=summary(a[:,2]),
              missing_xy_third_component=summary(a[~np.isfinite(a[:,:2]).all(1),2]))
    stats=dict(file_id=f'dataset/{ident}.json',vru_type=d['vru_type'],official_split=d['set'],
       original_frames=len(all_ts),original_span_seconds_inferred=(all_ts[-1]-all_ts[0])/1e6,
       selection=dict(rule='first contiguous <=10 seconds; no quality-based replacement',start_index=0,end_index_exclusive=n,
                      timestamp_first=int(ts[0]),timestamp_last=int(ts[-1]),span_seconds=t[-1],
                      timestamp_scale_seconds=1e-6,timestamp_unit_status='inferred_from_25Hz_not_officially_documented'),
       time=dict(dt_seconds=summary(dt),duplicates=int(np.sum(dt==0)),backward=int(np.sum(dt<0)),
                 gaps_after_indices=gaps-1,missing_nominal_slots=int(np.sum(np.maximum(0,np.rint(dt/.04).astype(int)-1)))),
       joints=jointstats,pose2d=two,bones_raw_units=bones,
       adjacent_root_relative_steps_raw_units=summary(step),
       jump_heuristic=dict(rule='root-relative adjacent displacement > 0.25 x median lower-leg/thigh segment length; diagnostic only',
                           reference_length=medleg,pairs=[dict(from_index=int(i),to_index=int(i+1),joint=JOINTS[j],distance=float(step[i,j])) for i,j in jump_pairs]),
       left_right_swap_heuristic=dict(rule='both full legs exchanged cost < half unchanged cost; not proof of swap',indices=suspect),
       knee_angles_degrees=dict(left=summary(knees[0]),right=summary(knees[1])),
       head_smoothed_shape=list(head.shape),head_raw_z=summary(p[:,0,2]),
       head_smoothed_xy_speed_mps_official_example_unit=summary(hs),raw_head_xy_speed_in_raw_units=summary(rs),
       head_smoothed_xy_path_length=float(np.linalg.norm(np.diff(head,axis=0),axis=1).sum()),
       label_statistics=states,onehot_label_violations=[i for i,l in enumerate(labels) if len(l)!=1],
       identity='unknown; file is a tracked trajectory, not verified actor identity',
       observed_complete_pedal_cycles='not established by numeric statistics; inspect raw previews',
       split_policy='integration_only; preserve official split but do not claim actor/session independence')
    with (out/'time_mapping.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['output_frame','source_frame','timestamp_original','seconds_from_window_start_inferred','gap_before_seconds','valid_3d_joints','labels'])
        for i in range(n):w.writerow([i,i,int(ts[i]),f'{t[i]:.6f}',f'{dt[i-1]:.6f}' if i else '',int(valid[i].sum()),'|'.join(labels[i])])
    with (out/'joint_validity.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['joint','valid_frames','total_frames','valid_ratio','longest_run_frames','longest_run_span_seconds'])
        for k,s in jointstats.items():r=s['longest_valid_run'] or {};w.writerow([k,s['valid_frames'],n,s['valid_ratio'],r.get('frames',0),r.get('span_seconds',0)])
    with (out/'frame_diagnostics.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['source_frame','seconds','left_knee_deg','right_knee_deg','left_foot_rel_x','left_foot_rel_y','left_foot_rel_z','right_foot_rel_x','right_foot_rel_y','right_foot_rel_z'])
        for i in range(n):w.writerow([i,t[i],*knees[:,i],*rel[i,12],*rel[i,13]])
    fig,ax=plt.subplots(3,1,figsize=(12,10),sharex=True)
    for k,col in [(0,'#176bb3'),(1,'#d56a13')]:
        ax[0].plot(t,knees[k],color=col,label=['left','right'][k],lw=1)
        ax[1].plot(t,rel[:,12+k,2],color=col,label=['left foot','right foot'][k],lw=1)
    ax[0].set_ylabel('Knee angle (deg)');ax[0].legend()
    ax[1].set_ylabel('Foot raw z minus hip midpoint');ax[1].legend()
    for lengths,name in zip(lens[-4:],[a+'--'+b for a,b in EDGE_NAMES[-4:]]):ax[2].plot(t,lengths,label=name,lw=1)
    ax[2].set_ylabel('Bone length (raw units)');ax[2].set_xlabel('Original relative time (s; microsecond unit inferred)');ax[2].legend(ncol=2)
    for a in ax:
        for idx in np.flatnonzero(~valid.all(1)):a.axvline(t[idx],color='red',alpha=.15,lw=1)
        for idx in gaps:a.axvspan(t[idx-1]+.04,t[idx],color='red',alpha=.3)
        a.grid(alpha=.25)
    fig.suptitle(f'APD {ident}: raw per-frame geometry, no filtering. Red = missing joints / time. Periodicity is not validation.')
    fig.tight_layout();fig.savefig(out/'pose_diagnostics.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(1,3,figsize=(16,5))
    ax[0].plot(p[:,0,0],p[:,0,1],'.-',ms=2,lw=.6,label='raw head pose3d')
    ax[0].plot(head[:,0],head[:,1],lw=1.5,label='provided RTS head_smoothed')
    ax[0].plot(hip[:,0],hip[:,1],lw=.7,label='derived hip midpoint (not floor root)')
    ax[0].set_xlabel('raw first coordinate');ax[0].set_ylabel('raw second coordinate');ax[0].axis('equal');ax[0].legend(fontsize=7)
    ax[1].plot(t[1:],rs,label='raw head XY',lw=.7);ax[1].plot(t[1:],hs,label='provided smoothed head XY',lw=1.3)
    ax[1].set_ylabel('XY speed (RTS: m/s per official example; raw: units/s)',fontsize=8);ax[1].legend(fontsize=7)
    for j,(state,values) in enumerate(d['motion_primitives'].items()):ax[2].step(t,np.asarray(values)[:n]+j*1.4,where='post',label=state)
    ax[2].set_yticks([]);ax[2].legend(fontsize=7);ax[2].set_title('Original manually annotated states')
    for a in ax[1:]:a.set_xlabel('Original relative time (s; unit inferred)')
    for a in ax:a.grid(alpha=.25)
    fig.suptitle(f'APD {ident}: raw coordinate paths, NOT an accepted ground/root trajectory. Head RTS is non-independent.')
    fig.tight_layout();fig.savefig(out/'trajectory.png',dpi=140);plt.close(fig)
    scale=min(300,200/max(.1,float(np.nanmax(np.abs(rel)))))
    stats['contact_sheets']=make_previews(out,ident,rel,valid,t,ts,labels,scale)
    stats['diagnostic_seconds']=time.perf_counter()-started
    dump(out/'statistics.json',stats)
    return stats


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path('outputs/audits/apd_feasibility_20261008'))
    args=parser.parse_args();started=time.perf_counter()
    assert (args.root/'gpu_before.txt').exists() and (args.root/'processes_before.txt').exists(), 'Record occupancy first'
    stats=[audit(args.root,i) for i in IDS]
    result=dict(schema='apd_bounded_audit_v1',ids=IDS,cpu_only=True,raw_files_extracted=3,
                selection_policy='6526 first cyclist-boundary hypothesis; 6900 and 7500 preselected dispersed numeric IDs before inspecting their arrays or labels. First <=10 seconds each. No rejected replacement candidates.',
                joint_order=JOINTS,bone_edges=EDGE_NAMES,
                time_scale_status='1e-6 seconds/tick inferred, not documented',
                input_3d_is_estimated=True,raw_rgb_available=False,source_group='APD_5724486_v1_identity_session_unknown',
                samples=[dict(id=i,statistics=f'samples/{i}/statistics.json',preview=f'samples/{i}/preview.html',selection=s['selection'],official_split=s['official_split']) for i,s in zip(IDS,stats)],
                total_diagnostic_seconds=time.perf_counter()-started)
    dump(args.root/'candidate_audit.json',result)
    links=''.join(f'<li><a href="samples/{i}/preview.html">{i}: exact-timestamp two-view skeleton</a> | <a href="samples/{i}/pose_diagnostics.png">pose diagnostics</a> | <a href="samples/{i}/trajectory.png">trajectory</a> | <a href="samples/{i}/statistics.json">statistics</a></li>' for i in IDS)
    (args.root/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>APD bounded audit</title><h1>APD three-window CPU audit</h1><p>Diagnostic estimated skeletons only; not SMPL-X, cloth acceptance or ground truth. Each window <=10 s. Original missing data retained. Click Play; playback stops at the end.</p><ul>'+links+'</ul>')
    print(json.dumps(clean(result),ensure_ascii=False,indent=2))


if __name__=='__main__': main()
