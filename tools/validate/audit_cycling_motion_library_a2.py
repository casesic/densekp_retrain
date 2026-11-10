"""Bounded, CPU-only native BVH FK audit and fixed-view review videos."""
import json
import math
import os
from pathlib import Path
import subprocess
import time

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
INPUT=ROOT/'outputs/motion_inputs/cycling_fit_20260929'
OUT=ROOT/'outputs/audits/cycling_motion_library_a2_20261009'
CASES=[('p2_straight','participant_2',5700,240,False),
       ('p1_comfortable','participant_1',7560,192,False),
       ('p1_alternate','participant_1',9600,192,True)]
NAMES=['Hips','Spine','Spine1','Spine2','Neck','Head',
       'LeftShoulder','LeftArm','LeftForeArm','LeftHand','RightShoulder','RightArm','RightForeArm','RightHand',
       'LeftUpLeg','LeftLeg','LeftFoot','LeftToeBase','RightUpLeg','RightLeg','RightFoot','RightToeBase']

def bvh_window(path,start,n):
    with open(path,'r') as f:
        header=[]
        while True:
            line=f.readline()
            if not line:raise ValueError('BVH missing MOTION')
            header.append(line.rstrip('\n'))
            if line.strip()=='MOTION':break
        total_line=f.readline().strip();time_line=f.readline().strip()
        assert total_line.startswith('Frames:') and time_line.startswith('Frame Time:')
        total=int(total_line.split(':',1)[1]);frame_time=float(time_line.split(':',1)[1]);assert start+n<=total
        for _ in range(start):assert f.readline()
        rows=np.array([[float(x) for x in f.readline().split()] for _ in range(n)],np.float64)
    assert rows.shape==(n,372) and np.isfinite(rows).all()
    return rows,frame_time,total,header

def native_fk(rows,joints):
    # Source BVH has six position/rotation channels per non-end joint.
    # Position channels replace OFFSET (they already contain it); local
    # rotations multiply in the listed Z-X-Y order, then parent FK.
    positions=np.empty((len(rows),len(joints),3),np.float64)
    for t,row in enumerate(rows):
        global_r=[]
        for i,j in enumerate(joints):
            local=np.array(j['offset'],float);local_r=np.eye(3)
            ch=j['channels'];values=row[j['channel_start']:j['channel_start']+len(ch)]
            for name,value in zip(ch,values):
                ax=name[0]
                if name.endswith('position'):
                    local['XYZ'.index(ax)]=value
                else:
                    angle=math.radians(value);c=math.cos(angle);s=math.sin(angle)
                    if ax=='X':q=np.array([[1,0,0],[0,c,-s],[0,s,c]])
                    elif ax=='Y':q=np.array([[c,0,s],[0,1,0],[-s,0,c]])
                    else:q=np.array([[c,-s,0],[s,c,0],[0,0,1]])
                    local_r=local_r@q
            parent=j['parent']
            if parent<0:
                global_r.append(local_r);positions[t,i]=local
            else:
                global_r.append(global_r[parent]@local_r)
                positions[t,i]=positions[t,parent]+global_r[parent]@local
    return positions

def render_case(name,source_name,start,n,from_original):
    began=time.monotonic();source=INPUT/source_name
    manifest=json.loads((source/'manifest.json').read_text());joints=manifest['joints']
    if from_original:
        data,dt,total,header=bvh_window(Path(manifest['source_bvh']),start,n)
    else:
        with np.load(source/'clip_native.npz',allow_pickle=False) as z:
            i=start-manifest['source_frame_range'][0]
            data=z['channels_native'][i:i+n].copy();cache=z['joint_positions_native'][i:i+n].copy()
            dt=float(z['frame_time']);np.testing.assert_array_equal(z['source_frame_ids'][i:i+n],np.arange(start,start+n))
        _,clip_dt,clip_total,header=bvh_window(source/'source_clip.bvh',i,n)
        assert clip_total==manifest['frames'] and dt==clip_dt
        total=manifest['source_total_frames']
    pos=native_fk(data,joints)
    if not from_original:
        err=float(np.max(np.abs(pos-cache)));assert err<1e-8
    else:
        err=None
        # Same participant's released full file and extracted clip use the same hierarchy.
        _,_,_,reference=bvh_window(source/'source_clip.bvh',0,1)
        assert header==reference
    assert np.isfinite(pos).all()
    names=[j['name'] for j in joints];idx={v:names.index(v) for v in NAMES};wanted=[idx[v] for v in NAMES]
    parent=[j['parent'] for j in joints];edges=[(parent[i],i) for i in wanted if parent[i] in wanted]
    # Fixed centre/scale for all frames and both views, retaining root drift.
    origin=pos[0,idx['Hips']].copy();relative=pos-origin
    bound=float(np.max(np.abs(relative[:,wanted,:])));scale=350/(2*max(bound,120))
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    folder=OUT/'previews'/name;folder.mkdir(parents=True,exist_ok=False)
    for t in range(n):
        im=Image.new('RGB',(800,456),'#f1f4f8');d=ImageDraw.Draw(im)
        d.text((8,6),f'{name} | BVH skeleton, native units (unit unconfirmed) | source {start+t} | {(start+t)*dt:.6f} nominal s',font=font,fill='#172030')
        for vi,(side,horizontal) in enumerate([('XY fixed','X'),('ZY fixed','Z')]):
            left=vi*400;d.text((left+10,30),side,font=font,fill='#172030')
            def uv(i):
                v=relative[t,i];return (left+200+v['XYZ'.index(horizontal)]*scale,249-v[1]*scale)
            for a,b in edges:
                limb='left' if names[b].startswith('Left') else 'right' if names[b].startswith('Right') else 'centre'
                d.line([uv(a),uv(b)],fill={'left':'#df5128','right':'#2371cc','centre':'#333944'}[limb],width=4)
            for i in wanted:
                u,v=uv(i);d.ellipse((u-3,v-3,u+3,v+3),fill='#151b27')
        d.text((8,432),f'Original Z-X-Y rotation order; centre = first Hips, one scale = {scale:.6f} px/native. No smoothing, loop, interpolation or frame-wise root removal.',font=font,fill='#243348')
        im.save(folder/f'frame_{t:03d}.jpg',quality=88)
    # Exact decimal frame_time in the published BVH header, not a rounded 24/30 assumption.
    rate=f'10000000/{round(dt*10000000)}'
    command=['ffmpeg','-nostdin','-v','error','-framerate',rate,'-i',str(folder/'frame_%03d.jpg'),'-frames:v',str(n),'-c:v','libx264','-crf','22','-preset','veryfast','-threads','2','-pix_fmt','yuv420p',str(folder/'native_skeleton.mp4')]
    subprocess.run(command,check=True)
    yleft=pos[:,idx['LeftFoot'],1]-pos[:,idx['Hips'],1];yright=pos[:,idx['RightFoot'],1]-pos[:,idx['Hips'],1]
    valid_step=np.linalg.norm(np.diff(pos[:,wanted],axis=0),axis=2)
    report=dict(name=name,source_participant=source_name,source_bvh=str(Path(manifest['source_bvh'])),source_frames_half_open=[start,start+n],frames=n,
        source_total_frames=total,frame_time_s=dt,frame_time_basis='BVH Frame Time header; nominal sample interval, no per-frame PTS',
        video_rate_rational=rate,first_last_nominal_span_s=(n-1)*dt,representation='BVH native FK skeleton, not SMPL-X or measured GT',
        joints=len(joints),channels=data.shape[1],native_position_channel_semantics='replace OFFSET',rotation_order='listed Zrotation Xrotation Yrotation intrinsic/local multiplication',
        cached_fk_error_native=err,all_finite=True,max_adjacent_selected_joint_step_native=float(valid_step.max()),
        left_right_pelvis_relative_foot_y_correlation=float(np.corrcoef(yleft,yright)[0,1]),
        left_right_foot_y_range_native=[float(np.ptp(yleft)),float(np.ptp(yright))],
        root_displacement_native=float(np.linalg.norm(pos[-1,idx['Hips']]-pos[0,idx['Hips']])),
        coordinates='native BVH; +Y up inferred; no scene/calibrated camera',unit='native BVH; centimetre conversion unconfirmed',
        fixed_view_axes=['X-Y','Z-Y'],fixed_origin='first source Hips',fixed_scale_px_per_native=scale,
        source_interval_contiguous=True,no_interpolation_or_smoothing=True,license='Zenodo 10668611 record CC BY 4.0; credit original authors',
        group='zenodo_10668611_'+source_name,preview=str(folder/'native_skeleton.mp4'),compute_seconds=time.monotonic()-began)
    (folder/'diagnostics.json').write_text(json.dumps(report,indent=2)+'\n')
    return report

def main():
    start=time.monotonic();results=[]
    for args in CASES:
        results.append(render_case(*args))
        assert time.monotonic()-start<900,'CPU budget exceeded'
    OUT.joinpath('cpu_execution.json').write_text(json.dumps(dict(cpu_only=True,preview_count=len(results),elapsed_seconds=time.monotonic()-start,budget_seconds=900,results=[dict(name=x['name'],seconds=x['compute_seconds']) for x in results]),indent=2)+'\n')
    print(json.dumps({x['name']:{'seconds':x['compute_seconds'],'fk_error':x['cached_fk_error_native'],'leg_correlation':x['left_right_pelvis_relative_foot_y_correlation']} for x in results}))

if __name__=='__main__':main()
