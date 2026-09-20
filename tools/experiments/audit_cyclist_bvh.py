"""Read BVH channels and perform forward kinematics for audit, not retargeting."""
import argparse
import json
import re
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def read_bvh(path):
    text=path.read_text(errors='strict')
    hierarchy,motion=text.split('MOTION',1)
    tokens=re.findall(r'[^\s{}]+|[{}]',hierarchy)
    cursor=0; joints=[]; channel_count=0
    def take():
        nonlocal cursor
        v=tokens[cursor];cursor+=1;return v
    def node(parent,kind):
        nonlocal channel_count
        name=take() if kind!='End' else joints[parent]['name']+'_EndSite'
        # End Site's Site token was consumed by the caller.
        assert take()=='{'
        index=len(joints)
        joint={'name':name,'parent':parent,'offset':None,'channels':[],
               'channel_start':channel_count,'end_site':kind=='End'}
        joints.append(joint)
        while True:
            token=take()
            if token=='}':break
            if token=='OFFSET':joint['offset']=[float(take()) for _ in range(3)]
            elif token=='CHANNELS':
                n=int(take());joint['channel_start']=channel_count
                joint['channels']=[take() for _ in range(n)];channel_count+=n
            elif token=='JOINT':node(index,token)
            elif token=='End':assert take()=='Site';node(index,'End')
            else:raise ValueError(token)
        assert joint['offset'] is not None
    assert take()=='HIERARCHY';assert take()=='ROOT';node(-1,'ROOT')
    assert cursor==len(tokens)
    match=re.match(r'\s*Frames:\s*(\d+)\s*Frame\s+Time:\s*([\d.eE+-]+)\s*',motion)
    assert match
    frame_count=int(match[1]);dt=float(match[2]);assert dt>0
    lines=[line for line in motion[match.end():].splitlines() if line.strip()]
    assert len(lines)==frame_count
    assert all(len(line.split())==channel_count for line in lines)
    values=np.fromstring(motion[match.end():],sep=' ')
    assert values.size==frame_count*channel_count,(values.size,frame_count,channel_count)
    values=values.reshape(frame_count,channel_count)
    assert np.isfinite(values).all()
    return joints,values,dt


def forward_kinematics(joints,values):
    count=len(values);positions=np.zeros((count,len(joints),3));rotations=[]
    for index,joint in enumerate(joints):
        # Match Blender's BVH importer: animated position replaces the static
        # offset on that axis. This export repeats offsets in position channels.
        local=np.broadcast_to(np.eye(3),(count,3,3)).copy()
        trans=np.broadcast_to(joint['offset'],(count,3)).copy().astype(float)
        for channel in joint['channels']:
            if channel.endswith('position'):trans[:,'XYZ'.index(channel[0])]=0
        for ci,channel in enumerate(joint['channels']):
            value=values[:,joint['channel_start']+ci]
            axis='XYZ'.index(channel[0])
            if channel.endswith('position'):
                shift=np.zeros((count,3));shift[:,axis]=value
                trans+=np.einsum('nij,nj->ni',local,shift)
            elif channel.endswith('rotation'):
                rotation=Rotation.from_euler(channel[0],value,degrees=True).as_matrix()
                local=local@rotation
            else:raise ValueError(channel)
        parent=joint['parent']
        if parent<0:positions[:,index]=trans;world=local
        else:
            positions[:,index]=positions[:,parent]+np.einsum('nij,nj->ni',rotations[parent],trans)
            world=rotations[parent]@local
        rotations.append(world)
    return positions


def main():
    p=argparse.ArgumentParser();p.add_argument('bvh',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    joints,values,dt=read_bvh(a.bvh)
    positions=forward_kinematics(joints,values)
    report={'path':str(a.bvh),'root':joints[0]['name'],'joint_count_excluding_end_sites':sum(not j['end_site'] for j in joints),
            'end_sites':sum(j['end_site'] for j in joints),'frames':len(values),'frame_time':dt,'fps':1/dt,
            'duration_sec':len(values)*dt,'channel_count':values.shape[1],
            'finite_channels':True,'hierarchy_valid':True,'joints':joints,
            'translation_semantics':'Position channels replace OFFSET; OFFSET used on absent axes/end sites (Blender importer convention)',
            'root_translation_min':positions[:,0].min(0).tolist(),
            'root_translation_max':positions[:,0].max(0).tolist(),
            'root_translation_range':np.ptp(positions[:,0],axis=0).tolist(),
            'position_bbox_min':positions.min((0,1)).tolist(),'position_bbox_max':positions.max((0,1)).tolist(),
            'units':'BVH has no unit declaration; infer separately from anatomy, do not silently scale',
            'coordinate_system':'Native BVH axes; up/forward/handedness require separate evidence',
            'euler_order_by_joint':{j['name']:''.join(c[0] for c in j['channels'] if c.endswith('rotation')) for j in joints if not j['end_site']}}
    (a.out/'bvh_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    np.save(a.out/'joint_positions_native.npy',positions)
    print(json.dumps({k:v for k,v in report.items() if k not in ('joints','euler_order_by_joint')},indent=2))


if __name__=='__main__':main()
