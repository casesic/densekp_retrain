"""Independent continuous torso/sleeve surface; no inherited anatomical folds."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from skimage.measure import marching_cubes

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.clothing.binding import clip_surface


def smooth_union(a,b,radius):
    h=np.maximum(radius-np.abs(a-b),0)/radius
    return np.minimum(a,b)-h*h*radius*.25


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--character',default='cycling_F02_v2')
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    char=args.source/'characters'/args.character;joints=np.load(char/'rest_joints.npy')
    female='F02' in args.character
    step=.008;origin=np.array([-.8,0.,-.25]);axes=[np.arange(a,b,step) for a,b in zip(origin,[.8,.62,.26])]
    x,y,z=np.meshgrid(*axes,indexing='ij');points=np.stack([x,y,z],axis=-1)
    heights=[.02,.15,.28,.39,.44,.49,.56,.61]
    rx=np.interp(y,heights,[.180,.177,.183,.205,.210,.115,.070,.065])
    rz=np.interp(y,heights,[.125,.120,.138,.150,.140,.090,.065,.06])
    if not female:rx*=1.12;rz*=1.12
    field=(np.sqrt((x/rx)**2+((z+.01)/rz)**2)-1)*np.minimum(rx,rz)
    field=np.maximum(field,np.maximum(.01-y,y-.605))
    for shoulder,elbow,wrist in [(16,18,20),(17,19,21)]:
        for a,b,r0,r1 in [(joints[shoulder],joints[elbow],.075,.047),
                           (joints[elbow],joints[wrist],.047,.034)]:
            if not female:r0*=1.08;r1*=1.08
            t=np.clip(((points-a)*(b-a)).sum(-1)/np.dot(b-a,b-a),0,1)
            center=a+t[...,None]*(b-a)
            distance=np.linalg.norm(points-center,axis=-1)-(r0+t*(r1-r0))
            field=smooth_union(field,distance,.045)
    vertices,faces,_,_=marching_cubes(field.astype(np.float32),0,spacing=(step,)*3,gradient_direction='ascent')
    vertices+=origin
    cuffleft=joints[20,0]-.018 if female else joints[16,0]+.58*(joints[18,0]-joints[16,0])
    cuffright=joints[21,0]+.018 if female else joints[17,0]+.58*(joints[19,0]-joints[17,0])
    neck=float(joints[12,1]+.075)
    idx,w,faces=clip_surface(vertices,faces,[(1,1,-.045),(1,-1,neck),(0,1,-cuffright),(0,-1,cuffleft)])
    vertices=(vertices[idx]*w[:,:,None]).sum(1).astype(np.float32)
    np.save(args.output/'clothing_rest_vertices.npy',vertices);np.save(args.output/'clothing_faces.npy',faces)
    (args.output/'pattern.json').write_text(json.dumps(dict(split='integration_only',character_id=args.character,
        source_joints=str(char/'rest_joints.npy'),method='smooth union of continuous elliptical torso and tapered sleeves; regular isosurface topology',
        purpose='remove inherited skin creases and discontinuous regional expansion, not a fitted body replacement',
        voxel_m=step,union_radius_m=.045,neck_height_m=neck,vertices=len(vertices),faces=len(faces),seconds=time.monotonic()-start,
        status='unfitted_pattern_candidate',body_changed=False),indent=2)+'\n')


if __name__=='__main__':main()
