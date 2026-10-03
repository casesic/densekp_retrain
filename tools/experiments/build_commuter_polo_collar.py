"""Curved five-row polo collar anchored to the actual trimmed shirt neckline."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from mathutils.bvhtree import BVHTree

ROOT=Path(__file__).resolve().parents[2];sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from tools.experiments.repair_commuter_top import edges_of,project
from densekp_retrain.rendering.blender.intersections import crossings


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--top',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--vertical-contact',action='store_true')
    p.add_argument('--front-drape',action='store_true')
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:])
    args.output.mkdir(parents=True,exist_ok=False);start=time.monotonic();z=np.load(args.top/'evidence.npz')
    top=z['final'];tf=z['cloth_faces'];edges,counts=edges_of(tf);boundary=np.unique(edges[counts==1])
    neck=top[boundary[(top[boundary,1]>.52)&(np.abs(top[boundary,0])<.12)]]
    center=neck.mean(0);angles=np.arctan2(neck[:,0]-center[0],neck[:,2]-center[2])%(2*np.pi)
    order=np.argsort(angles);angles=angles[order];neck=neck[order]
    angles=np.r_[angles[-1]-2*np.pi,angles,angles[0]+2*np.pi];neck=np.concatenate([neck[-1:],neck,neck[:1]])
    theta=np.linspace(.30,2*np.pi-.30,64);base=np.stack([np.interp(theta,angles,neck[:,i]) for i in range(3)],axis=1)
    rows=[]
    for width,height in zip([0,.004,.018,.032,.042],[.002,.009,.004,-.004,-.012]):
        extent=np.full(len(theta),width,dtype=float)
        if args.front_drape:extent*=1+np.maximum(np.cos(theta),0)-.15*np.maximum(-np.cos(theta),0)
        row=base+np.column_stack((np.sin(theta)*extent,np.full(len(theta),height),np.cos(theta)*extent))
        if args.front_drape:row[:,1]-=.040*np.maximum(np.cos(theta),0)*(width/.042)**2
        row[:,1]-=.010*(width/.042)*np.exp(-np.minimum(theta-.30,2*np.pi-.30-theta)**2/.12)
        rows.append(row)
    vertices=np.concatenate(rows);faces=[];n=len(theta)
    for j in range(4):
        for i in range(n-1):
            a=j*n+i;b=a+1;c=a+n;d=c+1;faces.extend([(a,c,d),(a,d,b)])
    faces=np.asarray(faces,np.int32)
    tree=BVHTree.FromPolygons(top.tolist(),tf.tolist(),all_triangles=True)
    initial=vertices.copy()
    if args.vertical_contact:
        for i,point in enumerate(vertices):
            q,_,_,_=tree.ray_cast((float(point[0]),1.,float(point[2])),(0.,-1.,0.))
            if q is not None:vertices[i,1]=max(vertices[i,1],q[1]+.002)
    else:vertices=project(vertices,tree,.002)
    own,_,_=crossings(vertices,faces);body,_,_=crossings(vertices,faces,z['body'],z['body_faces'])
    shirt,_,_=crossings(vertices,faces,top,tf)
    np.save(args.output/'clothing_rest_vertices.npy',vertices.astype(np.float32));np.save(args.output/'clothing_faces.npy',faces)
    combined=np.concatenate([top,vertices]);cf=np.concatenate([tf,faces+len(top)])
    np.savez_compressed(args.output/'evidence.npz',body=z['body'],body_faces=z['body_faces'],cloth_faces=cf,final=combined)
    report=dict(split='integration_only',parent_top=str(args.top),method='curved folded collar, five radial rows, open front, projected above actual shirt',
        contact_mode='upper_shirt_vertical_envelope' if args.vertical_contact else 'nearest_signed_normal',
        front_drape=args.front_drape,
        vertex_count=len(vertices),face_count=len(faces),self_crossings=len(own),body_crossings=len(body),shirt_crossings=len(shirt),
        max_pattern_contact_adjustment_m=float(np.linalg.norm(vertices-initial,axis=1).max()),seconds=time.monotonic()-start,
        status='reference_candidate_visual_pending' if not(len(own)+len(body)+len(shirt)) else 'failed_reference_contact',
        pending=['binding in two poses','placket connection','final whole-outfit contact'])
    (args.output/'pattern.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
