"""Blender oriented scanline volume for an explicit garment/body-region union.

No parity/XOR at body overlap. The body region is a collision input only.
This is a geometric garment envelope, not a physical cloth simulation.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from mathutils.bvhtree import BVHTree


def capped(vertices,faces):
    edges=np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]])
    _,inverse,counts=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=edges[counts[inverse]==1]
    following={int(a):int(b) for a,b in boundary}
    if len(following)!=len(boundary):raise ValueError('Nonmanifold boundary cannot be capped')
    vertices=vertices.tolist();faces=faces.tolist();caps=[]
    while following:
        first=next(iter(following));ring=[first];current=following.pop(first)
        while current!=first:
            ring.append(current);current=following.pop(current)
        center=np.mean(np.asarray(vertices)[ring],axis=0);index=len(vertices);vertices.append(center.tolist())
        for a,b in zip(ring,ring[1:]+ring[:1]):faces.append([b,a,index]);caps.append(len(faces)-1)
    return np.asarray(vertices,np.float32),np.asarray(faces,np.int32),np.asarray(caps,np.int32)


def scan(tree,origin,shape,step):
    occupancy=np.zeros(shape,bool);unclosed=0;direction=(1.,0.,0.)
    for j in range(shape[1]):
        for k in range(shape[2]):
            position=np.array([origin[0]-step,origin[1]+j*step,origin[2]+k*step])
            degree=0;previous=position[0]
            for _ in range(64):
                hit,normal,_,_=tree.ray_cast(position.tolist(),direction)
                if hit is None:break
                hit=np.asarray(hit)
                if degree>0:
                    lo=max(0,int(np.ceil((previous-origin[0])/step)))
                    hi=min(shape[0],int(np.floor((hit[0]-origin[0])/step))+1)
                    occupancy[lo:hi,j,k]=True
                degree+=1 if normal[0]<0 else -1
                previous=hit[0];position=hit+np.array([2e-6,0,0])
            else:raise RuntimeError('Scanline exceeded intersection bound')
            unclosed+=degree!=0
    return occupancy,int(unclosed)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--garment',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--step',type=float,default=.003)
    args=p.parse_args(sys.argv[sys.argv.index('--')+1:]);start=time.monotonic();args.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((args.garment/'repair.json').read_text());z=np.load(args.garment/'evidence.npz')
    old=Path(meta['parent_clothing']);oldmeta=json.loads((old/'clothing_manifest.json').read_text())
    component=next(c for c in oldmeta['components'] if c['name']==meta['component'])
    first,n=component['vertex_start'],component['vertex_count'];sl=slice(first,first+n)
    ids=np.load(old/'body_barycentric_indices.npy')[sl];w=np.load(old/'body_barycentric_weights.npy')[sl]
    covered=(z['body'][ids]*w[:,:,None]).sum(1)
    faces=np.load(old/'clothing_faces.npy');faces=faces[((faces>=first)&(faces<first+n)).all(1)]-first
    gv,gf,caps=capped(z['bound'],z['cloth_faces']);bv,bf,bcaps=capped(covered,faces)
    points=np.concatenate((gv,bv));step=args.step;origin=points.min(0)-step*4
    shape=tuple((np.ceil((points.max(0)-origin)/step).astype(int)+5).tolist())
    tree=lambda v,f:BVHTree.FromPolygons(v.tolist(),f.tolist(),all_triangles=True)
    garment,gu=scan(tree(gv,gf),origin,shape,step)
    body,bu=scan(tree(bv,bf),origin,shape,step)
    np.savez_compressed(args.output/'volume.npz',occupancy=garment|body,origin=origin,step=step,
                        garment_vertices=gv,garment_faces=gf,cap_faces=caps,
                        original_vertices=z['bound'],original_faces=z['cloth_faces'])
    (args.output/'volume.json').write_text(json.dumps(dict(split='integration_only',parent=str(args.garment),
        method='oriented scanline positive winding union, including explicitly capped covered body region',
        voxel_m=step,shape=shape,unclosed_garment_scanlines=gu,unclosed_body_scanlines=bu,
        body_changed=False,seconds=time.monotonic()-start,status='geometry_candidate_not_cloth_simulation'),indent=2)+'\n')


if __name__=='__main__':main()
