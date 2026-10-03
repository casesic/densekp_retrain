"""Minimal frozen DQS/scan algorithms for analytic Blender regression tests.

Not a production garment pipeline; expectations remain independent rigid
matrices, quaternion sign alignment and known inside/outside relationships.
"""

import numpy as np
from mathutils import Matrix


def qmultiply(a,b):
    return np.concatenate(((a[...,:1]*b[...,:1]-(a[...,1:]*b[...,1:]).sum(-1,keepdims=True)),
                           a[...,:1]*b[...,1:]+b[...,:1]*a[...,1:]+np.cross(a[...,1:],b[...,1:])),axis=-1)


def dual_quaternion_binding(vertices,skin,transforms):
    real=np.array([Matrix(t[:3,:3].tolist()).to_quaternion()[:] for t in transforms])
    dual=.5*qmultiply(np.column_stack((np.zeros(len(transforms)),transforms[:,:3,3])),real)
    reference=real[skin.argmax(1)]
    sign=np.where(reference@real.T<0,-1.,1.)
    qr=(skin*sign)@real;qd=(skin*sign)@dual
    length=np.linalg.norm(qr,axis=1,keepdims=True);qr/=length;qd/=length
    qd-=qr*(qr*qd).sum(1,keepdims=True)
    conjugate=qr*np.array([1,-1,-1,-1])
    translations=2*qmultiply(qd,conjugate)[:,1:]
    rotated=vertices+2*np.cross(qr[:,1:],np.cross(qr[:,1:],vertices)+qr[:,:1]*vertices)
    return rotated+translations,qr,translations


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
