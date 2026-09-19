"""NumPy implementation of the company Omni-Radtan camera model.

Camera coordinates are CV-style: +X right, +Y down and +Z optical forward.
``camera_pose`` is camera-to-world. Points and pose must use the same metric world
frame; projection depends only on ray direction.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class OmniRadtanIntrinsics:
    width: int = 3000; height: int = 3000
    cx: float = 1500.0; cy: float = 1500.0
    xi: float = 3.485643; f: float = 4495.392
    k1: float = 7.048e-1; k2: float = 7.548

    @classmethod
    def from_dict(cls, value): return cls(**value)
    def as_dict(self): return self.__dict__.copy()

def _camera_points(points_3d, camera_pose):
    p=np.asarray(points_3d,dtype=np.float64); pose=np.asarray(camera_pose,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or pose.shape!=(4,4): raise ValueError('expected points [N,3] and camera-to-world pose [4,4]')
    if not np.isfinite(p).all() or not np.isfinite(pose).all(): raise ValueError('non-finite input')
    return (p-pose[:3,3])@pose[:3,:3]

def project_points_omni_radtan(points_3d, camera_pose, intrinsics, return_details=False):
    """Project world points to top-left-origin pixels without using Blender."""
    q=_camera_points(points_3d,camera_pose); norm=np.linalg.norm(q,axis=1); ray=np.divide(q,norm[:,None],out=np.full_like(q,np.nan),where=norm[:,None]>0)
    rho=np.linalg.norm(ray[:,:2],axis=1); theta=np.arctan2(rho,ray[:,2]); phi=np.arctan2(ray[:,1],ray[:,0])
    den=np.cos(theta)+intrinsics.xi; m=np.divide(np.sin(theta),den,out=np.full_like(theta,np.nan),where=np.abs(den)>1e-15)
    r=intrinsics.f*(m+intrinsics.k1*m**3+intrinsics.k2*m**5)
    pixel=np.column_stack((intrinsics.cx+r*np.cos(phi),intrinsics.cy+r*np.sin(phi)))
    theta_limit=np.arccos(-1.0/intrinsics.xi) if intrinsics.xi>1 else np.pi
    model_domain=np.isfinite(pixel).all(1)&(norm>0)&(theta<=theta_limit+1e-12)&(m>=0)
    inside=model_domain&(pixel[:,0]>=0)&(pixel[:,0]<intrinsics.width)&(pixel[:,1]>=0)&(pixel[:,1]<intrinsics.height)
    if return_details: return pixel, {'camera_points':q,'rays':ray,'theta':theta,'phi':phi,'m':m,'radius':r,'model_domain':model_domain,'inside':inside}
    return pixel

def unproject_pixels_omni_radtan(pixels, intrinsics, iterations=64, return_valid=False):
    """Invert the monotone camera branch to unit CV rays for image resampling."""
    px=np.asarray(pixels,dtype=np.float64); d=px-np.array([intrinsics.cx,intrinsics.cy]); radius=np.linalg.norm(d,axis=1)
    mmax=1/np.sqrt(intrinsics.xi**2-1) if intrinsics.xi>1 else 10.0
    rmax=intrinsics.f*(mmax+intrinsics.k1*mmax**3+intrinsics.k2*mmax**5); valid=np.isfinite(px).all(1)&(radius<=rmax+1e-9)
    lo=np.zeros(len(px)); hi=np.full(len(px),mmax)
    target=np.minimum(radius,rmax)
    for _ in range(iterations):
        mid=(lo+hi)/2; value=intrinsics.f*(mid+intrinsics.k1*mid**3+intrinsics.k2*mid**5); take=value<target; lo[take]=mid[take]; hi[~take]=mid[~take]
    m=(lo+hi)/2; arg=np.clip(intrinsics.xi*m/np.sqrt(1+m*m),-1,1); theta=np.arctan(m)+np.arcsin(arg); phi=np.arctan2(d[:,1],d[:,0])
    rays=np.column_stack((np.sin(theta)*np.cos(phi),np.sin(theta)*np.sin(phi),np.cos(theta))); rays[~valid]=np.nan
    return (rays,valid) if return_valid else rays
