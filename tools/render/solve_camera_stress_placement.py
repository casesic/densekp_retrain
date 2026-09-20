"""Deterministic geometry-based camera translation solver with fixed intrinsics."""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiments/perspective_camera_stress'

def project(points,world,camera,k):
    q=np.c_[points,np.ones(len(points))]@(np.linalg.inv(camera)@world).T
    depth=-q[:,2]
    return np.c_[k['fx']*q[:,0]/depth+k['cx'],k['cy']-k['fy']*q[:,1]/depth],depth

def bbox_metrics(xy,w,h):
    x0,y0=xy.min(0);x1,y1=xy.max(0);bw=x1-x0;bh=y1-y0
    clipped=[float(np.clip(x0,0,w)),float(np.clip(y0,0,h)),float(np.clip(x1,0,w)),float(np.clip(y1,0,h))]
    return dict(full_body_bbox=[float(x0),float(y0),float(x1),float(y1)],in_frame_body_bbox=clipped,
                bbox_height_fraction=float(bh/h),bbox_width_fraction=float(bw/w),
                bbox_visible_fraction_x=max(0,clipped[2]-clipped[0])/bw,
                bbox_visible_fraction_y=max(0,clipped[3]-clipped[1])/bh,
                left_truncation_fraction=float(np.clip(-x0/bw,0,1)),
                right_truncation_fraction=float(np.clip((x1-w)/bw,0,1)),
                top_truncation_fraction=float(np.clip(-y0/bh,0,1)),
                bottom_truncation_fraction=float(np.clip((y1-h)/bh,0,1)))

def solve(vertices,clothing,row,config,safety):
    world=np.array(row['object_matrix_world']);camera=np.array(row['camera_matrix_world']);k=row['intrinsics']
    w,h=row['resolution_width'],row['resolution_height']
    q=np.c_[vertices,np.ones(len(vertices))]@(np.linalg.inv(camera)@world).T
    def residual(t):
        depth=-q[:,2]+t[2]
        x=k['fx']*(q[:,0]-t[0])/depth+k['cx'];y=k['cy']-k['fy']*(q[:,1]-t[1])/depth
        x0,x1=x.min(),x.max();y0,y1=y.min(),y.max()
        horizontal=config['horizontal'];vertical=config['vertical']
        rx=(x0+x1)/2-w/2
        if horizontal=='left_margin':rx=x0-config['margin']*w
        if horizontal=='right_margin':rx=x1-(1-config['margin'])*w
        if horizontal=='left_truncation':rx=x0+config['truncation']*(x1-x0)
        ry=(y0+y1)/2-h/2
        if vertical=='bottom_truncation':ry=y1-h-config['truncation']*(y1-y0)
        return np.array([rx/w,ry/h,(y1-y0)/h-config['height']])
    _,cd=project(clothing,world,camera,k)
    min_depth=min(float((-q[:,2]).min()),float(cd.min()))
    lower=row['near_clip']+safety-min_depth
    result=least_squares(residual,[0,0,max(0,lower+.01)],bounds=([-np.inf,-np.inf,lower],[np.inf]*3),
                         ftol=1e-12,xtol=1e-12,gtol=1e-12,max_nfev=100)
    assert result.success and max(abs(residual(result.x)))<1e-7
    actual=camera.copy();actual[:3,3]+=camera[:3,:3]@result.x
    xy,depth=project(vertices,world,actual,k);_,clothing_depth=project(clothing,world,actual,k)
    assert min(depth.min(),clothing_depth.min())>row['near_clip']+safety-1e-8
    return dict(camera_matrix_world=actual.tolist(),camera_local_translation=result.x.tolist(),
                body_world_transform=world.tolist(),solver_max_normalized_residual=float(max(abs(residual(result.x)))),
                camera_near_clip=row['near_clip'],min_body_camera_depth=float(depth.min()),
                min_clothing_camera_depth=float(clothing_depth.min()),**bbox_metrics(xy,w,h))

def main():
    cfg=json.loads((ROOT/'configs/perspective_camera_stress.json').read_text())
    base=ROOT/cfg['appearance_baseline'];gd=ROOT/cfg['geometry_source']
    rows=[json.loads(x) for x in (base/'manifest.jsonl').read_text().splitlines()]
    vertices=np.load(gd/'vertices.npy');clothing=np.load(gd/'clothing_vertices.npy')
    (OUT/'configs').mkdir(parents=True,exist_ok=True)
    for config in cfg['configs']:
        frames=[dict(frame_id=i,source_motion_frame=r['source_frame_30fps'],**solve(vertices[i],clothing[i],r,config,cfg['near_safety_margin_m'])) for i,r in enumerate(rows)]
        path=OUT/'configs'/f"{config['name']}.json"
        path.write_text(json.dumps(dict(config=config,frames=frames),indent=2)+'\n')
        print(config['name'],'height',np.mean([x['bbox_height_fraction'] for x in frames]),'min depth',min(x['min_body_camera_depth'] for x in frames))
    (OUT/'configs/definition.json').write_text(json.dumps(cfg,indent=2)+'\n')

if __name__=='__main__':main()
