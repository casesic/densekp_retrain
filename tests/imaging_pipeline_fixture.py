"""Small deterministic buffers exercising original post-processing edge behavior."""
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch
import warnings
import numpy as np
from PIL import Image


def capture(output, *, migrated=False):
    from tools.fisheye import run_fisheye_camera_p0 as old_omni
    from tools.fisheye import run_ds_scene_pair_pilot as old_ds
    if migrated:
        from densekp_retrain.imaging import omni, ds
        from densekp_retrain.imaging import source as source_module
    else:
        omni,ds=old_omni,old_ds
        source_module=None
    from densekp_retrain.camera import OmniRadtanIntrinsics
    from densekp_retrain.camera.double_sphere import DS_20260320
    from densekp_retrain.camera.rig import rig_poses
    output.mkdir(parents=True,exist_ok=False)
    height,width=24,48
    rgb=np.arange(height*width*3,dtype=np.float32).reshape(height,width,3)%31/31
    depth=np.ones((height,width),np.float32); depth[0,:]=np.inf; depth[-1,:]=np.nan
    instance=np.ones((height,width),np.float32);instance[:,:width//2]=2
    k=OmniRadtanIntrinsics(width=32,height=32,cx=16,cy=16,f=28)
    points=np.tile([[0.,0.,1.],[.2,.1,.97],[-.2,.1,.97]],(46,1))
    results={}
    for label in ('normal','person_ids','bad_person_ids','nonfinite_points','empty_points','zero_depth'):
        dest=output/label
        kwargs={}; pts=points.copy();distance=depth.copy()
        if label in ('person_ids','bad_person_ids'):
            kwargs=dict(person_ids=[1,2],point_person_ids=np.repeat([1,2],69) if label=='person_ids' else None)
        if label=='nonfinite_points': pts[0,0]=np.nan
        if label=='empty_points': pts=pts[:0]
        if label=='zero_depth': distance[:]=0
        owner=source_module if migrated else omni
        with patch.object(owner,'read_combined_depth_index',return_value=(rgb.copy(),distance,instance.copy(),['Combined','Depth','IndexOB'])):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                try:
                    omni.warp_case(Path('/fixture/source.exr'),dest,pts,np.eye(4),k,**kwargs)
                    results[label]=dict(exception=None)
                except Exception as error:
                    results[label]=dict(exception=type(error).__name__,message=str(error))
        results[label]['files']=sorted(p.name for p in dest.iterdir())
    intrinsics={n:replace(v,width=64,height=64,fx=v.fx/45,fy=v.fy/45,cx=v.cx/45,cy=v.cy/45) for n,v in DS_20260320.items()}
    poses,_,_=rig_poses([0,0,0],0,.18)
    vertices=np.array([[[0.,-1.,0.],[.2,1.,.1]]]);dense=np.tile([[[0.,.3,-2.],[0.,.3,2.]]],(1,69,1))
    people=[dict(id=1,frame=0,yaw_deg=0,location=[0,0,0])]
    for label in ('blocked_lens','bad_lens_shape'):
        dest=output/label; (dest/'source').mkdir(parents=True)
        for name in poses:
            Image.fromarray(np.full((height,width,3),100,np.uint8)).save(dest/'source'/f'{name}_preview.png')
        lenses={n:np.ones((64,64),bool) for n in poses};lenses['cam4'][:]=False
        if label=='bad_lens_shape': lenses['cam2']=np.ones((63,64),bool)
        owner=source_module if migrated else ds
        with patch.object(ds,'DS_20260320',intrinsics),patch('densekp_retrain.camera.domain.DS_20260320',intrinsics),patch.object(owner,'read_combined_depth_index',return_value=(rgb.copy(),depth.copy(),instance.copy(),[])):
            try:
                result=ds.process_scene(dest,people,poses,{n:np.ones((height,width),bool) for n in poses},vertices,dense,lens_masks=lenses,lens_metadata={n:{} for n in poses})
                (dest/'report.json').write_text(json.dumps(result,indent=2)+'\n')
                results[label]=dict(exception=None)
            except Exception as error:
                results[label]=dict(exception=type(error).__name__,message=str(error))
        results[label]['files']=sorted(str(p.relative_to(dest)) for p in dest.rglob('*') if p.is_file())
    (output/'exceptions.json').write_text(json.dumps(results,indent=2)+'\n')
