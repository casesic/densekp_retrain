"""Bounded static scene-ray clearance check at both paired-rig optical centers."""
import json
from pathlib import Path
import sys
import time

import bpy
from mathutils import Vector
import numpy as np


def main():
    root=Path(sys.argv[sys.argv.index('--root')+1]).resolve()
    common=json.loads((root/'recipe.json').read_text())
    scene_cfg=json.loads(Path(common['scene']).read_text())
    project=Path(__file__).resolve().parents[2]
    bpy.ops.wm.open_mainfile(filepath=str(project/scene_cfg['blend_path']))
    scene=bpy.context.scene;deps=bpy.context.evaluated_depsgraph_get()
    directions=[Vector(x) for x in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1),
                                   (1,1,0),(-1,1,0),(1,-1,0),(-1,-1,0))]
    directions=[d.normalized() for d in directions];records=[];start=time.monotonic()
    spec=json.loads((root/'multirig.json').read_text())
    for entry in spec['rigs']:
        rig_id=entry['rig_id']
        rig=json.loads((root/'rigs'/rig_id/'rig.json').read_text())
        for camera in rig['cameras']:
            center=np.asarray(camera['camera_to_world'],float)[:3,3]
            if not np.isfinite(center).all():raise ValueError('nonfinite_camera_center')
            hits=[]
            for direction in directions:
                hit,point,normal,face,obj,_=scene.ray_cast(deps,Vector(center.tolist()),direction,distance=.15)
                if hit:hits.append(dict(object=obj.name,distance_m=float((point-Vector(center.tolist())).length)))
            records.append(dict(rig_id=rig_id,camera_id=camera['id'],center_m=center.tolist(),
                                nearby_static_mesh_hits=hits,clearance_radius_m=.15,passed=not hits))
    result=dict(schema='camera_center_static_ray_clearance_v1',records=records,all_passed=all(x['passed'] for x in records),
                checked_directions=len(directions),elapsed_seconds=time.monotonic()-start,
                limitation='short rays against scene evaluated meshes; not a swept-volume or geometry-node instance collision proof')
    (root/'camera_clearance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if not result['all_passed']:raise ValueError('camera_center_static_clearance_failed')


if __name__=='__main__':main()
