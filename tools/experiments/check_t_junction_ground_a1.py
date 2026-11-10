"""Check the frozen three-rider turn against the delivered blend's native road."""
import json
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.data.dynamic_scene import geometry_paths


def main():
    prepared=Path(sys.argv[sys.argv.index('--prepared')+1]).resolve()
    output=Path(sys.argv[sys.argv.index('--output')+1]).resolve()
    recipe=json.loads((prepared/'recipe.json').read_text())
    config=json.loads(Path(recipe['scene']).read_text())
    blend=(ROOT/config['blend_path']).resolve()
    bpy.ops.wm.open_mainfile(filepath=str(blend))
    scene=bpy.context.scene
    matrices=np.load(prepared/'scene_matrices.npy')
    result=[]
    for j,person in enumerate(recipe['people']):
        asset=json.loads(Path(person['asset_manifest']).read_text())
        body=np.load(geometry_paths(asset)['body_vertices'],mmap_mode='r')
        allowed=set(person['support_checks']['road_object_names'])
        for k,index in enumerate(person['asset_frames']):
            M=matrices[k,j]
            world=body[index]@M[:3,:3].T+M[:3,3]
            low=world[world[:,2]<world[:,2].min()+.02][::10]
            hits=[]
            for point in low:
                hit,at,normal,face,obj,_=scene.ray_cast(
                    bpy.context.evaluated_depsgraph_get(),Vector((float(point[0]),float(point[1]),float(point[2]+.5))),
                    Vector((0.,0.,-1.)),distance=1.)
                if not hit or obj.name not in allowed or abs(at.z)>person['support_checks']['road_z_tolerance_m'] or normal.z<.9:
                    raise ValueError(f'native_road_support_conflict:p{j+1}:f{k}:object={obj.name if hit else None}')
                hits.append(dict(object=obj.name,gap_m=float(point[2]-at.z)))
            gaps=[v['gap_m'] for v in hits]
            low_limit,high_limit=person['support_checks']['body_min_clearance_m']
            if min(gaps)<low_limit or max(gaps)>high_limit:
                raise ValueError(f'body_road_gap_conflict:p{j+1}:f{k}')
            result.append(dict(person_id=j+1,frame_index=k,objects=sorted({v['object'] for v in hits}),
                               minimum_gap_m=min(gaps),maximum_gap_m=max(gaps),sample_count=len(hits)))
    output.write_text(json.dumps(dict(status='passed',source_blend=str(blend),
        checks=len(result),support_samples=sum(v['sample_count'] for v in result),frames=result,
        limitation='low body vertices every tenth, 31 discrete timestamps; not continuous collision'),indent=2)+'\n')
    print('NATIVE_ROAD_SUPPORT_PASSED',len(result))


if __name__=='__main__':main()
