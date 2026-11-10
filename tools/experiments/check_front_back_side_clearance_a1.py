"""Short static-ray clearance audit for the selected paired city rig centers."""
import json
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/experiments/front_back_side_multirig_a1_20261011'


def main():
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    result = []
    for rig in ('back', 'front', 'side'):
        data = json.loads((OUT / f'prepared_{rig}/rig.json').read_text())
        for camera in data['cameras']:
            center = Vector((camera['camera_to_world'][0][3],
                             camera['camera_to_world'][1][3],
                             camera['camera_to_world'][2][3]))
            hits=[]
            for j in range(12):
                angle = 2 * math.pi * j / 12
                direction = Vector((math.cos(angle), math.sin(angle), 0.0))
                hit, location, normal, face, obj, matrix = scene.ray_cast(depsgraph, center, direction, distance=0.25)
                if hit:
                    hits.append(dict(object=obj.name, distance_m=float((location-center).length),
                                     direction_index=j))
            result.append(dict(rig_id=f'rig_{rig}', camera_id=camera['id'],
                               center_world_m=list(center), horizontal_25cm_hits=hits,
                               short_ray_clearance_passed=not hits))
    evidence=dict(schema='selected_city_rig_static_short_ray_v1',
                  scene=str(bpy.data.filepath), ray_count_per_camera=12, ray_length_m=.25,
                  scope='static evaluated scene short horizontal rays; not full swept collision proof',
                  cameras=result, passed=all(x['short_ray_clearance_passed'] for x in result))
    (OUT / 'static_camera_clearance.json').write_text(json.dumps(evidence,indent=2)+'\n')
    if not evidence['passed']:
        raise RuntimeError('selected_rig_static_camera_clearance_failed')


if __name__ == '__main__':
    main()
