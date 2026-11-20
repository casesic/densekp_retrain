"""Check the selected new paired camera centers against static city geometry."""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/experiments/front_rig_size_repair_a1_20261011'


def main():
    rig = json.loads((OUT / 'prepared_front_revised/rig.json').read_text())
    scene = bpy.context.scene
    graph = bpy.context.evaluated_depsgraph_get()
    cameras = []
    for camera in rig['cameras']:
        matrix = camera['camera_to_world']
        center = Vector((matrix[0][3], matrix[1][3], matrix[2][3]))
        hits = []
        for j in range(12):
            angle = 2 * math.pi * j / 12
            direction = Vector((math.cos(angle), math.sin(angle), 0.0))
            hit, location, normal, face, obj, transform = scene.ray_cast(graph, center, direction, distance=0.25)
            if hit:
                hits.append(dict(direction_index=j, object=obj.name, distance_m=float((location-center).length)))
        cameras.append(dict(camera_id=camera['id'], center_world_m=list(center), horizontal_25cm_hits=hits))
    result = dict(schema='repaired_front_rig_city_static_short_ray_v1', scene=str(bpy.data.filepath),
                  ray_count_per_camera=12, ray_length_m=0.25,
                  scope='static evaluated scene horizontal rays, not a swept collision proof',
                  cameras=cameras, passed=all(not x['horizontal_25cm_hits'] for x in cameras))
    with (OUT / 'static_camera_clearance.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if not result['passed']:
        raise RuntimeError('repaired_front_camera_static_clearance_failed')


if __name__ == '__main__':
    main()
