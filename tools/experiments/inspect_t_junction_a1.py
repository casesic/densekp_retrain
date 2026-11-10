"""Read-only inspection of the two end junctions in the delivered city blend."""
import json
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def bounds(obj):
    points = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
    return [[min(p[i] for p in points), max(p[i] for p in points)] for i in range(3)]


def main():
    scene_path = Path(sys.argv[sys.argv.index('--scene') + 1])
    output_path = Path(sys.argv[sys.argv.index('--output') + 1])
    bpy.ops.wm.open_mainfile(filepath=str(scene_path))
    scene = bpy.context.scene
    objects = []
    for obj in scene.objects:
        if obj.type != 'MESH' or obj.hide_render:
            continue
        b = bounds(obj)
        if any(b[1][0] <= y <= b[1][1] for y in (-84., 84.)):
            objects.append(dict(name=obj.name, bounds_xyz=b,
                                collection=[c.name for c in obj.users_collection]))
    deps = bpy.context.evaluated_depsgraph_get()
    samples = []
    for endpoint in (-84., 84.):
        for y in (endpoint-8, endpoint-6, endpoint-4, endpoint-2, endpoint,
                  endpoint+2, endpoint+4, endpoint+6, endpoint+8):
            for x in (-12., -8., -5., -3., 0., 3., 5., 8., 12.):
                hit, point, normal, face, obj, matrix = scene.ray_cast(
                    deps, Vector((x, y, 4.)), Vector((0., 0., -1.)), distance=8.)
                samples.append(dict(x=x, y=y, object=obj.name if hit else None,
                                    z=float(point.z) if hit else None))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(dict(scene=str(scene_path), objects=objects,
                                           ground_samples=samples), indent=2) + '\n')
    print(json.dumps(dict(object_count=len(objects), sample_count=len(samples))))


if __name__ == '__main__':
    main()
