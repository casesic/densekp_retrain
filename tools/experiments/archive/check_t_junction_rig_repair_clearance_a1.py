"""Evaluate frozen paired camera centers against native city geometry."""
import argparse
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else [])
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))
    from densekp_retrain.camera.rig import rig_poses
    data = json.loads(args.candidates.read_text())
    depsgraph = bpy.context.evaluated_depsgraph_get()
    rows = []
    for entry in data["candidates"]:
        poses, _, _ = rig_poses(entry["origin_m"], entry["yaw_deg"], data["baseline_m"])
        cameras = []
        for name, matrix in poses.items():
            center = Vector(matrix[:3, 3])
            hits = []
            for j in range(16):
                a = 2 * math.pi * j / 16
                ray = Vector((math.cos(a), math.sin(a), 0))
                hit, loc, normal, face, obj, object_matrix = bpy.context.scene.ray_cast(
                    depsgraph, center, ray, distance=0.3)
                if hit:
                    hits.append(dict(object=obj.name, distance_m=float((loc - center).length), ray=j))
            cameras.append(dict(camera_id=name, center_world_m=list(center), horizontal_hits=hits))
        rows.append(dict(view=entry["view"], attempt=entry["attempt"], cameras=cameras,
                         passed=all(not c["horizontal_hits"] for c in cameras)))
    args.output.write_text(json.dumps(dict(schema="junction_repair_static_clearance_v1",
          scene=bpy.data.filepath, ray_count_per_camera=16, ray_length_m=0.3,
          scope="evaluated native scene short horizontal rays; not continuous collision proof",
          checks=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
