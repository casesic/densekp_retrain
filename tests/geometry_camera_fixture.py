"""Fixed CPU arrays for independent geometry/camera acceptance, without Blender."""

import importlib
import json
from pathlib import Path
import numpy as np

TARGETS = {
    "body.parameters": ["PARAMETERS", "evaluate", "zero_pose", "rest_body"],
    "body.measurements": ["measurement_rows", "measure"],
    "io.geometry": ["save_geometry"],
    "clothing.shells": ["vertex_normals", "outfit_regions", "shell", "clipped_shell"],
    "clothing.binding": [
        "clip_surface",
        "closest_surface",
        "bind_new_points",
        "pose_rotations",
        "CONV",
        "frame",
        "triangle_frame",
    ],
    "body.transforms": ["world_points"],
    "camera.erp": ["project_erp", "erp_points"],
    "camera.rig": ["rig_poses"],
    "camera.domain": ["point_in_view"],
    "camera.visibility": ["erp_visibility", "visible_at_instance_points"],
}


def api():
    functions = {}
    for destination, names in TARGETS.items():
        module = importlib.import_module("densekp_retrain." + destination)
        for name in names:
            functions[name] = getattr(module, name)
    return functions


class Capture:
    def __init__(self):
        self.arrays = {}
        self.records = {}

    def encode(self, value, path):
        if isinstance(value, (np.ndarray, np.generic)):
            a = np.asarray(value)
            self.arrays[path] = a.copy()
            return {
                "array": path,
                "dtype": str(a.dtype),
                "shape": list(a.shape),
                "type": type(value).__name__,
            }
        if isinstance(value, dict):
            return {k: self.encode(v, path + "/" + k) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return {
                "type": type(value).__name__,
                "items": [self.encode(v, path + "/" + str(i)) for i, v in enumerate(value)],
            }
        return value

    def call(self, name, function, *args, **kwargs):
        try:
            result = function(*args, **kwargs)
            self.records[name] = {"value": self.encode(result, name), "error": None}
            return result
        except Exception as e:
            self.records[name] = {
                "value": None,
                "error": {"type": type(e).__name__, "message": str(e)},
            }
            return None

    def save(self, folder):
        folder.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(folder / "arrays.npz", **self.arrays)
        (folder / "records.json").write_text(
            json.dumps(self.records, indent=2, allow_nan=True) + "\n"
        )


def synthetic(functions):
    c = Capture()
    f = functions
    rest = np.array(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], np.float32
    )
    faces = np.array([[a, b, c] for a in [0, 1] for b in [2, 3] for c in [4, 5]])
    for face in faces:
        v = rest[face]
        if np.cross(v[1] - v[0], v[2] - v[0]) @ v.mean(0) < 0:
            face[[1, 2]] = face[[2, 1]]
    joints = np.zeros((55, 3), np.float32)
    joints[12, 1] = 0.3
    joints[[1, 2], 1] = -0.15
    joints[[4, 5], 1] = -0.6
    joints[16, 0] = 0.5
    joints[18, 0] = 0.8
    joints[17, 0] = -0.5
    joints[19, 0] = -0.8
    joints[20, 0] = 1
    joints[21, 0] = -1
    joints[[7, 8], 1] = -0.95
    c.call("zero_pose", f["zero_pose"], np.zeros(10, np.float32), np.array([1, 2, 3], np.float32))
    c.call("normals", f["vertex_normals"], rest, faces)
    for offset in (0.004, 0, 0.011, np.nan):
        c.call("shell_" + str(offset), f["shell"], rest, faces, np.ones(len(faces), bool), offset)
    c.call(
        "undefined_normals",
        f["shell"],
        np.zeros_like(rest),
        faces,
        np.ones(len(faces), bool),
        0.004,
    )
    for profile in (None, dict(sleeve=0.7, leg=0.8, collar=0.02)):
        c.call("clipped_" + str(profile), f["clipped_shell"], rest, joints, rest, faces, profile)
    c.call(
        "clip_surface",
        f["clip_surface"],
        rest,
        faces,
        [(0, 1, 0.6), (0, -1, 0.6), (1, 1, 0.4), (1, -1, 0.4)],
    )
    c.call("closest_surface", f["closest_surface"], rest[:3] * 1.01, rest, faces)
    c.call("binding", f["bind_new_points"], rest[:3] * 1.01, rest, faces)
    c.call(
        "frame",
        f["frame"],
        np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]),
        np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
    )
    c.call("triangle_frame", f["triangle_frame"], rest, faces)
    c.call("world_points", f["world_points"], rest, rest, dict(yaw_deg=37, location=[2, -1, 0.3]))
    c.call("outfit_tiny_rejection", f["outfit_regions"], rest, joints, faces)
    c.call(
        "measurement_tiny_rejection", f["measurement_rows"], rest, joints, np.zeros(6), np.zeros(6)
    )
    for baseline in (0.18, 0, -1, np.nan):
        c.call("rig_" + str(baseline), f["rig_poses"], [0.2, -0.3, 0.4], 17, baseline)
    poses, _, _ = f["rig_poses"]([0, 0, 0], 0, 0.18)
    # Seam sides, poles, zero-length ray and nonfinite ray: retain old NaNs/warnings.
    rays = np.array(
        [
            [1e-12, 0, -1],
            [-1e-12, 0, -1],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 0],
            [np.nan, 0, 1],
        ]
    )
    for name, camera in poses.items():
        world = rays @ camera[:3, :3].T + camera[:3, 3]
        c.call("erp_" + name, f["project_erp"], world, camera, 32, 16)
        c.call("legacy_erp_" + name, f["erp_points"], world, camera, 32, 16)
        mask = np.ones((16, 32), bool)
        mask[:, :5] = False
        c.call("ds_domain_" + name, f["point_in_view"], world, camera, name, mask)
        c.call(
            "ds_empty_mask_" + name, f["point_in_view"], world, camera, name, np.zeros_like(mask)
        )
    depth = np.full((16, 32), 3.0, np.float32)
    index = np.ones_like(depth)
    index[5:12, 10:18] = 2
    depth[5:12, 10:18] = 5
    xy = np.tile(
        np.array([[0, 8], [31.9, 8], [10.5, 8], [32, 8], [-0.2, 8], [4, 16], [np.nan, 1]], float),
        (40, 1),
    )[:276]
    distances = np.full(276, 3.0)
    distances[138:] = 5.0
    c.call("erp_visibility", f["erp_visibility"], xy, distances, depth, index, [1, 2])
    for id in (1, 2):
        c.call(
            "instance_visibility_" + str(id),
            f["visible_at_instance_points"],
            xy,
            index == id,
            np.ones_like(index, bool),
        )
    c.call("CONV", lambda: f["CONV"])
    return c
