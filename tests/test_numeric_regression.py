"""Read-only regression checks against the frozen on-disk render baselines."""

import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from densekp_retrain.camera import OmniRadtanIntrinsics, project_points_omni_radtan


class FrozenNumericRegressionTest(unittest.TestCase):
    def test_perspective_densekp_and_camera_xyz(self):
        output = ROOT / "outputs/p0_dressed_eye_background_fixed"
        required = [
            output / "manifest.jsonl",
            output / "geometry/densekp_3d.npy",
            output / "annotations/densekp_2d.npy",
            output / "annotations/densekp_camera_xyz.npy",
        ]
        if not all(path.is_file() for path in required):
            self.skipTest("frozen perspective output is not available")
        rows = [json.loads(line) for line in required[0].read_text().splitlines()]
        dense = np.load(required[1]); saved_2d = np.load(required[2]); saved_xyz = np.load(required[3])
        max_2d = max_xyz = 0.0
        for frame in (0, 8, 15):
            row = rows[frame]
            points = np.c_[dense[frame], np.ones(len(dense[frame]))]
            world = (np.asarray(row["object_matrix_world"]) @ points.T).T
            camera = (np.linalg.inv(np.asarray(row["camera_matrix_world"])) @ world.T).T[:, :3]
            depth = -camera[:, 2]; intrinsics = row["intrinsics"]
            projected = np.column_stack((
                intrinsics["fx"] * camera[:, 0] / depth + intrinsics["cx"],
                intrinsics["cy"] - intrinsics["fy"] * camera[:, 1] / depth,
            ))
            max_xyz = max(max_xyz, float(np.max(np.abs(camera - saved_xyz[frame]))))
            max_2d = max(max_2d, float(np.max(np.abs(projected - saved_2d[frame]))))
        self.assertLessEqual(max_2d, 5e-5)
        self.assertLessEqual(max_xyz, 2e-7)

    def test_fisheye_densekp(self):
        output = ROOT / "outputs/fisheye_camera_p0"
        geometry = ROOT / "outputs/temporal_case_aaron_1095/geometry"
        required = [output / "camera_config.json", output / "densekp_3d_source.npy", geometry / "vertices.npy"]
        if not all(path.is_file() for path in required):
            self.skipTest("frozen fisheye output is not available")
        config = json.loads(required[0].read_text())
        intrinsics = OmniRadtanIntrinsics.from_dict(config["intrinsics"])
        local = np.load(required[1])[0]
        vertices = np.load(required[2])[0]
        lift = -float(vertices[:, 1].min())
        world = np.column_stack((local[:, 0], -local[:, 2], local[:, 1] + lift))
        case_dirs = {"fisheye_center": "center", "cam_left": "seam/cam_left", "cam_right": "seam/cam_right"}
        max_error = 0.0
        for camera in config["cameras"]:
            saved_path = output / case_dirs[camera["id"]] / "densekp_fisheye_2d.npy"
            if not saved_path.is_file():
                self.skipTest(f"frozen fisheye case is not available: {saved_path}")
            projected = project_points_omni_radtan(world, np.asarray(camera["camera_to_world"]), intrinsics)
            saved = np.load(saved_path)[0]
            max_error = max(max_error, float(np.max(np.abs(projected - saved))))
        self.assertLessEqual(max_error, 1.3e-4)


if __name__ == "__main__":
    unittest.main()
