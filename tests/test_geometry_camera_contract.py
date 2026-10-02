"""Frozen pre-migration arrays plus geometric meanings; no assets or Blender."""

import base64
import io
import json
from pathlib import Path
import unittest
import warnings
import numpy as np
from tests.geometry_camera_fixture import api, synthetic, TARGETS
from densekp_retrain.body.transforms import world_points
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.visibility import erp_visibility, visible_at_instance_points


class GeometryCameraContractTests(unittest.TestCase):
    def test_frozen_array_and_rejection_results(self):
        fixture = json.loads(
            (Path(__file__).parent / "fixtures/geometry_camera_v1.json").read_text()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = synthetic(api())
        self.assertEqual(json.dumps(result.records), json.dumps(fixture["records"]))
        with np.load(
            io.BytesIO(base64.b64decode(fixture["arrays_npz_base64"])), allow_pickle=False
        ) as old:
            self.assertEqual(list(result.arrays), old.files)
            for key, array in result.arrays.items():
                self.assertEqual(array.dtype, old[key].dtype, key)
                self.assertEqual(array.shape, old[key].shape, key)
                np.testing.assert_array_equal(array, old[key], err_msg=key)

    def test_clothing_lift_uses_body_and_does_not_modify_either_mesh(self):
        body = np.array([[0, -1, 0], [1, 1, 1]], np.float32)
        cloth = np.array([[0, 0.5, 0.25]], np.float32)
        before = cloth.copy()
        world = world_points(cloth, body, dict(yaw_deg=0, location=[1, 2, 3]))
        np.testing.assert_array_equal(world, [[1, 1.75, 4.5]])
        np.testing.assert_array_equal(cloth, before)
        self.assertEqual(world.dtype, np.float64)

    def test_physical_axes_erp_alignment_and_range_units(self):
        poses, lenses, yaw = rig_poses([0, 0, 0], 27, 0.18)
        np.testing.assert_array_equal(poses["cam2"][:3, :3], poses["cam4"][:3, :3])
        np.testing.assert_allclose(
            lenses["cam2"][:3, 2], -lenses["cam4"][:3, 2], atol=1e-15, rtol=0
        )
        self.assertAlmostEqual(np.linalg.norm(poses["cam2"][:3, 3] - poses["cam4"][:3, 3]), 0.18)
        for name, camera in poses.items():
            q = np.array([[1.0, 1.0, 1.0]])
            _, distance = project_erp(q @ camera[:3, :3].T + camera[:3, 3], camera, 32, 16)
            self.assertAlmostEqual(distance[0], np.sqrt(3))
            forward = lenses[name][:3, 3] + lenses[name][:3, 2] * 3
            self.assertTrue(
                point_in_view(forward[None], camera, name, np.ones((16, 32), bool))[-1][0]
            )
            self.assertFalse(
                point_in_view(forward[None], camera, name, np.zeros((16, 32), bool))[-1][0]
            )

    def test_periodic_depth_and_rounded_instance_rules_remain_different(self):
        xy = np.array([[7.99, 4.0]])
        depth = np.zeros((8, 8))
        depth[4, 0] = 3
        index = np.zeros((8, 8))
        index[4, 0] = 1
        valid, seen = erp_visibility(xy, np.array([3.0]), depth, index, [1])
        self.assertTrue(valid[0])
        self.assertTrue(seen[0])
        self.assertFalse(visible_at_instance_points(xy, index > 0, np.ones((8, 8), bool))[0])
        self.assertFalse(erp_visibility(xy, np.array([3.05]), depth, index, [1])[1][0])
        self.assertFalse(erp_visibility(xy, np.array([3.0]), depth, index, [2])[1][0])

    def test_old_imports_are_compatibility_exports(self):
        # Identity checks cover compatibility only; the frozen test above proves behavior.
        before_names = api(original=True)
        public = api()
        for name, value in public.items():
            self.assertIs(before_names[name], value, name)


if __name__ == "__main__":
    unittest.main()
