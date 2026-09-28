import unittest

import cv2
import numpy as np

from tools.annotation.build_omni_multiperson_tiny import roll_person
from tools.fisheye.build_ds_multiperson_pilot import remap_periodic_erp
from tools.fisheye.run_fisheye_multiperson_pilot import world_points


class MultipersonPilotTests(unittest.TestCase):
    def test_erp_seam_is_moved_outside_person(self):
        image = np.zeros((8, 16, 3), dtype=np.uint8)
        mask = np.zeros((8, 16), dtype=bool)
        mask[2:6, :2] = True
        mask[2:6, -2:] = True
        xy = np.array([[0.0, 3.0], [15.0, 4.0]])
        _, shifted_mask, shifted_xy, _ = roll_person(image, mask, xy)
        occupied = np.flatnonzero(shifted_mask.any(axis=0))
        self.assertLessEqual(int(occupied.max() - occupied.min()), 3)
        self.assertLessEqual(float(np.ptp(shifted_xy[:, 0])), 3)

    def test_world_transform_matches_blender_axis_convention(self):
        points = np.array([[1.0, 2.0, 3.0]])
        vertices = np.array([[0.0, -1.0, 0.0]])
        person = {"yaw_deg": 90.0, "location": [4.0, 5.0, 6.0]}
        transformed = world_points(points, vertices, person)
        np.testing.assert_allclose(transformed, [[7.0, 6.0, 9.0]], atol=1e-12)

    def test_erp_sampling_wraps_longitude_without_black_seam(self):
        image = np.array([[100, 0, 0, 200]], dtype=np.uint8)
        x = np.array([[3.5, -1]], dtype=np.float32)
        y = np.array([[0, -1]], dtype=np.float32)
        sampled = remap_periodic_erp(image, x, y, cv2.INTER_LINEAR)
        np.testing.assert_array_equal(sampled, [[150, 0]])


if __name__ == "__main__":
    unittest.main()
