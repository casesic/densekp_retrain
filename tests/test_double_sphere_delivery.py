"""Regression checks for the DenseKP delivery camera convention."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from densekp_retrain.camera.double_sphere import (  # noqa: E402
    CAMERA_EXTRINSIC, DS_20260320, ROTATION_MATRIX,
    build_erp_to_fisheye_lut, project_double_sphere, unproject_double_sphere,
)


class DoubleSphereDeliveryTest(unittest.TestCase):
    def test_production_axes(self):
        np.testing.assert_array_equal(ROTATION_MATRIX, np.eye(3))
        np.testing.assert_array_equal(CAMERA_EXTRINSIC["cam2"], np.eye(3))
        np.testing.assert_array_equal(CAMERA_EXTRINSIC["cam4"], np.diag([-1, 1, -1]))

    def test_center_erp_maps_to_respective_fisheye_centers(self):
        width, height = 360, 180
        for camera in ("cam2", "cam4"):
            with self.subTest(camera=camera):
                map_x, map_y = build_erp_to_fisheye_lut(
                    DS_20260320[camera], camera, width, height)
                x = 180 if camera == "cam2" else 0
                self.assertAlmostEqual(map_x[90, x], DS_20260320[camera].cx, places=3)
                self.assertAlmostEqual(map_y[90, x], DS_20260320[camera].cy, places=3)

    def test_projection_inverse(self):
        for camera in ("cam2", "cam4"):
            with self.subTest(camera=camera):
                params = DS_20260320[camera]
                pixels = np.array([[params.cx, params.cy],
                                   [params.cx + 300, params.cy + 200]])
                rays, valid = unproject_double_sphere(pixels, params)
                projected, projected_valid = project_double_sphere(rays, params)
                self.assertTrue(valid.all() and projected_valid.all())
                np.testing.assert_allclose(projected, pixels, atol=1e-9)

    def test_real_mask_matches_delivery_projection_when_available(self):
        masks_path = Path(__file__).resolve().parents[1] / "assets_external/real_erp_valid_masks.npz"
        if not masks_path.exists():
            self.skipTest("real ERP masks are optional local assets")
        with np.load(masks_path) as masks:
            for camera in ("cam2", "cam4"):
                with self.subTest(camera=camera):
                    measured = masks[f"valid_masks_{camera}"][::16, ::16]
                    map_x, map_y = build_erp_to_fisheye_lut(
                        DS_20260320[camera], camera, 360, 180)
                    projected = ((map_x >= 0) & (map_x < 2880) &
                                 (map_y >= 0) & (map_y < 2880))
                    self.assertEqual(measured.shape, projected.shape)
                    self.assertGreater(float(projected[measured].mean()), .999)


if __name__ == "__main__":
    unittest.main()
