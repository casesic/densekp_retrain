import tempfile
import unittest
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from densekp_retrain.body import compute_densekp_3d, write_geometry_arrays


class BodyGeometryTest(unittest.TestCase):
    def test_densekp_expression_is_strictly_equivalent(self):
        rng = np.random.default_rng(20260920)
        regressor = rng.standard_normal((138, 10475), dtype=np.float32)
        vertices = rng.standard_normal((2, 10475, 3), dtype=np.float32)
        expected = np.einsum(
            "kv,tvc->tkc", regressor, vertices, optimize=True
        ).astype(np.float32)

        actual = compute_densekp_3d(regressor, vertices)

        self.assertTrue(np.array_equal(actual, expected))
        self.assertEqual(actual.dtype, np.float32)

    def test_geometry_arrays_round_trip_without_conversion(self):
        vertices = np.zeros((1, 10475, 3), dtype=np.float32)
        faces = np.array([[0, 1, 2]], dtype=np.int32)
        densekp_3d = np.zeros((1, 138, 3), dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            write_geometry_arrays(
                output_dir,
                vertices=vertices,
                faces=faces,
                densekp_3d=densekp_3d,
            )
            self.assertTrue(np.array_equal(np.load(output_dir / "vertices.npy"), vertices))
            self.assertTrue(np.array_equal(np.load(output_dir / "faces.npy"), faces))
            self.assertTrue(
                np.array_equal(np.load(output_dir / "densekp_3d.npy"), densekp_3d)
            )


if __name__ == "__main__":
    unittest.main()
