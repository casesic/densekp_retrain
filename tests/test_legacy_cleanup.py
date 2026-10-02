"""Keep crop values, RNG and rejection checks until the crop API retirement review."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from tests.legacy_cleanup_fixture import capture_crop

ROOT = Path(__file__).resolve().parents[1]
CROP_FIELDS = (
    "get_example_False_False", "rng_False_False",
    "get_example_False_True", "rng_False_True",
    "get_example_True_False", "rng_True_False",
    "get_example_True_True", "rng_True_True",
    "resize", "skimage_0", "skimage_20", "skimage_rejection",
    "flipped_points", "rotation_aa", "keypoints3",
)


class LegacyCleanupTests(unittest.TestCase):
    def test_legacy_crop_values_rng_and_failure_behavior(self):
        from densekp.core.datasets import utils as crop

        frozen = json.loads((ROOT / "tests/fixtures/legacy_cleanup_v1.json").read_text())
        expected = {name: frozen[name] for name in CROP_FIELDS}
        with TemporaryDirectory() as temp:
            actual = capture_crop(Path(temp) / "capture", crop)
        self.assertEqual(actual, expected)
