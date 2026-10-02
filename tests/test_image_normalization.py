"""Current normalization checked against the original, unmodified legacy fixture."""

import base64
import json
from pathlib import Path
import unittest
import zlib

import torch

from densekp.core.image_normalization import denormalize_images


class ImageNormalizationTests(unittest.TestCase):
    def test_denormalize_matches_independent_original_bytes(self):
        fixture = Path(__file__).parent / "fixtures/legacy_cleanup_v1.json"
        expected = json.loads(fixture.read_text())["denormalize"]
        images = torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) / 10
        actual = denormalize_images(images).detach().cpu().numpy()
        self.assertEqual(str(actual.dtype), expected["dtype"])
        self.assertEqual(list(actual.shape), expected["shape"])
        self.assertEqual(actual.tobytes(), zlib.decompress(base64.b64decode(expected["bytes"])))
