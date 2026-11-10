"""Exact cache-view and material-description checks without Blender or models."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from densekp_retrain.assets.garment_contract import (
    describe_fixed, describe_simple, read_view_frame, select_asset_window,
    validate_description, write_new,
)

ROOT = Path(__file__).resolve().parents[1]
SIMPLE = ROOT / "outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/delivery_manifest.json"
FIXED = ROOT / "outputs/experiments/bedlam_m03_native_cloth_transfer_a2_20261010/builds/M03_KIT11_BEDLAM_native/delivery_manifest.json"


class GarmentContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.simple = validate_description(describe_simple(SIMPLE))
        cls.fixed = validate_description(describe_fixed(FIXED, SIMPLE))

    def test_descriptions_distinguish_geometry_and_material(self):
        self.assertNotEqual(self.simple["geometry_kind"], self.fixed["geometry_kind"])
        self.assertNotEqual(self.simple["material_kind"], self.fixed["material_kind"])
        self.assertEqual(self.fixed["compatibility"]["window"], [0, 120])

    def test_contiguous_subwindow_retains_original_indices_and_root(self):
        view = select_asset_window(self.fixed, [15, 106])
        self.assertEqual((view["frame_count"], view["source_asset_frame_indices"][0],
                          view["source_asset_frame_indices"][-1]), (91, 15, 105))
        self.assertEqual(view["source_root_reference_asset_frame"], 0)
        self.assertEqual(view["timeline"][0]["view_episode_time_tick"], 0)
        self.assertEqual(view["timeline"][-1]["view_episode_time_tick"], 90)
        self.assertEqual(read_view_frame(view, 0)["cloth_vertices"].shape[1], 3)

    def test_invalid_windows_and_unknown_types_rejected(self):
        for bounds in ([-1, 10], [15, 15], [20, 121], [106, 15], [0, 120, 1], [0.0, 10]):
            with self.subTest(bounds=bounds), self.assertRaisesRegex(ValueError, "invalid_contiguous_window"):
                select_asset_window(self.fixed, bounds)
        bad = copy.deepcopy(self.fixed)
        bad["material_kind"] = "unknown"
        with self.assertRaisesRegex(ValueError, "type_or_material"):
            validate_description(bad)
        bad = copy.deepcopy(self.fixed)
        bad["material"]["faces_uv"] = str(ROOT / "assets/densekp_regressor_R.npy")
        with self.assertRaisesRegex(ValueError, "native_face_corner_uv"):
            validate_description(bad)

    def test_missing_dependency_and_lock_rejected(self):
        bad = copy.deepcopy(self.fixed)
        bad["material"]["diffuse"] = "/no/such/texture.png"
        with self.assertRaises(FileNotFoundError):
            validate_description(bad)
        bad = copy.deepcopy(self.fixed)
        bad["compatibility"]["motion_id"] = "KIT425_walk"
        with self.assertRaisesRegex(ValueError, "identity_mismatch"):
            validate_description(bad)

    def test_write_is_append_only_and_has_no_build_side_effect(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "view.json"
            view = select_asset_window(self.simple, [0, 120])
            write_new(path, view)
            self.assertEqual(json.loads(path.read_text())["frame_count"], 120)
            with self.assertRaises(FileExistsError):
                write_new(path, view)
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == "__main__":
    unittest.main()
