"""Small contract checks for the A2 composition planner and append-only cache."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from densekp_retrain.assets.composition import build, cached_folder, plan
from densekp_retrain.data.dynamic_scene import person_matrices


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/asset_composition_a2_20261010"
REGISTRY = json.loads((CONFIG / "components.json").read_text())
RECIPES = json.loads((CONFIG / "recipes.json").read_text())["recipes"]


class CompositionContractTest(unittest.TestCase):
    def test_plan_chooses_components_without_loading_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("densekp_retrain.assets.composition.load_smplx_model", side_effect=AssertionError("plan constructed model")):
                a = plan(ROOT, REGISTRY, RECIPES["A_f02_p2"])
                b = plan(ROOT, REGISTRY, RECIPES["B_f02_p2_m02_cloth"])
                c = plan(ROOT, REGISTRY, RECIPES["C_f02_p2_m02_cloth_hair"])
            self.assertEqual((a["components"]["body"]["id"], a["components"]["motion"]["id"]), ("F02", "p2_straight"))
            self.assertNotEqual(a["clothing_reference"], b["clothing_reference"])
            self.assertNotEqual(b["hair_reference"], c["hair_reference"])
            self.assertEqual((a["frame_count"], b["frame_count"], c["frame_count"]), (120, 120, 120))
            self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_unsupported_root_or_native_cache_rejected(self):
        wrong = dict(RECIPES["A_f02_p2"], root_policy="source_root_relative_to_start_plus_fixed_scene_matrix")
        with self.assertRaisesRegex(ValueError, "root"):
            plan(ROOT, REGISTRY, wrong)
        native = dict(RECIPES["A_f02_p2"], clothing_id="BEDLAM_native_locked_example")
        with self.assertRaisesRegex(ValueError, "native_cache_locked"):
            plan(ROOT, REGISTRY, native)

    def test_missing_dependency_and_version_change_rejected(self):
        bad = copy.deepcopy(REGISTRY)
        bad["motions"]["p2_straight"]["parameters"] = "outputs/this_motion_does_not_exist.npz"
        with self.assertRaises(FileNotFoundError):
            plan(ROOT, bad, RECIPES["A_f02_p2"])
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "cache"
            self.assertFalse(cached_folder(folder, {"version": 1}))
            (folder / "sentinel").write_text("keep")
            self.assertTrue(cached_folder(folder, {"version": 1}))
            with self.assertRaisesRegex(RuntimeError, "cache_collision"):
                cached_folder(folder, {"version": 2})
            self.assertEqual((folder / "sentinel").read_text(), "keep")

    def test_semantic_face_order_and_apd_native_time(self):
        original_load = np.load
        def altered(path, *args, **kwargs):
            value = original_load(path, *args, **kwargs)
            if str(path).endswith("cycling_M02_v2/reference/geometry/faces.npy"):
                return value[::-1].copy()
            return value
        with patch("densekp_retrain.assets.composition.np.load", side_effect=altered):
            with self.assertRaisesRegex(ValueError, "semantic_topology_face_order_conflict"):
                plan(ROOT, REGISTRY, RECIPES["B_f02_p2_m02_cloth"])
        apd = dict(RECIPES["A_f02_p2"], motion_id="APD6526", window=[0, 76])
        self.assertEqual(plan(ROOT, REGISTRY, apd)["frame_count"], 76)

    def test_walking_root_window_and_cycling_regression(self):
        walking = RECIPES["M04_KIT425_walk_replay_v2"]
        full = plan(ROOT, REGISTRY, walking)
        self.assertEqual(full["root_policy"], "source_root_relative_to_start_plus_fixed_scene_matrix")
        self.assertEqual(full["root_reference_source_frame"], 9)
        self.assertEqual(full["frame_count"], 94)
        sub = plan(ROOT, REGISTRY, dict(walking, window=[3, 20]))
        self.assertEqual(sub["window"], [3, 20])
        self.assertEqual(sub["root_reference_source_frame"], 9)
        with self.assertRaisesRegex(ValueError, "unsupported_root_ownership"):
            plan(ROOT, REGISTRY, dict(walking, root_policy="fixed_pelvis_scene_owns_progression"))
        bad = copy.deepcopy(REGISTRY)
        bad["motions"]["KIT425_walk"]["root_reference_source_frame"] = 12
        with self.assertRaisesRegex(ValueError, "source_root_reference_conflict"):
            plan(ROOT, bad, walking)
        self.assertEqual(plan(ROOT, REGISTRY, RECIPES["A_f02_p2"])["root_policy"], "fixed_pelvis_scene_owns_progression")

    def test_kit11_m03_walk_uses_its_own_source_and_root(self):
        recipe = RECIPES["M03_KIT11_walk"]
        resolved = plan(ROOT, REGISTRY, recipe)
        self.assertEqual(resolved["frame_count"], 120)
        self.assertEqual(resolved["root_reference_source_frame"], 0)
        self.assertEqual(resolved["root_policy"], "source_root_relative_to_start_plus_fixed_scene_matrix")
        self.assertEqual(resolved["components"]["body"]["id"], "M03")
        self.assertEqual(resolved["components"]["motion"]["id"], "KIT11_walk")
        self.assertNotEqual(resolved["components"]["motion"]["source_group"],
                            plan(ROOT, REGISTRY, RECIPES["M04_KIT425_walk_replay_v2"])["components"]["motion"]["source_group"])

    def test_m03_bedlam_cloth_only_reuses_exact_accepted_combination(self):
        recipe = RECIPES["M03_KIT11_BEDLAM_transfer"]
        resolved = plan(ROOT, REGISTRY, recipe)
        self.assertEqual(resolved["selection_mode"], "reuse_precomputed_locked_asset")
        self.assertEqual(resolved["frame_count"], 120)
        with tempfile.TemporaryDirectory() as tmp:
            out, result = build(ROOT, resolved, tmp, "locked_selection")
            self.assertEqual(result["status"], "reused_precomputed_locked_asset")
            self.assertEqual(result["dressed_human_status"], "accepted_user_visual_20261010")
            self.assertTrue((out / "delivery_manifest.json").is_file())
            self.assertEqual(list(Path(tmp).iterdir()), [])
        for change in ({"body_id": "M02"}, {"motion_id": "KIT425_walk"},
                       {"hair_id": "M04_groom"}, {"window": [1, 120]}):
            with self.subTest(change=change):
                with self.assertRaisesRegex(ValueError, "clothing_locked_body_motion_hair_window_mismatch"):
                    plan(ROOT, REGISTRY, dict(recipe, **change))

    def test_walking_scene_matrix_remains_fixed_for_nonfirst_subwindow(self):
        relative = np.array([[0., 0., 0.], [.1, 0., .2], [.2, 0., .5]])
        start = 1
        rotation = np.eye(3)
        anchor = np.array([2., .8, 4.])
        matrix = np.eye(4)
        matrix[:3, 3] = anchor - rotation @ relative[start]
        actual = person_matrices({"root_mode": "source_root_relative_to_start_plus_fixed_scene_matrix",
                                  "model_to_world": matrix.tolist()}, 2)
        np.testing.assert_array_equal(actual[0], actual[1])
        np.testing.assert_allclose(relative[start:start+2] @ rotation.T + actual[0, :3, 3],
                                   anchor + (relative[start:start+2] - relative[start]) @ rotation.T)
        with self.assertRaisesRegex(ValueError, "double_root_ownership"):
            person_matrices({"root_mode": "source_root_relative_to_start_plus_fixed_scene_matrix",
                             "model_to_world": matrix.tolist(), "scene_from_model": [matrix.tolist()] * 2}, 2)


if __name__ == "__main__":
    unittest.main()
