import json
import unittest
from pathlib import Path

import numpy as np

from tools.experiments.plan_t_junction_local_rigs_a1 import root_clearance
from tools.experiments.prepare_urban_three_multirig_a1 import coverage_pass


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs/experiments/t_junction_multirig_a1_20261011/prepared_scene"
OUT = ROOT / "outputs/experiments/t_junction_local_rig_delivery_a1_20261011"


class JunctionLocalRigTests(unittest.TestCase):
    def test_true_paired_centers_and_local_replay(self):
        matrices = np.load(SOURCE / "scene_matrices.npy")
        frozen = json.loads((OUT / "frozen_local_proposals.json").read_text())
        proposals = frozen["proposals"]
        self.assertEqual(len(proposals), 512)
        self.assertEqual(len({(round(r["origin_m"][0], 8), round(r["origin_m"][1], 8),
                               r["yaw_deg"]) for r in proposals}), 512)
        checks = json.loads((OUT / "local_checks.json").read_text())["attempts"]
        for i in (0, 211, 256, 488, 511):
            row = checks[i]
            near, cams = root_clearance(row["origin_m"], row["yaw_deg"], matrices, .18)
            self.assertAlmostEqual(near["distance_m"], row["minimum_root_distance_m"], places=10)
            self.assertEqual({c["camera"] for c in cams}, {"cam2", "cam4"})
            self.assertGreaterEqual(near["distance_m"], .5)

    def test_edge_preference_and_ordinary_fallback_are_distinct(self):
        selected = json.loads((OUT / "rig_selection.json").read_text())
        self.assertEqual(selected["status"], "ready_ordinary_no_edge_quota")
        self.assertIsNone(selected["edge_selected_rig_id"])
        self.assertEqual(set(selected["selected"]), {"back", "front", "side"})
        self.assertTrue(all(v["status"] == "passed" for v in selected["selected"].values()))
        self.assertTrue(all(v["edge_window_frames"] < 6 for v in selected["selected"].values()))
        for view in ("back", "front", "side"):
            self.assertGreaterEqual(selected["selected"][view]["view_window_frames"], 6)

    def test_old_back_default_and_explicit_ordinary_are_separate(self):
        common = [True] * 31
        view = [True] * 6 + [False] * 25
        edge = [False] * 31
        base = dict(minimum_common_physical_frames=6, minimum_view_common_frames=6,
                    minimum_view_common_run=6)
        self.assertFalse(coverage_pass(common, view, edge, view_goal="back",
                                       config=base, road_pass=False))
        ordinary = dict(base, minimum_dual_edge_frames=0, minimum_dual_edge_run=0)
        self.assertTrue(coverage_pass(common, view, edge, view_goal="back",
                                      config=ordinary, road_pass=False))
        self.assertFalse(coverage_pass(common, [False] * 31, edge, view_goal="back",
                                       config=ordinary, road_pass=False))

    def test_prepared_rigs_share_unchanged_scene(self):
        parent_world = np.load(SOURCE / "world_densekp.npy")
        parent_scene = np.load(SOURCE / "scene_matrices.npy")
        for view in ("back", "front", "side"):
            prepared = Path(json.loads((OUT / "rig_selection.json").read_text())["selected"][view]["prepared_ref"])
            self.assertTrue(np.array_equal(np.load(prepared / "world_densekp.npy"), parent_world))
            self.assertTrue(np.array_equal(np.load(prepared / "scene_matrices.npy"), parent_scene))
            self.assertEqual(np.load(prepared / "expected_in_view.npy").shape, (31, 3, 2, 138))

    def test_float32_world_matrix_precision_still_rejects_wrong_root(self):
        all_matrices = np.load(SOURCE / "scene_matrices.npy")[:, 0]
        index = int(np.argmax(np.abs(all_matrices[:, 1, 3].astype(np.float32) -
                                     all_matrices[:, 1, 3])))
        matrix = all_matrices[index]
        world = np.load(SOURCE / "world_densekp.npy")[index, 0]
        # Blender stores this matrix at float32 precision. One representable
        # step at the junction Y coordinate exceeds the old fixed 2 um bound.
        quantized = matrix.astype(np.float32).astype(float)
        self.assertGreater(abs(quantized[1, 3]-matrix[1, 3]), 2e-6)
        tolerance = np.maximum(2e-6, np.spacing(np.float32(np.abs(world))))
        self.assertGreater(float(tolerance[:, 1].max()), 2e-6)
        changed = matrix.copy()
        changed[1, 3] += .001
        self.assertGreater(abs(changed[1, 3]-matrix[1, 3]),
                           float(tolerance[:, 1].max()) * 100)


if __name__ == "__main__":
    unittest.main()
