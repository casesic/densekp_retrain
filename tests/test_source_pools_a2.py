"""Narrow metadata checks for proposed A2 source pools."""

import copy
import json
import unittest
from pathlib import Path

from densekp_retrain.assets.source_pools import validate_batch, validate_episode, validate_proposal


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "outputs/plans/asset_source_pools_a2_20261010"


def person(catalog, cid, window=None, role="target"):
    c = catalog["combinations"][cid]
    return {"role": role, "combo_id": cid, "body_id": c["body_id"],
            "motion_id": c["motion_id"], "clothing_id": c["clothing_id"],
            "hair_id": c["hair_id"], "asset_window": window or c["asset_window"],
            "declared_source_group": catalog["motions"][c["motion_id"]]["source_group"]}


class PoolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads((BASE / "source_catalog.json").read_text())
        cls.pools = json.loads((BASE / "pool_proposals.json").read_text())

    def episode(self, cid, pool="train_candidate", proposal="primary_recording_and_character_isolated"):
        return {"schema": "densekp_source_episode_candidate_v1", "proposal_id": proposal,
                "pool": pool, "scene_episode_id": "scene001", "rig_id": "rigA", "seed": 1, "expected_person_count": 1,
                "persons": [person(self.catalog, cid)]}

    def codes(self, result):
        return {x["code"] for x in result["issues"]}

    def test_primary_source_graph_and_pending_human_scope(self):
        self.assertEqual(validate_proposal(self.catalog, self.pools, "primary_recording_and_character_isolated"), [])
        self.assertTrue(validate_episode(self.catalog, self.pools, self.episode("p2_M01"))["eligible_under_proposed_policy"])
        self.assertFalse(validate_episode(self.catalog, self.pools, self.episode("KIT425_M04"))["eligible_under_proposed_policy"])

    def test_cross_pool_target_and_occluder(self):
        e = self.episode("p2_M01")
        e["persons"].append(person(self.catalog, "p1_comfortable_M02", role="occluder"))
        e["expected_person_count"] = 2
        self.assertIn("outside_pool", self.codes(validate_episode(self.catalog, self.pools, e)))
        p = copy.deepcopy(self.pools)
        p["proposals"]["primary_recording_and_character_isolated"]["train_candidate"].append("p1_comfortable_M02")
        self.assertIn("cross_pool_motion", self.codes({"issues": validate_proposal(self.catalog, p, "primary_recording_and_character_isolated")}))

    def test_same_body_and_p1_two_windows_cannot_split(self):
        p = copy.deepcopy(self.pools)
        p["proposals"]["primary_recording_and_character_isolated"]["train_candidate"].append("p1_alternate_M02")
        codes = self.codes({"issues": validate_proposal(self.catalog, p, "primary_recording_and_character_isolated")})
        self.assertIn("cross_pool_identity", codes)
        self.assertIn("cross_pool_motion", codes)

    def test_fixed_cache_subwindow_and_lock(self):
        p = copy.deepcopy(self.pools)
        p["proposals"]["primary_recording_and_character_isolated"]["val_candidate"].append("KIT11_M03_BEDLAM")
        e = self.episode("KIT11_M03_BEDLAM", "val_candidate")
        e["persons"] = [person(self.catalog, "KIT11_M03_BEDLAM", [15, 106])]
        self.assertFalse(self.codes(validate_episode(self.catalog, p, e)))
        e["persons"][0]["hair_id"] = "M02_groom"
        self.assertIn("fixed_cache_lock", self.codes(validate_episode(self.catalog, p, e)))
        e["persons"][0]["hair_id"] = "M03_groom"
        e["persons"][0]["asset_window"] = [15, 121]
        self.assertIn("invalid_window", self.codes(validate_episode(self.catalog, p, e)))

    def test_built_acceptance_and_unknown_license(self):
        p = copy.deepcopy(self.pools)
        p["proposals"]["primary_recording_and_character_isolated"]["train_candidate"].append("planned_M04_p1")
        e = self.episode("planned_M04_p1")
        self.assertFalse(validate_episode(self.catalog, p, e)["persons"][0]["built"])
        p["proposals"]["primary_recording_and_character_isolated"]["train_candidate"].append("bikefit_F02")
        self.assertIn("license_unresolved", self.codes({"issues": validate_proposal(self.catalog, p, "primary_recording_and_character_isolated")}))
        e2 = self.episode("p2_M01")
        e2["persons"][0]["asset_window"] = [0, 120]
        e2["persons"][0]["combo_id"] = "planned_M04_p1"
        self.assertIn("component_mismatch", self.codes(validate_episode(self.catalog, p, e2)))

    def test_short_preview_acceptance_cannot_cover_full_cache(self):
        p = copy.deepcopy(self.pools)
        p["proposals"]["primary_recording_and_character_isolated"]["train_candidate"].append("A_f02_p2")
        e = self.episode("A_f02_p2")
        e["persons"] = [person(self.catalog, "A_f02_p2", [0, 60])]
        first = validate_episode(self.catalog, p, e)["persons"][0]
        self.assertTrue(first["built"])
        self.assertTrue(first["human_accepted_scope"])
        e["persons"][0]["asset_window"] = [0, 120]
        self.assertFalse(validate_episode(self.catalog, p, e)["persons"][0]["human_accepted_scope"])

    def test_unknown_actor_and_episode_rig_variant(self):
        p = copy.deepcopy(self.pools)
        p["proposals"]["primary_recording_and_character_isolated"]["claims_actor_independent_walking"] = True
        self.assertIn("actor_independence_unknown", self.codes({"issues": validate_proposal(self.catalog, p, "primary_recording_and_character_isolated")}))
        a = self.episode("p2_M01")
        b = self.episode("p1_comfortable_M02", "val_candidate")
        b["rig_id"] = "rigB"
        self.assertIn("episode_cross_pool", self.codes(validate_batch(self.catalog, self.pools, [a, b])[1]))

    def test_missing_fields_and_versions(self):
        e = self.episode("p2_M01")
        e["schema"] = "unknown"
        e["persons"][0]["declared_source_group"] = "unknown"
        e["persons"][0]["asset_window"] = [-1, 2]
        e["persons"].append({"combo_id": "missing", "role": "occluder"})
        codes = self.codes(validate_episode(self.catalog, self.pools, e))
        self.assertTrue({"wrong_version", "source_mismatch", "invalid_window", "dangling_combo"} <= codes)


if __name__ == "__main__":
    unittest.main()
