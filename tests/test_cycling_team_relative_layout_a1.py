"""Bounded CPU checks for the explicitly diagnostic team layout policy."""
import json
import tempfile
import unittest
from pathlib import Path

from densekp_retrain.data.cycling_team import relative_layout_candidates, projected_occlusion_gate, preflight_team
from tools.experiments.prepare_cycling_team_relative_layout_a1 import frozen_people, make_recipe


ROOT = Path(__file__).resolve().parents[1]


class RelativeTeamTest(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((ROOT/'configs/cycling_team_relative_layout_a1_20261010.json').read_text())

    def test_relative_seed_and_anchor(self):
        anchor = self.cfg['riverfront_anchor_parameters']
        a = relative_layout_candidates(anchor, 20261023)
        self.assertEqual(a, relative_layout_candidates(anchor, 20261023))
        self.assertNotEqual(a[0]['parameters'], relative_layout_candidates(anchor, 20261024)[0]['parameters'])
        self.assertEqual(len(a), 32)
        self.assertEqual(len({tuple(sorted(x['parameters'].items())) for x in a}), 32)
        for row in a:
            q = row['parameters']
            for key in anchor:
                self.assertEqual(q[key], anchor[key])
            self.assertGreater(q['p2_y_m']-q['p1_y_m'], 1.35)
            self.assertLess(q['p2_y_m']-q['p1_y_m'], 2.35)
            self.assertGreater(q['p2_speed_m_s'], 0)
        with self.assertRaises(ValueError):
            relative_layout_candidates(anchor, 20261023, limit=33)

    def test_strict_default_and_diagnostic_only(self):
        absent = {'cam2': False, 'cam4': False}
        self.assertFalse(projected_occlusion_gate(absent))
        self.assertTrue(projected_occlusion_gate(absent, 'diagnostic_only'))
        self.assertTrue(projected_occlusion_gate({'cam2': True, 'cam4': False}))
        with self.assertRaises(ValueError):
            projected_occlusion_gate(absent, 'skip_checks')

    def test_two_previous_combinations_are_verified_hits(self):
        _, plan, people, assets, placements, checks = frozen_people()
        self.assertEqual([p['source_origin_record'] for p in plan['people'][:2]], [0,4])
        self.assertEqual([p['component_ids'] for p in people[:2]],
                         [p['selection'] for p in plan['people'][:2]])
        self.assertEqual([c['cache_state'] for c in checks[:2]],
                         ['delivery_verified_hit']*2)
        self.assertEqual(len({a['character_family'] for a in assets[:2]}),2)
        self.assertNotEqual(placements[0]['height_m'], placements[1]['height_m'])

    def test_diagnostic_policy_still_rejects_old_erp_seam(self):
        policy, plan, people, assets, placements, _ = frozen_people()
        source = ROOT/'outputs/experiments/seeded_cycling_team_a1_20261010/riverfront_2_overlap_gate_correction2/prepared/candidate_checks.json'
        old = json.loads(source.read_text())['attempts'][0]
        candidate = {key:old[key] for key in ('attempt','child_seed','parameters')}
        with tempfile.TemporaryDirectory() as temp:
            stage = Path(temp)
            ep, timelines, geo = make_recipe(policy,plan,people,2,'riverfront',stage)
            result = preflight_team(ep,timelines,assets[:2],placements[:2],[candidate],geo,
                                    stage/'prepared',projected_occlusion_policy='diagnostic_only')
            self.assertIsNone(result)
            check = json.loads((stage/'prepared/candidate_checks.json').read_text())['attempts'][0]
            self.assertEqual(check['rejections'], ['ERP_geometry_seam_crossing'])
            self.assertEqual(check['failure_context']['camera_id'], 'cam2')
            self.assertTrue(check['failure_context']['per_person_support']['2']['seam_crossing'])


if __name__ == '__main__': unittest.main()
