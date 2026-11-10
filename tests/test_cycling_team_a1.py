"""Targeted CPU checks for the explicit cycling-team recipe."""
import copy
import json
import unittest
from pathlib import Path

import numpy as np

from densekp_retrain.assets import composition as a2
from densekp_retrain.data.cycling_team import CLOCK, candidate_parameters, heading_angle_degrees
from densekp_retrain.data.dynamic_scene import frame_record, person_frame_record, person_matrices
from tools.experiments.seed_cycling_team_a1 import load_config, plan, ROOT


class CyclingTeamTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT/'configs/seeded_cycling_team_a1_20261010.json')
        cls.registry = a2.read(ROOT/cls.config['registry'])

    def test_seed_and_source_diversity(self):
        a = plan(self.config,self.registry)
        self.assertEqual(a,plan(self.config,self.registry))
        self.assertEqual([x['person_ids'] for x in a['episodes']],[[1,2],[1,2,3]])
        self.assertEqual(len({p['selection']['body_id'] for p in a['people']}),3)
        self.assertEqual(len({p['source_group'] for p in a['people']}),3)
        self.assertEqual(len({p['selection']['clothing_id'] for p in a['people']}),3)
        changed=copy.deepcopy(self.config);changed['master_seed']=20261023
        b=plan(changed,self.registry)
        self.assertNotEqual([p['selection'] for p in a['people']], [p['selection'] for p in b['people']])

    def test_candidate_substream_and_limit(self):
        ranges=self.config['environments']['riverfront']['parameter_ranges']
        a=candidate_parameters(20261022,0,2,ranges,limit=12)
        self.assertEqual(a,candidate_parameters(20261022,0,2,ranges,limit=12))
        self.assertEqual(len({x['child_seed'] for x in a}),12)
        with self.assertRaises(ValueError):candidate_parameters(20261022,0,4,ranges)
        with self.assertRaises(ValueError):candidate_parameters(20261022,0,2,ranges,limit=13)

    def test_real_three_person_prepared_clock_and_matrices(self):
        prepared=ROOT/'outputs/experiments/seeded_cycling_team_a1_20261010/urban_3_implementation_correction1/prepared'
        recipe=json.loads((prepared/'recipe.json').read_text())
        self.assertEqual(recipe['episode_timeline'],CLOCK)
        self.assertEqual(len(recipe['people']),3)
        m=np.load(prepared/'scene_matrices.npy')
        world=np.load(prepared/'world_densekp.npy')
        expected=np.load(prepared/'expected_in_view.npy')
        self.assertEqual(m.shape,(31,3,4,4))
        self.assertEqual(world.shape,(31,3,138,3))
        self.assertEqual(expected.shape,(31,3,2,138))
        self.assertTrue(np.isfinite(m).all())
        for k in (0,15,30):
            records=frame_record(recipe,k)['person_records']
            self.assertEqual([r['person_id'] for r in records],[1,2,3])
            for j,person in enumerate(recipe['people']):
                self.assertEqual(person_frame_record(recipe,k,person)['asset_frame'],person['asset_frames'][k])
                np.testing.assert_allclose(person_matrices(person,31)[k],m[k,j])
                asset=json.loads(Path(person['asset_manifest']).read_text())
                original=np.load(asset['densekp'],mmap_mode='r')[person['asset_frames'][k]]
                np.testing.assert_allclose(world[k,j],original@m[k,j,:3,:3].T+m[k,j,:3,3],atol=1e-5)
        for j,person in enumerate(recipe['people']):
            asset=json.loads(Path(person['asset_manifest']).read_text())
            self.assertLessEqual(float(heading_angle_degrees(asset,person['asset_frames'],m[0,j]).max()),25)
        self.assertEqual(len({tuple(p['asset_frames']) for p in recipe['people']}),3)

    def test_old_single_and_mixed_recipes_still_read(self):
        for relative,count in [
            ('outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront/recipe.json',1),
            ('outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009/run/recipe.json',2),
        ]:
            recipe=json.loads((ROOT/relative).read_text())
            self.assertEqual(len(recipe['people']),count)
            for k in (0,recipe['frame_count']-1):
                self.assertEqual(frame_record(recipe,k)['frame_index'],k)

    def test_frozen_bounded_rejections_remain_explicit(self):
        root=ROOT/'outputs/experiments/seeded_cycling_team_a1_20261010'
        for name in ('riverfront_2','urban_3'):
            checks=json.loads((root/(name+'_overlap_gate_correction2')/'prepared/candidate_checks.json').read_text())
            self.assertEqual(len(checks['attempts']),12)
            self.assertTrue(all(row['rejections'] for row in checks['attempts']))
        city=json.loads((root/'urban_3_overlap_gate_correction2/prepared/candidate_checks.json').read_text())
        self.assertEqual(city['attempts'][5]['rejections'],['no_projected_neighbor_overlap_rise_recovery'])


if __name__=='__main__':unittest.main()
