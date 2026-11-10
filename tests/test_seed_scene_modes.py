import json
import copy
import tempfile
import unittest
from pathlib import Path

from densekp_retrain.data.mixed_road_pass import plan_mixed,preflight_mixed
from tools.experiments.seed_scene_modes import make_config, valid_origins


ROOT=Path(__file__).resolve().parents[1]
POOL=ROOT/'configs/seeded_scene_modes_a1_20261010.json'


class SeededSceneModesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pool=json.loads(POOL.read_text())

    def test_same_seed_restores_assets_origins_layout_candidates(self):
        a,record=make_config(self.pool,'single_cyclist_boundary_pass',20261010,0)
        b,repeated=make_config(self.pool,'single_cyclist_boundary_pass',20261010,0)
        self.assertEqual((a,record),(b,repeated))
        self.assertEqual(len(a['reference_recipes']),1)
        self.assertNotIn('explicit_candidate_parameters',a)
        self.assertEqual(len(plan_mixed(a)['candidate_sets'][0]['candidates']),12)
        self.assertEqual(record['source_groups'],['APD6526_original_recording'])

    def test_second_seed_changes_real_recipe_fields(self):
        a,_=make_config(self.pool,'single_cyclist_boundary_pass',20261010,0)
        b,_=make_config(self.pool,'single_cyclist_boundary_pass',20261011,0)
        self.assertNotEqual(a['episode_id'],b['episode_id'])
        self.assertNotEqual(a['selected_asset_id'],b['selected_asset_id'])
        self.assertNotEqual(a['source_origin_records'],b['source_origin_records'])
        self.assertNotEqual(plan_mixed(a)['candidate_sets'],plan_mixed(b)['candidate_sets'])

    def test_multiple_children_get_distinct_ids_and_layout_streams(self):
        a,first=make_config(self.pool,'mixed_walk_cycle_crossing',20261010,0)
        b,second=make_config(self.pool,'mixed_walk_cycle_crossing',20261010,1)
        self.assertNotEqual(first['episode_id'],second['episode_id'])
        self.assertNotEqual(first['episode_child_seed'],second['episode_child_seed'])
        self.assertNotEqual(plan_mixed(a)['candidate_sets'],plan_mixed(b)['candidate_sets'])

    def test_mixed_has_two_original_clocks_and_root_owners(self):
        a,record=make_config(self.pool,'mixed_walk_cycle_crossing',20261010,0)
        p=plan_mixed(a);ep=p['episodes'][0]
        self.assertEqual(len(ep['people']),2)
        self.assertEqual(ep['scene_mode'],'mixed_walk_cycle_crossing')
        self.assertEqual([x['root_mode'] for x in ep['people']],
            ['fixed_pelvis_scene_owns_progression','source_root_relative_to_start_plus_fixed_scene_matrix'])
        self.assertEqual(len(ep['person_timelines']['1']),31)
        self.assertEqual(len(ep['person_timelines']['2']),31)
        self.assertNotEqual(ep['person_timelines']['1'][1]['source_time_base'],ep['person_timelines']['2'][1]['source_time_base'])
        self.assertEqual(len(record['source_groups']),2)

    def test_too_short_original_sequence_rejected(self):
        # The actual F02 origin set is nonempty; a four-record original source
        # cannot cover a 31-time, three-second episode without looping.
        rows=[dict(asset_frame=i,source_frame=i,source_time_tick=i,time_base=[1,30]) for i in range(4)]
        from densekp_retrain.data.person_time import map_timeline
        clock=[dict(frame_index=k,episode_time_tick=k,time_base=[1,10]) for k in range(31)]
        with self.assertRaisesRegex(ValueError,'desired_time_outside_source'):
            map_timeline(rows,clock,origin_record=0)

    def test_unaccepted_or_disallowed_bound_asset_rejected(self):
        for key,value in [('accepted_internal',False),('license','not_permitted')]:
            pool=copy.deepcopy(self.pool)
            pool['riders'][0][key]=value
            with self.assertRaisesRegex(ValueError,'unaccepted_asset_combination|asset_license_not_permitted'):
                make_config(pool,'single_cyclist_boundary_pass',20261010,0)

    def test_old_two_person_config_remains_plannable(self):
        old=json.loads((ROOT/'outputs/experiments/f02_mtb_mixed_scene_repair_a1_20261010/freeze/config.json').read_text())
        old.pop('explicit_candidate_parameters',None)
        ep=plan_mixed(old)['episodes'][0]
        self.assertEqual(len(ep['people']),2)
        self.assertEqual(ep['frame_count'],31)

    def test_no_legal_layout_retains_rejection_instead_of_switching_asset(self):
        cfg,_=make_config(self.pool,'single_cyclist_boundary_pass',20261010,0)
        cfg['attempt_limit']=1
        cfg['parameter_ranges']['target_x_m']=[10.,10.]
        cfg['explicit_candidate_parameters']=[dict(target_x_m=10.,target_y_m=1.9,cycling_speed_m_s=.5)]
        cfg['episode_id']='synthetic_test_only_no_legal_road_layout'
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(preflight_mixed(plan_mixed(cfg),directory))
            output=Path(directory)/cfg['episode_id']
            attempts=json.loads((output/'candidate_checks.json').read_text())[0]['attempts']
            self.assertEqual(len(attempts),1)
            self.assertIn('outside_road',attempts[0]['rejections'])
            self.assertEqual(json.loads((output/'geometry_preflight.json').read_text())['status'],'rejected')


if __name__=='__main__':unittest.main()
