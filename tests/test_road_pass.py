"""Targeted CPU tests only: no models, rendering, optimization or old fixtures."""
import copy
import unittest
import numpy as np
from densekp_retrain.data.road_pass import plan, longest_true_run
from densekp_retrain.body.sequence_binding import head_transform


class RoadPassTests(unittest.TestCase):
    def config(self):
        return dict(mode='road_pass',master_seed=17,episode_count=2,person_counts=[1,2],scene='fixture_scene',
            rig=dict(baseline_m=.18,road_yaw_deg=90),render=dict(samples=64),asset_pool=[dict(asset_id='synthetic_test_only',manifest='fixture.json',character_family='fixture',available_combinations_verified=True,allowed_start_frames=[9,12])])

    def test_replay_and_changes(self):
        c=self.config();self.assertEqual(plan(c),plan(copy.deepcopy(c)))
        d=copy.deepcopy(c);d['master_seed']+=1;self.assertNotEqual(plan(c)['episodes'],plan(d)['episodes'])

    def test_counts_and_modes(self):
        c=self.config();c['episode_count']=1
        for n in range(1,6):
            c['person_counts']=[n];self.assertEqual(len(plan(c)['episodes'][0]['people']),n)
        for n in [0,6,True,1.5]:
            c['person_counts']=[n]
            with self.assertRaises(ValueError):plan(c)
        c=self.config();c['mode']='three_way_junction'
        with self.assertRaises(ValueError):plan(c)

    def test_explicit_root_time_and_bound_combinations(self):
        c=self.config();ep=plan(c)['episodes'][0]
        self.assertEqual(ep['source_stride'],3);self.assertEqual(ep['time_base'],[1,10]);self.assertEqual(ep['crop_expansion'],1)
        self.assertTrue(89<=ep['rig']['yaw_deg']<=91)
        for p in ep['people']:
            self.assertEqual(p['root_mode'],'source_root_relative_to_start_plus_fixed_scene_matrix')
            self.assertEqual(p['speed_policy'],'source_rate_unchanged')
        c['asset_pool'][0]['available_combinations_verified']=False
        with self.assertRaises(ValueError):plan(c)

    def test_rigid_head_includes_root_once(self):
        local=np.tile(np.eye(3),(22,1,1));local[0]=[[0,-1,0],[1,0,0],[0,0,1]]
        parents=np.arange(22)-1;rest=np.array([0,1,0]);posed=np.array([3,4,5])
        H=head_transform(local,parents,posed,rest)
        np.testing.assert_allclose(H[:3,:3]@rest+H[:3,3],posed)
        T=np.eye(4);T[:3,3]=[8,9,10]
        np.testing.assert_allclose((T@H)@np.r_[rest,1],np.r_[posed+[8,9,10],1])
        local[0,0,0]=np.nan
        with self.assertRaises(ValueError):head_transform(local,parents,posed,rest)

    def test_consecutive_support_not_scattered_total(self):
        self.assertEqual(longest_true_run([True,False,True,False,True]),1)
        self.assertEqual(longest_true_run([False,True,True,True,False]),3)
        self.assertEqual(longest_true_run([]),0)

if __name__=='__main__':unittest.main()
