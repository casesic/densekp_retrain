"""Bounded CPU checks for the component-sampling road interface."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from tools.experiments.seed_asset_composition_a1 import (
    ROOT, CONFIG, OUTPUT, allowed_components, checked_build, freeze_plans,
    load_policy, sampled_selection, derive_rotation_and_height,
)
from tools.experiments.seed_scene_modes import make_config
from densekp_retrain.assets.composition import read
from densekp_retrain.data.dynamic_scene import frame_record
from densekp_retrain.data.mixed_road_pass import plan_mixed


class SeedAssetCompositionA1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.policy=load_policy(CONFIG)
        cls.registry=read(ROOT / cls.policy['registry'])

    def test_frozen_eight_plan_replays_without_building(self):
        plans=freeze_plans(self.policy,self.registry)
        frozen=read(OUTPUT/'frozen_plans.json')
        self.assertEqual(plans,frozen)
        self.assertEqual(len(plans),8)
        self.assertNotEqual(plans[0]['selected_components']['body_id'],plans[1]['selected_components']['body_id'])
        self.assertNotEqual(plans[0]['selected_components']['motion_id'],plans[1]['selected_components']['motion_id'])
        self.assertTrue(any(p['selected_components'][kind].split('_')[0]!=p['selected_components']['body_id']
                            for p in plans[:2] for kind in ('clothing_id','hair_id')))

    def test_other_master_seed_changes_component_selection(self):
        groups=allowed_components(self.policy,self.registry)
        original,_,_,_=sampled_selection(self.policy,self.registry,groups,0)
        changed_policy=copy.deepcopy(self.policy);changed_policy['master_seed']+=1
        changed,_,_,_=sampled_selection(changed_policy,self.registry,groups,0)
        self.assertNotEqual(original,changed)

    def test_source_gate_runs_before_component_sample(self):
        policy=copy.deepcopy(self.policy)
        policy['allowed']['motion_source_groups']=['unavailable']
        with self.assertRaisesRegex(ValueError,'empty_source'):
            allowed_components(policy,self.registry)
        policy=copy.deepcopy(self.policy)
        policy['allowed']['walker_source_groups']=[]
        with self.assertRaisesRegex(ValueError,'walker_source_group_excluded'):
            allowed_components(policy,self.registry,mode='mixed_walk_cycle_crossing')
        policy=copy.deepcopy(self.policy)
        policy['allowed']['clothing'].append('BEDLAM_native_locked_example')
        with self.assertRaisesRegex(ValueError,'unsupported_component_representation'):
            allowed_components(policy,self.registry)

    def test_short_motion_and_unknown_representation_rejected(self):
        from tools.experiments.seed_scene_modes import valid_origins
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'short_synthetic_test_only.json'
            path.write_text(json.dumps([dict(asset_frame=i,source_frame=i,source_time_tick=i,time_base=[1,30]) for i in range(4)]))
            self.assertEqual(valid_origins({'timeline_ref':str(path)},
                [dict(frame_index=k,episode_time_tick=k,time_base=[1,10]) for k in range(31)]),[])
        recipe=dict(body_id='M01',motion_id='APD6526',clothing_id='BEDLAM_native_locked_example',
                    hair_id='M01_groom',window=[0,2],root_policy='fixed_pelvis_scene_owns_progression')
        from densekp_retrain.assets.composition import plan
        with self.assertRaisesRegex(ValueError,'native_cache_locked'):
            plan(ROOT,self.registry,recipe)

    def test_verified_delivery_hit_and_nonfirst_original_time(self):
        plans=read(OUTPUT/'frozen_plans.json')
        out=OUTPUT/'composition_cache'
        for plan in plans[:2]:
            _,_,adapter,state=checked_build(plan,out)
            self.assertEqual(state,'delivery_verified_hit')
            self.assertEqual(adapter['root_mode'],'fixed_pelvis_scene_owns_progression')
            self.assertEqual(adapter['components'] if 'components' in adapter else plan['selected_components'],plan['selected_components'])
        cfg=read(OUTPUT/'episode_00/config.json')
        ep=plan_mixed(cfg)['episodes'][0]
        self.assertEqual(ep['component_provenance']['selection'],plans[0]['selected_components'])
        self.assertEqual(ep['person_timelines']['1'][0]['source_record_index'],plans[0]['source_time_origin_record'])
        self.assertEqual(frame_record(ep,1)['person_records'][0]['asset_frame'],ep['people'][0]['asset_frames'][1])

    def test_occupied_delivery_change_or_partial_is_rejected(self):
        plan=read(OUTPUT/'frozen_plans.json')[0]
        changed=copy.deepcopy(plan)
        changed['resolved']['dependencies'][0]['size']+=1
        with self.assertRaisesRegex(ValueError,'occupied_delivery_recipe_or_dependency_changed'):
            checked_build(changed,OUTPUT/'composition_cache')
        with tempfile.TemporaryDirectory() as d:
            selected=plan['selected_components']
            name=f"a1_00_{selected['body_id']}_{selected['motion_id']}_{selected['clothing_id']}_{selected['hair_id']}"
            partial=Path(d)/'builds'/name
            partial.mkdir(parents=True)
            (partial/'frozen_recipe.json').write_text(json.dumps(plan['resolved']))
            with self.assertRaises(FileNotFoundError):
                checked_build(plan,Path(d))

    def test_fixed_heading_correction_reuses_motion_and_keeps_25_degree_gate(self):
        import numpy as np
        from scipy.spatial.transform import Rotation
        plan=read(OUTPUT/'frozen_plans.json')[1]
        asset=read(OUTPUT/'episode_01_correction1/asset_adapter.json')
        old,old_h,_,rows=derive_rotation_and_height(asset,plan['source_time_origin_record'],'heading_mean_v1')
        new,new_h,_,again=derive_rotation_and_height(asset,plan['source_time_origin_record'],'heading_minimax_fixed_yaw_v2')
        self.assertEqual(rows,again)
        self.assertEqual(old_h,new_h)
        self.assertNotEqual(old,new)
        with np.load(asset['parameters'],allow_pickle=False) as z:
            local=Rotation.from_rotvec(z['global_orient'][[r['asset_frame'] for r in rows]]).apply([0.,0.,1.])
        heading=local @ np.asarray(new)[:2,:].T
        angles=np.degrees(np.arctan2(heading[:,0],heading[:,1]))
        self.assertLessEqual(float(np.abs(angles).max()),25.)

    def test_legacy_bound_pool_keeps_old_selection_path(self):
        old=read(ROOT/'configs/seeded_scene_modes_a1_20261010.json')
        cfg,record=make_config(old,'single_cyclist_boundary_pass',20261010,0)
        self.assertEqual(record['selected_asset_id'],'cycling_M01_v2_APD6526_30_106_simple_v1')
        self.assertNotIn('component_provenance',cfg)


if __name__=='__main__': unittest.main()
