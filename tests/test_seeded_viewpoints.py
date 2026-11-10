import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.data.dynamic_scene import scene_trajectory
from densekp_retrain.data.mixed_road_pass import candidates,crossing_transition,coverage_goal,plan_mixed,preflight_mixed
from densekp_retrain.data.viewpoint import classify_view,view_angle,view_coverage
from tools.experiments.seed_scene_modes import load_pool,make_config


POOL=Path(__file__).resolve().parents[1]/'configs/seeded_viewpoints_a1_20261010.json'


class SeededViewpointTests(unittest.TestCase):
    def test_binning_and_degenerate_vectors(self):
        self.assertEqual([classify_view(v) for v in (0,44.99,45,79.99,80,99.99,100,134.99,135,180)],
                         ['front','front','front_oblique','front_oblique','side','side','rear_oblique','rear_oblique','back','back'])
        self.assertAlmostEqual(view_angle([0,0,0],[0,1,0],[0,1,0]),0)
        self.assertAlmostEqual(view_angle([0,0,0],[0,1,0],[0,-1,0]),180)
        with self.assertRaisesRegex(ValueError,'degenerate'):view_angle([0,0,0],[0,1,0],[0,0,1])

    def test_camera_stream_does_not_move_asset_or_layout(self):
        pool=load_pool(POOL)
        a,ra=make_config(pool,'single_cyclist_boundary_pass',20261012,0,'front')
        b,rb=make_config(pool,'single_cyclist_boundary_pass',20261012,0,'front')
        self.assertEqual((a,ra),(b,rb))
        self.assertEqual(a['requested_view_class'],'front')
        self.assertNotEqual(candidates(a,a['master_seed'])[0]['camera_parameters'],
                            candidates(a,a['master_seed'])[1]['camera_parameters'])
        changed=copy.deepcopy(a)
        changed['camera_parameter_ranges']['yaw_deg']=[90.,90.]
        old=candidates(a,a['master_seed']);new=candidates(changed,changed['master_seed'])
        self.assertEqual([r['parameters'] for r in old],[r['parameters'] for r in new])
        self.assertEqual([r['child_seed'] for r in old],[r['child_seed'] for r in new])
        self.assertNotEqual([r['camera_parameters'] for r in old],[r['camera_parameters'] for r in new])
        self.assertNotEqual(make_config(pool,'single_cyclist_boundary_pass',20261013,0,'front')[0]['camera_seed'],a['camera_seed'])
        self.assertNotEqual(make_config(pool,'single_cyclist_boundary_pass',20261013,0,'front')[0]['episode_child_seed'],a['episode_child_seed'])

    def test_old_recipe_keeps_fixed_rig_and_positive_y(self):
        pool=load_pool(POOL)
        old,_=make_config(pool,'single_cyclist_boundary_pass',20261010,0)
        self.assertNotIn('camera_seed',old)
        self.assertNotIn('requested_view_class',old)
        self.assertNotIn('travel_direction_sign',candidates(old,old['master_seed'])[0])
        ep=plan_mixed(old)['episodes'][0]
        self.assertNotIn('requested_view_class',ep)

    def test_rig_moves_as_one_unit_without_cam4_erp_flip(self):
        a,al,_=rig_poses([0,0,0],90,.18)
        b,bl,_=rig_poses([-1.2,.1,0],94,.18)
        np.testing.assert_allclose(np.linalg.inv(a['cam2'])@a['cam4'],np.linalg.inv(b['cam2'])@b['cam4'],atol=1e-12)
        np.testing.assert_allclose(np.linalg.inv(al['cam2'])@al['cam4'],np.linalg.inv(bl['cam2'])@bl['cam4'],atol=1e-12)
        np.testing.assert_allclose(b['cam2'][:3,:3],b['cam4'][:3,:3],atol=1e-12)
        self.assertAlmostEqual(b['cam2'][2,3],1.15)
        point=np.array([[0,2,1.2]])
        from densekp_retrain.camera.erp import project_erp
        from densekp_retrain.camera.domain import point_in_view
        mask=np.ones((100,200),dtype=bool)
        xy=project_erp(point,b['cam2'],200,100)[0]
        np.testing.assert_allclose(xy,point_in_view(point,b['cam2'],'cam2',mask)[0])

    def test_positive_speed_reverse_time_and_crossing(self):
        clock=[dict(episode_time_tick=k,time_base=[1,10]) for k in (0,10,30)]
        M=scene_trajectory(clock,[0,3,1.2],[0,-1],.5,np.eye(3))
        np.testing.assert_allclose(M[:,1,3],[3,2.5,1.5])
        exit_record={c:dict(first_outside=2,last_outside=24,first_distance_p10=20,last_distance_p10=-30) for c in ('cam2','cam4')}
        self.assertEqual(crossing_transition(exit_record,'either'),'exit_valid')
        self.assertIsNone(crossing_transition(exit_record,'enter_valid'))
        enter_record={c:dict(first_outside=25,last_outside=2,first_distance_p10=-30,last_distance_p10=20) for c in ('cam2','cam4')}
        self.assertEqual(crossing_transition(enter_record,'enter_valid'),'enter_valid')
        with self.assertRaisesRegex(ValueError,'invalid_crossing'):crossing_transition(exit_record,'reverse_video')

    def test_both_cameras_and_edge_overlap_are_required(self):
        rows=[dict(views=dict(cam2=dict(view_class='front'),cam4=dict(view_class='front'))) for _ in range(31)]
        edge=[False]*10+[True]*11+[False]*10
        self.assertTrue(view_coverage(rows,edge,'front')['passed'])
        rows[14]['views']['cam4']['view_class']='back'
        rows[17]['views']['cam4']['view_class']='back'
        self.assertFalse(view_coverage(rows,edge,'front')['passed'])
        with self.assertRaisesRegex(ValueError,'invalid_requested'):view_coverage(rows,edge,'facial',16)

    def test_coverage_goal_defaults_and_invalid(self):
        self.assertEqual(coverage_goal({},True),'boundary_transition')
        self.assertEqual(coverage_goal({},False),'edge_coverage')
        with self.assertRaisesRegex(ValueError,'invalid_coverage_goal'):coverage_goal({'coverage_goal':'skip_checks'},True)
        with self.assertRaisesRegex(ValueError,'mixed_boundary_transition_unsupported'):
            coverage_goal({'coverage_goal':'boundary_transition'},False)
        pool=load_pool(POOL)
        cfg,_=make_config(pool,'single_cyclist_boundary_pass',20261012,0,'front',coverage_goal='edge_coverage')
        self.assertEqual(cfg['coverage_goal'],'edge_coverage')
        self.assertIn('edge_coverage',cfg['episode_id'])

    def test_frozen_candidate_passes_edge_goal_but_strict_goal_rejects(self):
        root=Path(__file__).resolve().parents[1]
        new=root/'outputs/experiments/front_edge_coverage_a1_20261010/config.json'
        old=root/'outputs/experiments/seeded_viewpoints_a1_20261010/preflight/single_20261012_retry1/single_cyclist_boundary_pass_front_s20261012_e00/candidate_checks.json'
        cfg=json.loads(new.read_text())
        frozen=json.loads(old.read_text())[0]['attempts'][0]
        self.assertEqual(candidates(cfg,cfg['master_seed'])[0]['camera_child_seed'],frozen['camera_child_seed'])
        self.assertEqual(candidates(cfg,cfg['master_seed'])[0]['parameters'],frozen['parameters'])
        cfg['attempt_limit']=1
        cfg['explicit_candidate_parameters']=[frozen['parameters']]
        with tempfile.TemporaryDirectory() as tmp:
            edge=copy.deepcopy(cfg);edge['episode_id']='synthetic_test_only_edge_goal'
            passed=preflight_mixed(plan_mixed(edge),tmp)
            self.assertIsNotNone(passed)
            self.assertFalse(passed['boundary_transition_passed'])
            self.assertTrue(passed['actual_view_coverage']['passed'])
            strict=copy.deepcopy(cfg);strict['episode_id']='synthetic_test_only_transition_goal'
            strict['coverage_goal']='boundary_transition'
            self.assertIsNone(preflight_mixed(plan_mixed(strict),tmp))
            attempted=json.loads((Path(tmp)/strict['episode_id']/'candidate_checks.json').read_text())[0]['attempts'][0]
            self.assertIn('boundary_crossing_not_proved',attempted['rejections'])
            tiny=copy.deepcopy(cfg);tiny['episode_id']='synthetic_test_only_edge_tiny_ROI'
            tiny['max_union_roi_pixels']=1
            self.assertIsNone(preflight_mixed(plan_mixed(tiny),tmp))
            attempted=json.loads((Path(tmp)/tiny['episode_id']/'candidate_checks.json').read_text())[0]['attempts'][0]
            self.assertIn('union_ROI_pixel_budget',attempted['rejections'])


if __name__=='__main__':unittest.main()
