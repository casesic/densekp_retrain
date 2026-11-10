import math
import unittest

import numpy as np

from densekp_retrain.data.t_junction import (
    check_trajectory, road_point, road_supported, scene_trajectory,
)
from tools.experiments.prepare_urban_three_multirig_a1 import coverage_pass


class TJunctionTrajectoryTests(unittest.TestCase):
    def test_road_is_a_union_not_its_bounding_rectangle(self):
        self.assertTrue(road_supported(0., 79.95, margin=.04))
        self.assertTrue(road_supported(5., 80.05, margin=.04))
        self.assertFalse(road_supported(5., 79.95, margin=.04))
        self.assertFalse(road_supported(5., 88., margin=.04))

    def test_quarter_arc_has_position_and_tangent_continuity(self):
        radius = 4.
        join = math.pi * radius / 2
        for s in (0., join):
            before, previous_tangent = road_point(s-1e-5, turn_y=78., radius=radius)
            after, next_tangent = road_point(s+1e-5, turn_y=78., radius=radius)
            self.assertLess(np.linalg.norm(before-after), 3e-5)
            self.assertLess(np.linalg.norm(previous_tangent-next_tangent), 3e-5)

    def test_sampled_turn_rotates_rigid_person_with_path_tangent(self):
        records = [dict(frame_index=i, episode_time_tick=i, time_base=[1, 10])
                   for i in range(31)]
        base = np.array([[-1., 0., 0.], [0., 0., 1.], [0., 1., 0.]])
        matrices = scene_trajectory(records, lane='turn_main_to_positive_x',
                                    start=.05, speed=2.07, lateral=0., height=1.2,
                                    base_rotation=base, turn_y=78., radius=4.)
        check = check_trajectory(matrices)
        self.assertLess(check['maximum_sampled_heading_step_deg'], 8.)
        self.assertGreater(matrices[-1, 0, 3], 3.8)
        self.assertGreater(matrices[-1, 1, 3], 81.9)
        np.testing.assert_allclose(matrices[:, 2, 3], 1.2)
        headings = matrices[:, :3, :3] @ np.array([0., 0., 1.])
        self.assertGreater(headings[0, 1], .99)
        self.assertGreater(headings[-1, 0], .99)

    def test_invalid_turn_and_old_straight_mode_are_separate(self):
        with self.assertRaises(ValueError):
            road_point(1., turn_y=78., radius=2.)
        records = [dict(frame_index=i, episode_time_tick=i, time_base=[1, 10])
                   for i in range(31)]
        base = np.eye(3)
        with self.assertRaises(ValueError):
            scene_trajectory(records, lane='unspecified', start=0., speed=2.,
                             lateral=0., height=1., base_rotation=base)

    def test_optional_junction_rig_gates_preserve_old_defaults(self):
        common = [True] * 31
        view = [True] * 8 + [False] * 23
        edge = [False] * 31
        base = dict(minimum_common_physical_frames=6, minimum_view_common_run=6)
        self.assertFalse(coverage_pass(common, view, edge, view_goal='back',
                                       config=base, road_pass=False))
        junction = dict(base, minimum_view_common_frames=6,
                        minimum_dual_edge_frames=0, minimum_dual_edge_run=0)
        self.assertTrue(coverage_pass(common, view, edge, view_goal='back',
                                      config=junction, road_pass=False))
        self.assertFalse(coverage_pass(common, view, edge, view_goal='side',
                                       config=base, road_pass=False))
        self.assertTrue(coverage_pass(common, view, edge, view_goal='side',
                                      config=dict(base, require_road_cross_section=False), road_pass=False))


if __name__ == '__main__':
    unittest.main()
