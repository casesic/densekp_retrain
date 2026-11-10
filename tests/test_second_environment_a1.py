import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.experiments.prepare_second_environment_a1 import (
    centerline, prepare, rigid_placement, transform_points,
)
from densekp_retrain.data.sequence_resume import inspect_parent


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/second_environment_boundary_pass_a1_20261010.json'


class SecondEnvironmentTests(unittest.TestCase):
    def test_rigid_road_and_rig_geometry(self):
        for name in ('cycling_forest', 'riverfront'):
            transform, angle = rigid_placement(name, -4.25 if name == 'cycling_forest' else 0.0,
                                                 [0.3, -0.2, 0.0])
            anchor = transform_points([0.3, -0.2, 0.0], transform)
            self.assertAlmostEqual(anchor[0], centerline(name, anchor[1]), places=8)
            self.assertAlmostEqual(anchor[2], 0.0)
            np.testing.assert_allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(transform[:3, :3]), 1.0)
            self.assertLess(abs(angle), 25)

    def test_frozen_seed_and_source_time_are_replayed(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            base = Path(temp)
            a = prepare(CONFIG, 'cycling_forest', base / 'one')
            b = prepare(CONFIG, 'cycling_forest', base / 'two')
            self.assertIsNotNone(a)
            self.assertEqual(a['people'][0]['asset_frames'], b['people'][0]['asset_frames'])
            self.assertEqual(a['people'][0]['source_frames'], b['people'][0]['source_frames'])
            self.assertEqual(a['environment_road_to_world'], b['environment_road_to_world'])
            self.assertEqual(json.loads((base/'one/candidate_checks.json').read_text())['attempts'],
                             json.loads((base/'two/candidate_checks.json').read_text())['attempts'])
            self.assertEqual(a['coverage_goal'] if 'coverage_goal' in a else 'boundary_transition',
                             'boundary_transition')
            self.assertLess(json.loads((base/'one/geometry_preflight.json').read_text())
                            ['selected_attempt']['max_full_erp_projection_delta_px'], 0.002)
            self.assertEqual(a['environment_blend_dependency']['path'],
                             'assets/scenes/photoreal/cycling_forest/cycling_forest.blend')

    def test_invalid_environment_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            with self.assertRaisesRegex(ValueError, 'unknown_environment'):
                prepare(CONFIG, 'not_a_road', Path(temp) / 'x')

    def test_missing_dependency_and_road_bounds_refused(self):
        with tempfile.TemporaryDirectory(dir=ROOT / 'outputs') as temp:
            base = Path(temp)
            config = json.loads(CONFIG.read_text())
            config['environments']['riverfront']['blend_path'] = 'assets/scenes/riverfront/absent.blend'
            (base / 'missing.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'environment_dependency_missing'):
                prepare(base / 'missing.json', 'riverfront', base / 'missing_output')
            config['environments']['riverfront']['blend_path'] = 'assets/scenes/riverfront/riverfront.blend'
            config['environments']['riverfront']['road_half_width_m'] = 0.01
            (base / 'narrow.json').write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, 'invalid_native_road_config'):
                prepare(base / 'narrow.json', 'riverfront', base / 'narrow_output')

    def test_probe_prefix_resume_is_read_only(self):
        probe = ROOT / 'outputs/experiments/second_environment_boundary_pass_a1_20261010/probe_riverfront'
        if not (probe / 'task_wall.json').exists():
            self.skipTest('probe has not completed')
        resume = inspect_parent(probe)
        self.assertEqual(resume['reused'], [0, 1, 2])
        self.assertEqual(resume['new'], list(range(3, 31)))
        self.assertTrue(resume['parent_outputs_readonly'])


if __name__ == '__main__':
    unittest.main()
