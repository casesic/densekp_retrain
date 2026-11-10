"""Focused contracts for a fixed five-person scene observed by three paired rigs."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from tools.experiments.prepare_front_back_side_a1 import CONFIG, OUT, candidates, child_seed
from tools.experiments.run_multirig_roi_a1 import prepare


class FrontBackSideContractTest(unittest.TestCase):
    def test_camera_stream_changes_only_camera_plan(self):
        cfg = json.loads(CONFIG.read_text())
        world = np.load(OUT / 'prepared_back/world_densekp.npy')
        a = candidates(cfg, world, 'front', cfg['camera_master_seed'])
        self.assertEqual(a, candidates(cfg, world, 'front', cfg['camera_master_seed']))
        self.assertNotEqual(a, candidates(cfg, world, 'front', cfg['comparison_seed']))
        self.assertNotEqual(child_seed(cfg['camera_master_seed'], 'front'),
                            child_seed(cfg['camera_master_seed'], 'side'))
        self.assertEqual(len(a), 16)
        self.assertEqual(len({tuple(x['origin_m']) for x in a}), 16)

    def test_frozen_view_and_support_evidence(self):
        back = json.loads((OUT / 'back_view_check.json').read_text())
        self.assertEqual(back['status'], 'passed')
        self.assertGreaterEqual(back['back_common_frames'], 16)
        self.assertGreaterEqual(back['dual_edge_frames'], 16)
        front = json.loads((OUT / 'prepared_front/geometry_preflight.json').read_text())
        side = json.loads((OUT / 'prepared_side/geometry_preflight.json').read_text())
        self.assertEqual(front['coverage']['view_goal'], 'front')
        self.assertGreaterEqual(front['coverage']['view_common_count'], 16)
        self.assertGreaterEqual(front['coverage']['longest_view_common_run'], 6)
        self.assertEqual(side['coverage']['view_goal'], 'side')
        self.assertTrue(side['coverage']['road_cross_section_passed'])
        self.assertGreaterEqual(side['coverage']['common_frames'], 16)
        self.assertGreaterEqual(side['coverage']['longest_view_common_run'], 3)
        self.assertTrue(front['exact_groom'] and side['exact_groom'])
        for view in ('back', 'front', 'side'):
            self.assertEqual(np.load(OUT / f'prepared_{view}/expected_in_view.npy').shape,
                             (31, 5, 2, 138))

    def test_three_rig_cpu_prepare_preserves_people_and_ids(self):
        manifest = OUT / 'rig_list_manifest.json'
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'prepared'
            config = prepare(root, gpu_uuid='GPU-test-only', budget_seconds=3600,
                             rig_list_manifest=manifest)
            self.assertEqual(config['max_camera_calls'], 186)
            spec = json.loads((root / 'multirig.json').read_text())
            frozen = json.loads(manifest.read_text())
            self.assertEqual([x['rig_id'] for x in spec['rigs']],
                             ['rig_back', 'rig_front', 'rig_side'])
            self.assertEqual(spec['maximum_budget_seconds'], 3600)
            shared = np.load(root / 'world_densekp.npy')
            self.assertEqual(shared.shape, (31, 5, 138, 3))
            for entry, expected in zip(spec['rigs'], frozen['rigs']):
                p = Path(entry['root'])
                recipe = json.loads((p / 'recipe.json').read_text())
                self.assertEqual(recipe['scene_episode_id'], spec['scene_episode_id'])
                self.assertEqual(recipe['rig_id'], entry['rig_id'])
                self.assertEqual(recipe['camera_seed'], expected['camera_seed'])
                self.assertEqual(entry['camera_seed'], expected['camera_seed'])
                self.assertEqual(len(recipe['readback_observation_ids']), 12)
                self.assertEqual(recipe['neighbor_evidence_policy'], 'central_ray_consensus_v1')
                np.testing.assert_array_equal(np.load(p / 'world_densekp.npy'), shared)

    def test_duplicate_rig_id_rejected(self):
        manifest = json.loads((OUT / 'rig_list_manifest.json').read_text())
        manifest['rigs'][1]['rig_id'] = manifest['rigs'][0]['rig_id']
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.json'
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'invalid_rig_list'):
                prepare(Path(folder) / 'task', gpu_uuid='GPU-test-only', budget_seconds=3600,
                        rig_list_manifest=path)


if __name__ == '__main__':
    unittest.main()
