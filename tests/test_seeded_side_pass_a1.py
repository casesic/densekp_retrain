import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.data.viewpoint import classify_view, side_pass_edge_coverage, side_view_control
from densekp_retrain.data.sequence_index import SequenceIndexError, validate_sequence_index
from tools.experiments.seed_fixed_person_side_pass_a1 import plan
from tests.test_sequence_index import sequence_fixture, write_fixture_files


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/seeded_side_pass_a1_20261010.json'


class SeededSidePassTest(unittest.TestCase):
    def test_camera_stream_changes_without_person_plan_change(self):
        with tempfile.TemporaryDirectory() as directory:
            a = plan(CONFIG, Path(directory) / 'a.json', seed=20261024)
            b = plan(CONFIG, Path(directory) / 'b.json', seed=20261024)
            c = plan(CONFIG, Path(directory) / 'c.json', seed=20261025)
        self.assertEqual(a, b)
        self.assertEqual(a['scene_episode_id'], c['scene_episode_id'])
        self.assertEqual(a['person_world_ref'], c['person_world_ref'])
        self.assertEqual(a['person_matrices_ref'], c['person_matrices_ref'])
        self.assertNotEqual(a['candidates'], c['candidates'])
        self.assertEqual(len(a['candidates']), 32)

    def test_whole_rig_rotation_keeps_both_relative_matrices(self):
        old, old_lenses, _ = rig_poses((0., 0., 0.), 90.7, .18)
        new, new_lenses, _ = rig_poses((3.15, 2.5, 0.), 175., .18)
        for a, b in ((old, new), (old_lenses, new_lenses)):
            np.testing.assert_allclose(np.linalg.inv(a['cam2']) @ a['cam4'],
                                       np.linalg.inv(b['cam2']) @ b['cam4'], atol=1e-12)
        self.assertNotEqual(old['cam2'][0, 3], new['cam2'][0, 3])
        self.assertNotEqual(old['cam4'][0, 3], new['cam4'][0, 3])

    def test_side_window_does_not_relax_edge_or_road_pass(self):
        rows = [{'views': {'cam2': {'view_class': 'side'}, 'cam4': {'view_class': 'side'}}}
                for _ in range(31)]
        edge = [True] * 16 + [False] * 15
        positions = np.linspace(-.7, .7, 31)
        valid = side_pass_edge_coverage(rows, edge, positions)
        self.assertTrue(valid['passed'])
        self.assertFalse(side_pass_edge_coverage(rows, [False] * 31, positions)['passed'])
        self.assertFalse(side_pass_edge_coverage(rows, edge, np.ones(31))['passed'])
        self.assertEqual(classify_view(80), 'side')
        self.assertEqual(classify_view(100), 'rear_oblique')

    def test_side_control_is_distinct_from_dual_edge_goal(self):
        rows=[dict(views={c:dict(view_class='side') for c in ('cam2','cam4')},common_physical_points=9)
              for _ in range(31)]
        positions=np.linspace(-.7,.7,31)
        self.assertTrue(side_view_control(rows,positions)['passed'])
        self.assertFalse(side_pass_edge_coverage(rows,[False]*31,positions)['passed'])
        rows[1]['common_physical_points']=0
        self.assertEqual(side_view_control(rows,positions)['common_physical_count'],30)
        self.assertFalse(side_view_control(rows,np.ones(31))['passed'])

    def test_optional_scene_and_rig_identity_in_sequence_rows(self):
        ep, table = sequence_fixture()
        ep.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        table.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        for camera in ep['cameras']:camera['rig_id']='rig_side_1'
        for obs in table['observations']:obs.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        for box in table['boxes']:box.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        validate_sequence_index(ep, table, base_dir='/fixture', check_files=False)
        table['boxes'][0]['rig_id']='wrong_rig'
        with self.assertRaises(SequenceIndexError) as error:
            validate_sequence_index(ep, table, base_dir='/fixture', check_files=False)
        self.assertEqual(error.exception.code, 'box_scene_rig_identity_mismatch')

    def test_packed_rig_columns_preserve_loader_compatibility(self):
        ep, table = sequence_fixture()
        ep.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        table.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        for camera in ep['cameras']:camera['rig_id']='rig_side_1'
        for obs in table['observations']:obs.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        for box in table['boxes']:box.update(scene_episode_id='shared_scene', rig_id='rig_side_1')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_fixture_files(root, ep, table)
            with np.load(root / 'packed.npz', allow_pickle=False) as archive:
                columns = {key: archive[key] for key in archive.files}
            columns.update(scene_episode_id=np.array(['shared_scene'] * 6),
                           rig_id=np.array(['rig_side_1'] * 6))
            np.savez(root / 'packed.npz', **columns)
            validate_sequence_index(ep, table, base_dir=root, check_files=True, check_packed=True)
            from densekp_retrain.data.loading import load_box_dataset
            item = load_box_dataset(root / 'packed.npz')[0]
            self.assertEqual(item['img_size'].tolist(), [60, 80])


if __name__ == '__main__':
    unittest.main()
