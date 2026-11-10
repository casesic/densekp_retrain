"""Focused CPU checks for bounded 4/5-person planning and rig identity."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from densekp_retrain.data.cycling_team import team_recipe
from tools.experiments.prepare_dispersed_cyclists_a1 import candidates, camera_candidates
from tools.experiments.run_multirig_roi_a1 import prepare


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'outputs/experiments/cycling_team_relative_layout_a1_20261010/city_attempt5_cpu_only_correction1/prepared'


class DispersedCyclistsTest(unittest.TestCase):
    def test_layout_seed_replays_and_changes_new_people_only(self):
        anchor = json.loads((OLD / 'recipe.json').read_text())['sampled_parameters']
        for count, seed in ((4, 20261030), (5, 20261031)):
            first = candidates(count, seed, anchor)
            self.assertEqual(first, candidates(count, seed, anchor))
            self.assertNotEqual(first, candidates(count, seed + 1, anchor))
            self.assertEqual(len(first), 12)
            self.assertEqual(len({tuple(sorted(c['parameters'].items())) for c in first}), 12)
            for item in first:
                for key, value in anchor.items():
                    self.assertEqual(item['parameters'][key], value)
                self.assertGreater(item['parameters']['p4_y_m'], anchor['p3_y_m'])

    def test_camera_seed_changes_rig_without_changing_person_plan(self):
        source = ROOT / 'outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/rigs/rig_front_oblique/recipe.json'
        rig = json.loads(source.read_text())['rig']
        base = dict(origin_m=rig['origin_m'], yaw_deg=rig['yaw_deg'])
        first = camera_candidates(4, 20261032, base)
        self.assertEqual(first, camera_candidates(4, 20261032, base))
        self.assertNotEqual(first[1:], camera_candidates(4, 20261035, base)[1:])
        self.assertEqual(first[0]['origin_m'], rig['origin_m'])
        self.assertEqual(first[0]['yaw_deg'], rig['yaw_deg'])

    def test_four_five_people_keep_individual_original_clocks(self):
        recipe = json.loads((OLD / 'recipe.json').read_text())
        stage = ROOT / 'outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011'
        people = [dict(p) for p in recipe['people']]
        for person_id, motion in ((4, 'p1_alternate'), (5, 'p2_straight')):
            adapter = stage / f'city_{person_id}p/person{person_id}_asset_adapter.json'
            if not adapter.exists():
                self.skipTest('combination adapter not yet generated')
            person = dict(people[0], person_id=person_id,
                character_family=f'cycling_M0{person_id - 1}',
                source_group='zenodo_10668611_participant_1' if person_id == 4
                             else 'zenodo_10668611_participant_2',
                asset_manifest=str(adapter))
            people.append(person)
            episode, _ = team_recipe(recipe, people, [0, 4, 19, 0, 0][:person_id],
                                     f'fixture_{person_id}', 'urban_street_assets')
            self.assertEqual(len(episode['people']), person_id)
            self.assertEqual(len({p['character_family'] for p in episode['people']}), person_id)
            self.assertEqual(len({p['source_group'] for p in episode['people']}), 3)
            self.assertEqual(len(episode['person_timelines']), person_id)
            self.assertNotEqual(episode['person_timelines']['1'][10]['source_frame'],
                                episode['person_timelines'][str(person_id)][10]['source_frame'])

    def test_prepared_four_person_axes_and_explicit_readback_ids(self):
        prepared = ROOT / 'outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_4p/prepared_edge'
        if not (prepared / 'geometry_preflight.json').exists():
            self.skipTest('production fixture not yet generated')
        self.assertEqual(np.load(prepared / 'world_densekp.npy').shape, (31, 4, 138, 3))
        self.assertEqual(np.load(prepared / 'expected_in_view.npy').shape, (31, 4, 2, 138))
        slots = [f'f15_{c}_p{i}' for c in ('cam2', 'cam4') for i in range(1, 5)]
        slots += [f'f0_{c}_p1' for c in ('cam2', 'cam4')]
        slots += [f'f30_{c}_p2' for c in ('cam2', 'cam4')]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'rigs.json'
            manifest.write_text(json.dumps(dict(schema='scene_episode_rig_list_v1',
                scene_episode_id='fixture_scene', rigs=[dict(rig_id='rig_edge',
                prepared=str(prepared), coverage_goal='edge_coverage',
                readback_observation_ids=slots)])))
            prepare(root / 'run', gpu_uuid='fixture-GPU', budget_seconds=30,
                    rig_list_manifest=manifest)
            stored = json.loads((root / 'run/rigs/rig_edge/recipe.json').read_text())
            self.assertEqual(stored['readback_observation_ids'], slots)
            self.assertEqual(stored['readback_row_cap'], 12)

    def test_five_person_geometry_has_all_independent_axes(self):
        prepared = ROOT / 'outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p/prepared_edge'
        if not (prepared / 'geometry_preflight.json').exists():
            self.skipTest('five-person CPU preflight not yet generated')
        recipe = json.loads((prepared / 'recipe.json').read_text())
        matrix = np.load(prepared / 'scene_matrices.npy')
        world = np.load(prepared / 'world_densekp.npy')
        in_view = np.load(prepared / 'expected_in_view.npy')
        self.assertEqual(matrix.shape, (31, 5, 4, 4))
        self.assertEqual(world.shape, (31, 5, 138, 3))
        self.assertEqual(in_view.shape, (31, 5, 2, 138))
        self.assertEqual([p['person_id'] for p in recipe['people']], [1, 2, 3, 4, 5])
        self.assertEqual(len({p['character_family'] for p in recipe['people']}), 5)
        self.assertGreaterEqual(len({p['component_ids']['motion_id'] for p in recipe['people']}), 3)
        self.assertEqual([p['asset_frames'][0] for p in recipe['people'][:3]], [0, 4, 19])
        self.assertTrue(all(np.isfinite(array).all() for array in (matrix, world)))
        self.assertFalse(np.array_equal(matrix[:, 3], matrix[:, 4]))


if __name__ == '__main__':
    unittest.main()
