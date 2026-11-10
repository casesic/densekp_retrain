"""Small CPU checks for a fixed three-person scene and a list of paired rigs."""
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np

from densekp_retrain.runtime.frame_consumer import serve_frames
from tools.experiments.prepare_urban_three_multirig_a1 import plan, _longest
from tools.experiments.run_multirig_roi_a1 import prepare


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/urban_three_cyclists_multirig_a1_20261011.json'
PREPARED = ROOT / json.loads(CONFIG.read_text())['source_prepared']


class ThreePersonMultiRigTest(unittest.TestCase):
    def test_camera_seed_does_not_change_three_people_or_their_original_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = plan(CONFIG, root / 'a.json')
            replay = plan(CONFIG, root / 'b.json')
            alternate = plan(CONFIG, root / 'c.json', seed=20261029)
        self.assertEqual(first, replay)
        self.assertEqual(first['fixed_people'], alternate['fixed_people'])
        self.assertNotEqual(first['candidates'], alternate['candidates'])
        self.assertEqual(len(first['candidates']), 24)
        self.assertEqual(len({p['asset_id'] for p in first['fixed_people']}), 3)
        self.assertEqual([p['asset_frames'][0] for p in first['fixed_people']], [0, 4, 19])

    def test_front_window_requires_six_consecutive_true_values(self):
        self.assertEqual(_longest([True]*3+[False]+[True]*5), 5)
        self.assertEqual(_longest([False, True, True, True, True, True, True]), 6)

    def test_rig_list_preserves_person_and_time_axes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / 'rigs.json'
            manifest.write_text(json.dumps(dict(schema='scene_episode_rig_list_v1',
                scene_episode_id='fixture_shared_scene',rigs=[
                    dict(rig_id='rig_edge', prepared=str(PREPARED), coverage_goal='edge_coverage'),
                    dict(rig_id='rig_front_oblique',prepared=str(PREPARED),coverage_goal='front_oblique_view_control')])))
            config = prepare(root / 'production',gpu_uuid='fixture-GPU',budget_seconds=30,
                             rig_list_manifest=manifest)
            self.assertEqual(config['max_camera_calls'],124)
            scene = json.loads((root/'production/recipe.json').read_text())
            self.assertEqual([p['person_id'] for p in scene['people']],[1,2,3])
            self.assertEqual([p['asset_frames'][0] for p in scene['people']],[0,4,19])
            for rig_id in ('rig_edge','rig_front_oblique'):
                child=root/'production/rigs'/rig_id
                self.assertEqual(np.load(child/'expected_in_view.npy').shape,(31,3,2,138))
                self.assertEqual(json.loads((child/'recipe.json').read_text())['scene_episode_id'],'fixture_shared_scene')
            with self.assertRaises(FileExistsError):
                prepare(root/'production',gpu_uuid='fixture-GPU',budget_seconds=30,
                        rig_list_manifest=manifest)

    def test_duplicate_rig_identity_rejected_before_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/'rigs.json'
            manifest.write_text(json.dumps(dict(schema='scene_episode_rig_list_v1',scene_episode_id='scene',
                rigs=[dict(rig_id='rig_edge',prepared=str(PREPARED),coverage_goal='edge_coverage')]*2)))
            with self.assertRaisesRegex(ValueError,'invalid_rig_list'):
                prepare(root/'production',gpu_uuid='fixture-GPU',budget_seconds=30,rig_list_manifest=manifest)

    def test_changed_second_person_time_is_not_broadcast_from_person_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);changed=root/'changed';shutil.copytree(PREPARED,changed)
            recipe=json.loads((changed/'recipe.json').read_text())
            recipe['people'][1]['source_frames'][0]+=1
            (changed/'recipe.json').write_text(json.dumps(recipe))
            manifest=root/'rigs.json'
            manifest.write_text(json.dumps(dict(schema='scene_episode_rig_list_v1',scene_episode_id='scene',
                rigs=[dict(rig_id='rig_edge',prepared=str(PREPARED),coverage_goal='edge_coverage'),
                      dict(rig_id='rig_front_oblique',prepared=str(changed),coverage_goal='front_oblique_view_control')])))
            with self.assertRaisesRegex(ValueError,'frozen_scene_mismatch'):
                prepare(root/'production',gpu_uuid='fixture-GPU',budget_seconds=30,rig_list_manifest=manifest)

    def test_previous_two_rig_invocation_keeps_legacy_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=prepare(Path(tmp)/'legacy',gpu_uuid='fixture-GPU',budget_seconds=30)
            self.assertEqual(config['max_camera_calls'],124)
            plan=json.loads((Path(tmp)/'legacy/multirig.json').read_text())
            self.assertEqual([x['rig_id'] for x in plan['rigs']],['rig_edge','rig_side'])
            self.assertEqual(json.loads((Path(tmp)/'legacy/recipe.json').read_text())['people'][0]['person_id'],1)

    def test_one_consumer_routes_by_rig_and_rejects_duplicate_unit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'recipe.json').write_text(json.dumps(dict(frame_count=2)))
            rigs=[]
            for rig_id in ('rig_edge','rig_front_oblique'):
                child=root/'rigs'/rig_id
                (child/'frame_000').mkdir(parents=True)
                rigs.append(dict(rig_id=rig_id,root=str(child)))
            (root/'multirig.json').write_text(json.dumps(dict(rigs=rigs)))
            calls=[]
            incoming=io.StringIO(''.join(json.dumps(x)+'\n' for x in [
                dict(command='frame',frame=0,rig_id='rig_edge'),
                dict(command='frame',frame=0,rig_id='rig_front_oblique'),
                dict(command='frame',frame=0,rig_id='rig_edge')]))
            outgoing=io.StringIO()
            code=serve_frames(root,lambda task,frame:calls.append((task.name,frame)),
                              incoming=incoming,outgoing=outgoing)
            self.assertEqual(code,1)
            self.assertEqual(calls,[('rig_edge',0),('rig_front_oblique',0)])
            self.assertEqual([json.loads(x)['status'] for x in outgoing.getvalue().splitlines()],
                             ['ready','ok','ok','error'])


if __name__ == '__main__':
    unittest.main()
