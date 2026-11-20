"""Check three actual paired rigs against one frozen five-rider city scene."""
import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.data.viewpoint import classify_view, view_angle
from tools.experiments.validate_dispersed_cyclists_a1 import validate as validate_slots


RIGS = ('rig_back', 'rig_front', 'rig_side')


def read(path):
    return json.loads(Path(path).read_text())


def longest(values):
    best = run = 0
    for value in values:
        run = run + 1 if value else 0
        best = max(best, run)
    return best


def validate(root):
    root = Path(root).resolve()
    plan = read(root.parent / 'frozen_execution.json')
    if plan['selected_rig_order'] != list(RIGS) or plan['planned_camera_calls'] != 186:
        raise ValueError('frozen_three_rig_plan_changed')
    original = Path(plan['source_prepared_ref'])
    old_production = Path(root.parent / 'frozen_scene.json')
    old_production = Path(read(old_production)['parent_production'])
    shared = read(root / 'recipe.json')
    previous = read(old_production / 'recipe.json')
    if shared['people'] != previous['people'] or shared['episode_timeline'] != previous['episode_timeline']:
        raise ValueError('person_or_time_changed')
    if not np.array_equal(np.load(root / 'world_densekp.npy'), np.load(old_production / 'world_densekp.npy')):
        raise ValueError('person_world_changed')
    base = validate_slots(root, expected_rigs=list(RIGS))
    if (base['planned_camera_calls'] != 186 or base['planned_observation_slots'] != 930 or
        base['rig_pair_relative_transform_max_abs'] > 1e-9 or
        any(not base['rigs'][r]['target_neighbor_evidence_available'] for r in RIGS)):
        raise ValueError('three_rig_count_relative_or_occlusion_evidence_conflict')
    matrices = np.load(original / 'scene_matrices.npy')
    if matrices.shape != (31, 5, 4, 4):
        raise ValueError('source_matrix_person_axis_conflict')
    asset = read(shared['people'][0]['asset_manifest'])
    with np.load(asset['parameters']) as params:
        orientation = params['global_orient'][shared['people'][0]['asset_frames']]
    headings = [matrices[k, 0, :3, :3] @ cv2.Rodrigues(orientation[k])[0] @ [0., 0., 1.]
                for k in range(31)]
    windows = {}
    for rig_id, view in zip(RIGS, ('back','front','side')):
        folder = root / 'rigs' / rig_id
        poses = {c['id']:np.asarray(c['camera_to_world']) for c in read(folder / 'rig.json')['cameras']}
        index = read(folder / 'pack/sequence_index.json')
        boxes = {box['observation_id'] for box in index['boxes']}
        class_flags, common_flags, actual_dual_input, rows = [], [], [], []
        for k in range(31):
            angles = {c:view_angle(matrices[k, 0, :3, 3], headings[k], poses[c][:3, 3])
                      for c in ('cam2','cam4')}
            classes = {c:classify_view(angles[c]) for c in angles}
            label = {}
            for c in ('cam2','cam4'):
                with np.load(folder / f'frame_{k:03d}' / c / 'annotations.npz') as z:
                    label[c] = z['in_view'][0].copy()
            common = int((label['cam2'] & label['cam4']).sum())
            class_flags.append(common >= 8 and all(classes[c] == view for c in classes))
            common_flags.append(common >= 8)
            paired = all(f'f{k}_{c}_p1' in boxes for c in ('cam2','cam4'))
            actual_dual_input.append(class_flags[-1] and paired)
            rows.append(dict(frame_index=k, angles_deg=angles, classes=classes,
                             common_physical_points=common, both_native_boxes=paired))
        if view in ('front','back'):
            passed = sum(class_flags) >= 16 and longest(class_flags) >= 6
        else:
            signed = [float(matrices[k, 0, 1, 3] - read(folder / 'recipe.json')['rig']['origin_m'][1])
                      for k in (0,30)]
            passed = (sum(common_flags) >= 16 and longest(class_flags) >= 3 and signed[0] < 0 < signed[1])
        if rig_id == 'rig_back':
            old = read(old_production / 'rigs/rig_edge/rig.json')
            if old['cameras'] != read(folder / 'rig.json')['cameras']:
                raise ValueError('back_camera_not_inherited_exactly')
            for k in range(31):
                for c in ('cam2','cam4'):
                    with (np.load(old_production / f'rigs/rig_edge/frame_{k:03d}/{c}/annotations.npz') as old_z,
                          np.load(folder / f'frame_{k:03d}/{c}/annotations.npz') as new_z):
                        if not np.array_equal(old_z['xy'], new_z['xy']) or not np.array_equal(old_z['in_view'], new_z['in_view']):
                            raise ValueError('inherited_back_GT_or_physical_changed')
            passed = passed and base['rigs'][rig_id]['target_dual_edge_frames'] >= 16
        windows[rig_id] = dict(goal=view,passed=passed,view_common_frames=sum(class_flags),
            longest_view_common_run=longest(class_flags),common_physical_frames=sum(common_flags),
            actual_two_camera_native_box_view_frames=sum(actual_dual_input),
            longest_actual_two_camera_native_box_view_run=longest(actual_dual_input),
            target_dual_edge_frames=base['rigs'][rig_id]['target_dual_edge_frames'],
            all_people_visible_some_rig=base['all_people_content_passed'],frames=rows)
    base['three_view_goals'] = windows
    base['all_three_view_goals_passed'] = all(x['passed'] for x in windows.values())
    base['back_GT_and_physical_exactly_match_old_edge'] = True
    base['frozen_36_dataset_slots_actual_reads'] = base['actual_Dataset_reads']
    (root / 'front_back_side_acceptance.json').write_text(json.dumps(base,indent=2,allow_nan=False)+'\n')
    if not base['all_three_view_goals_passed']:
        raise ValueError('actual_three_view_goal_not_met')
    return base


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(validate(args.root)['status'])
