"""Validate the replacement front paired rig and quantify actual cam4 scale."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.data.viewpoint import classify_view, view_angle
from tools.experiments.validate_dispersed_cyclists_a1 import validate as validate_slots


def read(path):
    return json.loads(Path(path).read_text())


def longest(flags):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def native_target_height(folder, camera, frame):
    box = read(folder / f'frame_{frame:03d}' / camera / 'boxes.json')[0]['bbox_xyxy_exclusive']
    return None if box is None else int(box[3] - box[1])


def validate(root):
    root = Path(root).resolve()
    task = root.parent
    frozen = read(task / 'frozen_execution.json')
    old = (ROOT / frozen['old_front_production']).resolve()
    source = (ROOT / frozen['source_prepared']).resolve()
    rig_id = 'rig_front_revised'
    new = root / 'rigs' / rig_id
    if read(task / 'static_camera_clearance.json')['passed'] is not True:
        raise ValueError('new_camera_static_clearance_not_passed')
    original_recipe, new_recipe = read(old / 'recipe.json'), read(new / 'recipe.json')
    if (new_recipe['people'] != original_recipe['people'] or
            new_recipe['episode_timeline'] != original_recipe['episode_timeline'] or
            not np.array_equal(np.load(new / 'world_densekp.npy'), np.load(old / 'world_densekp.npy')) or
            not np.array_equal(np.load(source / 'scene_matrices.npy'), np.load(new / 'scene_matrices.npy'))):
        raise ValueError('five_person_world_or_time_changed')
    base = validate_slots(root, expected_rigs=[rig_id])
    if (base['planned_camera_calls'] != 62 or base['planned_observation_slots'] != 310 or
            base['rig_pair_relative_transform_max_abs'] != 0.0):
        raise ValueError('single_new_rig_count_or_relative_conflict')
    cameras = {c['id']:np.asarray(c['camera_to_world']) for c in read(new / 'rig.json')['cameras']}
    old_cameras = {c['id']:np.asarray(c['camera_to_world']) for c in read(old / 'rig.json')['cameras']}
    new_relative = np.linalg.inv(cameras['cam2']) @ cameras['cam4']
    old_relative = np.linalg.inv(old_cameras['cam2']) @ old_cameras['cam4']
    relative_error = float(np.max(np.abs(new_relative - old_relative)))
    if relative_error > 1e-9:
        raise ValueError('paired_camera_calibration_changed')
    matrices = np.load(source / 'scene_matrices.npy')
    asset = read(new_recipe['people'][0]['asset_manifest'])
    with np.load(asset['parameters']) as params:
        orient = params['global_orient'][new_recipe['people'][0]['asset_frames']]
    flags, dual_input = [], []
    rows = []
    packed = read(new / 'pack/sequence_index.json')
    boxes = {b['observation_id'] for b in packed['boxes']}
    for k in range(31):
        heading = matrices[k,0,:3,:3] @ cv2.Rodrigues(orient[k])[0] @ [0.,0.,1.]
        angles = {c:view_angle(matrices[k,0,:3,3],heading,pose[:3,3]) for c,pose in cameras.items()}
        classes = {c:classify_view(angle) for c,angle in angles.items()}
        inside = {}
        for c in ('cam2','cam4'):
            with np.load(new / f'frame_{k:03d}' / c / 'annotations.npz') as z:
                inside[c] = z['in_view'][0].copy()
        common = int((inside['cam2'] & inside['cam4']).sum())
        front = common >= 8 and all(classes[c] == 'front' for c in ('cam2','cam4'))
        paired = all(f'f{k}_{c}_p1' in boxes for c in ('cam2','cam4'))
        flags.append(front)
        dual_input.append(front and paired)
        rows.append(dict(frame_index=k, angles_deg=angles, classes=classes,
                         common_physical_points=common, front_common=front, both_native_boxes=paired))
    old_height = [native_target_height(old,'cam4',k) for k in range(31)]
    new_height = [native_target_height(new,'cam4',k) for k in range(31)]
    old_median = float(np.median([x for x in old_height if x is not None]))
    new_median = float(np.median([x for x in new_height if x is not None]))
    size_ratio = new_median / old_median
    if (sum(flags) < 16 or longest(flags) < 6 or sum(dual_input) < 16 or
            size_ratio < frozen['minimum_actual_native_height_ratio']):
        raise ValueError('front_view_or_actual_cam4_size_goal_not_met')
    video = new / 'pack/dual_roi.mp4'
    probe = json.loads(subprocess.run(['ffprobe','-v','error','-count_frames','-select_streams','v:0',
        '-show_entries','stream=nb_read_frames,r_frame_rate,duration','-of','json',str(video)],
        capture_output=True,text=True,check=True).stdout)['streams'][0]
    if probe['nb_read_frames'] != '31' or probe['r_frame_rate'] != '10/1':
        raise ValueError('video_frame_count_or_rate_changed')
    result = dict(schema='front_rig_size_repair_acceptance_a1_v1',status='technical_passed',
        scene_episode_id=frozen['scene_episode_id'],rig_id=rig_id,
        person_world_and_time_exactly_unchanged=True,
        paired_relative_calibration_max_abs=relative_error,
        original_front_common_frames=read(task.parent / 'front_back_side_multirig_a1_20261011/production/front_back_side_acceptance.json')['three_view_goals']['rig_front']['view_common_frames'],
        revised_front_common_frames=sum(flags),revised_longest_front_run=longest(flags),
        revised_actual_dual_native_box_front_frames=sum(dual_input),
        original_cam4_p1_native_bbox_height_median_px=old_median,
        revised_cam4_p1_native_bbox_height_median_px=new_median,
        cam4_p1_native_height_ratio=size_ratio,
        frame15_cam4_p1_old_new_height_px=[old_height[15],new_height[15]],
        old_cam4_p1_heights_px=old_height,new_cam4_p1_heights_px=new_height,
        planned_camera_calls=62,actual_camera_calls=base['actual_completed_camera_calls'],
        planned_observation_slots=310,packed_rows=base['packed_rows'],
        actual_Dataset_reads=base['actual_Dataset_reads'],
        all_five_people_legal_visible_content=base['all_people_content_passed'],
        neighbor_evidence_available=base['rigs'][rig_id]['target_neighbor_evidence_available'],
        video_frames=31,video_rate='10Hz',video_duration_seconds=float(probe['duration']),
        frame_view_records=rows,base_acceptance_ref=str(root / 'acceptance.json'))
    (root / 'front_rig_size_acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    print(validate(parser.parse_args().root)['status'])
