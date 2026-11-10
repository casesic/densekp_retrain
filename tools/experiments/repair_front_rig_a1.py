"""Plan a closer front paired rig for the frozen city five-rider episode.

Only the camera changes. Exact support and production use the existing
five-person planner and multirig renderer, respectively.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.data.viewpoint import classify_view, view_angle
from tools.experiments.prepare_urban_three_multirig_a1 import preflight as exact_preflight

CONFIG = ROOT / 'configs/front_rig_size_repair_a1_20261011.json'
OUT = ROOT / 'outputs/experiments/front_rig_size_repair_a1_20261011'


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write('\n')


def longest(flags):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def candidates(config, world):
    cells = [(sign * x, y) for sign in config['roadside_signs']
             for x in config['roadside_x_abs_m'] for y in config['along_road_y_m']]
    if len(cells) != config['candidate_limit'] or len(set(cells)) != len(cells):
        raise ValueError('invalid_frozen_cells')
    rng = random.Random(f"front_rig_size_repair_a1_v1|{config['camera_master_seed']}")
    rng.shuffle(cells)
    target = world[15, 0].mean(axis=0)
    result = []
    for attempt, (cx, cy) in enumerate(cells):
        seed = rng.getrandbits(63)
        local = random.Random(seed)
        x = cx + local.uniform(-config['jitter_xy_m'], config['jitter_xy_m'])
        y = cy + local.uniform(-config['jitter_xy_m'], config['jitter_xy_m'])
        yaw = math.degrees(math.atan2(target[1] - y, target[0] - x))
        yaw += local.uniform(-config['jitter_yaw_deg'], config['jitter_yaw_deg'])
        result.append(dict(attempt=attempt, camera_child_seed=seed,
                           nominal_cell=[cx, cy], origin_m=[x, y, 0.0], yaw_deg=yaw,
                           construction='roadside_grid_to_frozen_p1_midpoint'))
    return result


def source_inputs(config):
    source = (ROOT / config['source_prepared']).resolve()
    recipe = read(source / 'recipe.json')
    world = np.load(source / 'world_densekp.npy')
    matrices = np.load(source / 'scene_matrices.npy')
    if (recipe['episode_id'] != config['scene_episode_id'] or
            world.shape != (31, 5, 138, 3) or matrices.shape != (31, 5, 4, 4)):
        raise ValueError('frozen_scene_identity_or_person_axis_changed')
    return source, recipe, world, matrices


def headings(recipe, matrices):
    asset = read(recipe['people'][0]['asset_manifest'])
    with np.load(asset['parameters']) as params:
        orient = params['global_orient'][recipe['people'][0]['asset_frames']]
    return np.stack([matrices[k, 0, :3, :3] @ cv2.Rodrigues(orient[k])[0] @ [0., 0., 1.]
                     for k in range(31)])


def proxy_height(world, pose, mask, camera):
    xy, _, _, _, inside = point_in_view(world, pose, camera, mask)
    if inside.sum() < 8:
        return None, int(inside.sum())
    return float(np.ptp(xy[inside, 1])), int(inside.sum())


def freeze():
    cfg = read(CONFIG)
    source, recipe, world, matrices = source_inputs(cfg)
    old = (ROOT / cfg['old_front_production']).resolve()
    old_recipe = read(old / 'recipe.json')
    if (recipe['people'] != old_recipe['people'] or
            recipe['episode_timeline'] != old_recipe['episode_timeline'] or
            not np.array_equal(world, np.load(old / 'world_densekp.npy')) or
            not np.array_equal(matrices, np.load(old / 'scene_matrices.npy'))):
        raise ValueError('old_front_scene_differs_from_frozen_source')
    OUT.mkdir(parents=True, exist_ok=False)
    write_new(OUT / 'frozen_inputs.json', dict(schema=cfg['schema'], source_prepared=str(source),
        old_front_production=str(old), scene_episode_id=cfg['scene_episode_id'],
        scene_matrices_ref=str(source / 'scene_matrices.npy'), world_ref=str(source / 'world_densekp.npy'),
        people=recipe['people'], episode_timeline=recipe['episode_timeline'],
        camera_master_seed=cfg['camera_master_seed'], source_grouping=cfg['source_grouping'],
        only_camera_geometry_changes=True))
    write_new(OUT / 'frozen_candidates.json', dict(schema='front_back_side_camera_rig_v1',
        scene_episode_id=cfg['scene_episode_id'], source_prepared=str(source),
        camera_seed=cfg['camera_master_seed'], candidates=candidates(cfg, world),
        candidate_generation='seed_shuffled_roadside_grid_with_independent_child_jitter',
        quality_rule='cam4_p1_visible_R138_median_height_ratio>=1.35_old; front_common>=16; run>=6'))


def screen():
    cfg = read(CONFIG)
    frozen = read(OUT / 'frozen_candidates.json')
    source, recipe, world, matrices = source_inputs(cfg)
    if frozen['candidates'] != candidates(cfg, world):
        raise ValueError('candidate_freeze_changed')
    old = (ROOT / cfg['old_front_production']).resolve()
    old_rig = read(old / 'rig.json')
    old_pose = {c['id']: np.asarray(c['camera_to_world']) for c in old_rig['cameras']}
    masks = {c: np.asarray(Image.open(source / 'calibration' / f'{c}_valid_mask.png')) > 0
             for c in ('cam2', 'cam4')}
    old_proxy = [proxy_height(world[k, 0], old_pose['cam4'], masks['cam4'], 'cam4')[0]
                 for k in range(31)]
    if any(x is None for x in old_proxy):
        raise ValueError('old_front_cam4_target_has_insufficient_R138')
    old_median = float(np.median(old_proxy))
    forward = headings(recipe, matrices)
    results = []
    for candidate in frozen['candidates']:
        poses, _, _ = rig_poses(candidate['origin_m'], candidate['yaw_deg'], recipe['rig']['baseline_m'])
        flags, common_count, heights = [], [], []
        for k in range(31):
            inside = {}
            classes = {}
            for cam in ('cam2', 'cam4'):
                height, count = proxy_height(world[k, 0], poses[cam], masks[cam], cam)
                if cam == 'cam4':
                    heights.append(height)
                inside[cam] = point_in_view(world[k, 0], poses[cam], cam, masks[cam])[-1]
                angle = view_angle(matrices[k, 0, :3, 3], forward[k], poses[cam][:3, 3])
                classes[cam] = classify_view(angle)
            common = int((inside['cam2'] & inside['cam4']).sum())
            common_count.append(common)
            flags.append(common >= cfg['minimum_common_physical_points'] and
                         classes['cam2'] == classes['cam4'] == 'front')
        finite_heights = [x for x in heights if x is not None]
        median_height = float(np.median(finite_heights)) if finite_heights else None
        ratio = median_height / old_median if median_height is not None else None
        meets = (sum(flags) >= cfg['minimum_front_common_frames'] and
                 longest(flags) >= cfg['minimum_front_common_run'] and
                 ratio is not None and ratio >= cfg['minimum_cam4_target_proxy_height_ratio_to_old'])
        results.append(dict(**candidate, front_common_frames=sum(flags),
            longest_front_common_run=longest(flags), common_physical_frames=sum(x >= 8 for x in common_count),
            cam4_target_proxy_height_median_px=median_height,
            cam4_target_proxy_height_ratio_to_old=ratio, screen_passed=meets))
    ranked = sorted((x for x in results if x['screen_passed']),
                    key=lambda x: (-x['cam4_target_proxy_height_ratio_to_old'], x['attempt']))
    write_new(OUT / 'screen_results.json', dict(schema=cfg['schema'], old_cam4_p1_R138_proxy_median_height_px=old_median,
        candidate_count=len(results), screen_passed_count=len(ranked), candidates=results,
        ranked_attempts=[x['attempt'] for x in ranked]))
    if not ranked:
        return
    selected = ranked[:cfg['maximum_exact_candidates']]
    write_new(OUT / 'exact_candidates.json', dict(schema='front_back_side_camera_rig_v1',
        scene_episode_id=cfg['scene_episode_id'], source_prepared=str(source),
        camera_seed=cfg['camera_master_seed'], candidates=selected,
        selected_from_all_frozen_attempts=[x['attempt'] for x in selected]))
    write_new(OUT / 'exact_settings.json', dict(schema='front_back_side_camera_rig_v1',
        view_goal='front', rig_id='rig_front_revised', coverage_goal='front_view_control',
        minimum_common_points=8, minimum_common_physical_frames=16,
        minimum_view_common_frames=16, minimum_view_common_run=6,
        minimum_camera_to_person_root_m=cfg['minimum_camera_to_person_root_m'],
        minimum_camera_road_x_abs_m=cfg['minimum_camera_road_x_abs_m'],
        maximum_union_roi_pixels=cfg['maximum_union_roi_pixels'],
        maximum_exact_support_candidates=cfg['maximum_exact_candidates']))


def prepare():
    if not (OUT / 'exact_candidates.json').exists():
        raise ValueError('no_larger_front_candidate_passed_screen')
    result = exact_preflight(OUT / 'exact_settings.json', OUT / 'exact_candidates.json',
                             OUT / 'prepared_front_revised', time_limit_seconds=600)
    if result is None:
        raise ValueError('no_larger_front_candidate_passed_full_support')
    cfg = read(CONFIG)
    old_recipe = read(ROOT / cfg['old_front_production'] / 'recipe.json')
    path = OUT / 'prepared_front_revised/recipe.json'
    recipe = read(path)
    for key in ('occlusion_metric', 'neighbor_evidence_policy'):
        recipe[key] = old_recipe[key]
    path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    slots = [f'f15_{cam}_p{person}' for cam in ('cam2','cam4') for person in range(1,6)]
    slots += [f'f0_{cam}_p1' for cam in ('cam2','cam4')]
    write_new(OUT / 'rig_list_manifest.json', dict(schema='scene_episode_rig_list_v1',
        scene_episode_id=cfg['scene_episode_id'], source_grouping=cfg['source_grouping'],
        integration_only=True, maximum_budget_seconds=cfg['render_budget_seconds'],
        rigs=[dict(rig_id='rig_front_revised', prepared=str(OUT / 'prepared_front_revised'),
                   coverage_goal='front_view_control', camera_seed=cfg['camera_master_seed'],
                   readback_observation_ids=slots)]))
    write_new(OUT / 'readback_plan_frozen.json', dict(planned_slots=12, rig_id='rig_front_revised',
        selected_observation_ids=slots, rigs={'rig_front_revised':slots},
        no_replacement_for_unavailable=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','screen','prepare'))
    action = parser.parse_args().action
    {'freeze':freeze,'screen':screen,'prepare':prepare}[action]()
