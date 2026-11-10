"""Freeze camera-only candidates around an unchanged five-rider city episode."""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.data.viewpoint import classify_view, view_angle
from tools.experiments.prepare_urban_three_multirig_a1 import preflight as exact_preflight

CONFIG = ROOT / 'configs/front_back_side_multirig_a1_20261011.json'
OUT = ROOT / 'outputs/experiments/front_back_side_multirig_a1_20261011'
PARENT = ROOT / 'outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False, ensure_ascii=False)
        stream.write('\n')


def child_seed(master, view):
    return random.Random(f'front_back_side_camera_v1|{master}|{view}').getrandbits(63)


def candidates(config, world, view, master):
    """No projection here: freeze unique, road-derived rig poses first."""
    spec = config[view]
    target = world[15, 0].mean(axis=0)
    cells = [(x, y, offset) for x in spec['roadside_x_centers_m']
             for y in spec['along_road_y_centers_m']
             for offset in spec['seam_yaw_offsets_deg']]
    if len(cells) != config['candidate_limit_each_view'] or len(set(cells)) != len(cells):
        raise ValueError('candidate_cell_count_or_duplicate')
    rng = random.Random(child_seed(master, view))
    rng.shuffle(cells)
    result = []
    for attempt, (cx, cy, offset) in enumerate(cells):
        seed = rng.getrandbits(63)
        local = random.Random(seed)
        x = cx + local.uniform(-config['jitter_xy_m'], config['jitter_xy_m'])
        y = cy + local.uniform(-config['jitter_xy_m'], config['jitter_xy_m'])
        bearing = math.degrees(math.atan2(target[1] - y, target[0] - x))
        result.append(dict(attempt=attempt, camera_child_seed=seed,
            origin_m=[x, y, 0.0], yaw_deg=bearing + offset + local.uniform(-config['jitter_yaw_deg'], config['jitter_yaw_deg']),
            nominal_cell=[cx, cy, offset], construction='roadside_grid_to_frozen_target_midpoint_bearing'))
    return result


def headings(source):
    recipe = read(source / 'recipe.json')
    matrices = np.load(source / 'scene_matrices.npy')
    asset = read(recipe['people'][0]['asset_manifest'])
    with np.load(asset['parameters']) as params:
        orient = params['global_orient'][recipe['people'][0]['asset_frames']]
    return np.stack([matrices[k, 0, :3, :3] @ cv2.Rodrigues(orient[k])[0] @ [0., 0., 1.]
                     for k in range(31)])


def back_check(source, config):
    recipe = read(source / 'recipe.json')
    world = np.load(source / 'world_densekp.npy')
    matrices = np.load(source / 'scene_matrices.npy')
    direction = headings(source)
    rig = read(source / 'rig.json')
    poses = {c['id']: np.asarray(c['camera_to_world']) for c in rig['cameras']}
    masks = {c: np.asarray(Image.open(source / 'calibration' / f'{c}_valid_mask.png')) > 0 for c in poses}
    flags, edge, common, rows = [], [], [], []
    for k in range(31):
        inside = {c: point_in_view(world[k, 0], pose, c, masks[c])[-1] for c, pose in poses.items()}
        angles = {c: view_angle(matrices[k, 0, :3, 3], direction[k], pose[:3, 3]) for c, pose in poses.items()}
        shared = int((inside['cam2'] & inside['cam4']).sum())
        common.append(shared >= 8)
        flags.append(shared >= 8 and all(classify_view(angles[c]) == 'back' for c in poses))
        edge.append(shared >= 8 and all(inside[c].sum() >= 8 and (~inside[c]).sum() >= 5 for c in poses))
        rows.append(dict(frame_index=k, camera_angles_deg=angles, common_physical_points=shared,
                         in_view={c:int(inside[c].sum()) for c in poses}))
    def longest(values):
        best = run = 0
        for value in values:
            run = run + 1 if value else 0
            best = max(best, run)
        return best
    passed = (sum(flags) >= config['back']['minimum_view_common_frames'] and
              longest(flags) >= config['back']['minimum_view_common_run'] and
              sum(edge) >= config['back']['minimum_dual_edge_frames'] and longest(edge) >= 6)
    return dict(status='passed' if passed else 'rejected', source_prepared=str(source),
        source_scene_episode_id=recipe['episode_id'], back_common_frames=sum(flags),
        longest_back_common_run=longest(flags), dual_edge_frames=sum(edge),
        longest_dual_edge_run=longest(edge), common_physical_frames=sum(common), frames=rows)


def freeze():
    cfg = read(CONFIG)
    source = (ROOT / cfg['source_prepared']).resolve()
    recipe = read(source / 'recipe.json')
    world = np.load(source / 'world_densekp.npy')
    matrices = np.load(source / 'scene_matrices.npy')
    original = read(PARENT / 'production/recipe.json')
    if (recipe['episode_id'] != cfg['scene_episode_id'] or len(recipe['people']) != 5 or
        world.shape != (31, 5, 138, 3) or matrices.shape != (31, 5, 4, 4) or
        not np.array_equal(world, np.load(PARENT / 'production/world_densekp.npy')) or
        recipe['people'] != original['people'] or recipe['episode_timeline'] != original['episode_timeline']):
        raise ValueError('frozen_five_person_scene_changed')
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT / 'frozen_scene.json', dict(schema=cfg['schema'], source_prepared=str(source),
        parent_production=str(PARENT / 'production'), scene_episode_id=cfg['scene_episode_id'],
        people=recipe['people'], episode_timeline=recipe['episode_timeline'],
        world_ref=str(source / 'world_densekp.npy'), scene_matrices_ref=str(source / 'scene_matrices.npy'),
        inherited_camera_rig='old_rig_edge_to_rig_back', camera_master_seed=cfg['camera_master_seed'],
        camera_child_seeds={view:child_seed(cfg['camera_master_seed'], view) for view in ('front','back','side')},
        render_order=cfg['render_order'], source_grouping=cfg['source_grouping']))
    write(OUT / 'back_view_check.json', back_check(source, cfg))
    for view in ('front', 'side'):
        rows = candidates(cfg, world, view, cfg['camera_master_seed'])
        write(OUT / f'frozen_{view}_candidates.json', dict(schema=cfg['schema'],
            scene_episode_id=cfg['scene_episode_id'], source_prepared=str(source),
            camera_seed=child_seed(cfg['camera_master_seed'], view),
            view_goal=view, candidate_generation='stable_independent_child_stream_then_seed_shuffled_roadside_cells',
            candidates=rows))
    write(OUT / 'comparison_seed_plan.json', dict(master_seed=cfg['comparison_seed'], projection_or_render=False,
        people_plan_unchanged=True,
        camera_child_seeds={view:child_seed(cfg['comparison_seed'], view) for view in ('front','back','side')},
        different_first_front_candidate=candidates(cfg, world, 'front', cfg['comparison_seed'])[0] !=
                                        candidates(cfg, world, 'front', cfg['camera_master_seed'])[0]))


def preflight(view):
    cfg = read(CONFIG)
    frozen = OUT / f'frozen_{view}_candidates.json'
    src = Path(read(frozen)['source_prepared'])
    if read(frozen)['candidates'] != candidates(cfg, np.load(src / 'world_densekp.npy'), view, cfg['camera_master_seed']):
        raise ValueError('frozen_camera_candidates_changed')
    goal = cfg[view]
    settings = dict(schema=cfg['schema'], view_goal=view, rig_id=f'rig_{view}',
        coverage_goal=f'{view}_view_control', minimum_common_points=8,
        minimum_common_physical_frames=goal['minimum_common_physical_frames'],
        minimum_view_common_frames=goal.get('minimum_view_common_frames', 0),
        minimum_view_common_run=goal['minimum_view_common_run'],
        minimum_camera_to_person_root_m=cfg['minimum_camera_to_person_root_m'],
        minimum_camera_road_x_abs_m=cfg['minimum_camera_road_x_abs_m'],
        maximum_union_roi_pixels=cfg['maximum_union_roi_pixels'],
        maximum_exact_support_candidates=cfg['maximum_exact_support_candidates_each_view'])
    write(OUT / f'{view}_preflight_settings.json', settings)
    return exact_preflight(OUT / f'{view}_preflight_settings.json', frozen,
                           OUT / f'prepared_{view}', time_limit_seconds=cfg['cpu_preflight_budget_seconds'] / 2)


def prepare_manifest():
    cfg = read(CONFIG)
    source = (ROOT / cfg['source_prepared']).resolve()
    back = read(OUT / 'back_view_check.json')
    if back['status'] != 'passed':
        raise ValueError('inherited_edge_not_back')
    for view in ('front', 'side'):
        if read(OUT / f'prepared_{view}/geometry_preflight.json')['status'] != 'passed':
            raise ValueError(f'{view}_not_preflighted')
    original = read(PARENT / 'production/recipe.json')
    # Old prepared recipe lacked this optional evidence switch; apply it only
    # to owned copies before compact render, preserving both old sources.
    metadata = {key:copy.deepcopy(original[key]) for key in ('occlusion_metric','neighbor_evidence_policy')}
    shutil.copytree(source, OUT / 'prepared_back')
    for view in ('back','front','side'):
        path = OUT / f'prepared_{view}/recipe.json'
        value = read(path)
        value.update(metadata)
        value['scene_episode_id'] = cfg['scene_episode_id']
        value['rig_id'] = f'rig_{view}'
        if view == 'back':
            value['coverage_goal'] = 'back_view_and_edge_coverage'
            value['requested_view_class'] = 'back'
            value['inherited_rig_ref'] = str(source)
        path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    slots=[f'f15_{camera}_p{person}' for camera in ('cam2','cam4') for person in range(1,6)]
    slots += [f'f0_{camera}_p1' for camera in ('cam2','cam4')]
    assert len(slots) == 12
    rigs = [dict(rig_id=f'rig_{view}',prepared=str(OUT/f'prepared_{view}'),
                 coverage_goal={'back':'back_view_and_edge_coverage','front':'front_view_control',
                                'side':'side_view_control'}[view],
                 camera_seed=child_seed(cfg['camera_master_seed'],view),
                 readback_observation_ids=slots) for view in ('back','front','side')]
    write(OUT / 'rig_list_manifest.json', dict(schema='scene_episode_rig_list_v1',
          scene_episode_id=cfg['scene_episode_id'],source_grouping=cfg['source_grouping'],
          integration_only=True,maximum_budget_seconds=cfg['render_and_finish_budget_seconds'],rigs=rigs))
    write(OUT / 'readback_plan_frozen.json', dict(scene_episode_id=cfg['scene_episode_id'],
          planned_slots=36,rigs={r['rig_id']:slots for r in rigs},no_replacement_for_unavailable=True))


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze','preflight_front','preflight_side','manifest'))
    args=parser.parse_args()
    if args.action=='freeze': freeze()
    elif args.action=='manifest': prepare_manifest()
    else: preflight(args.action.removeprefix('preflight_'))
