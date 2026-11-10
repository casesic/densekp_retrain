"""Freeze first-trial road inputs; use the existing mixed road planner for CPU checks.

This entry never builds an asset or starts Blender. Each invocation prepares one
of the four source-frozen episodes and records all twelve layout/camera trials.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from densekp_retrain.data.dynamic_scene import adapt_proposal
from densekp_retrain.data.mixed_road_pass import plan_mixed, preflight_mixed

PLAN = ROOT / 'outputs/plans/first_dataset_preparation_a1_20261011/episode_plan.json'
MIXED = ROOT / 'outputs/experiments/mixed_walk_cycling_edge_a1_20261009/config.json'
OLD_CYCLE = ROOT / 'outputs/experiments/cycling_roi_integration_a1_20261009/run/recipe.json'
OLD_WALK = ROOT / 'outputs/experiments/pedestrian_seeded_pilot_a1_20261009/layout_attempt_2/road_pass_00/recipe.json'
RIVER = ROOT / 'outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront/recipe.json'
FRONT = ROOT / 'outputs/experiments/front_edge_coverage_a1_20261010/preflight/single_cyclist_boundary_pass_front_edge_coverage_s20261012_e00/recipe.json'
PROPOSALS = {
    'p2_M01': ROOT / 'outputs/experiments/bvh_p2_dressed_asset_a2_20261010/a1_proposal.json',
    'p1_alternate_M02': ROOT / 'outputs/experiments/bvh_p1_dressed_assets_a2_20261010/p1_alternate/a1_proposal.json',
}
WALK_VIEWS = {
    'KIT425_M04': ROOT / 'outputs/experiments/walking_composition_a2_20261010/builds/M04_KIT425_walk_replay_v2/a1_walking_view.json',
    'KIT11_M03_simple': ROOT / 'outputs/experiments/walking_composition_a2_20261010/builds/M03_KIT11_walk/a1_walking_view.json',
}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def heading_rotation(adapter, indices):
    with np.load(adapter['parameters']) as par:
        orient = par['global_orient'][indices]
    c = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
    forward = np.array([c @ cv2.Rodrigues(o)[0] @ [0., 0., 1.] for o in orient])
    theta = math.pi / 2 - math.atan2(forward[:, 1].mean(), forward[:, 0].mean())
    yaw = np.array([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])
    rotated = forward @ yaw.T
    if float((rotated[:, 1] / np.linalg.norm(rotated[:, :2], axis=1)).min()) < math.cos(math.radians(25)):
        raise ValueError('cycling_forward_outside_25_degree_road_gate')
    return yaw @ c, math.degrees(theta)


def freeze(index, out):
    plan = json.loads(PLAN.read_text())['episodes'][index]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'config.json').exists():
        raise FileExistsError('frozen_episode_config_exists')
    config = json.loads(MIXED.read_text())
    old_rider = json.loads(OLD_CYCLE.read_text())
    rider = plan['persons'][0]
    adapter = adapt_proposal(PROPOSALS[rider['combo_id']])
    dump(out / 'rider_adapter.json', adapter)
    reference = copy.deepcopy(old_rider)
    person = reference['people'][0]
    person.update(asset_id=adapter['asset_id'], asset_manifest=str(out / 'rider_adapter.json'),
                  character_family=adapter['character_family'], root_mode=adapter['root_mode'],
                  support_policy=adapter['support_policy'])
    person.pop('scene_from_model', None)
    dump(out / 'rider_reference.json', reference)
    indices = [r['asset_frame'] for r in rider['exact_time_mapping']]
    rotation, yaw = heading_rotation(adapter, indices)
    body = np.load(adapter['body_vertices'], mmap_mode='r')[indices]
    height = round(-float(body[:, :, 1].min()) + .10, 3)
    config.update(episode_id=plan['scene_episode_id'], master_seed=plan['master_seed'] + index,
                  planning_seeds=[plan['master_seed'] + index], attempt_limit=12,
                  observation_policy='record_unavailable_v1', roi_planning_policy='per_person_square_union_v1',
                  scene_mode=plan['mode'], coverage_goal='edge_coverage',
                  min_dual_edge_times=7 if len(plan['persons']) == 1 else 16,
                  require_dual_edge=len(plan['persons']) == 1,
                  camera_moves_without_scene_translation=True,
                  placement_height_m=height, fixed_scene_rotation=rotation.tolist(), fixed_scene_yaw_deg=yaw,
                  source_origin_records=[p['proposed_start_asset_frame'] for p in plan['persons']],
                  reference_recipes=[str(out / 'rider_reference.json')],
                  box_augmentation_policy=plan['box_policy'],
                  max_union_roi_pixels=7_000_000,
                  separation_policy='aabb_then_vertex_3d_v1')
    config['support_checks']['pelvis_height_above_road_m'] = [height - .06, height + .06]
    if len(plan['persons']) == 2:
        walker = plan['persons'][1]
        walk_asset = json.loads(WALK_VIEWS[walker['combo_id']].read_text())
        old_walker = json.loads(OLD_WALK.read_text())
        wp = old_walker['people'][0]
        wp.update(asset_id=walk_asset['asset_id'], asset_manifest=str(WALK_VIEWS[walker['combo_id']]),
                  character_family=walk_asset['character_family'], root_mode=walk_asset['root_mode'])
        wp.pop('scene_from_model', None)
        dump(out / 'walker_reference.json', old_walker)
        config['reference_recipes'].append(str(out / 'walker_reference.json'))
        config['walker_fixed_rotation_deg'] = -90
    base = json.loads((RIVER if plan['environment'] == 'riverfront' else FRONT).read_text())
    config['scene'] = base['scene']
    config['rig'] = copy.deepcopy(base['rig'])
    if len(plan['persons']) == 1:
        config['rig']['origin_m'] = copy.deepcopy(base['rig']['origin_m'])
    config['road_bounds_xyxy'] = [-2.25, -15, 2.25, 15] if plan['environment'] == 'riverfront' else [-3.1, -15, 3.1, 15]
    config['support_checks']['road_object_names'] = base['people'][0]['support_checks']['road_object_names']
    config['support_checks']['road_object_prefixes'] = base['people'][0]['support_checks']['road_object_prefixes']
    config['render']['gpu_uuid'] = None  # Select and record a free device immediately before production.
    initial = plan['rigs'][0]
    reference_start = base['people'][0]['road_start_m']
    reference_speed = base['people'][0]['speed_m_s']
    rig_start = base['rig']['origin_m']
    original_camera = dict(origin_x_m=rig_start[0] + initial['rig_translation_x_m'],
                           origin_y_m=rig_start[1] + initial['rig_translation_y_m'],
                           yaw_deg=base['rig']['yaw_deg'] + initial['rig_yaw_offset_deg'])
    nominal_camera = dict(origin_x_m=rig_start[0], origin_y_m=rig_start[1], yaw_deg=base['rig']['yaw_deg'])
    rng = random.Random(f'first_four_layout_v1:{plan["master_seed"]}:{index}')
    parameters = []
    cameras = []
    for j in range(12):
        delta = 0 if j < 2 else min(.12, (j - 1) * .018)
        parameters.append(dict(target_x_m=float(reference_start[0] + (0 if j < 2 else rng.uniform(-delta, delta))),
                               target_y_m=float(reference_start[1] + (0 if j < 2 else rng.uniform(-delta, delta))),
                               cycling_speed_m_s=float(reference_speed + (0 if j < 2 else rng.uniform(-.04, .04)))))
        cameras.append(copy.deepcopy(original_camera if j == 0 else nominal_camera if j == 1 else
                                     dict(origin_x_m=rig_start[0] + rng.uniform(-.18, .18),
                                          origin_y_m=rig_start[1] + rng.uniform(-.15, .15),
                                          yaw_deg=base['rig']['yaw_deg'] + rng.uniform(-8, 8))))
        if len(plan['persons']) == 2:
            parameters[-1].update(walker_x_m=float(-1.7 if j < 2 else rng.uniform(-1.9, -1.4)),
                                  walker_y_m=float(1.1 if j < 2 else rng.uniform(.7, 1.5)))
    config['parameter_ranges'] = {key: [min(p[key] for p in parameters), max(p[key] for p in parameters)] for key in parameters[0]}
    config['camera_parameter_ranges'] = {key: [min(p[key] for p in cameras), max(p[key] for p in cameras)] for key in cameras[0]}
    config['explicit_candidate_parameters'] = parameters
    config['explicit_camera_parameters'] = cameras
    config['camera_seed'] = plan['rigs'][0]['camera_seed']
    config['travel_direction_options'] = [1]
    # The proposal's angle labels are reported, not silently turned into a
    # stricter gate than this batch requested. Camera poses are still sampled.
    dump(out / 'config.json', config)
    dump(out / 'execution_inventory.json', dict(schema='first_four_execution_inventory_v1', plan_ref=str(PLAN),
         episode_index=index, split_pool=plan['split_pool'], first_scene_consumption=True,
         combinations=[p['combo_id'] for p in plan['persons']], exact_source_starts=config['source_origin_records'],
         asset_refs=[str(PROPOSALS[rider['combo_id']])] + ([str(WALK_VIEWS[plan['persons'][1]['combo_id']])] if len(plan['persons']) == 2 else []),
         camera_seeds=[r['camera_seed'] for r in plan['rigs']], user_review_status='pending_new_road_video',
         training_status='not_started', proposed_layout_candidates=12, source_status='development_candidate_only'))
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode-index', type=int, choices=range(4), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    config = freeze(args.episode_index, args.output)
    if args.preflight:
        result = preflight_mixed(plan_mixed(config), args.output)
        print(json.dumps(dict(prepared=str(args.output), geometry_status='passed' if result else 'rejected')))
    else:
        print(json.dumps(dict(config=str(args.output / 'config.json'), geometry_status='pending')))


if __name__ == '__main__':
    main()
