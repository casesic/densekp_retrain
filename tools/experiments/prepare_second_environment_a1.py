"""Place a frozen boundary-pass episode into a native road environment.

The existing rider, action clock, relative rig and projected crops are kept.
Only one rigid road-to-world transform is introduced; Blender and Dataset
continue through the shared road_pass and ROI-only consumers.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.data.dynamic_scene import geometry_paths

VERSION = 'rigid_native_road_environment_v1'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def centerline(environment, y):
    if environment == 'cycling_forest':
        return 15.0 + 10.0 * math.sin(y / 32.0)
    if environment == 'riverfront':
        return 0.0
    raise ValueError('unknown_environment')


def rigid_placement(environment, anchor_y, old_rig_origin):
    yaw = math.atan(10.0 / 32.0 * math.cos(anchor_y / 32.0)) if environment == 'cycling_forest' else 0.0
    c, s = math.cos(yaw), math.sin(yaw)
    transform = np.eye(4)
    transform[:3, :3] = [[c, s, 0], [-s, c, 0], [0, 0, 1]]
    target = np.array([centerline(environment, anchor_y), anchor_y, 0.0])
    transform[:3, 3] = target - transform[:3, :3] @ np.asarray(old_rig_origin, float)
    return transform, -math.degrees(yaw)


def transform_points(points, matrix):
    return np.asarray(points) @ matrix[:3, :3].T + matrix[:3, 3]


def prepare(config_path, environment, output):
    config = json.loads(Path(config_path).read_text())
    if environment not in config['environments']:
        raise ValueError('unknown_environment')
    env = config['environments'][environment]
    if not (ROOT / env['scene_config']).is_file() or not (ROOT / env['blend_path']).is_file():
        raise ValueError('environment_dependency_missing')
    if env['road_half_width_m'] <= config['road_edge_margin_m'] or env['road_y_range_m'][0] >= env['road_y_range_m'][1]:
        raise ValueError('invalid_native_road_config')
    source = (ROOT / config['reference_prepared_episode']).resolve()
    old = json.loads((source / 'recipe.json').read_text())
    if old['frame_count'] != config['frame_count'] or len(old['people']) != 1:
        raise ValueError('reference_episode_mismatch')
    projection = json.loads((source / 'selected_projection.json').read_text())
    crossing = projection['boundary_crossing']
    if not all(v['first_outside'] >= 20 and v['last_outside'] <= 5 and
               v['first_distance_p10'] < 0 < v['last_distance_p10'] for v in crossing.values()):
        raise ValueError('reference_boundary_transition_not_proved')
    if not (projection['dual_edge']['passed'] and
            projection['dual_edge']['count'] >= config['edge_min_count'] and
            any(window['duration_seconds'] >= config['edge_min_continuous_seconds']
                for window in projection['dual_edge']['windows'])):
        raise ValueError('reference_dual_edge_not_passed')
    if config['transition'] != 'enter_valid':
        raise ValueError('unsupported_environment_reference_transition')
    regions = json.loads((source / 'planned_regions.json').read_text())
    if any((r['roi_xyxy'][2]-r['roi_xyxy'][0])*(r['roi_xyxy'][3]-r['roi_xyxy'][1]) > config['max_roi_pixels_per_camera']
           for frame in regions for r in frame.values()):
        raise ValueError('reference_ROI_budget_exceeded')
    old_rig = json.loads((source / 'rig.json').read_text())
    old_matrices = np.load(source / 'scene_matrices.npy')
    old_world = np.load(source / 'world_densekp.npy')
    asset = json.loads(Path(old['people'][0]['asset_manifest']).read_text())
    geometry = geometry_paths(asset)
    indices = old['people'][0]['asset_frames']
    body = np.load(geometry['body_vertices'], mmap_mode='r')
    cloth = np.load(geometry['cloth_vertices'], mmap_mode='r')
    candidate_rng = random.Random(f'{VERSION}:{config["master_seed"]}:{environment}:layout')
    attempts = []
    chosen = None
    for number in range(config['candidate_limit_per_environment']):
        child = candidate_rng.getrandbits(63)
        anchor_y = random.Random(child).uniform(*env['anchor_y_range_m'])
        matrix, yaw_deg = rigid_placement(environment, anchor_y, old['rig']['origin_m'])
        scene_matrices = matrix @ old_matrices
        min_margin = float('inf')
        reasons = []
        for frame, asset_frame in enumerate(indices):
            for part in (body[asset_frame], cloth[asset_frame]):
                world = transform_points(part, scene_matrices[frame, 0])
                road_x = np.array([centerline(environment, float(y)) for y in world[:, 1]])
                margin = np.min(env['road_half_width_m'] - np.abs(world[:, 0] - road_x))
                min_margin = min(min_margin, float(margin))
                if (world[:, 1].min() < env['road_y_range_m'][0] or
                        world[:, 1].max() > env['road_y_range_m'][1]):
                    reasons.append('native_road_length_exceeded')
        if min_margin < config['road_edge_margin_m']:
            reasons.append('native_road_width_exceeded')
        # A shared rigid transform of the full scene state and both cameras
        # preserves the previously measured physical-domain transition.
        new_rig = copy.deepcopy(old_rig)
        for camera in new_rig['cameras']:
            for key in ('camera_to_world', 'physical_lens_to_world'):
                camera[key] = (matrix @ np.asarray(camera[key])).tolist()
        new_world = transform_points(old_world, matrix)
        old_cam = {x['id']: x for x in old_rig['cameras']}
        max_projection_delta = 0.0
        for camera in new_rig['cameras']:
            cid = camera['id']
            for frame in range(config['frame_count']):
                before = project_erp(old_world[frame, 0], np.asarray(old_cam[cid]['camera_to_world']), 5760, 2880)[0]
                after = project_erp(new_world[frame, 0], np.asarray(camera['camera_to_world']), 5760, 2880)[0]
                delta = np.abs(before-after)
                delta[:, 0] = np.minimum(delta[:, 0], 5760-delta[:, 0])
                max_projection_delta = max(max_projection_delta, float(delta.max()))
        if max_projection_delta > 0.002:
            reasons.append('rigid_projection_changed')
        attempt = dict(attempt=number, child_seed=child, anchor_y_m=anchor_y,
                       road_to_world=matrix.tolist(), road_yaw_deg=yaw_deg,
                       minimum_road_edge_margin_m=min_margin,
                       max_full_erp_projection_delta_px=max_projection_delta,
                       rejections=sorted(set(reasons)))
        attempts.append(attempt)
        if not reasons:
            chosen = (attempt, matrix, scene_matrices, new_world, new_rig)
            break
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'candidate_checks.json', dict(schema=VERSION, environment_id=environment,
        master_seed=config['master_seed'], max_candidates=config['candidate_limit_per_environment'], attempts=attempts))
    if chosen is None:
        write(output / 'geometry_preflight.json', dict(status='rejected', reasons=attempts[-1]['rejections']))
        return None
    attempt, matrix, scene_matrices, new_world, new_rig = chosen
    for name in ('episode_timeline.json', 'person1_timeline.json', 'person1_source_timeline.json',
                 'expected_in_view.npy', 'planned_regions.json', 'selected_projection.json'):
        shutil.copy2(source / name, output / name)
    np.save(output / 'scene_matrices.npy', scene_matrices)
    np.save(output / 'world_densekp.npy', new_world)
    write(output / 'rig.json', new_rig)
    shutil.copytree(source / 'calibration', output / 'calibration')
    scene_config = json.loads((ROOT / env['scene_config']).read_text())
    if scene_config['blend_path'] != env['blend_path']:
        raise ValueError('environment_blend_path_conflict')
    blend_stat = (ROOT / env['blend_path']).stat()
    scene_config['expected_blend_stat'] = dict(size_bytes=blend_stat.st_size, mtime_ns=blend_stat.st_mtime_ns)
    scene_config['scene_settings_after_configure_render'] = env['scene_settings_after_configure_render']
    write(output / 'scene_config.json', scene_config)
    recipe = copy.deepcopy(old)
    recipe['episode_id'] = f'{old["episode_id"]}_{environment}_s{config["master_seed"]}'
    recipe['sequence_id'] = recipe['episode_id']
    recipe['environment_id'] = environment
    recipe['environment_version'] = VERSION
    recipe['environment_candidate_seed'] = attempt['child_seed']
    recipe['environment_road_to_world'] = matrix.tolist()
    recipe['environment_blend_dependency'] = dict(path=env['blend_path'], **scene_config['expected_blend_stat'])
    recipe['scene'] = str(output / 'scene_config.json')
    recipe['parent_recipe_ref'] = str(source / 'recipe.json')
    recipe['scene_matrices_ref'] = str(output / 'scene_matrices.npy')
    recipe['episode_timeline_ref'] = str(output / 'episode_timeline.json')
    recipe['person_timeline_refs'] = {'1': str(output / 'person1_timeline.json')}
    recipe['source_timeline_refs'] = {'1': str(output / 'person1_source_timeline.json')}
    recipe['expected_in_view_ref'] = str(output / 'expected_in_view.npy')
    recipe['rig']['origin_m'] = transform_points(np.asarray(old['rig']['origin_m']), matrix).tolist()
    recipe['rig']['yaw_deg'] += attempt['road_yaw_deg']
    recipe['rig']['road_yaw_deg'] += attempt['road_yaw_deg']
    person = recipe['people'][0]
    person['scene_from_model'] = scene_matrices[:, 0].tolist()
    person['road_start_m'] = scene_matrices[0, 0, :3, 3].tolist()
    person['path_direction_xy'] = (matrix[:2, :2] @ np.asarray(person['path_direction_xy'])).tolist()
    person['scene_yaw_deg'] += attempt['road_yaw_deg']
    person['support_checks']['road_object_names'] = env['road_object_names']
    person['support_checks']['road_object_prefixes'] = env['road_object_prefixes']
    recipe['road_bounds_xyxy'] = [float(new_world[..., 0].min()-1), env['road_y_range_m'][0],
                                   float(new_world[..., 0].max()+1), env['road_y_range_m'][1]]
    write(output / 'recipe.json', recipe)
    write(output / 'geometry_preflight.json', dict(status='passed', selected_attempt=attempt,
        reference_transition=crossing, reference_edge=projection['dual_edge'],
        identical_relative_projection_within_px=attempt['max_full_erp_projection_delta_px'],
        full_frame_native_road_analytic_clearance_m=attempt['minimum_road_edge_margin_m'],
        actual_native_road_ray_and_material_gate='required_in_Blender_probe',
        projection_invariance_reason='same rigid transform applied to all world geometry and both ERP/physical cameras'))
    return recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, default=ROOT/'configs/second_environment_boundary_pass_a1_20261010.json')
    parser.add_argument('--environment', choices=['cycling_forest', 'riverfront'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    recipe = prepare(args.config, args.environment, args.output)
    print('ENVIRONMENT_PREPARED' if recipe else 'ENVIRONMENT_REJECTED', args.output)


if __name__ == '__main__':
    main()
