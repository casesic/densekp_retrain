"""Freeze a rider trajectory and seed only the *whole* paired rig pose.

`plan` does no projection or rendering. `preflight` evaluates only its frozen
32-or-fewer candidate table, and writes a standard prepared ROI episode.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.data.dynamic_scene import geometry_paths
from densekp_retrain.data.mixed_road_pass import edge_summary, crossing_transition
from densekp_retrain.data.viewpoint import classify_view, side_pass_edge_coverage, side_view_control, view_angle
from densekp_retrain.imaging.roi_planning import plan_native_crop_region

VERSION = 'fixed_person_seeded_rig_side_pass_v1'


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _relative(a, b):
    return np.linalg.inv(a) @ b


def frozen_candidates(config, old_recipe, old_world, *, seed):
    """Seed changes order and small within-cell rig offsets; rider is untouched."""
    root = np.asarray(old_recipe['rig']['origin_m'], float)
    middle = np.asarray(old_world[15, 0], float).mean(0)
    old_bearing = math.degrees(math.atan2(middle[1] - root[1], middle[0] - root[0]))
    seam_offset = float(old_recipe['rig']['yaw_deg']) - old_bearing
    cells = [(x, y, d) for x in config['camera_origin_x_m']
             for y in config['camera_origin_y_m'] for d in config['seam_yaw_offset_deg']]
    if len(cells) > config['candidate_limit'] or len(cells) != len(set(cells)):
        raise ValueError('invalid_candidate_cells')
    rng = random.Random(f'{VERSION}|{seed}|camera_only')
    rng.shuffle(cells)
    result = []
    for i, (x, y, offset) in enumerate(cells):
        child = rng.getrandbits(63)
        local = random.Random(child)
        # The bounded perturbation remains inside the frozen cell's small range.
        x = float(x + local.uniform(-.035, .035))
        y = float(y + local.uniform(-.035, .035))
        offset = float(offset + local.uniform(-.2, .2))
        bearing = math.degrees(math.atan2(middle[1] - y, middle[0] - x))
        result.append(dict(attempt=i, child_seed=child, origin_m=[x, y, root[2]],
                           yaw_deg=bearing + seam_offset + offset,
                           nominal_cell=dict(x_m=x, y_m=y, seam_yaw_offset_deg=offset),
                           construction='bearing_to_frozen_midpoint_plus_reference_seam_offset'))
    return result


def plan(config_path, output, *, seed=None):
    config = json.loads(Path(config_path).read_text())
    if config['schema'] != VERSION or config['coverage_goal'] != 'side_pass_edge_coverage':
        raise ValueError('invalid_side_pass_config')
    source = ROOT / config['reference_prepared_episode']
    recipe = json.loads((source / 'recipe.json').read_text())
    if recipe['frame_count'] != 31 or len(recipe['people']) != 1 or recipe.get('environment_id') != 'riverfront':
        raise ValueError('fixed_reference_mismatch')
    world = np.load(source / 'world_densekp.npy')
    if world.shape != (31, 1, 138, 3) or not np.isfinite(world).all():
        raise ValueError('invalid_frozen_world_points')
    actual_seed = config['master_seed'] if seed is None else seed
    result = dict(schema=VERSION, coverage_goal=config['coverage_goal'], master_seed=actual_seed,
                  camera_seed_rule=f'{VERSION}|{actual_seed}|camera_only',
                  scene_episode_id=recipe['episode_id'],
                  person_plan_ref=str(source / 'recipe.json'), person_world_ref=str(source / 'world_densekp.npy'),
                  person_matrices_ref=str(source / 'scene_matrices.npy'),
                  person_frozen_fields=['asset_id', 'asset_frames', 'source_frames', 'scene_from_model',
                                        'road_start_m', 'scene_yaw_deg', 'placement_height_m'],
                  construction=dict(road_side=config['camera_side'], road_half_width_m=config['road_half_width_m'],
                                    x_centers_m=config['camera_origin_x_m'], y_centers_m=config['camera_origin_y_m'],
                                    yaw_offsets_deg=config['seam_yaw_offset_deg'], jitter_xy_m=.035, jitter_yaw_deg=.2),
                  candidates=frozen_candidates(config, recipe, world, seed=actual_seed),
                  projected_candidate_count=0, production_frame_count=31)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        raise FileExistsError(out)
    write(out, result)
    return result


def _world_part(local, matrix):
    return np.asarray(local) @ matrix[:3, :3].T + matrix[:3, 3]


def _active_groom(asset):
    groom = Path(asset['groom_reference'])
    points = np.load(groom / 'points.npy')
    active = np.repeat(np.load(groom / 'active_curve_mask.npy'),
                       np.diff(np.load(groom / 'curve_offsets.npy')))
    return points[active]


def preflight(config_path, frozen_path, output):
    import cv2
    from PIL import Image
    from densekp_retrain.data.boxes import crop_from_box
    begin = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    frozen = json.loads(Path(frozen_path).read_text())
    if frozen['schema'] != config['schema'] or len(frozen['candidates']) > 32:
        raise ValueError('frozen_candidate_version_or_count')
    goal=config.get('coverage_goal','side_pass_edge_coverage')
    if goal not in ('side_pass_edge_coverage','side_view_control'):
        raise ValueError('unknown_side_coverage_goal')
    source = Path(frozen['person_plan_ref']).parent
    old = json.loads((source / 'recipe.json').read_text())
    world = np.load(source / 'world_densekp.npy')[:, 0]
    mats = np.load(source / 'scene_matrices.npy')[:, 0]
    if not np.array_equal(world, np.load(frozen['person_world_ref'])[:, 0]):
        raise ValueError('person_world_changed_since_freeze')
    person = old['people'][0]
    asset = json.loads(Path(person['asset_manifest']).read_text())
    geometry = geometry_paths(asset)
    indices = person['asset_frames']
    body = np.load(geometry['body_vertices'], mmap_mode='r')
    cloth = np.load(geometry['cloth_vertices'], mmap_mode='r')
    head = np.load(geometry['head_transforms'], mmap_mode='r')
    hair = _active_groom(asset)
    with np.load(asset['parameters']) as pars:
        global_orient = pars['global_orient'][indices]
    headings = np.array([mats[k, :3, :3] @ cv2.Rodrigues(global_orient[k])[0] @ [0., 0., 1.]
                         for k in range(31)])
    masks = {c: np.asarray(Image.open(source / 'calibration' / f'{c}_valid_mask.png')) > 0
             for c in ('cam2', 'cam4')}
    old_rig = json.loads((source / 'rig.json').read_text())
    old_cam = {c['id']: np.asarray(c['camera_to_world']) for c in old_rig['cameras']}
    old_lens = {c['id']: np.asarray(c['physical_lens_to_world']) for c in old_rig['cameras']}
    signed = {c: cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) -
             cv2.distanceTransform((~m).astype(np.uint8), cv2.DIST_L2, 5) for c, m in masks.items()}
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    attempts = []
    selected = None
    for candidate in frozen['candidates']:
        attempt = dict(candidate, rejections=[], frames=[])
        poses, lenses, _ = rig_poses(candidate['origin_m'], candidate['yaw_deg'], old['rig']['baseline_m'])
        try:
            if candidate['origin_m'][0] - config['road_half_width_m'] < config['camera_road_edge_clearance_min_m']:
                raise ValueError('rig_inside_road_clearance')
            if (not np.allclose(_relative(poses['cam2'], poses['cam4']),
                                _relative(old_cam['cam2'], old_cam['cam4']), atol=1e-9)
                or not np.allclose(_relative(lenses['cam2'], lenses['cam4']),
                                   _relative(old_lens['cam2'], old_lens['cam4']), atol=1e-9)):
                raise ValueError('paired_rig_relative_transform_changed')
            signed_road = [float(mats[k, 1, 3] - candidate['origin_m'][1]) for k in range(31)]
            if not (signed_road[0] < 0 < signed_road[-1] or signed_road[0] > 0 > signed_road[-1]):
                raise ValueError('no_road_cross_section_pass')
            for k in range(31):
                views, inside = {}, {}
                for c in ('cam2', 'cam4'):
                    pose = poses[c]
                    xy, _, _, _, inside[c] = point_in_view(world[k], pose, c, masks[c])
                    angle = view_angle(mats[k, :3, 3], headings[k], pose[:3, 3])
                    yy = np.clip(np.floor(xy[:, 1]).astype(int), 0, 2879)
                    xx = np.floor(xy[:, 0]).astype(int) % 5760
                    dist = signed[c][yy, xx]
                    views[c] = dict(in_view=int(inside[c].sum()), outside_domain=int((~inside[c]).sum()),
                                    view_angle_deg=angle, view_class=classify_view(angle),
                                    signed_boundary_distance_px=dist.tolist())
                attempt['frames'].append(dict(frame_index=k, views=views,
                                              common_physical_points=int((inside['cam2'] & inside['cam4']).sum())))
            attempt['dual_edge'] = edge_summary(attempt['frames'], old['episode_timeline'])
            attempt['side_pass'] = side_pass_edge_coverage(attempt['frames'], attempt['dual_edge']['flags'], signed_road,
                minimum_edge_count=config['edge_min_count'], minimum_overlap_frames=config['side_min_overlap_frames'])
            attempt['side_control'] = side_view_control(attempt['frames'],signed_road)
            first, last = attempt['frames'][0], attempt['frames'][-1]
            attempt['boundary_crossing'] = {c: dict(first_outside=first['views'][c]['outside_domain'],
                last_outside=last['views'][c]['outside_domain'],
                first_distance_p10=float(np.percentile(first['views'][c]['signed_boundary_distance_px'], 10)),
                last_distance_p10=float(np.percentile(last['views'][c]['signed_boundary_distance_px'], 10)))
                for c in ('cam2', 'cam4')}
            attempt['boundary_transition'] = crossing_transition(attempt['boundary_crossing'], 'either')
            if goal=='side_pass_edge_coverage' and not attempt['side_pass']['passed']:
                raise ValueError('side_pass_edge_coverage_gate')
            if goal=='side_view_control' and not attempt['side_control']['passed']:
                raise ValueError('side_view_control_gate')
            # Exact active groom and all body/cloth vertices, same planner as production.
            regions = []
            expected = []
            for k in range(31):
                row, valid = {}, []
                body_world = _world_part(body[indices[k]], mats[k])
                cloth_world = _world_part(cloth[indices[k]], mats[k])
                groom_world = _world_part(_world_part(hair, head[indices[k]]), mats[k])
                for c in ('cam2', 'cam4'):
                    pose = poses[c]
                    parts = {n: project_erp(v, pose, 5760, 2880)[0]
                             for n, v in (('body', body_world), ('cloth', cloth_world), ('groom', groom_world))}
                    region = plan_native_crop_region(parts, (5760, 2880))
                    x0, y0, x1, y1 = region['roi_xyxy']
                    if (x1 - x0) * (y1 - y0) > config['max_roi_pixels_per_camera']:
                        raise ValueError(f'ROI_area_exceeded:frame={k}:camera={c}')
                    visible_xy, _, _, _, visible = point_in_view(
                        np.concatenate((body_world, cloth_world)), pose, c, masks[c])
                    if visible.any():
                        support = visible_xy[visible]
                        box = np.r_[support.min(0), support.max(0)]
                        crop_from_box(box, image_size_wh=(5760, 2880), crop_policy='deployment',
                                      boundary_policy='erp_reject_seam', min_side_px=2)
                    row[c] = region
                    valid.append(point_in_view(world[k], pose, c, masks[c])[-1])
                regions.append(row)
                # [frame, person, camera, point], matching the production label check.
                expected.append(np.asarray(valid)[None,None])
            selected = (candidate, attempt, poses, lenses, regions, np.concatenate(expected))
        except ValueError as error:
            attempt['rejections'].append(str(error))
        attempts.append(attempt)
        write(out / 'candidate_checks.json', dict(schema=VERSION, frozen_candidates_ref=str(frozen_path),
              attempts=attempts, projected_candidate_count=len(attempts)))
        if selected is not None:
            break
    if selected is None:
        write(out / 'geometry_preflight.json', dict(status='rejected', reason='no_candidate_passed',
              seconds=time.monotonic()-begin, projected_candidate_count=len(attempts)))
        return None
    candidate, check, poses, lenses, regions, expected = selected
    for name in ('episode_timeline.json', 'person1_timeline.json', 'person1_source_timeline.json',
                 'scene_matrices.npy', 'world_densekp.npy'):
        shutil.copy2(source / name, out / name)
    shutil.copytree(source / 'calibration', out / 'calibration')
    shutil.copy2(source / 'scene_config.json', out / 'scene_config.json')
    np.save(out / 'expected_in_view.npy', expected)
    write(out / 'planned_regions.json', regions)
    write(out / 'selected_projection.json', check)
    write(out / 'rig.json', dict(calibration_target=old_rig['calibration_target'],
        baseline_kind='synthetic_not_measured', cameras=[dict(id=c, camera_to_world=poses[c].tolist(),
        physical_lens_to_world=lenses[c].tolist()) for c in ('cam2', 'cam4')]))
    recipe = copy.deepcopy(old)
    recipe['episode_id'] = f'single_cyclist_{goal}_s{frozen["master_seed"]}_e00'
    recipe['sequence_id'] = recipe['episode_id']
    recipe['scene_episode_id'] = frozen['scene_episode_id']
    recipe['rig_id'] = f'rig_side_s{frozen["master_seed"]}_c{candidate["child_seed"]}'
    recipe['rig_seed'] = candidate['child_seed']
    recipe['rig_plan_version'] = VERSION
    recipe['parent_recipe_ref'] = str(source / 'recipe.json')
    recipe['coverage_goal'] = goal
    recipe['requested_view_class'] = 'side'
    recipe['seed_algorithm_version'] = VERSION
    recipe['root_master_seed'] = frozen['master_seed']
    recipe['camera_child_seed'] = candidate['child_seed']
    recipe['selected_attempt'] = candidate['attempt']
    recipe['sampled_camera_parameters'] = dict(origin_m=candidate['origin_m'], yaw_deg=candidate['yaw_deg'])
    recipe['rig'].update(origin_m=candidate['origin_m'], yaw_deg=candidate['yaw_deg'])
    recipe['actual_view_coverage'] = check['side_pass'] if goal=='side_pass_edge_coverage' else check['side_control']
    recipe['boundary_transition_passed'] = check['boundary_transition'] is not None
    recipe['actual_crossing_transition'] = check['boundary_transition']
    recipe['view_angles_by_frame'] = [{c:dict(angle_deg=row['views'][c]['view_angle_deg'],
        view_class=row['views'][c]['view_class']) for c in ('cam2', 'cam4')} for row in check['frames']]
    recipe['scene'] = str(out / 'scene_config.json')
    for key, name in [('scene_matrices_ref', 'scene_matrices.npy'), ('expected_in_view_ref', 'expected_in_view.npy'),
                      ('episode_timeline_ref', 'episode_timeline.json')]:
        recipe[key] = str(out / name)
    recipe['person_timeline_refs'] = {'1':str(out / 'person1_timeline.json')}
    recipe['source_timeline_refs'] = {'1':str(out / 'person1_source_timeline.json')}
    write(out / 'recipe.json', recipe)
    write(out / 'geometry_preflight.json', dict(status='passed', seconds=time.monotonic()-begin,
        selected_attempt=candidate['attempt'], dual_edge=check['dual_edge'], side_pass=check['side_pass'],side_control=check['side_control'],
        boundary_transition=check['boundary_transition'],
        fixed_world_points_identical=True, fixed_scene_matrices_identical=True,
        exact_active_groom_regions=True, max_roi_pixels_per_camera=config['max_roi_pixels_per_camera']))
    return recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['plan', 'preflight'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/seeded_side_pass_a1_20261010.json')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'plan':
        result = plan(args.config, args.output, seed=args.seed)
        print('SIDE_PASS_FROZEN', len(result['candidates']), args.output)
    else:
        if args.frozen is None:parser.error('--frozen is required for preflight')
        recipe = preflight(args.config, args.frozen, args.output)
        print('SIDE_PASS_PREPARED' if recipe else 'SIDE_PASS_REJECTED', args.output)


if __name__ == '__main__':
    main()
