"""Freeze and preflight a second paired rig for a fixed three-rider city scene.

Planning does not evaluate models or touch the GPU. Every projected candidate is
listed before preflight; the three people and their original clocks stay fixed.
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
from densekp_retrain.data.dynamic_scene import frame_record
from densekp_retrain.data.scene_geometry import person_geometry as _person_geometry, transform as _transform
from densekp_retrain.data.viewpoint import classify_view, view_angle
from densekp_retrain.imaging.roi_planning import plan_person_crop_union

SCHEMA = 'urban_three_cyclists_multirig_a1_v1'
SUPPORTED_SCHEMAS = (SCHEMA, 'dispersed_cyclists_camera_rig_v1', 'front_back_side_camera_rig_v1',
                     't_junction_multirig_camera_v1')


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source(config):
    return (ROOT / config['source_prepared']).resolve()


def frozen_candidates(config, world, *, seed):
    target_mid = world[15, 0].mean(0)
    cells = [(x, y, yaw) for x in config['camera_x_m'] for y in config['camera_y_m']
             for yaw in config['seam_yaw_offset_deg']]
    if len(cells) != config['candidate_limit'] or len(cells) != len(set(cells)) or len(cells) > 24:
        raise ValueError('invalid_frozen_camera_cells')
    rng = random.Random(f'{SCHEMA}|camera|{seed}')
    rng.shuffle(cells)
    result = []
    for index, (cx, cy, offset) in enumerate(cells):
        child = rng.getrandbits(63)
        local = random.Random(child)
        x = cx + local.uniform(-.02, .02)
        y = cy + local.uniform(-.02, .02)
        yaw_offset = offset + local.uniform(-.2, .2)
        bearing = math.degrees(math.atan2(target_mid[1] - y, target_mid[0] - x))
        result.append(dict(attempt=index, camera_child_seed=child,
                           nominal_cell=[cx, cy, offset], origin_m=[x, y, 0.0],
                           yaw_deg=bearing + yaw_offset,
                           construction='target_midpoint_bearing_plus_seeded_seam_offset'))
    return result


def plan(config_path, output, *, seed=None):
    config = json.loads(Path(config_path).read_text())
    if config['schema'] not in SUPPORTED_SCHEMAS:
        raise ValueError('unknown_config_schema')
    src = source(config)
    recipe = json.loads((src / 'recipe.json').read_text())
    world = np.load(src / 'world_densekp.npy')
    matrices = np.load(src / 'scene_matrices.npy')
    expected = np.load(src / 'expected_in_view.npy', mmap_mode='r')
    count = len(recipe['people'])
    if recipe['frame_count'] != 31 or not 1 <= count <= 5 or [p['person_id'] for p in recipe['people']] != list(range(1, count + 1)):
        raise ValueError('source_person_contract')
    if world.shape != (31, count, 138, 3) or matrices.shape != (31, count, 4, 4) or expected.shape != (31, count, 2, 138):
        raise ValueError('source_person_axis_conflict')
    if json.loads((src / 'geometry_preflight.json').read_text())['status'] != 'passed':
        raise ValueError('source_geometry_not_passed')
    if not all(np.isfinite(x).all() for x in (world, matrices)):
        raise ValueError('nonfinite_source_geometry')
    seed = config['camera_seed'] if seed is None else seed
    result = dict(schema=SCHEMA, scene_episode_id=config['scene_episode_id'],
                  source_prepared=str(src), camera_seed=seed, target_person_id=1,
                  fixed_people=[dict(person_id=p['person_id'], asset_id=p['asset_id'],
                                     asset_manifest=p['asset_manifest'], asset_frames=p['asset_frames'],
                                     source_frames=p['source_frames']) for p in recipe['people']],
                  candidate_generation='separate_camera_random_stream; sorted_cells_then_seeded_shuffle_and_bounded_jitter',
                  candidates=frozen_candidates(config, world, seed=seed),
                  limits=dict(max_candidates=24, front_oblique_deg=[45, 80],
                              simultaneous_window_frames=6, common_points=8,
                              common_frames=16, maximum_union_roi_pixels=7_000_000))
    out = Path(output)
    if out.exists():
        raise FileExistsError(out)
    write(out, result)
    return result


def _longest(flags):
    run = best = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def coverage_pass(common, view_common, edge, *, view_goal, config, road_pass):
    """Keep historical back-edge and side-cross-section defaults explicit."""
    if view_goal not in ('front', 'back', 'side', 'front_oblique'):
        raise ValueError('invalid_camera_view_goal')
    view_count_gate = (view_goal not in ('front', 'back') or
                       sum(view_common) >= config.get('minimum_view_common_frames', 16))
    edge_gate = (view_goal != 'back' or
                 (sum(edge) >= config.get('minimum_dual_edge_frames', 16) and
                  _longest(edge) >= config.get('minimum_dual_edge_run', 6)))
    road_gate = (view_goal != 'side' or not config.get('require_road_cross_section', True) or road_pass)
    return (sum(common) >= config['minimum_common_physical_frames'] and
            _longest(view_common) >= config.get('minimum_view_common_run',
                                                config.get('minimum_front_oblique_common_run', 6)) and
            view_count_gate and edge_gate and road_gate)



def preflight(config_path, frozen_path, output, *, time_limit_seconds=900):
    import cv2
    from PIL import Image
    from densekp_retrain.data.boxes import crop_from_box
    start = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    frozen = json.loads(Path(frozen_path).read_text())
    if config['schema'] != frozen['schema'] or config['schema'] not in SUPPORTED_SCHEMAS or len(frozen['candidates']) > 24:
        raise ValueError('frozen_config_conflict')
    src = Path(frozen['source_prepared'])
    recipe = json.loads((src / 'recipe.json').read_text())
    world = np.load(src / 'world_densekp.npy')
    matrices = np.load(src / 'scene_matrices.npy')
    persons = _person_geometry(recipe)
    old_rig = json.loads((src / 'rig.json').read_text())
    old_pose = {c['id']: np.asarray(c['camera_to_world']) for c in old_rig['cameras']}
    old_lens = {c['id']: np.asarray(c['physical_lens_to_world']) for c in old_rig['cameras']}
    masks = {c: np.asarray(Image.open(src / 'calibration' / f'{c}_valid_mask.png')) > 0
             for c in ('cam2', 'cam4')}
    # Body forward is the accepted model +Z transformed by its own source
    # global_orient and this person's scene matrix, never by camera yaw.
    headings = []
    for k in range(31):
        vector = matrices[k, 0, :3, :3] @ cv2.Rodrigues(persons[0]['orientation'][k])[0] @ [0., 0., 1.]
        headings.append(vector)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checks = []
    selected = None
    exact_attempts = 0
    view_goal = config.get('view_goal', 'front_oblique')
    if view_goal not in ('front', 'back', 'side', 'front_oblique'):
        raise ValueError('invalid_camera_view_goal')
    for candidate in frozen['candidates']:
        if time.monotonic() - start >= time_limit_seconds:
            break
        check = dict(attempt=candidate['attempt'], origin_m=candidate['origin_m'],
                     yaw_deg=candidate['yaw_deg'], camera_child_seed=candidate['camera_child_seed'],
                     rejections=[], frames=[])
        try:
            poses, lenses, _ = rig_poses(candidate['origin_m'], candidate['yaw_deg'], recipe['rig']['baseline_m'])
            for source_m, current_m in ((old_pose, poses), (old_lens, lenses)):
                relative_old = np.linalg.inv(source_m['cam2']) @ source_m['cam4']
                relative_new = np.linalg.inv(current_m['cam2']) @ current_m['cam4']
                if not np.allclose(relative_old, relative_new, atol=1e-9):
                    raise ValueError('paired_rig_relative_transform_changed')
            # Basic placement gate precedes expensive projection. The static
            # environment ray clearance is checked separately before render.
            for k in range(31):
                for camera in poses.values():
                    distance = np.linalg.norm(matrices[k, :, :2, 3] - camera[:2, 3], axis=1)
                    if distance.min() < config['minimum_camera_to_person_root_m']:
                        raise ValueError(f'camera_near_person_root:frame={k}')
            if (config.get('minimum_camera_road_x_abs_m') is not None and
                abs(candidate['origin_m'][0]) < config['minimum_camera_road_x_abs_m']):
                raise ValueError('camera_inside_road_clearance')
            for k in range(31):
                inside = {}
                views = {}
                for camera, pose in poses.items():
                    _, _, _, _, inside[camera] = point_in_view(world[k, 0], pose, camera, masks[camera])
                    angle = view_angle(matrices[k, 0, :3, 3], headings[k], pose[:3, 3])
                    views[camera] = dict(in_view=int(inside[camera].sum()),
                                         outside_domain=int((~inside[camera]).sum()),
                                         view_angle_deg=angle, view_class=classify_view(angle))
                check['frames'].append(dict(frame_index=k, views=views,
                                            common_physical_points=int((inside['cam2'] & inside['cam4']).sum())))
            common = [row['common_physical_points'] >= config['minimum_common_points'] for row in check['frames']]
            view_common = [both and all(row['views'][c]['view_class'] == view_goal for c in ('cam2', 'cam4'))
                           for both, row in zip(common, check['frames'])]
            edge = [all(row['views'][c]['in_view'] >= 8 and row['views'][c]['outside_domain'] >= 5
                        for c in ('cam2', 'cam4')) and row['common_physical_points'] >= 8
                    for row in check['frames']]
            road_pass = bool(matrices[0, 0, 1, 3] < candidate['origin_m'][1] < matrices[-1, 0, 1, 3])
            passed = coverage_pass(common, view_common, edge, view_goal=view_goal,
                                   config=config, road_pass=road_pass)
            check['coverage'] = dict(common_frames=sum(common), view_goal=view_goal,
                view_common_flags=view_common, view_common_count=sum(view_common),
                longest_view_common_run=_longest(view_common), dual_edge_flags=edge,
                dual_edge_count=sum(edge), longest_dual_edge_run=_longest(edge),
                road_cross_section_passed=road_pass if view_goal == 'side' else None,
                passed=passed)
            if view_goal == 'front_oblique':
                check['coverage'].update(front_oblique_common_flags=view_common,
                                         longest_front_oblique_common_run=_longest(view_common))
            if not check['coverage']['passed']:
                raise ValueError(f'{view_goal}_or_common_physical_gate')
            if exact_attempts >= config.get('maximum_exact_support_candidates', 24):
                raise ValueError('exact_support_candidate_limit_reached')
            exact_attempts += 1
            # Full active hair, body and cloth per person; exactly the union
            # planner used by the production renderer, for all 31 frames.
            regions = []
            expected = []
            for k in range(31):
                row = {}
                per_inside = []
                for c, pose in poses.items():
                    grouped = {}
                    masks_for_camera = []
                    for j, person in enumerate(persons):
                        M = matrices[k, j]
                        parts = dict(body=_transform(person['body'][k], M),
                                     cloth=_transform(person['cloth'][k], M),
                                     groom=_transform(_transform(person['hair'], person['head'][k]), M))
                        grouped[str(j+1)] = {name: project_erp(points, pose, 5760, 2880)[0]
                                             for name, points in parts.items()}
                        in_view = point_in_view(world[k, j], pose, c, masks[c])[-1]
                        masks_for_camera.append(in_view)
                        proxy, _, _, _, visible = point_in_view(
                            np.concatenate((parts['body'], parts['cloth'])), pose, c, masks[c])
                        if visible.any():
                            support = proxy[visible]
                            box = np.r_[support.min(0), support.max(0)]
                            crop_from_box(box, image_size_wh=(5760, 2880), crop_policy='deployment',
                                          boundary_policy='erp_reject_seam', min_side_px=2)
                    region = plan_person_crop_union(grouped, (5760, 2880))
                    x0, y0, x1, y1 = region['roi_xyxy']
                    if (x1-x0)*(y1-y0) > config['maximum_union_roi_pixels']:
                        raise ValueError(f'union_ROI_pixel_budget:frame={k}:camera={c}')
                    row[c] = region
                    per_inside.append(np.stack(masks_for_camera, axis=0))
                expected.append(np.stack(per_inside, axis=1))
                regions.append(row)
            selected = (candidate, check, poses, lenses, regions, np.asarray(expected))
            check['exact_active_groom_and_all_person_roi_passed'] = True
        except ValueError as error:
            check['rejections'].append(str(error))
        checks.append(check)
        write(output / 'candidate_checks.json', dict(schema=config['schema'], attempts=checks,
                                                       attempted_count=len(checks), candidate_count=len(frozen['candidates'])))
        if selected is not None:
            break
    if selected is None:
        write(output / 'geometry_preflight.json', dict(status='rejected', attempted_count=len(checks),
                                                       elapsed_seconds=time.monotonic()-start,
                                                       reason='no_front_oblique_rig_with_full_support'))
        return None
    candidate, check, poses, lenses, regions, expected = selected
    if expected.shape != (31, len(recipe['people']), 2, 138):
        raise ValueError('preflight_person_axis_conflict')
    for item in src.iterdir():
        if item.name in ('recipe.json', 'rig.json', 'planned_regions.json', 'selected_projection.json',
                         'expected_in_view.npy', 'geometry_preflight.json', 'candidate_checks.json'):
            continue
        if item.is_file():
            shutil.copy2(item, output / item.name)
        elif item.name == 'calibration':
            shutil.copytree(item, output / item.name)
    np.save(output / 'expected_in_view.npy', expected)
    write(output / 'planned_regions.json', regions)
    write(output / 'selected_projection.json', check)
    write(output / 'rig.json', dict(calibration_target=old_rig['calibration_target'],
                                    baseline_kind='synthetic_not_measured',
                                    cameras=[dict(id=c, camera_to_world=poses[c].tolist(),
                                                  physical_lens_to_world=lenses[c].tolist())
                                             for c in ('cam2', 'cam4')]))
    prepared_recipe = copy.deepcopy(recipe)
    rig_id = config.get('rig_id', 'rig_front_oblique')
    prepared_recipe.update(scene_episode_id=frozen['scene_episode_id'], rig_id=rig_id,
                           coverage_goal=config.get('coverage_goal', 'front_oblique_view_control'), requested_view_class=view_goal,
                           camera_child_seed=candidate['camera_child_seed'], selected_camera_attempt=candidate['attempt'],
                           view_angles_by_frame=[{c:dict(angle_deg=row['views'][c]['view_angle_deg'],
                                                         view_class=row['views'][c]['view_class'])
                                                  for c in ('cam2', 'cam4')} for row in check['frames']],
                           expected_in_view_ref=str((output / 'expected_in_view.npy').resolve()))
    prepared_recipe['rig'].update(origin_m=candidate['origin_m'], yaw_deg=candidate['yaw_deg'])
    write(output / 'recipe.json', prepared_recipe)
    write(output / 'geometry_preflight.json', dict(status='passed', selected_attempt=candidate['attempt'],
                                                   coverage=check['coverage'], exact_groom=True,
                                                   all_person_ROI=True, exact_candidates=exact_attempts,
                                                   elapsed_seconds=time.monotonic()-start))
    return prepared_recipe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['plan', 'preflight'])
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/urban_three_cyclists_multirig_a1_20261011.json')
    parser.add_argument('--frozen', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int)
    args = parser.parse_args()
    if args.command == 'plan':
        plan(args.config, args.output, seed=args.seed)
    else:
        if args.frozen is None:
            parser.error('--frozen is required')
        preflight(args.config, args.frozen, args.output)


if __name__ == '__main__':
    main()
