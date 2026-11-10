#!/usr/bin/env python3
"""Freeze and preflight 4/5 distinct riders in the existing city scene."""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.data.cycling_team import team_recipe, preflight_team
from tools.experiments.prepare_cycling_team_relative_layout_a1 import frozen_people
from tools.experiments.seed_asset_composition_a1 import derive_rotation_and_height, motion_origins

CONFIG = ROOT / 'configs/dispersed_cyclists_4_5_multirig_a1_20261011.json'
OUT = ROOT / 'outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011'
OLD = ROOT / 'outputs/experiments/cycling_team_relative_layout_a1_20261010/city_attempt5_cpu_only_correction1/prepared'


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def candidates(count, seed, anchor):
    """Seed-order a bounded, structured, dispersed placement table."""
    if count not in (4, 5) or type(seed) is not int:
        raise ValueError('unsupported_count_or_seed')
    cells = [(4.9, .35), (5.2, .35), (5.5, .35),
             (4.9, .55), (5.2, .55), (5.5, .55),
             (4.9, -.75), (5.2, -.75), (5.5, -.75),
             (4.9, .8), (5.2, .8), (5.5, .8)]
    rng = random.Random(f'dispersed_cyclists_v1|layout|{seed}|{count}')
    rng.shuffle(cells)
    output = []
    for i, (y4, x4) in enumerate(cells):
        child = rng.getrandbits(63)
        local = random.Random(child)
        q = dict(anchor)
        q.update(p4_y_m=y4 + local.uniform(-.035, .035),
                 p4_x_m=x4 + local.uniform(-.025, .025),
                 p4_speed_m_s=.47 + local.uniform(-.04, .04))
        if count == 5:
            q.update(p5_y_m=6.9 + local.uniform(-.18, .18),
                     p5_x_m=(-.65 if x4 > 0 else .65) + local.uniform(-.035, .035),
                     p5_speed_m_s=.49 + local.uniform(-.04, .04))
        output.append(dict(attempt=i, child_seed=child, parameters=q,
                           structured_cell=dict(person4_y_m=y4, person4_x_m=x4)))
    return output


def freeze():
    cfg = read(CONFIG)
    base = read(OLD / 'recipe.json')
    anchor = base['sampled_parameters']
    for count, seed in ((4, 20261030), (5, 20261031)):
        episode = OUT / f'city_{count}p'
        write_new(episode / 'frozen_layout_candidates.json', candidates(count, seed, anchor))
        write_new(episode / 'frozen_policy.json', dict(config=cfg, count=count,
            inherited_scene=str(OLD), layout_seed=seed,
            first_three_parameters=anchor,
            construction='preserve first three; seed-shuffle structured forward road cells with bounded jitter',
            first_passing_hard_geometry_wins=True,
            projected_occlusion_policy='diagnostic_only'))
    second = candidates(4, 20261034, anchor)
    write_new(OUT / 'different_seed_plan_check.json', dict(seed=20261034,
              person_plan_unchanged=True, camera_plan_not_executed=True,
              first_layout_differs=second[0]['parameters'] != candidates(4, 20261030, anchor)[0]['parameters']))


def prepare_people(count, stage):
    policy, plan, people, assets, placements, checks = frozen_people()
    builds = read(OUT / 'composition_builds.json')
    registry = read(ROOT / 'configs/asset_composition_a2_20261010/components.json')
    origins = [p['source_origin_record'] for p in plan['people']]
    for slot in range(4, count + 1):
        record = builds[slot - 4]
        if record['person_id'] != slot or record['status'] != 'passed' or record['validation'] != 'numeric_passed':
            raise ValueError(f'new_person_{slot}_not_verified')
        manifest = read(record['delivery'])
        adapter = copy.deepcopy(record['adapter'])
        component = next(p for p in read(CONFIG)['people'] if p['person_id'] == slot)
        ids = {k: component[k] for k in ('body_id', 'motion_id', 'clothing_id', 'hair_id')}
        if any(manifest['components'][kind]['id'] != ids[kind + '_id']
               for kind in ('body', 'motion', 'clothing', 'hair')):
            raise ValueError('new_person_component_conflict')
        origins_allowed, _ = motion_origins(registry, ids['motion_id'])
        local = random.Random(f'dispersed_cyclists_v1|time|{slot}')
        origin = origins_allowed[local.randrange(len(origins_allowed))]
        rotation, height, clearance, mapped = derive_rotation_and_height(
            adapter, origin, 'heading_minimax_fixed_yaw_v2')
        adapter.update(components=ids, component_versions={k: manifest['components'][k]['version']
                       for k in ('body', 'motion', 'clothing', 'hair')},
                       composition_delivery_ref=record['delivery'],
                       human_acceptance='pending_road_video_review',
                       technical_use_policy='integration_only_numeric_passed')
        adapter_path = stage / f'person{slot}_asset_adapter.json'
        write_new(adapter_path, adapter)
        person = copy.deepcopy(people[0])
        for key in ('scene_from_model', 'model_to_world', 'asset_frames', 'source_frames',
                    'road_start_m', 'path_direction_xy', 'speed_m_s'):
            person.pop(key, None)
        person.update(person_id=slot, role='occluder', asset_id=adapter['asset_id'],
            asset_manifest=str(adapter_path), character_family=adapter['character_family'],
            root_mode=adapter['root_mode'], support_policy=adapter['support_policy'],
            component_ids=ids, source_group=adapter['source_group'],
            composition_delivery_ref=record['delivery'],
            outfit_policy='fixed_components_entire_episode')
        reference = copy.deepcopy(read(people[0]['reference_recipe']))
        reference['people'] = [copy.deepcopy(person)]
        reference_path = stage / f'person{slot}_reference_recipe.json'
        person['reference_recipe'] = str(reference_path)
        reference['people'][0]['reference_recipe'] = str(reference_path)
        write_new(reference_path, reference)
        support = copy.deepcopy(placements[0]['support_checks'])
        support.update(body_min_clearance_m=[.05, .7],
                       pelvis_height_above_road_m=[height - .05, height + .05])
        people.append(person); assets.append(adapter)
        placements.append(dict(rotation=rotation, height_m=height,
                               support_checks=support, body_clearance_m=clearance))
        origins.append(origin)
        checks.append(dict(person_id=slot, components=ids, delivery=record['delivery'],
             source_group=adapter['source_group'], origin_record=origin,
             selected_asset_frames=[x['asset_frame'] for x in mapped],
             cache_state=record['cache_state'], validation=record['validation'],
             human_acceptance='pending_road_video_review'))
    write_new(stage / 'asset_reuse_and_new_combinations.json', checks)
    write_new(stage / 'placements.json', placements)
    return policy, people, assets, placements, origins


def preflight(count):
    stage = OUT / f'city_{count}p'
    frozen = read(stage / 'frozen_policy.json')
    if frozen['config'] != read(CONFIG) or frozen['count'] != count:
        raise ValueError('frozen_policy_changed')
    candidate_list = read(stage / 'frozen_layout_candidates.json')
    if candidate_list != candidates(count, frozen['layout_seed'], read(OLD / 'recipe.json')['sampled_parameters']):
        raise ValueError('frozen_candidate_algorithm_changed')
    policy, people, assets, placements, origins = prepare_people(count, stage)
    base = read(OLD / 'recipe.json')
    episode, sources = team_recipe(read(people[0]['reference_recipe']), people, origins,
        f'city_dispersed_{count}p_s{frozen["layout_seed"]}', 'urban_street_assets')
    episode.update(scene=base['scene'], render=copy.deepcopy(base['render']),
        rig=copy.deepcopy(base['rig']), layout_master_seed=frozen['layout_seed'],
        projected_occlusion_policy='diagnostic_only', max_union_roi_pixels=7_000_000,
        component_provenance=[dict(person_id=i+1, components=people[i]['component_ids'],
             source_group=people[i]['source_group'], origin_record=origins[i]) for i in range(count)])
    geo = copy.deepcopy(policy['environments']['urban_street_assets'])
    geo.update(environment_id='urban_street_assets', baseline_m=episode['rig']['baseline_m'],
        rig_origin_z_m=episode['rig']['origin_m'][2], physical_masks=policy['physical_masks'])
    for placement in placements:
        placement['support_checks']['road_object_names'] = geo['road_object_names']
        placement['support_checks']['road_object_prefixes'] = geo['road_object_prefixes']
    started = time.monotonic()
    result = preflight_team(episode, sources, assets, placements, candidate_list,
        geo, stage / 'prepared_edge', max_roi_pixels=7_000_000,
        projected_occlusion_policy='diagnostic_only')
    write_new(stage / 'edge_preflight_cost.json', dict(elapsed_seconds=time.monotonic() - started,
        status='passed' if result else 'rejected',
        attempted_count=len(read(stage / 'prepared_edge/candidate_checks.json')['attempts'])))
    return result


def camera_candidates(count, camera_seed, base):
    if count not in (4, 5) or type(camera_seed) is not int:
        raise ValueError('invalid_camera_candidate_count_or_seed')
    randomizer = random.Random(f'dispersed_cyclists_v1|camera|{camera_seed}|{count}')
    rows = []
    for attempt in range(12):
        child = randomizer.getrandbits(63)
        local = random.Random(child)
        if attempt == 0:
            origin, yaw = base['origin_m'], base['yaw_deg']
        else:
            origin = [base['origin_m'][0] + local.uniform(-.35, .35),
                      base['origin_m'][1] + local.uniform(-.4, .4),
                      base['origin_m'][2]]
            yaw = base['yaw_deg'] + local.uniform(-4, 4)
        rows.append(dict(attempt=attempt, camera_child_seed=child,
             origin_m=origin, yaw_deg=yaw,
             construction='old_three_person_front_oblique_rig_plus_bounded_seeded_offset'))
    return rows


def freeze_camera(count):
    stage = OUT / f'city_{count}p'
    source = stage / 'prepared_edge'
    if read(source / 'geometry_preflight.json')['status'] != 'passed':
        raise ValueError('edge_rig_not_preflighted')
    camera_seed = {4: 20261032, 5: 20261033}[count]
    old = read(ROOT / 'outputs/experiments/urban_three_cyclists_multirig_a1_20261011/production/rigs/rig_front_oblique/recipe.json')['rig']
    base = dict(origin_m=old['origin_m'], yaw_deg=old['yaw_deg'])
    rows = camera_candidates(count, camera_seed, base)
    write_new(stage / 'frozen_camera_candidates.json', dict(
        schema='dispersed_cyclists_camera_rig_v1',
        scene_episode_id=read(source / 'recipe.json')['episode_id'],
        source_prepared=str(source), camera_seed=camera_seed,
        fixed_people=[dict(person_id=p['person_id'], asset_id=p['asset_id'])
                      for p in read(source / 'recipe.json')['people']],
        candidates=rows))
    write_new(stage / 'camera_preflight_config.json', dict(
        schema='dispersed_cyclists_camera_rig_v1',
        minimum_common_points=8, minimum_common_physical_frames=16,
        minimum_front_oblique_common_run=6, maximum_union_roi_pixels=7_000_000,
        minimum_camera_to_person_root_m=.5))


def preflight_camera(count):
    from tools.experiments.prepare_urban_three_multirig_a1 import preflight as check_camera
    stage = OUT / f'city_{count}p'
    return check_camera(stage / 'camera_preflight_config.json',
                        stage / 'frozen_camera_candidates.json',
                        stage / 'prepared_front_oblique', time_limit_seconds=900)


def rig_manifest(count):
    stage = OUT / f'city_{count}p'
    edge = stage / 'prepared_edge'
    front = stage / 'prepared_front_oblique'
    if any(read(path / 'geometry_preflight.json')['status'] != 'passed' for path in (edge, front)):
        raise ValueError('rig_not_preflighted')
    def slots(rig_id):
        items = [f'f15_{camera}_p{person}' for camera in ('cam2', 'cam4') for person in range(1, count + 1)]
        if count == 4:
            if rig_id == 'rig_edge':
                items.extend(f'f0_{camera}_p1' for camera in ('cam2', 'cam4'))
                items.extend(f'f30_{camera}_p2' for camera in ('cam2', 'cam4'))
            else:
                items.extend(f'f30_{camera}_p1' for camera in ('cam2', 'cam4'))
                items.extend(f'f0_{camera}_p3' for camera in ('cam2', 'cam4'))
        else:
            frame = 0 if rig_id == 'rig_edge' else 30
            items.extend(f'f{frame}_{camera}_p1' for camera in ('cam2', 'cam4'))
        if len(items) != 12:
            raise ValueError('readback_slot_count_conflict')
        return items
    scene_id = read(edge / 'recipe.json')['episode_id']
    entries = [dict(rig_id='rig_edge', prepared=str(edge), coverage_goal='edge_coverage',
                    camera_seed={4: 20261032, 5: 20261033}[count],
                    readback_observation_ids=slots('rig_edge')),
               dict(rig_id='rig_front_oblique', prepared=str(front),
                    coverage_goal='front_oblique_view_control',
                    camera_seed={4: 20261032, 5: 20261033}[count],
                    readback_observation_ids=slots('rig_front_oblique'))]
    manifest = dict(schema='scene_episode_rig_list_v1', scene_episode_id=scene_id,
                    integration_only=True, source_grouping='all_people_all_rigs_one_scene', rigs=entries)
    write_new(stage / 'rig_list_manifest.json', manifest)
    write_new(stage / 'readback_plan_frozen.json', dict(scene_episode_id=scene_id,
        planned_slots=24, rigs={r['rig_id']: r['readback_observation_ids'] for r in entries},
        no_box_slots_remain_unread=True))
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'preflight4', 'preflight5',
                                         'freeze_camera4', 'freeze_camera5',
                                         'preflight_camera4', 'preflight_camera5',
                                         'rig_manifest4', 'rig_manifest5'))
    args = parser.parse_args()
    if args.action == 'freeze': freeze()
    elif args.action.startswith('preflight_camera'): preflight_camera(int(args.action[-1]))
    elif args.action.startswith('freeze_camera'): freeze_camera(int(args.action[-1]))
    elif args.action.startswith('rig_manifest'): rig_manifest(int(args.action[-1]))
    else: preflight(int(args.action[-1]))


if __name__ == '__main__':
    main()
