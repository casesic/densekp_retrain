#!/usr/bin/env python3
"""Read-only reuse of two frozen combinations; bounded relative river layout."""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from densekp_retrain.assets import composition as a2
from densekp_retrain.data.cycling_team import team_recipe, preflight_team, relative_layout_candidates
from tools.experiments.seed_asset_composition_a1 import checked_build
from tools.experiments.seed_scene_modes import load_pool

OLD = ROOT/'outputs/experiments/seeded_cycling_team_a1_20261010'
OUT = ROOT/'outputs/experiments/cycling_team_relative_layout_a1_20261010'
CONFIG = ROOT/'configs/cycling_team_relative_layout_a1_20261010.json'


def read(path): return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def frozen_people():
    policy = read(ROOT/'configs/seeded_cycling_team_a1_20261010.json')
    plan = read(OLD/'frozen_team_plan.json')
    builds = read(OLD/'team_assets/builds.json')
    placements = read(OLD/'team_assets/placements.json')
    registry = a2.read(ROOT/policy['registry'])
    pool = load_pool(ROOT/policy['legacy_pool'])
    people = []; assets = []; checks = []
    for i in (1, 2, 3):
        entry = plan['people'][i-1]; build = builds[i-1]
        directory = Path(build['delivery']).parent
        if not directory.is_dir() or not (directory/'frozen_recipe.json').is_file():
            raise ValueError('missing_previous_verified_build')
        resolved = a2.plan(ROOT, registry, dict(entry['recipe'], output_dir=str(OLD/'composition_cache')))
        if a2.read(directory/'frozen_recipe.json') != resolved:
            raise ValueError('previous_build_dependency_or_recipe_changed')
        fake = dict(episode_index=i+10, selected_components=entry['selection'], resolved=resolved)
        verified, validation, adapted, state = checked_build(fake, OLD/'composition_cache')
        if verified != directory or state != 'delivery_verified_hit':
            raise ValueError('previous_build_not_a_verified_cache_hit')
        adapter = read(build['asset_manifest'])
        for key in ('asset_id', 'character_family', 'root_mode', 'source_group', 'timeline_ref',
                    'parameters', 'groom_reference'):
            if adapted[key] != adapter[key]:
                raise ValueError('old_adapter_build_mismatch:'+key)
        template = next((x for x in pool['riders'] if x['character_family']==entry['character_family']), None)
        if template is None and entry['character_family']=='cycling_M02':
            root = ROOT/'outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1'
            template = dict(reference_recipe=str(root/'reference_recipe.json'))
        if template is None: raise ValueError('missing_scene_template')
        person = copy.deepcopy(read(template['reference_recipe'])['people'][0])
        for key in ('scene_from_model','model_to_world','asset_frames','source_frames'):
            person.pop(key, None)
        person.update(asset_id=adapter['asset_id'], asset_manifest=build['asset_manifest'],
                      character_family=adapter['character_family'], root_mode=adapter['root_mode'],
                      support_policy=adapter['support_policy'], component_ids=entry['selection'],
                      source_group=adapter['source_group'], composition_delivery_ref=build['delivery'],
                      outfit_policy='fixed_seed_selected_components_entire_episode',
                      reference_recipe=template['reference_recipe'])
        people.append(person); assets.append(adapter)
        checks.append(dict(person_id=i, delivery=build['delivery'], asset_manifest=build['asset_manifest'],
                           selection=entry['selection'], origin_record=entry['source_origin_record'],
                           validation=validation['status'], cache_state=state,
                           human_acceptance='not_inherited_new_combination'))
    return policy, plan, people, assets, placements, checks


def make_recipe(policy, plan, people, count, environment, stage):
    base = read(people[0]['reference_recipe'])
    episode_id = f'cycling_team_relative_layout_{environment}_s20261023_{count}p'
    episode, sources = team_recipe(base, copy.deepcopy(people[:count]),
        [p['source_origin_record'] for p in plan['people'][:count]], episode_id, environment)
    episode.update(master_seed=policy['master_seed'], layout_master_seed=20261023,
                   projected_occlusion_policy='diagnostic_only',
                   component_provenance=[dict(person_id=i+1, selection=x['selection'],
                       source_group=x['source_group'], origin_record=x['source_origin_record'])
                       for i,x in enumerate(plan['people'][:count])],
                   occlusion_metric=copy.deepcopy(policy['occlusion_metric']),
                   max_union_roi_pixels=7_000_000)
    scene = read(ROOT/policy['environments'][environment]['scene_config'])
    episode['scene'] = str(stage/'scene_config.json')
    scene['scene_settings_after_configure_render'] = policy['environments'][environment].get('render_settings',
        scene.get('scene_settings_after_configure_render'))
    blend = ROOT/scene['blend_path']; stat = blend.stat()
    scene['expected_blend_stat'] = dict(size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
    write_new(stage/'scene_config.json', scene)
    geo = copy.deepcopy(policy['environments'][environment]); geo.update(environment_id=environment,
        baseline_m=episode['rig']['baseline_m'], rig_origin_z_m=episode['rig']['origin_m'][2],
        physical_masks=policy['physical_masks'])
    return episode, sources, geo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('action', choices=('freeze','river_preflight','city_recheck','old_failure_diagnostics'))
    args = ap.parse_args()
    cfg = read(CONFIG)
    if cfg['schema'] != 'cycling_team_relative_layout_v1' or cfg['layout_master_seed'] != 20261023:
        raise ValueError('invalid_relative_layout_config')
    t0 = time.monotonic()
    policy, plan, people, assets, placements, checks = frozen_people()
    if args.action == 'freeze':
        anchor = cfg['riverfront_anchor_parameters']
        candidates = relative_layout_candidates(anchor, cfg['layout_master_seed'], limit=cfg['candidate_limit'],
            cells_config=cfg['relative_cells'])
        write_new(OUT/'frozen_asset_reuse.json', checks[:2])
        write_new(OUT/'frozen_candidates.json', candidates)
        write_new(OUT/'frozen_execution_policy.json', cfg)
        other = relative_layout_candidates(anchor, cfg['different_seed_plan_check'], limit=2,
            cells_config=cfg['relative_cells'])
        write_new(OUT/'different_seed_plan_check.json', dict(seed=cfg['different_seed_plan_check'],
                  changes_parameters=other[0]['parameters']!=candidates[0]['parameters'], candidates=other))
    elif args.action == 'river_preflight':
        if read(OUT/'frozen_execution_policy.json') != cfg: raise ValueError('frozen_policy_changed')
        if read(OUT/'frozen_candidates.json') != relative_layout_candidates(
                cfg['riverfront_anchor_parameters'], cfg['layout_master_seed'],
                limit=cfg['candidate_limit'], cells_config=cfg['relative_cells']):
            raise ValueError('frozen_candidate_algorithm_changed')
        stage = OUT/'riverfront_2'; stage.mkdir(exist_ok=False)
        ep, sources, geo = make_recipe(policy, plan, people, 2, 'riverfront', stage)
        for p in placements[:2]:
            p['support_checks']['road_object_names']=geo['road_object_names']
            p['support_checks']['road_object_prefixes']=geo['road_object_prefixes']
        result = preflight_team(ep, sources, assets[:2], placements[:2],
             read(OUT/'frozen_candidates.json'), geo, stage/'prepared', max_roi_pixels=7_000_000,
             projected_occlusion_policy='diagnostic_only')
        write_new(OUT/'riverfront_cpu_cost.json', dict(seconds=time.monotonic()-t0,
                  candidate_count=len(read(stage/'prepared/candidate_checks.json')['attempts']),
                  status='passed' if result else 'rejected'))
    elif args.action == 'old_failure_diagnostics':
        old = read(OLD/'riverfront_2_overlap_gate_correction2/prepared/candidate_checks.json')['attempts']
        stage = OUT/'old_failure_diagnostics'; stage.mkdir(exist_ok=False)
        ep, sources, geo = make_recipe(policy, plan, people, 2, 'riverfront', stage)
        for p in placements[:2]:
            p['support_checks']['road_object_names']=geo['road_object_names']
            p['support_checks']['road_object_prefixes']=geo['road_object_prefixes']
        old_candidates = [{k:old[i][k] for k in ('attempt','child_seed','parameters')} for i in (0,3,6)]
        preflight_team(ep, sources, assets[:2], placements[:2], old_candidates,
            geo, stage/'prepared', max_roi_pixels=7_000_000,
            projected_occlusion_policy='require_event')
        write_new(OUT/'old_failure_cpu_cost.json',dict(seconds=time.monotonic()-t0,
            representative_old_attempts=[0,3,6]))
    else:
        original = read(OLD/'urban_3_overlap_gate_correction2/prepared/candidate_checks.json')['attempts'][5]
        old = {k:original[k] for k in ('attempt','child_seed','parameters')}
        stage = OUT/'city_attempt5_cpu_only_correction1'; stage.mkdir(exist_ok=False)
        ep,sources,geo = make_recipe(policy,plan,people,3,'urban_street_assets',stage)
        for p in placements:
            p['support_checks']['road_object_names']=geo['road_object_names']
            p['support_checks']['road_object_prefixes']=geo['road_object_prefixes']
        result = preflight_team(ep,sources,assets,placements,[old],geo,stage/'prepared',
            max_roi_pixels=7_000_000,projected_occlusion_policy='diagnostic_only')
        write_new(OUT/'city_attempt5_cpu_cost.json',dict(seconds=time.monotonic()-t0,
                  status='passed' if result else 'rejected', old_attempt=5))
    print(json.dumps(dict(action=args.action, seconds=time.monotonic()-t0)))


if __name__ == '__main__': main()
