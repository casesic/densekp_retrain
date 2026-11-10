#!/usr/bin/env python3
"""Seed compatible components, then explicitly build and preflight road episodes.

The plan action is CPU-only: it calls A2's public plan contract, never build,
SMPL-X construction, Blender, or GPU discovery. Existing bound-asset seeding
continues through seed_scene_modes.py unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT)]
from densekp_retrain.assets import composition as a2
from densekp_retrain.data.dynamic_scene import adapt_proposal
from densekp_retrain.data.mixed_road_pass import plan_mixed, preflight_mixed
from tools.experiments.seed_scene_modes import load_pool, valid_origins, make_config

SCHEMA = 'seeded_asset_composition_a1_v1'
OUTPUT = ROOT / 'outputs/experiments/seeded_asset_composition_a1_20261010'
CONFIG = ROOT / 'configs/seeded_asset_composition_a1_20261010.json'
CLOCK = [dict(frame_index=k, episode_time_tick=k, time_base=[1, 10]) for k in range(31)]


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def load_policy(path=CONFIG):
    policy = json.loads(Path(path).read_text())
    if policy['schema'] != SCHEMA or policy['plan_count'] != 8 or policy['production_quota'] != 2:
        raise ValueError('unsupported_bounded_component_policy')
    if policy['view_quota'] != ['front', 'back'] or policy['coverage_goal'] != 'edge_coverage':
        raise ValueError('unsupported_scene_quota')
    if policy.get('placement_policy_version','heading_mean_v1') not in ('heading_mean_v1','heading_minimax_fixed_yaw_v2'):
        raise ValueError('unknown_placement_policy')
    return policy


def stable_rng(policy, episode, stream):
    return random.Random(f"{policy['algorithm_version']}|{policy['master_seed']}|{episode}|{stream}")


def allowed_components(policy, registry, *, mode='single_cyclist_boundary_pass'):
    """Source and representation gates run before a component is sampled."""
    allow = policy['allowed']
    for category in ('bodies','motions','clothing','hair'):
        unknown=set(allow[category])-set(registry[category])
        if unknown:raise ValueError('unregistered_component_in_allowed_pool:' + ','.join(sorted(unknown)))
    if mode not in ('single_cyclist_boundary_pass', 'mixed_walk_cycle_crossing'):
        raise ValueError('unsupported_scene_mode')
    if mode == 'mixed_walk_cycle_crossing':
        walker = load_pool(ROOT / policy['legacy_pool'])['walker']
        if walker['source_group'] not in allow['walker_source_groups']:
            raise ValueError('walker_source_group_excluded')
    groups = {}
    specs = [('bodies', 'character_families', 'character_family'),
             ('motions', 'motion_source_groups', 'source_group')]
    for category, source_allow, source_key in specs:
        groups[category] = [key for key in sorted(allow[category])
                            if key in registry[category] and registry[category][key][source_key] in allow[source_allow]]
    bad_root=[key for key in groups['motions']
              if registry['motions'][key].get('root_mode') != policy['compatibility']['root_policy']]
    if bad_root:raise ValueError('unsupported_root_ownership:' + ','.join(bad_root))
    groups['motions'] = [key for key in groups['motions']
                         if registry['motions'][key].get('license') in allow['motion_licenses']]
    for category, kind in [('clothing', 'clothing_kind'), ('hair', 'hair_kind')]:
        unsupported=[key for key in sorted(allow[category])
                     if registry[category][key].get('kind') != policy['compatibility'][kind]]
        if unsupported:raise ValueError('unsupported_component_representation:' + ','.join(unsupported))
        groups[category] = sorted(allow[category])
    if any(not value for value in groups.values()):
        raise ValueError('empty_source_or_compatible_component_pool')
    return groups


def motion_origins(registry, motion_id):
    motion = registry['motions'][motion_id]
    rows = a2.read(a2.resolve(ROOT, motion['timeline']))
    return valid_origins({'timeline_ref': str(a2.resolve(ROOT, motion['timeline']))}, CLOCK), rows


def sampled_selection(policy, registry, groups, episode, earlier=()):
    """Independent shuffled component orders, conditional on frozen quotas.

    This is deterministic conditional sampling, not a whitelist of completed
    four-part assets and not a claim of uniform samples over all combinations.
    """
    orders = {}
    for category in ('bodies', 'motions', 'clothing', 'hair'):
        values = list(groups[category])
        stable_rng(policy, episode, category).shuffle(values)
        orders[category] = values
    # Each stream contributes its own rank. A separate order stream resolves
    # equal rank totals without changing any component's random stream.
    tuples = [(b, m, c, h) for b in orders['bodies'] for m in orders['motions']
              for c in orders['clothing'] for h in orders['hair']]
    rank = {category: {value: i for i, value in enumerate(values)} for category, values in orders.items()}
    tie = stable_rng(policy, episode, 'conditional_order')
    ranked = sorted(((sum(rank[cat][val] for cat, val in zip(('bodies','motions','clothing','hair'), combo)),
                      tie.random(), combo) for combo in tuples))
    reasons = []
    for _, _, combo in ranked:
        b, m, c, h = combo
        if episode == 1 and earlier and (b == earlier[0]['selected_components']['body_id'] or m == earlier[0]['selected_components']['motion_id']):
            reasons.append('second_quota_requires_different_body_and_motion'); continue
        if episode == 1 and earlier and c.endswith(b + '_simple') and h.endswith(b + '_groom') and not (
                earlier[0]['selected_components']['clothing_id'].split('_')[0] != earlier[0]['selected_components']['body_id'] or
                earlier[0]['selected_components']['hair_id'].split('_')[0] != earlier[0]['selected_components']['body_id']):
            reasons.append('cross_body_appearance_quota'); continue
        origins, _ = motion_origins(registry, m)
        if not origins:
            reasons.append('motion_too_short_for_3s_10Hz'); continue
        selection = dict(body_id=b, motion_id=m, clothing_id=c, hair_id=h)
        origin = origins[stable_rng(policy, episode, 'time').randrange(len(origins))]
        return selection, origin, orders, dict(conditional_rejections={k: reasons.count(k) for k in sorted(set(reasons))},
                                               eligible_origins=len(origins), candidate_combinations=len(ranked))
    raise ValueError('no_compatible_selection_after_source_and_quota_filters')


def freeze_plans(policy, registry, output_root=OUTPUT):
    groups = allowed_components(policy, registry)
    result = []
    for episode in range(policy['plan_count']):
        selection, origin, orders, evidence = sampled_selection(policy, registry, groups, episode, result)
        motion = registry['motions'][selection['motion_id']]
        rows = a2.read(a2.resolve(ROOT, motion['timeline']))
        recipe = dict(selection, window=[0, len(rows)], root_policy=policy['compatibility']['root_policy'],
                      output_dir=str(Path(output_root).resolve() / 'composition_cache'))
        resolved = a2.plan(ROOT, registry, recipe)
        result.append(dict(schema=SCHEMA, episode_index=episode, master_seed=policy['master_seed'],
                           selected_components=selection, component_versions={k: resolved['components'][k]['version'] for k in selection_key_order()},
                           component_order=orders, source_group=resolved['source_group'], character_family=resolved['character_family'],
                           source_time_origin_record=origin, source_time_origin_tick=rows[origin]['source_time_tick'],
                           source_time_base=rows[origin]['time_base'], frame_count=resolved['frame_count'],
                           recipe=recipe, resolved=resolved, selection_evidence=evidence,
                           requested_view_class=policy['view_quota'][episode] if episode < policy['production_quota'] else None,
                           use_policy=policy['scene']['technical_candidate_use'] if episode < policy['production_quota'] else 'plan_only'))
    return result


def selection_key_order():
    return ('body', 'motion', 'clothing', 'hair')


def checked_build(plan, output_root):
    """A2 append-only build with an explicit verified-delivery hit path."""
    resolved = plan['resolved']
    ids = plan['selected_components']
    name = f"a1_{plan['episode_index']:02d}_{ids['body_id']}_{ids['motion_id']}_{ids['clothing_id']}_{ids['hair_id']}"
    out = Path(output_root) / 'builds' / name
    if out.exists():
        if a2.read(out / 'frozen_recipe.json') != resolved:
            raise ValueError('occupied_delivery_recipe_or_dependency_changed')
        manifest = a2.read(out / 'delivery_manifest.json')
        check = a2.read(out / 'a1_adapter_check.json')
        validation = a2.read(out / 'validation.json')
        if manifest['components'] != resolved['components'] or validation['status'] != 'numeric_passed' or not check['status'].startswith('passed'):
            raise ValueError('occupied_delivery_incomplete_or_invalid')
        for key in ('parameters','body_vertices','body_faces','body_R138','cloth_vertices','cloth_faces','groom_reference','head_transforms','timeline'):
            path = Path(manifest[key])
            if not path.exists(): raise ValueError('occupied_delivery_missing_persistent_file:' + key)
        n=resolved['frame_count']
        for key,tail in (('body_vertices',(10475,3)),('body_R138',(138,3)),('head_transforms',(4,4))):
            value=np.load(manifest[key],mmap_mode='r')
            if value.shape!=(n,*tail) or not np.isfinite(value).all():
                raise ValueError('occupied_delivery_geometry_invalid:' + key)
        cloth=np.load(manifest['cloth_vertices'],mmap_mode='r')
        if cloth.ndim!=3 or cloth.shape[0]!=n or cloth.shape[2]!=3 or not np.isfinite(cloth).all():
            raise ValueError('occupied_delivery_clothing_invalid')
        if len(a2.read(manifest['timeline']))!=n:
            raise ValueError('occupied_delivery_timeline_invalid')
        adapted = adapt_proposal(out / 'a1_proposal.json')
        return out, validation, adapted, 'delivery_verified_hit'
    built, validation = a2.build(ROOT, resolved, output_root, name)
    if built != out or validation['status'] != 'numeric_passed':
        raise ValueError('composition_build_not_verified')
    return out, validation, adapt_proposal(out / 'a1_proposal.json'), 'new_build'


def derive_rotation_and_height(asset, origin, placement_policy='heading_mean_v1'):
    """Fit one fixed yaw and height from the actual selected motion geometry."""
    from densekp_retrain.data.person_time import map_timeline
    rows = a2.read(asset['timeline_ref'])
    mapped = map_timeline(rows, CLOCK, origin_record=origin)
    ix = [row['asset_frame'] for row in mapped]
    body = np.load(asset['body_vertices'], mmap_mode='r')[ix]
    with np.load(asset['parameters'], allow_pickle=False) as z:
        orient = z['global_orient'][ix]
    from scipy.spatial.transform import Rotation
    local = Rotation.from_rotvec(orient).apply(np.array([0., 0., 1.]))
    # Existing model Y-up to world Z-up basis; yaw then aligns mean forward
    # with +Y. The existing road preflight checks all 31 headings <=25deg.
    axis = np.array([[1.,0.,0.],[0.,0.,-1.],[0.,1.,0.]])
    h = local @ axis[:2].T
    if (np.linalg.norm(h, axis=1)<1e-9).any():
        raise ValueError('degenerate_body_forward')
    angles=np.unwrap(np.arctan2(h[:,0],h[:,1]))
    if placement_policy=='heading_mean_v1':
        mean=np.mean(h / np.linalg.norm(h,axis=1)[:,None],axis=0)
        yaw=float(np.arctan2(mean[0],mean[1]))
    elif placement_policy=='heading_minimax_fixed_yaw_v2':
        # A mean yaw can fail the existing per-frame 25deg gate even when one
        # fixed yaw satisfies it. Center the actual finite angle interval.
        yaw=float((angles.min()+angles.max())/2)
    else:
        raise ValueError('unknown_placement_policy')
    turn = np.array([[np.cos(yaw),-np.sin(yaw),0.],[np.sin(yaw),np.cos(yaw),0.],[0.,0.,1.]])
    rotation = turn @ axis
    min_y = float(body[:,:,1].min())
    height = round(0.20 - min_y, 4)
    if not np.isfinite(height) or height < .8 or height > 1.8:
        raise ValueError('computed_fixed_riding_height_outside_engineering_bounds')
    clearance = np.array(body[:,:,1].min(axis=1)) + height
    if clearance.min() < .05 or clearance.max() > .7:
        raise ValueError('fixed_riding_height_body_clearance_outside_support')
    return rotation.tolist(), height, [float(clearance.min()), float(clearance.max())], mapped


def prepared_scene(plan, policy, output_root, *, stage_name=None):
    built, validation, adapted, cache_state = checked_build(plan, output_root / 'composition_cache')
    index = plan['episode_index']
    stage = output_root / (stage_name or f'episode_{index:02d}')
    stage.mkdir(exist_ok=False)
    adapted.update(components=plan['selected_components'], component_versions=plan['component_versions'],
                   resolved_recipe_ref=str(built / 'frozen_recipe.json'), composition_delivery_ref=str(built / 'delivery_manifest.json'),
                   technical_use_policy=plan['use_policy'], human_acceptance='not_inherited_new_combination')
    adapter = stage / 'asset_adapter.json'
    write(adapter, adapted)
    old_pool = load_pool(ROOT / policy['legacy_pool'])
    body = plan['selected_components']['body_id']
    template = next((r for r in old_pool['riders'] if r['character_family'] == 'cycling_' + body), old_pool['riders'][0])
    reference = a2.read(template['reference_recipe'])
    person = reference['people'][0]
    person.update(asset_id=adapted['asset_id'], asset_manifest=str(adapter), character_family=adapted['character_family'],
                  root_mode=adapted['root_mode'], support_policy=adapted['support_policy'],
                  outfit_policy='fixed_seed_selected_components_entire_episode',
                  component_ids=plan['selected_components'], composition_delivery_ref=str(built / 'delivery_manifest.json'))
    for key in ('scene_from_model','model_to_world','asset_frames','source_frames'):
        person.pop(key, None)
    ref = stage / 'reference_recipe.json'
    write(ref, reference)
    placement_policy=('heading_minimax_fixed_yaw_v2' if stage_name=='episode_01_correction1'
                      else policy.get('placement_policy_version','heading_mean_v1'))
    rotation, height, clearance, mapped = derive_rotation_and_height(adapted, plan['source_time_origin_record'],placement_policy)
    viewpoint = load_pool(ROOT / policy['viewpoint_pool'])
    rider = copy.deepcopy(template)
    rider.update(asset_id=adapted['asset_id'], asset_manifest=str(adapter), reference_recipe=str(ref),
                 source_group=adapted['source_group'], accepted_internal=True, license='internal_shared_source_only',
                 fixed_scene_rotation=rotation, placement_height_m=height,
                 support_checks=dict(template['support_checks'], body_min_clearance_m=[.05,.7],
                                     pelvis_height_above_road_m=[max(.5,height-.05),height+.05]))
    # This is an explicit integration-only technical-use policy. The adapter
    # retains human_acceptance=not_inherited_new_combination.
    pool = dict(riders=[rider], walker=old_pool['walker'], viewpoint_ranges=viewpoint['viewpoint_ranges'])
    cfg, seed_record = make_config(pool, policy['mode'], policy['master_seed'], index,
                                    requested_view_class=plan['requested_view_class'], coverage_goal=policy['coverage_goal'])
    cfg['source_origin_records'] = [plan['source_time_origin_record']]
    cfg['placement_policy_version']=placement_policy
    cfg['component_provenance'] = dict(selection=plan['selected_components'], versions=plan['component_versions'],
                                       resolved_recipe_ref=str(built/'frozen_recipe.json'), delivery_ref=str(built/'delivery_manifest.json'),
                                       source_group=adapted['source_group'], character_family=adapted['character_family'],
                                       selected_source_origin_record=plan['source_time_origin_record'],
                                       use_policy=plan['use_policy'])
    cfg['episode_id'] = cfg['episode_id'].replace('_s', '_components_s', 1)
    seed_record.update(selected_asset_id=adapted['asset_id'], selected_asset_manifest=str(adapter),
                       source_origin_records=cfg['source_origin_records'], component_provenance=cfg['component_provenance'])
    write(stage/'config.json', cfg)
    write(stage/'seed_record.json', seed_record)
    write(stage/'placement_derivation.json', dict(rotation=rotation, fixed_height_m=height,
          body_clearance_m=clearance, selected_source_frames=[row['source_frame'] for row in mapped],
          method=placement_policy+'; actual selected body and orientation; one fixed yaw and height; heading checked across all times'))
    write(stage/'build_record.json', dict(state=cache_state, validation=validation, delivery=str(built),
          body_reference=plan['selected_components']['body_id'], clothing_reference=adapted['components']['clothing_id'] if 'components' in adapted else None))
    return stage, cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('plan','prepare','correct-second-heading'))
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    policy = load_policy(args.config)
    out = args.output.resolve()
    if args.action == 'plan':
        out.mkdir(parents=True, exist_ok=True)
        if (out/'frozen_plans.json').exists(): raise FileExistsError('frozen_plans_exist')
        registry = a2.read(ROOT / policy['registry'])
        plans = freeze_plans(policy, registry, out)
        write(out/'frozen_policy.json', policy)
        write(out/'frozen_plans.json', plans)
        print(json.dumps(dict(output=str(out), plans=len(plans), selections=[x['selected_components'] for x in plans])))
        return
    plans = a2.read(out/'frozen_plans.json')
    if a2.read(out/'frozen_policy.json') != policy: raise ValueError('frozen_policy_changed')
    selected=plans[:policy['production_quota']] if args.action=='prepare' else [plans[1]]
    for plan in selected:
        stage, cfg = prepared_scene(plan, policy, out,
                                    stage_name='episode_01_correction1' if args.action=='correct-second-heading' else None)
        candidate = plan_mixed(cfg)
        write(stage/'planned_road_pass.json', candidate)
        (stage/'preflight').mkdir()
        ep = preflight_mixed(candidate, stage/'preflight')
        print(json.dumps(dict(episode=cfg['episode_id'], preflight='passed' if ep else 'rejected',
                              prepared=str(stage/'preflight'/cfg['episode_id']))), flush=True)


if __name__ == '__main__': main()
