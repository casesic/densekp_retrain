"""Freeze bounded road scene recipes; rendering is a separate explicit step.

The asset pool describes accepted *bound combinations*, not interchangeable
body, clothing, and groom parts. No renderer or model is imported here.
"""
import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np

from densekp_retrain.data.person_time import map_timeline

VERSION='seeded_scene_modes_v1'
VIEW_VERSION='seeded_viewpoints_v1'
MODES={'single_cyclist_boundary_pass':1,'mixed_walk_cycle_crossing':2}
INTERNAL_LICENSE_STATES={'unknown','internal_shared_source_only'}


def load_pool(path):
    """An optional small viewpoint overlay keeps the accepted asset pool frozen."""
    path=Path(path)
    pool=json.loads(path.read_text())
    if pool.get('schema')=='seeded_viewpoint_pool_overlay_v1':
        base=Path(pool['base_pool_ref'])
        if not base.is_absolute():base=path.parent/base
        accepted=json.loads(base.read_text())
        accepted['viewpoint_ranges']=pool['viewpoint_ranges']
        accepted['viewpoint_range_basis']=pool['range_basis']
        return accepted
    return pool


def source_rows(asset):
    if asset.get('timeline_ref'):
        return json.loads(Path(asset['timeline_ref']).read_text())
    with np.load(asset['parameters']) as pars: ids=pars['source_frame_ids'].tolist()
    return [dict(asset_frame=i,source_frame=int(f),source_time_tick=int(f),time_base=[1,30]) for i,f in enumerate(ids)]


def valid_origins(asset,clock):
    rows=source_rows(asset)
    return [i for i in range(len(rows)) if _origin_works(rows,clock,i)]


def _origin_works(rows,clock,i):
    try:map_timeline(rows,clock,origin_record=i)
    except ValueError:return False
    return True


def make_config(pool,mode,master_seed,episode_index,requested_view_class=None,coverage_goal=None):
    if mode not in MODES:raise ValueError('unsupported_scene_mode')
    if type(master_seed)is not int or master_seed<0 or type(episode_index)is not int or episode_index<0:raise ValueError('invalid_seed_or_index')
    riders=pool['riders']; walker=pool['walker']
    if not riders or any(x.get('accepted_internal') is not True for x in riders+[walker]):raise ValueError('unaccepted_asset_combination')
    if any(x.get('license','unknown') not in INTERNAL_LICENSE_STATES for x in riders+[walker]):
        raise ValueError('asset_license_not_permitted_even_for_internal_trial')
    if len({x['asset_id'] for x in riders})!=len(riders):raise ValueError('duplicate_asset_id')
    version=VIEW_VERSION if requested_view_class is not None else VERSION
    if requested_view_class is not None:
        from densekp_retrain.data.viewpoint import VIEW_CLASSES
        if requested_view_class not in VIEW_CLASSES or 'viewpoint_ranges' not in pool:
            raise ValueError('unsupported_viewpoint_request')
    if coverage_goal is not None:
        if coverage_goal not in ('edge_coverage','boundary_transition') or (mode=='mixed_walk_cycle_crossing' and coverage_goal=='boundary_transition'):
            raise ValueError('invalid_coverage_goal')
    asset_rng=random.Random(f'{version}|{mode}|{master_seed}|{episode_index}|asset')
    layout_rng=random.Random(f'{version}|{mode}|{master_seed}|{episode_index}|layout')
    camera_rng=random.Random(f'{version}|{mode}|{master_seed}|{episode_index}|camera')
    rider=riders[asset_rng.randrange(len(riders))]
    count=MODES[mode]
    refs=[rider['reference_recipe']]+([walker['reference_recipe']] if count==2 else [])
    assets=[rider]+([walker] if count==2 else [])
    clock=[dict(frame_index=k,episode_time_tick=k,time_base=[1,10]) for k in range(31)]
    origin_lists=[]
    for x in assets:
        manifest=json.loads(Path(x['asset_manifest']).read_text())
        if manifest['asset_id']!=x['asset_id']:raise ValueError('asset_manifest_id_conflict')
        if manifest.get('available_combinations_verified') is not True:raise ValueError('unverified_bound_combination')
        eligible=valid_origins(manifest,clock)
        if not eligible:raise ValueError('asset_too_short_for_31_times:'+x['asset_id'])
        origin_lists.append(eligible)
    origins=[indices[asset_rng.randrange(len(indices))] for indices in origin_lists]
    episode_seed=layout_rng.getrandbits(63)
    episode_id=f'{mode}{"_"+requested_view_class if requested_view_class else ""}{"_"+coverage_goal if coverage_goal else ""}_s{master_seed}_e{episode_index:02d}'
    cfg=copy.deepcopy(json.loads(Path(rider['base_config']).read_text()))
    cfg.update(schema='road_pass_mixed_sources_v1',mode='road_pass',scene_mode=mode,episode_id=episode_id,
        master_seed=episode_seed,root_master_seed=master_seed,episode_child_seed=episode_seed,
        episode_index=episode_index,planning_seeds=[episode_seed],attempt_limit=12,
        reference_recipes=refs,source_origin_records=origins,
        parameter_ranges=copy.deepcopy(rider['parameter_ranges'][mode]),
        fixed_scene_rotation=rider['fixed_scene_rotation'],placement_height_m=rider['placement_height_m'],
        support_checks=copy.deepcopy(rider['support_checks']),
        roi_planning_policy='per_person_square_union_v1',observation_policy='record_unavailable_v1',
        max_union_roi_pixels=7_000_000,
        seed_algorithm_version=version,asset_pool_order=[x['asset_id'] for x in riders],
        selected_asset_id=rider['asset_id'],selected_asset_manifest=rider['asset_manifest'],
        selected_source_groups=[x['source_group'] for x in assets],
        source_group_policy='all_people_in_episode; seeds_are_not_independent_source_splits',
        permitted_people=count)
    cfg.pop('explicit_candidate_parameters',None)
    cfg.pop('separation_policy',None)
    if any(not isinstance(v,list) or len(v)!=2 or not np.isfinite(v).all() or v[0]>v[1]
           for v in cfg['parameter_ranges'].values()):raise ValueError('invalid_bounded_parameter_range')
    if count==2:
        cfg['walker_fixed_rotation_deg']=walker['fixed_rotation_deg']
        cfg['separation_policy']='aabb_then_vertex_3d_v1'
    else:
        cfg.pop('walker_fixed_rotation_deg',None)
    if requested_view_class is not None:
        policy=pool['viewpoint_ranges'][mode][requested_view_class]
        cfg['parameter_ranges'].update(copy.deepcopy(policy['layout_ranges']))
        cfg['requested_view_class']=requested_view_class
        cfg['camera_parameter_ranges']=copy.deepcopy(policy['camera_ranges'])
        cfg['camera_seed']=camera_rng.getrandbits(63)
        cfg['travel_direction_options']=list(policy['travel_direction_options'])
        cfg['crossing_transition']=policy.get('crossing_transition','either')
        cfg['viewpoint_policy_version']=VIEW_VERSION
        if sorted(cfg['travel_direction_options']) not in ([-1], [1], [-1,1]):
            raise ValueError('invalid_travel_direction_options')
        if cfg['crossing_transition'] not in ('enter_valid','exit_valid','either'):
            raise ValueError('invalid_crossing_transition')
        if any(not isinstance(v,list) or len(v)!=2 or not np.isfinite(v).all() or v[0]>v[1]
               for v in cfg['camera_parameter_ranges'].values()):
            raise ValueError('invalid_camera_parameter_range')
        if set(cfg['camera_parameter_ranges'])!={'origin_x_m','origin_y_m','yaw_deg'}:
            raise ValueError('invalid_camera_parameter_keys')
    if coverage_goal is not None:cfg['coverage_goal']=coverage_goal
    record=dict(schema=version,mode=mode,master_seed=master_seed,episode_index=episode_index,
        episode_id=episode_id,episode_child_seed=episode_seed,asset_selection_seed_rule=f'{version}|{mode}|{master_seed}|{episode_index}|asset',
        layout_seed_rule=f'{version}|{mode}|{master_seed}|{episode_index}|layout',
        camera_seed_rule=f'{version}|{mode}|{master_seed}|{episode_index}|camera' if requested_view_class else None,
        requested_view_class=requested_view_class,
        selected_asset_id=rider['asset_id'],selected_asset_manifest=rider['asset_manifest'],
        source_origin_records=origins,eligible_origin_counts=[len(x) for x in origin_lists],
        asset_pool_order=[x['asset_id'] for x in riders],source_groups=[x['source_group'] for x in assets],
        sampled_dimensions=list(cfg['parameter_ranges']),candidate_limit=12,
        candidate_seed_policy='separate layout stream; each candidate has a child seed',
        render_authorized_by_this_entry=False)
    if coverage_goal is not None:record['coverage_goal']=coverage_goal
    return cfg,record


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config',type=Path,required=True,help='Verified bound-combination asset pool and mode ranges')
    ap.add_argument('--mode',choices=sorted(MODES),required=True)
    ap.add_argument('--master-seed',type=int,required=True)
    ap.add_argument('--episode-count',type=int,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allowed-source-group',action='append')
    ap.add_argument('--view-classes',help='Comma-separated per-episode quota; deterministically shuffled by master seed')
    ap.add_argument('--coverage-goal',choices=['edge_coverage','boundary_transition'],help='Explicit physical coverage objective; old mode defaults remain unchanged')
    args=ap.parse_args()
    if not 1<=args.episode_count<=8:ap.error('episode_count must be 1..8; rendering remains separately authorized')
    pool=load_pool(args.config)
    view_classes=None
    if args.view_classes:
        from densekp_retrain.data.viewpoint import VIEW_CLASSES
        view_classes=args.view_classes.split(',')
        if len(view_classes)!=args.episode_count or any(v not in VIEW_CLASSES for v in view_classes):
            ap.error('view-classes must give one supported class per episode')
        random.Random(f'{VIEW_VERSION}|{args.mode}|{args.master_seed}|quota').shuffle(view_classes)
    allowed=set(args.allowed_source_group or pool['internal_source_groups'])
    pool['riders']=[x for x in pool['riders'] if x['source_group'] in allowed and x.get('accepted_internal') is True
                    and x.get('license','unknown') in INTERNAL_LICENSE_STATES]
    if not pool['riders'] or (args.mode=='mixed_walk_cycle_crossing' and pool['walker']['source_group'] not in allowed):
        ap.error('no accepted asset after source-group filter')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    entries=[]
    for i in range(args.episode_count):
        cfg,record=make_config(pool,args.mode,args.master_seed,i,
                               requested_view_class=view_classes[i] if view_classes else None,
                               coverage_goal=args.coverage_goal)
        ep=out/cfg['episode_id'];ep.mkdir()
        (ep/'config.json').write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
        (ep/'seed_record.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
        entries.append(record)
    (out/'index.json').write_text(json.dumps(dict(schema=VIEW_VERSION if view_classes else VERSION,mode=args.mode,master_seed=args.master_seed,
        requested_episode_count=args.episode_count,asset_pool_ref=str(args.config.resolve()),
        allowed_source_groups=sorted(allowed),requested_view_quota=view_classes,episodes=entries),indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(episodes=[x['episode_id'] for x in entries],output=str(out))))


if __name__=='__main__':main()
