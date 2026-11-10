"""Metadata-only first-trial planning; deliberately cannot build or render."""
from __future__ import annotations

import json
import random
import copy
from pathlib import Path

from densekp_retrain.assets.source_pools import EPISODE_SCHEMA, validate_batch
from densekp_retrain.data.box_augmentation import DEFAULT_POLICY

VERSION = 'first_dataset_seed_recipe_v1'
MODES = {'single_cyclist_boundary_pass': (1, 1),
         'mixed_walk_cycle_crossing': (2, 2),
         'dispersed_cyclists': (2, 5),
         'city_t_junction': (3, 3)}


def stream(seed, episode_index, name):
    return random.Random(f'{VERSION}|{seed}|{episode_index}|{name}')


def _choice(rng, values):
    values = list(values)
    if not values:
        raise ValueError('empty_allowed_choice')
    return values[rng.randrange(len(values))]


def plan(config, catalog, proposals, registry=None):
    if config.get('schema') != VERSION or len(config.get('episodes', [])) > 8:
        raise ValueError('invalid_first_dataset_config')
    proposal_id = config['source_proposal_id']
    catalog=copy.deepcopy(catalog);proposals=copy.deepcopy(proposals)
    proposal = proposals['proposals'][proposal_id]
    seed = config['master_seed']
    if type(seed) is not int:
        raise ValueError('invalid_master_seed')
    plans, source_input = [], []
    for index, spec in enumerate(config['episodes']):
        mode, environment, pool = spec['mode'], spec['environment'], spec['split_pool']
        if mode not in MODES or environment not in ('urban_street_assets', 'riverfront'):
            raise ValueError('unsupported_mode_or_environment')
        if mode == 'city_t_junction' and environment != 'urban_street_assets':
            raise ValueError('junction_requires_city')
        if pool not in ('train_candidate', 'val_candidate'):
            raise ValueError('invalid_split_pool')
        allowed_counts=list(spec['allowed_person_counts'])
        weights=spec.get('person_count_weights',[1]*len(allowed_counts))
        if (len(weights)!=len(allowed_counts) or not weights
                or any(type(w) not in (int,float) or w<0 for w in weights)
                or sum(weights)<=0):
            raise ValueError('invalid_person_count_weights')
        count=stream(seed,index,'count').choices(allowed_counts,weights=weights,k=1)[0]
        lo, hi = MODES[mode]
        if not lo <= count <= hi:
            raise ValueError('person_count_outside_mode')
        if spec.get('selection_mode','registered_combo')=='component_pool':
            if registry is None:raise ValueError('component_registry_required')
            cp=spec['component_pools']
            allowed_combos=[catalog['combinations'][cid] for cid in proposal[pool]]
            choices={
                'body_id':sorted(set(cp['body_ids']) & {c['body_id'] for c in allowed_combos}),
                'motion_id':sorted(set(cp['motion_ids']) & {c['motion_id'] for c in allowed_combos}),
                'clothing_id':sorted(set(cp['clothing_ids']) & set(registry['clothing'])),
                'hair_id':sorted(set(cp['hair_ids']) & set(registry['hair']))}
            chosen=[]
            for person_index in range(count):
                local_choices=dict(choices)
                if mode=='mixed_walk_cycle_crossing':
                    want_walk=person_index==1
                    local_choices['motion_id']=[mid for mid in choices['motion_id']
                                                if ('walk' in mid)==want_walk]
                selected={field:_choice(stream(seed,index,f'{field}:{person_index}'),values)
                          for field,values in local_choices.items()}
                motion=catalog['motions'][selected['motion_id']]
                if (registry['clothing'][selected['clothing_id']]['kind']!='simple_fixed_barycentric_normal_offset'
                        or registry['hair'][selected['hair_id']]['kind']!='static_curves_head_rigid'):
                    raise ValueError('component_representation_locked_or_unsupported')
                existing=next((cid for cid,c in catalog['combinations'].items()
                    if all(c.get(k)==v for k,v in selected.items()) and cid in proposal[pool]),None)
                if existing is not None:
                    chosen.append(existing);continue
                cid='planned_'+'_'.join(selected[k] for k in ('body_id','motion_id','clothing_id','hair_id'))
                if cid not in catalog['combinations']:
                    catalog['combinations'][cid]=dict(**selected,asset_window=[0,motion['clock']['frame_count']],
                        compatible=True,built=False,human_accepted_scope=None,scene_consumed=False,
                        authority_ref=None,license_status=motion['license_status'])
                    proposal[pool].append(cid)
                chosen.append(cid)
        else:
            if spec.get('selection_mode','registered_combo')!='registered_combo':
                raise ValueError('unknown_component_selection_mode')
            allowed = sorted(set(spec['allowed_combinations']) & set(proposal[pool]))
            cyclists = [cid for cid in allowed if catalog['motions'][catalog['combinations'][cid]['motion_id']]['root_mode'] == 'fixed_pelvis_scene_owns_progression']
            walkers = [cid for cid in allowed if cid not in cyclists]
            rng = stream(seed,index,'components'); rng.shuffle(cyclists); rng.shuffle(walkers)
            chosen = cyclists[:count if mode != 'mixed_walk_cycle_crossing' else 1]
            if mode == 'mixed_walk_cycle_crossing':
                chosen += walkers[:1]
        rejects = []
        if len(chosen) != count:
            rejects.append('insufficient_allowed_components')
        if mode == 'dispersed_cyclists' and len({catalog['combinations'][c]['motion_id'] for c in chosen}) < 3:
            rejects.append('fewer_than_three_distinct_cycling_motions')
        persons=[]
        side_bodies={catalog['combinations'][cid]['body_id'] for cid in proposal[pool]
                     if cid in catalog['combinations'] and not cid.startswith('planned_')}
        for person_index,cid in enumerate(chosen):
            combo=catalog['combinations'][cid]; motion=catalog['motions'][combo['motion_id']]
            window=combo['asset_window']; n=window[1]-window[0]
            # Approximate 3-second feasibility only. The exact integer-tick
            # nearest-record map is deferred to the existing person_time path.
            clock=motion.get('clock',{}); base=clock.get('time_base',[1,30]);
            nominal=clock.get('interval_ticks_max',1)*base[0]/base[1]
            required=int(3/nominal)+2 if nominal>0 else n+1
            last_start=max(window[0],window[1]-required)
            if n < required: rejects.append(f'window_too_short:{cid}')
            start=stream(seed,index,f'time:{person_index}').randint(window[0],last_start)
            appearance_parents={kind:catalog[kind][combo[key]].get('reference_body')
                for kind,key in (('clothing','clothing_id'),('hair','hair_id'))}
            appearance_review=any(parent not in side_bodies for parent in appearance_parents.values())
            persons.append(dict(person_id=person_index+1,role='occluder' if mode=='mixed_walk_cycle_crossing' and person_index==1 else 'target',
                combo_id=cid,**{k:combo[k] for k in ('body_id','motion_id','clothing_id','hair_id')},
                declared_source_group=motion['source_group'],asset_window=window,
                proposed_start_asset_frame=start, time_mapping_status='pending_exact_integer_timeline_check',
                combination_status='reuse_registered' if combo['built'] else 'build_required',
                appearance_parents=appearance_parents,appearance_parent_review_required=appearance_review))
        scene_id=f'{mode}_s{seed}_e{index:02d}'
        layout_rng=stream(seed,index,'layout'); camera_rng=stream(seed,index,'camera')
        rig_ids=spec['rig_ids'];
        if len(set(rig_ids))!=len(rig_ids) or not rig_ids: raise ValueError('invalid_rig_list')
        layout=dict(anchor_x_m=round(layout_rng.uniform(-2,2),6),anchor_y_m=round(layout_rng.uniform(0,5),6),
                    lane_offset_m=round(layout_rng.uniform(-.8,.8),6),speed_mps=round(layout_rng.uniform(1.4,2.3),6),
                    status='unverified_parameter_proposal')
        rigs=[dict(rig_id=rid,camera_seed=camera_rng.getrandbits(63),
                   rig_translation_x_m=round(camera_rng.uniform(-1.5,1.5),6),
                   rig_translation_y_m=round(camera_rng.uniform(-1.5,1.5),6),
                   rig_yaw_offset_deg=round(camera_rng.uniform(-15,15),6),
                   status='unverified_parameter_proposal') for rid in rig_ids]
        source_input.append(dict(schema=EPISODE_SCHEMA,proposal_id=proposal_id,pool=pool,
            scene_episode_id=scene_id,rig_id=rig_ids[0],persons=persons,expected_person_count=count))
        plans.append(dict(scene_episode_id=scene_id,mode=mode,environment=environment,split_pool=pool,
            master_seed=seed,episode_index=index,child_seed_keys=['count','components','time:<person>','layout','camera','box'],
            person_count=count,allowed_person_counts=allowed_counts,person_count_weights=weights,
            persons=persons,layout=layout,rigs=rigs,box_seed=stream(seed,index,'box').getrandbits(63),
            box_policy=DEFAULT_POLICY,frame_count=31,time_base=[1,10],
            planned_observation_slots=31*2*len(rigs)*count,
            plan_valid=not rejects,plan_rejections=rejects,geometry_checked=False,
            rendered=False,dataset_verified=False,production_status='pending_review_not_authorized'))
    validations=validate_batch(catalog,proposals,source_input)
    for item,result in zip(plans,validations):
        item['source_validation']=result
        item['source_eligible']=result['eligible_under_proposed_policy'] and not any(
            p['appearance_parent_review_required'] for p in item['persons'])
        item['asset_ready']=all(p['built'] and p['human_accepted_scope'] and p['scene_consumed'] for p in result['persons'])
        if not item['asset_ready']: item['production_status']='requires_asset_or_scene_or_acceptance_gate'
    return dict(schema=VERSION,review_status='pending_review',master_seed=seed,episodes=plans,
                source_proposal_id=proposal_id,geometry_checked=False,rendered=False,dataset_verified=False)


def read_catalog(root):
    base=Path(root)/'outputs/plans/asset_source_pools_a2_20261010'
    return json.loads((base/'source_catalog.json').read_text()),json.loads((base/'pool_proposals.json').read_text())
