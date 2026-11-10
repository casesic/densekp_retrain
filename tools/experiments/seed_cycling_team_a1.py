#!/usr/bin/env python3
"""Seed distinct cycling components, then explicitly build and preflight teams.

The plan action only reads the registry and original timelines.  Build and
render are separate explicit actions; no global RNG or directory-order choice.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.assets import composition as a2
from densekp_retrain.data.dynamic_scene import adapt_proposal
from densekp_retrain.data.cycling_team import VERSION, candidate_parameters, team_recipe, preflight_team
from tools.experiments.seed_asset_composition_a1 import allowed_components, motion_origins, derive_rotation_and_height, checked_build
from tools.experiments.seed_scene_modes import load_pool

CONFIG=ROOT/'configs/seeded_cycling_team_a1_20261010.json'
OUTPUT=ROOT/'outputs/experiments/seeded_cycling_team_a1_20261010'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def write_or_verify(path,value):
    path=Path(path)
    if path.exists():
        if json.loads(path.read_text())!=value:raise ValueError('partial_output_conflict:'+str(path))
    else:write(path,value)


def load_config(path):
    value=json.loads(Path(path).read_text())
    if value['schema']!=VERSION or type(value['master_seed']) is not int:
        raise ValueError('invalid_frozen_team_schema_or_seed')
    if [(x['environment_id'],x['person_count']) for x in value['episodes']]!=[('riverfront',2),('urban_street_assets',3)]:
        raise ValueError('unsupported_team_quota')
    if value['component_candidate_limit']>4 or value['layout_candidate_limit']>12 or value['max_roi_pixels']>7_000_000:
        raise ValueError('team_candidate_or_ROI_limit')
    return value


def _ranked_slot(policy,registry,groups,slot,used):
    orders={}
    for category in ('bodies','motions','clothing','hair'):
        order=list(groups[category]);random.Random(f'{VERSION}|{policy["master_seed"]}|{slot}|{category}').shuffle(order)
        orders[category]=order
    ranks={key:{value:i for i,value in enumerate(order)} for key,order in orders.items()}
    combos=[]
    for body in orders['bodies']:
        for motion in orders['motions']:
            for clothing in orders['clothing']:
                for hair in orders['hair']:
                    score=sum(ranks[key][v] for key,v in zip(('bodies','motions','clothing','hair'),(body,motion,clothing,hair)))
                    combos.append((score,body,motion,clothing,hair))
    tie=random.Random(f'{VERSION}|{policy["master_seed"]}|{slot}|tie')
    ranked=sorted((score,tie.random(),b,m,c,h) for score,b,m,c,h in combos)
    rejected=[]
    for _,_,b,m,c,h in ranked:
        group=registry['motions'][m]['source_group']
        if b in used['bodies'] or group in used['source_groups'] or c in used['clothing']:
            rejected.append('identity_motion_group_or_clothing_already_used');continue
        origins,rows=motion_origins(registry,m)
        if not origins:rejected.append('motion_too_short_for_3s');continue
        origin=origins[random.Random(f'{VERSION}|{policy["master_seed"]}|{slot}|time').randrange(len(origins))]
        chosen=dict(body_id=b,motion_id=m,clothing_id=c,hair_id=h)
        recipe=dict(chosen,window=[0,len(rows)],root_policy=policy['compatibility']['root_policy'])
        return chosen,origin,recipe,orders,dict(conditional_rejections={x:rejected.count(x) for x in set(rejected)},eligible_origins=len(origins))
    raise ValueError('no_distinct_compatible_team_components')


def plan(policy,registry):
    groups=allowed_components(policy,registry)
    used=dict(bodies=set(),source_groups=set(),clothing=set());people=[]
    for slot in range(1,4):
        selection,origin,recipe,orders,evidence=_ranked_slot(policy,registry,groups,slot,used)
        source_group=registry['motions'][selection['motion_id']]['source_group']
        used['bodies'].add(selection['body_id']);used['source_groups'].add(source_group);used['clothing'].add(selection['clothing_id'])
        people.append(dict(person_id=slot,selection=selection,source_group=source_group,
                           character_family=registry['bodies'][selection['body_id']]['character_family'],
                           source_origin_record=origin,recipe=recipe,component_orders=orders,
                           selection_evidence=evidence,
                           substreams={category:f'{VERSION}|{policy["master_seed"]}|{slot}|{category}' for category in ('bodies','motions','clothing','hair','time')}))
    return dict(schema=VERSION,master_seed=policy['master_seed'],people=people,
                episodes=[dict(episode_index=i,environment_id=x['environment_id'],person_count=x['person_count'],
                               person_ids=list(range(1,x['person_count']+1))) for i,x in enumerate(policy['episodes'])],
                candidate_limit=policy['component_candidate_limit'],selection_policy='separate shuffled component streams, conditional distinct family/source/clothing; first eligible tuple',
                source_policy='all people including occluders belong to the same integration-only episode')


def _cache_or_build(person,policy,registry,output):
    ids=person['selection'];base_recipe=person['recipe']
    old=ROOT/'outputs/experiments/seeded_asset_composition_a1_20261010/composition_cache'
    for directory in sorted((old/'builds').glob('*')):
        frozen=directory/'frozen_recipe.json'
        if not frozen.is_file():continue
        raw=a2.read(frozen)
        if [raw['components'][key]['id'] for key in ('body','motion','clothing','hair')]!=[ids[key+'_id'] for key in ('body','motion','clothing','hair')]:continue
        resolved=a2.plan(ROOT,registry,dict(base_recipe,output_dir=str(old)))
        if raw!=resolved:raise ValueError('existing_matching_component_dependency_changed')
        index=int(directory.name.split('_')[1]);fake=dict(episode_index=index,selected_components=ids,resolved=resolved)
        built,validation,adapted,state=checked_build(fake,old)
        return built,validation,adapted,'verified_previous_a1_'+state
    cache=output/'composition_cache'
    resolved=a2.plan(ROOT,registry,dict(base_recipe,output_dir=str(cache)))
    fake=dict(episode_index=person['person_id']+10,selected_components=ids,resolved=resolved)
    return checked_build(fake,cache)


def prepare_assets(policy,plan_record,registry,output):
    stage=output/'team_assets';stage.mkdir(parents=True,exist_ok=True)
    old_pool=load_pool(ROOT/policy['legacy_pool'])
    people=[];assets=[];placements=[];builds=[]
    for entry in plan_record['people']:
        i=entry['person_id'];built,validation,adapted,state=_cache_or_build(entry,policy,registry,output)
        adapted.update(components=entry['selection'],composition_delivery_ref=str(built/'delivery_manifest.json'),
                       human_acceptance='not_inherited_new_combination',technical_use_policy='integration_only_numeric_passed')
        asset_ref=stage/f'person{i}_asset_adapter.json';write_or_verify(asset_ref,adapted)
        template=next((x for x in old_pool['riders'] if x['character_family']==entry['character_family']),None)
        if template is None and entry['character_family']=='cycling_M02':
            root=ROOT/'outputs/experiments/seeded_asset_composition_a1_20261010/episode_01_correction1'
            prior=next((root/'preflight').glob('*/recipe.json'))
            previous=json.loads(prior.read_text())['people'][0]
            template=dict(reference_recipe=str(root/'reference_recipe.json'),support_checks=previous['support_checks'])
        if template is None:raise ValueError('missing_verified_body_scene_template')
        reference=json.loads(Path(template['reference_recipe']).read_text())
        person=copy.deepcopy(reference['people'][0])
        for key in ('scene_from_model','model_to_world','asset_frames','source_frames'):
            person.pop(key,None)
        person.update(asset_id=adapted['asset_id'],asset_manifest=str(asset_ref),
                      character_family=adapted['character_family'],root_mode=adapted['root_mode'],
                      support_policy=adapted['support_policy'],component_ids=entry['selection'],
                      source_group=adapted['source_group'],composition_delivery_ref=str(built/'delivery_manifest.json'),
                      outfit_policy='fixed_seed_selected_components_entire_episode',
                      reference_recipe=template['reference_recipe'])
        rotation,height,clearance,mapped=derive_rotation_and_height(adapted,entry['source_origin_record'],
                                                                   placement_policy='heading_minimax_fixed_yaw_v2')
        support=copy.deepcopy(template['support_checks']);support.update(body_min_clearance_m=[.05,.7],
                    pelvis_height_above_road_m=[height-.05,height+.05])
        people.append(person);assets.append(adapted)
        placements.append(dict(rotation=rotation,height_m=height,support_checks=support,body_clearance_m=clearance))
        builds.append(dict(person_id=i,selection=entry['selection'],source_group=entry['source_group'],
                           origin_record=entry['source_origin_record'],cache_state=state,
                           delivery=str(built/'delivery_manifest.json'),technical_validation=validation['status'],
                           asset_manifest=str(asset_ref),selected_source_frames=[x['source_frame'] for x in mapped]))
    if (stage/'builds.json').exists():
        previous=json.loads((stage/'builds.json').read_text())
        if len(previous)!=len(builds) or any({k:v for k,v in a.items() if k!='cache_state'} !=
                                             {k:v for k,v in b.items() if k!='cache_state'}
                                             for a,b in zip(previous,builds)):
            raise ValueError('partial_build_evidence_conflict')
    else:write(stage/'builds.json',builds)
    write_or_verify(stage/'placements.json',placements)
    return people,assets,placements


def prepare_episode(policy,plan_record,people,assets,placements,index,output,*,stage_suffix=''):
    quota=plan_record['episodes'][index];count=quota['person_count'];env=quota['environment_id']
    stage=output/(('riverfront_2' if index==0 else 'urban_3')+stage_suffix);stage.mkdir(parents=True,exist_ok=False)
    base=json.loads(Path(people[0]['reference_recipe']).read_text())
    episode_id=f'cycling_team_boundary_pass_{env}_s{policy["master_seed"]}_e{index:02d}'
    ep,sources=team_recipe(base,copy.deepcopy(people[:count]),
                           [x['source_origin_record'] for x in plan_record['people'][:count]],episode_id,env)
    ep['master_seed']=policy['master_seed'];ep['component_algorithm_version']=VERSION
    ep['component_provenance']=[dict(person_id=i+1,selection=x['selection'],source_group=x['source_group'],
                                     origin_record=x['source_origin_record']) for i,x in enumerate(plan_record['people'][:count])]
    ep['occlusion_metric']=copy.deepcopy(policy['occlusion_metric']);ep['max_union_roi_pixels']=policy['max_roi_pixels']
    template=ROOT/policy['environments'][env]['scene_config']
    scene=json.loads(template.read_text());ep['scene']=str(stage/'scene_config.json')
    if env=='riverfront':
        render_env=policy['environments'][env]
        scene['scene_settings_after_configure_render']=render_env['render_settings']
    blend=ROOT/scene['blend_path'];stat=blend.stat()
    scene['expected_blend_stat']=dict(size_bytes=stat.st_size,mtime_ns=stat.st_mtime_ns)
    write(stage/'scene_config.json',scene)
    geo=copy.deepcopy(policy['environments'][env]);geo.update(environment_id=env,
          baseline_m=ep['rig']['baseline_m'],rig_origin_z_m=ep['rig']['origin_m'][2],
          physical_masks=policy['physical_masks'])
    for p,placement in zip(ep['people'],placements):
        placement['support_checks']['road_object_names']=geo['road_object_names']
        placement['support_checks']['road_object_prefixes']=geo['road_object_prefixes']
    candidates=candidate_parameters(policy['master_seed'],index,count,geo['parameter_ranges'],
                                     limit=policy['layout_candidate_limit'])
    write(stage/'frozen_candidates.json',candidates)
    prepared=stage/'prepared'
    return preflight_team(ep,sources,assets[:count],placements[:count],candidates,geo,prepared,
                          max_roi_pixels=policy['max_roi_pixels'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=('plan','prepare'))
    ap.add_argument('--config',type=Path,default=CONFIG);ap.add_argument('--output',type=Path,default=OUTPUT)
    args=ap.parse_args();policy=load_config(args.config);output=args.output.resolve()
    registry=a2.read(ROOT/policy['registry'])
    if args.action=='plan':
        output.mkdir(parents=True,exist_ok=True)
        frozen=plan(policy,registry)
        write(output/'frozen_policy.json',policy);write(output/'frozen_team_plan.json',frozen)
        print(json.dumps(dict(planned_people=[x['selection'] for x in frozen['people']],output=str(output))))
        return
    if a2.read(output/'frozen_policy.json')!=policy:raise ValueError('frozen_team_policy_changed')
    frozen=a2.read(output/'frozen_team_plan.json')
    people,assets,placements=prepare_assets(policy,frozen,registry,output)
    for index in range(2):
        base='riverfront_2' if index==0 else 'urban_3'
        suffix=('_overlap_gate_correction2' if (output/(base+'_implementation_correction1')).exists()
                else '_implementation_correction1' if (output/base).exists() else '')
        result=prepare_episode(policy,frozen,people,assets,placements,index,output,stage_suffix=suffix)
        print(json.dumps(dict(episode_index=index,preflight='passed' if result else 'rejected')),flush=True)


if __name__=='__main__':main()
