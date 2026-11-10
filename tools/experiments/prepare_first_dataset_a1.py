"""CPU metadata planner. No imports of Blender, GPU, body models or renderers."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from densekp_retrain.data.first_dataset_plan import MODES, VERSION, plan, read_catalog


def review_tables(catalog,proposals,planned):
    proposal=proposals['proposals'][planned['source_proposal_id']]
    pools={}
    for side in ('train_candidate','val_candidate'):
        rows=[]
        for cid in proposal[side]:
            combo=catalog['combinations'][cid]
            motion=catalog['motions'][combo['motion_id']]
            body=catalog['bodies'][combo['body_id']]
            rows.append(dict(combo_id=cid,body_id=combo['body_id'],character_family=body.get('character_family'),
                appearance_conservative_group=body.get('appearance_conservative_group'),
                shape_source_group=body.get('shape_source_group'),skin_texture_source=body.get('texture_source',{}).get('source_id'),
                texture_license=body.get('texture_license'),historical_box400_split=body.get('historical_box400_split'),
                motion_id=combo['motion_id'],recording_group=motion.get('recording_group'),
                participant_group=motion.get('participant_group'),source_group=motion.get('source_group'),
                motion_license_status=motion.get('license_status'),motion_known_usage=motion.get('known_usage'),
                clothing_id=combo['clothing_id'],hair_id=combo['hair_id'],
                clothing_parent=catalog['clothing'][combo['clothing_id']].get('reference_body'),
                hair_parent=catalog['hair'][combo['hair_id']].get('reference_body'),
                clothing_license=catalog['clothing'][combo['clothing_id']].get('license'),
                hair_license=catalog['hair'][combo['hair_id']].get('license'),
                asset_window=combo['asset_window'],built=combo['built'],human_accepted_scope=combo.get('human_accepted_scope'),
                scene_consumed=combo['scene_consumed'],license_status=combo.get('license_status'),
                authority_ref=combo.get('authority_ref')))
        pools[side]=rows
    counts={s:dict(cycling_motion_ids=sorted({r['motion_id'] for r in rows if 'walk' not in r['motion_id']}),
                   cycling_body_ids=sorted({r['body_id'] for r in rows if 'walk' not in r['motion_id']}))
            for s,rows in pools.items()}
    source=dict(schema='first_dataset_source_review_v1',review_status='pending_review',proposal_id=planned['source_proposal_id'],
        pools=pools,counts=counts,walking_actor_independence=catalog.get('unknowns',{}).get('KIT11_vs_KIT425_actor'),
        diagnostic_only=proposal['diagnostic_only'],
        rules=['entire episode including occluders on one side','all rigs and neighbor frames share scene source',
               'clothing/hair/seed changes do not create new identity','different motion IDs may share participant'])
    coverage=dict(schema='first_dataset_coverage_targets_v1',review_status='pending_review',
        denominator='planned person-camera-time slots unless specified; all no-box slots retained',
        targets=[dict(name='two_camera_physical_edge',fraction=.20,denominator='target-person episode-time within edge-design episodes'),
                 dict(name='one_camera_truncation',fraction=.20,denominator='target-person camera-time'),
                 dict(name='light_or_full_control',fraction=.25,denominator='target-person camera-time'),
                 dict(name='neighbor_occlusion',fraction=.15,denominator='mixed target-person camera-time; requires ID and range'),
                 dict(name='complete_occlusion_or_no_box',fraction_range=[.02,.10],denominator='mixed target-person camera-time; diagnostic goal, never delete excess'),
                 dict(name='front_back_side',fraction=None,denominator='target-person rig-time; report separately'),
                 dict(name='minimum_person_height',minimum_px=180,denominator='legal native box camera-time; smaller side also reported')],
        planned_slots=sum(e['planned_observation_slots'] for e in planned['episodes']),
        note='physical geometry is CPU-previsible; native masks, legal boxes and neighbor occlusion require rendered evidence')
    budget=dict(schema='first_dataset_budget_v1',review_status='pending_review',episode_count=len(planned['episodes']),
        total_camera_calls=sum(31*2*len(e['rigs']) for e in planned['episodes']),
        planned_observation_slots=coverage['planned_slots'],
        assumptions=['31 frames/episode at 10Hz','single rigs 62 calls; mixed two rigs 124 calls',
                     'ROI sizes and environment/rig pose may change cost','no new asset build assumed; pending scene-bound combos need separate validation'],
        cpu_preflight_minutes_range=[5,20],asset_completion_minutes='unknown_until_authority_and_acceptance_review',
        assembly_minutes_per_episode_range=[.4,1.5],render_consume_finish_minutes_total_range=[30,80],
        transient_peak_megabytes_range=[200,600],long_term_megabytes_total_range=[400,1200],
        evidence=['riverfront one-rig 31 frames about 5.3min','city 5-person two-rig 31 frames about 28.5min',
                  'city T-junction 3-person three-rig 31 frames about 22.6min'])
    gaps=dict(schema='first_dataset_gaps_v1',review_status='pending_review',
        blocking_before_formal_batch=['pool and license/use review','KIT425_M04 accepted window missing in authority catalog',
           'source-qualified combinations need actual scene consumption and ROI reservation',
           'exact source timeline and geometry preflight per episode','common eligible parent-row set for box on/off comparison'],
        effect_study_limits=['KIT11/KIT425 performer independence unknown','original DenseKP pretraining exposure unknown',
           'participant_1 comfortable/alternate are distinct motion IDs but share participant and body M02',
           '3-5 cyclists with distinct identities cannot be assembled within either proposed pool'],
        mode_by_pool={
            'single_cyclist_boundary_pass':{'train':'candidate_pending_scene','val':'candidate_pending_scene'},
            'mixed_walk_cycle_crossing':{'train':'blocked_walker_acceptance_and_scene','val':'candidate_pending_scene'},
            'dispersed_cyclists_3_to_5':{'train':'insufficient_distinct_bodies_and_motions','val':'insufficient_distinct_bodies'},
            'city_t_junction_3p':{'train':'insufficient_distinct_bodies_and_motions','val':'insufficient_distinct_bodies'}},
        diagnostic_existing_episodes=[
            'outputs/experiments/second_environment_boundary_pass_a1_20261010/riverfront_complete',
            'outputs/experiments/mixed_walk_cycling_edge_continuation_a1_20261009',
            'outputs/experiments/dispersed_cyclists_4_5_multirig_a1_20261011/city_5p',
            'outputs/experiments/t_junction_local_rig_delivery_a1_20261011/production_retry3'])
    return dict(source_pools=source,coverage_targets=coverage,budget=budget,gaps=gaps,
                box_policy=dict(schema='first_dataset_box_policy_v1',review_status='pending_review',
                    render_reservation=planned['episodes'][0]['box_policy'],training_enabled_by_default=False,
                    deployment_crop_expansion=1.0,historical_trial_expansion=1.2,
                    unsupported=['rotation','prefilter','flip','Albumentations','ERP_wrap']))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',nargs='?',choices=('list','save','preflight'),default='list')
    parser.add_argument('--config',type=Path,default=ROOT/'configs/first_dataset_preparation_a1_20261011.json')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.command=='list':
        print(json.dumps(dict(schema=VERSION,modes={k:list(v) for k,v in MODES.items()},
            environments=['urban_street_assets','riverfront'],production_command='not_authorized_pending_review',
            stages=['plan_valid','source_eligible','asset_ready','geometry_checked','rendered','dataset_verified']),indent=2))
        return
    if args.output is None:parser.error('--output is required for save/preflight')
    cfg=json.loads(args.config.read_text());catalog,proposals=read_catalog(ROOT)
    registry=json.loads((ROOT/'configs/asset_composition_a2_20261010/components.json').read_text())
    result=plan(cfg,catalog,proposals,registry=registry)
    if args.command=='preflight':
        from densekp_retrain.assets.composition import plan as composition_plan
        from densekp_retrain.data.person_time import map_timeline
        clock=[dict(frame_index=i,episode_time_tick=i,time_base=[1,10]) for i in range(31)]
        for episode in result['episodes']:
            for person in episode['persons']:
                motion=catalog['motions'][person['motion_id']]
                timeline=json.loads((ROOT/motion['timeline_ref']).read_text())
                try:
                    origin=next(i for i,row in enumerate(timeline)
                                if row['asset_frame']==person['proposed_start_asset_frame'])
                    mapped=map_timeline(timeline,clock,origin_record=origin)
                    person['exact_time_mapping']=mapped
                    person['time_mapping_status']='checked_exact_integer_timeline'
                except (ValueError,StopIteration) as error:
                    person['time_mapping_status']='rejected:'+str(error)
                    episode['plan_valid']=False
                    episode['plan_rejections'].append('time_mapping:'+str(error))
                if person['combination_status']!='build_required':continue
                motion=registry['motions'][person['motion_id']]
                request={k:person[k] for k in ('body_id','motion_id','clothing_id','hair_id')}
                request.update(window=person['asset_window'],root_policy=motion['root_mode'],
                    output_dir=str(args.output.parent/'planned_combinations'))
                try:
                    resolved=composition_plan(ROOT,registry,request)
                    person['a2_compatibility']=dict(status='plan_passed_build_required',
                        selection_mode=resolved.get('selection_mode','new_composition'),frame_count=resolved['frame_count'])
                except (ValueError,FileNotFoundError,KeyError) as error:
                    person['a2_compatibility']=dict(status='rejected',reason=str(error))
                    episode['plan_valid']=False
                    episode['plan_rejections'].append('a2_composition_plan_rejected:'+str(error))
        result['cpu_preflight']=dict(source_and_window_metadata_checked=True,
            exact_time_mapping='checked_using_existing_person_time_for_every_person',
            body_projection='not_run_no_model_evaluation',
            roi_support='pending_actual_projected_geometry',
            render_permission=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    if args.command=='preflight':
        for name,value in review_tables(catalog,proposals,result).items():
            (args.output.parent/f'{name}.json').write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(args.output),episodes=len(result['episodes']),
        plan_valid=sum(e['plan_valid'] for e in result['episodes']),
        source_eligible=sum(e['source_eligible'] for e in result['episodes']),
        asset_ready=sum(e['asset_ready'] for e in result['episodes'])),ensure_ascii=False))

if __name__=='__main__':main()
