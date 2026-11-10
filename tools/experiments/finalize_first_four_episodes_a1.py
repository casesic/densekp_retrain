"""Read-only accounting and viewing index for the bounded four-episode run."""
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np
from densekp_retrain.data.boxes import change_box_for_image, crop_from_box
from densekp_retrain.imaging.regions import require_crop_in_region

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/first_four_episodes_a1_20261011'
PLAN=ROOT/'outputs/plans/first_dataset_preparation_a1_20261011/episode_plan.json'


def load(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def video_probe(path):
    if not path.is_file():return None
    raw=subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-count_frames',
        '-show_entries','stream=nb_read_frames,r_frame_rate,duration','-of','json',str(path)],text=True)
    return dict(path=str(path),bytes=path.stat().st_size,**load_json(raw)['streams'][0])


def load_json(value):return json.loads(value)


def run_summary(ep,index):
    folder=BASE/f'e{index:02d}'
    run=folder/'run'
    if not (run/'task_wall.json').is_file():
        prepared=folder/ep['scene_episode_id']/'geometry_preflight.json'
        status=load(prepared) if prepared.is_file() else {}
        checks=folder/ep['scene_episode_id']/'candidate_checks.json'
        attempts=[x for batch in load(checks) for x in batch['attempts']] if checks.is_file() else []
        refinement=folder/'rig_refinement1'
        refined=[];refined_seconds=0.0
        for item in sorted(refinement.glob('batch_*/'+ep['scene_episode_id']+'/candidate_checks.json')):
            refined.extend(x for batch in load(item) for x in batch['attempts'])
            preflight=item.with_name('geometry_preflight.json')
            refined_seconds+=float(load(preflight)['seconds'])
        return dict(episode_id=ep['scene_episode_id'],split_pool=ep['split_pool'],status='geometry_rejected' if status.get('status')=='rejected' else 'not_produced',
            layout_candidates_evaluated=len(attempts),rejections=[x['rejections'] for x in attempts],
            rig_only_refinement_trials=len(refined),rig_only_refinement_rejections=[x['rejections'] for x in refined],
            preflight_seconds=float(status.get('seconds') or 0)+refined_seconds,
            camera_calls=0,observation_slots=0,legal_parent_rows=0)
    multi=(run/'multirig.json').is_file()
    spec=load(run/'multirig.json') if multi else None
    items=[(x['rig_id'],Path(x['root']),Path(x['source_prepared_ref'])) for x in spec['rigs']] if multi else [
        ('rig_edge',run,folder/'corrected_replay1'/ep['scene_episode_id'])]
    calls=[load_json(line) for line in (run/'render_calls.jsonl').read_text().splitlines()]
    completed=[x for x in calls if x['event']=='completed']
    started=[x for x in calls if x['event']=='started']
    rigs=[];challenges=[];read_total=0
    for rig_id,child,prepared in items:
        pack=child/'pack'
        actual=load(pack/'multiperson_acceptance.json')
        index_table=load(pack/'sequence_index.json')
        with np.load(pack/'packed.npz') as data:
            row_count=len(data['imgname']);support=np.asarray(data['box_augmentation_family_supported'],bool)
        lookup={o['observation_id']:o for o in index_table['observations']}
        box_lookup={b['observation_id']:b for b in index_table['boxes']}
        heights={str(p['person_id']):[] for p in ep['persons']}
        for box in index_table['boxes']:
            pid=str(lookup[box['observation_id']]['person_id'])
            heights[pid].append(float(box['bbox_xyxy'][3]-box['bbox_xyxy'][1]))
        height_summary={p:dict(count=len(a),median_px=statistics.median(a) if a else None,
                               range_px=[min(a),max(a)] if a else None) for p,a in heights.items()}
        read=load(pack/'readback.json')
        audit_ref=folder/(f'{rig_id}_box_readback.json' if multi else 'box_readback.json')
        audit=load(audit_ref) if audit_ref.is_file() else None
        read_total+=read['actual_Dataset_reads']+(audit['actual_additional_Dataset_item_reads'] if audit else 0)
        if audit:
            for record in audit['records']:
                if record['status']!='readback_passed':continue
                parent=box_lookup[record['observation_id']]['bbox_xyxy']
                roi=lookup[record['observation_id']]['roi_xyxy']
                changed=change_box_for_image(parent,image_size_wh=(5760,2880),
                    boundary_policy='erp_reject_seam',min_side_px=2,
                    shift_reference_expansion=1.,dx=.03,dy=-.03,factor=1.,edge=None,fraction=0.)
                center,scale=crop_from_box(changed,image_size_wh=(5760,2880),
                    boundary_policy='erp_reject_seam',crop_policy='deployment',min_side_px=2)
                require_crop_in_region(center,scale,roi,(5760,2880),interpolation_margin_px=2)
                challenges.append(dict(rig_id=rig_id,observation_id=record['observation_id'],
                    packed_row=record['packed_row'],parent_bbox_xyxy=parent,
                    fixed_bbox_xyxy=np.asarray(changed,float).tolist(),
                    center=np.asarray(center,float).tolist(),scale=float(scale),
                    box_kind='synthetic_perturbation',parameters={'dx':.03,'dy':-.03,'factor':1.,'edge':None,'fraction':0.},
                    derived_from_native_parent=True,support_checked=True,
                    model_evaluation_status='not_run'))
        regions=load(prepared/'planned_regions.json')
        rig_calls=[x for x in started if not multi or x['rig_id']==rig_id]
        roi_mismatch=[]
        for call in rig_calls:
            cam=call.get('camera_id',call.get('camera'))
            if call['roi_xyxy']!=regions[call['frame_index']][cam]['roi_xyxy']:
                roi_mismatch.append(dict(frame=call['frame_index'],camera=cam))
        video=video_probe(pack/'dual_roi.mp4')
        rigs.append(dict(rig_id=rig_id,prepared_ref=str(prepared),pipeline_status=actual['pipeline_status'],
            frame_count=actual['frames'],observation_slots=actual['observation_slots'],
            legal_parent_rows=row_count,missing_status_counts=actual['observation_status_counts'],
            whole_box_family_supported=int(support.sum()),whole_box_family_unsupported=int((~support).sum()),
            dual_edge=actual.get('dual_edge'),neighbor_events=actual.get('events'),
            native_box_height_by_person=height_summary,camera_calls=len(rig_calls),
            planned_vs_actual_ROI_mismatch_count=len(roi_mismatch),video=video,
            prior_native_Dataset_reads=read['actual_Dataset_reads'],
            additional_box_Dataset_reads=audit['actual_additional_Dataset_item_reads'] if audit else 0,
            readback_ref=str(audit_ref) if audit else None))
    wall=load(run/'task_wall.json')
    seconds=wall.get('elapsed_seconds',wall.get('total_generation_wall_seconds'))
    batch_config=load(folder/'config.json')
    checks=folder/ep['scene_episode_id']/'geometry_preflight.json'
    preflight=load(checks) if checks.is_file() else {}
    size=sum(path.stat().st_size for path in folder.rglob('*') if path.is_file())
    return dict(episode_id=ep['scene_episode_id'],split_pool=ep['split_pool'],status='rendered_candidate_pending_review',
        environment=ep['environment'],people=[p['combo_id'] for p in ep['persons']],rigs=rigs,
        camera_calls=len(completed),planned_camera_calls=31*2*len(rigs),
        observation_slots=sum(r['observation_slots'] for r in rigs),legal_parent_rows=sum(r['legal_parent_rows'] for r in rigs),
        whole_box_family_supported=sum(r['whole_box_family_supported'] for r in rigs),
        whole_box_family_unsupported=sum(r['whole_box_family_unsupported'] for r in rigs),
        batch_dual_edge_requirement=dict(required=bool(batch_config.get('require_dual_edge',len(ep['persons'])==1)),
            minimum_times=int(batch_config['min_dual_edge_times']),
            actual_primary_rig_count=int(rigs[0]['dual_edge']['count']),
            passed=(not batch_config.get('require_dual_edge',len(ep['persons'])==1) or rigs[0]['dual_edge']['count']>=batch_config['min_dual_edge_times'])),
        real_Dataset_item_reads=read_total,render_consume_finish_seconds=seconds,
        preflight_seconds=preflight.get('seconds'),
        disk_bytes=size,shared_scene_start_count=1,
        strict_source_split='pending_review_not_formal',human_scene_acceptance='pending_video_review',
        training_status='not_started',fixed_challenge_boxes=challenges)


def main():
    planned=load(PLAN)['episodes']
    rows=[run_summary(ep,i) for i,ep in enumerate(planned)]
    completed=[r for r in rows if r['status']=='rendered_candidate_pending_review']
    challenge=[c for r in completed for c in r['fixed_challenge_boxes']]
    parent_rows=[]
    for ep in completed:
        for rig in ep['rigs']:
            pack=Path(rig['video']['path']).parent
            index=load(pack/'sequence_index.json')
            observations={o['observation_id']:o for o in index['observations']}
            for box in index['boxes']:
                observation=observations[box['observation_id']]
                parent_rows.append(dict(scene_episode_id=ep['episode_id'],rig_id=rig['rig_id'],
                    split_pool=ep['split_pool'],observation_id=box['observation_id'],
                    packed_row_id=box['packed_row_id'],person_id=observation['person_id'],
                    camera_id=observation['camera_id'],frame_index=observation['frame_index'],
                    source_ref=observation['source_ref'],parent_bbox_xyxy=box['bbox_xyxy'],
                    whole_box_family_supported=observation['box_augmentation_family_supported'],
                    on_off_same_parent_eligible=observation['box_augmentation_family_supported']))
    write(BASE/'acceptance.json',dict(schema='first_four_acceptance_v1',episodes=rows,
        planned_episodes=4,completed_episodes=len(completed),planned_camera_calls=372,
        actual_camera_calls=sum(x['camera_calls'] for x in rows),planned_observation_slots=620,
        actual_observation_slots=sum(x['observation_slots'] for x in rows),
        legal_parent_rows=sum(x['legal_parent_rows'] for x in rows),
        whole_box_family_unsupported=sum(x.get('whole_box_family_unsupported',0) for x in rows),
        real_Dataset_item_reads=sum(x.get('real_Dataset_item_reads',0) for x in rows),
        training_started=False,model_evaluation_started=False))
    write(BASE/'fixed_box_challenge.json',dict(schema='fixed_box_challenge_candidates_v1',status='frozen_inputs_no_model_evaluation',items=challenge))
    write(BASE/'common_parent_rows.json',dict(schema='first_four_common_parent_rows_v1',
        on_and_off_use_same_parent_rows=True,comparison_status='pending_review_no_training',
        total_native_parent_rows=len(parent_rows),unsupported_whole_family=sum(not row['whole_box_family_supported'] for row in parent_rows),
        rows=parent_rows))
    coverage=[]
    for ep in completed:
        for rig in ep['rigs']:
            pack=Path(rig['video']['path']).parent
            actual=load(pack/'multiperson_acceptance.json')
            coverage.append(dict(scene_episode_id=ep['episode_id'],rig_id=rig['rig_id'],
                planned_times=31,dual_edge_times=actual['dual_edge']['count'],
                dual_edge_windows=actual['dual_edge']['windows'],
                common_crop_at_least8_times=actual['pair_input_availability']['common_at_least8_count'],
                native_input_status=actual['observation_status_counts'],
                neighbor_event_status=actual['event_coverage_status'],
                neighbor_events=actual['events'],
                native_box_height_by_person=rig['native_box_height_by_person']))
    write(BASE/'coverage_summary.json',dict(schema='first_four_coverage_summary_v1',
        denominator='31 planned times per rig; no-frame and no-box times retained',rigs=coverage))
    preflight_paths=[BASE/'e00'/planned[0]['scene_episode_id']/'geometry_preflight.json',
        *sorted((BASE/'e00/rig_refinement1').glob('batch_*/'+planned[0]['scene_episode_id']+'/geometry_preflight.json')),
        BASE/'e01'/planned[1]['scene_episode_id']/'geometry_preflight.json',
        BASE/'e01/corrected_replay1'/planned[1]['scene_episode_id']/'geometry_preflight.json',
        BASE/'e02'/planned[2]['scene_episode_id']/'geometry_preflight.json',
        BASE/'e02/rig_front'/planned[2]['scene_episode_id']/'geometry_preflight.json',
        BASE/'e03'/planned[3]['scene_episode_id']/'geometry_preflight.json',
        BASE/'e03/rig_front'/planned[3]['scene_episode_id']/'geometry_preflight.json']
    cpu_preflight=sum(float(load(path)['seconds']) for path in preflight_paths)
    render_finish=sum(float(ep['render_consume_finish_seconds']) for ep in completed)
    write(BASE/'cost_summary.json',dict(schema='first_four_cost_summary_v1',
        cpu_geometry_preflight_seconds=cpu_preflight,
        render_consume_pack_video_seconds=render_finish,
        total_bounded_compute_seconds=cpu_preflight+render_finish,
        cpu_budget_seconds=1200,production_budget_seconds=4800,total_budget_seconds=6000,
        production_episode_seconds={ep['episode_id']:ep['render_consume_finish_seconds'] for ep in completed},
        gpu_uuid='GPU-d481acc7-106f-29e5-1298-8c0d398261b2',
        temporary_peak_bytes=None,temporary_peak_status='not_reliably_instrumented'))
    write(BASE/'training_candidate_ticks.json',dict(status='pending_review',per_episode={
        p['scene_episode_id']:dict(training_even_ticks=list(range(0,31,2)),evaluation_all_ticks=list(range(31)),
                                   eligible_for_training=False) for p in planned}))
    links=[]
    for i,ep in enumerate(rows):
        if 'rigs' not in ep:
            links.append(f'<h2>{ep["episode_id"]}</h2><p>几何预检未通过；未渲染，无视频。</p>')
            continue
        links.append(f'<h2>{ep["episode_id"]} — {ep["environment"]}</h2>')
        for rig in ep['rigs']:
            relative=Path(rig['video']['path']).relative_to(BASE)
            filename=f'{ep["episode_id"]}_{rig["rig_id"]}.mp4'
            links.append(f'<h3>{rig["rig_id"]} — {rig["legal_parent_rows"]} legal rows</h3>'
                f'<video controls width="720" src="{relative}"></video>'
                f'<p><a class="download" href="{relative}" download="{filename}">下载 MP4</a></p>')
    (BASE/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>首轮候选视频</title>'
        '<style>body{font-family:sans-serif;background:#eee;margin:2rem}video{display:block}'
        '.download{display:inline-block;padding:.55rem .9rem;margin-bottom:2rem;background:#175b9a;color:white;'
        'border-radius:.35rem;text-decoration:none}.download:hover{background:#104775}</style>'
        '<h1>首轮候选数据：仅高质量ROI，灰区/黑区不表示完整场景RGB</h1>'+'\n'.join(links)+'</html>')
    print(json.dumps(dict(completed=len(completed),calls=sum(x['camera_calls'] for x in rows),
                          rows=sum(x['legal_parent_rows'] for x in rows)),ensure_ascii=False))


if __name__=='__main__':main()
