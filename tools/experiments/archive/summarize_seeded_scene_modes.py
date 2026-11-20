"""Bounded manifest-only acceptance for two explicit seeded ROI episodes."""
import argparse
import json
import statistics
from pathlib import Path


def read(path):return json.loads(path.read_text())
def size(path):return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def check_one(root,mode,eid):
    p=root/'runs'/eid;prepared=root/'preflight'/f'{mode}_s20261010'/eid
    recipe=read(p/'recipe.json');index=read(p/'pack/sequence_index.json');accept=read(p/'pack/multiperson_acceptance.json');back=read(p/'pack/readback.json')
    validation=read(p/'pack/validation.json');cost=read(p/'task_wall.json');blender=read(p/'blender_process.json')
    regions=read(prepared/'planned_regions.json')
    calls=[json.loads(s) for s in (p/'render_calls.jsonl').read_text().splitlines()];completed=[x for x in calls if x['event']=='completed']
    if len(completed)!=62 or len(index['frames'])!=31 or len(back['video_pts_seconds'])!=31:raise ValueError('incomplete_episode')
    if back['status']!='passed' or validation['status']!='passed':raise ValueError('dataset_or_index_validation')
    mismatches=[];areas=[];frame_cost=[];consume_cost=[];consumer_rss=[]
    for k in range(31):
        r=read(p/f'frame_{k:03d}/render_frame.json');frame_cost.append(r['source_manifest']['render_seconds'])
        report=read(p/f'frame_{k:03d}/report.json');consume_cost.append(report['consume_seconds']);consumer_rss.append(report['peak_rss_kib'])
        for c in ('cam2','cam4'):
            expected=regions[k][c]['roi_xyxy'];actual=r['plan']['cameras'][c]['roi_xyxy']
            if actual!=expected:mismatches.append(dict(frame=k,camera=c,planned=expected,actual=actual))
            areas.append((actual[2]-actual[0])*(actual[3]-actual[1]))
    if mismatches or max(areas)>7_000_000:raise ValueError('roi_plan_actual_mismatch_or_budget')
    if back['video_pts_seconds']!=[k/10 for k in range(31)]:raise ValueError('video_clock_mismatch')
    if any(not read(q/'generation_status.json')['success'] for q in p.glob('frame_*') if (q/'generation_status.json').exists()):raise ValueError('failed_frame')
    if any(q.exists() for q in p.glob('frame_*/_temporary_ds_source')):raise ValueError('temporary_source_remains')
    people=len(recipe['people'])
    if len(index['observations'])!=31*2*people:raise ValueError('observation_slot_count')
    setup=read(p/'single_process_setup.json')
    by_camera={c:[x['seconds'] for x in completed if x['camera']==c] for c in ('cam2','cam4')}
    return dict(status='data_passed' if back['actual_Dataset_reads']<=8 else 'data_passed_readback_cap_exceeded',
        mode=mode,episode_id=eid,asset_ids=[x['asset_id'] for x in recipe['people']],
        frames=31,camera_calls=len(completed),observation_slots=len(index['observations']),packed_rows=index['packed']['row_count'],
        observation_status_counts=accept['observation_status_counts'],dataset_reads=back['actual_Dataset_reads'],
        dataset_read_cap=8,dataset_read_cap_met=back['actual_Dataset_reads']<=8,
        native_image_count=len({x['image'] for x in index['observations'] if x['image']}),
        dual_physical_edge=accept['dual_edge'],common_at_least8=accept['pair_input_availability']['common_at_least8_count'],
        event_coverage_status=accept['event_coverage_status'],events=accept['events'],
        roi_exact_matches=62,max_roi_pixels=max(areas),mean_roi_pixels=statistics.mean(areas),
        blender_starts=cost['blender_start_count'],setup_callback_seconds=setup['setup_callback_seconds'],blender_wall_seconds=blender['wall_seconds'],
        complete_generation_wall_seconds=cost['total_generation_wall_seconds'],
        first_frame_blender_block_seconds=frame_cost[0],subsequent_frame_blender_block_mean_seconds=statistics.mean(frame_cost[1:]),
        subsequent_frame_blender_block_range_seconds=[min(frame_cost[1:]),max(frame_cost[1:])],
        first_frame_cpu_consume_seconds=consume_cost[0],subsequent_frame_cpu_consume_mean_seconds=statistics.mean(consume_cost[1:]),
        camera_call_combined_seconds={c:dict(total=sum(v),first=v[0],subsequent_mean=statistics.mean(v[1:])) for c,v in by_camera.items()},
        finish_seconds=cost['finish_process_seconds'],sampled_peak_gpu_mib=blender['sampled_peak_GPU_MiB'],
        sampled_peak_temporary_bytes=max(x['temporary_bytes'] for x in blender['gpu_samples']),
        sampled_peak_disk_bytes=max(x['disk_bytes'] for x in blender['gpu_samples']),
        peak_blender_rss_mib=max(x['rss_kib'] for x in completed)/1024,peak_consumer_rss_mib=max(consumer_rss)/1024,
        retained_bytes=size(p),temporary_source_remaining=0,
        video_ref=str((p/'pack/dual_roi.mp4').resolve()),sequence_index_ref=str((p/'pack/sequence_index.json').resolve()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);a=ap.parse_args();root=a.root.resolve()
    pairs=[('single_cyclist_boundary_pass','single_cyclist_boundary_pass_s20261010_e00'),('mixed_walk_cycle_crossing','mixed_walk_cycle_crossing_s20261010_e00')]
    results=[check_one(root,m,e) for m,e in pairs]
    result=dict(schema='seeded_scene_modes_acceptance_v1',status='passed' if all(x['dataset_read_cap_met'] for x in results) else 'data_passed_with_bounded_readback_excess',episodes=results,
        total_camera_calls=sum(x['camera_calls'] for x in results),total_slots=sum(x['observation_slots'] for x in results),
        total_packed_rows=sum(x['packed_rows'] for x in results),total_dataset_reads=sum(x['dataset_reads'] for x in results),
        total_generation_wall_seconds=sum(x['complete_generation_wall_seconds'] for x in results),
        total_retained_bytes=size(root),data_kind='integration_only/shared_source_diagnostic',
        exposure_sync_validated=False,model_forwards=0,optimizer_updates=0)
    (root/'acceptance.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='episodes'},indent=2))


if __name__=='__main__':main()
