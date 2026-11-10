"""Summarize a completed single multiperson run; no rendering, models or Dataset reads."""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);args=ap.parse_args();root=args.root.resolve()
    begin=time.monotonic()
    read=lambda p:json.loads((root/p).read_text())
    job=read('job_progress.json');render=read('render_progress.json');process=read('blender_process.json');wall=read('task_wall.json')
    recipe=read('recipe.json');acceptance=read('pack/multiperson_acceptance.json');readback=read('pack/readback.json')
    count=recipe['frame_count'];assert len(job['frames'])==count==len(render['frames'])
    def summary(values):
        return dict(count=len(values),mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),maximum=max(values),sum=sum(values))
    consume=[read(f'frame_{k:03d}/consume_timing.json') for k in range(count)]
    calls=[json.loads(line) for line in (root/'render_calls.jsonl').read_text().splitlines()]
    completed=[c for c in calls if c['event']=='completed'];started=[c for c in calls if c['event']=='started']
    assert len(completed)==len(started)==2*count and all(c['kind']=='roi' for c in started)
    assert not list(root.rglob('_temporary_ds_source')) and not list(root.rglob('*.exr')) and not list(root.rglob('normal_erp.png'))
    rows=read('pack/per_person_time_statistics.json');statistics_by_person={}
    import numpy as np
    import cv2
    from PIL import Image
    signed={}
    for camera in ['cam2','cam4']:
        mask=np.asarray(Image.open(root/'calibration'/f'{camera}_valid_mask.png'))>0
        signed[camera]=cv2.distanceTransform(mask.astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform((~mask).astype(np.uint8),cv2.DIST_L2,5)
    edge_rows=[]
    for k in range(count):
        for camera in signed:
            with np.load(root/f'frame_{k:03d}'/camera/'annotations.npz') as ann:
                for j,pid in enumerate(ann['person_ids']):
                    xy=ann['xy'][j];x=np.floor(xy[:,0]).astype(int)%5760;y=np.clip(np.floor(xy[:,1]).astype(int),0,2879)
                    distance=signed[camera][y,x]
                    edge_rows.append(dict(frame_index=k,person_id=int(pid),camera_id=camera,boundary_signed_distance_final_ERP_px=distance.tolist(),near_boundary_64px=int((np.abs(distance)<=64).sum())))
    (root/'pack/edge_statistics.json').write_text(json.dumps(edge_rows,indent=2,allow_nan=False)+'\n')
    for person in recipe['people']:
        pid=person['person_id'];statistics_by_person[str(pid)]={}
        for camera in ['cam2','cam4']:
            rs=[r for r in rows if r['person_id']==pid and r['camera_id']==camera]
            values=lambda key:[r[key] for r in rs if r[key] is not None]
            statistics_by_person[str(pid)][camera]=dict(observations=len(rs),observed=sum(r['status']=='observed' for r in rs),
                in_view_range=[min(values('in_view')),max(values('in_view'))],visible_range=[min(values('visible')),max(values('visible'))],
                common_physical_range=[min(values('common_physical_points')),max(values('common_physical_points'))],
                common_supervised_range=[min(values('common_supervised_points')),max(values('common_supervised_points'))] if values('common_supervised_points') else None,
                truncated_times=sum(r['outside_domain']>0 for r in rs),near_boundary_times=sum(r['near_boundary_64px']>0 for r in edge_rows if r['person_id']==pid and r['camera_id']==camera),
                totals={key:sum(values(key)) for key in ['supervised','supervised_occluded','neighbor_occluded','self_instance_occluded','scene_occluded','unresolved_occluded','outside_domain']},
                box_width_range=[min(values('box_width')),max(values('box_width'))] if values('box_width') else None,
                box_height_range=[min(values('box_height')),max(values('box_height'))] if values('box_height') else None,
                distinct_native_boxes=len({tuple(r['box_xyxy']) for r in rs if r['box_xyxy'] is not None}))
    camera_stats={}
    for camera in ['cam2','cam4']:
        rs=[r for f in render['frames'] for r in f['records'] if r['id']==camera]
        pixel=lambda r:(r['roi_xyxy'][2]-r['roi_xyxy'][0])*(r['roi_xyxy'][3]-r['roi_xyxy'][1])
        camera_stats[camera]=dict(first_call_seconds=rs[0]['seconds'],subsequent_call_seconds=summary([r['seconds'] for r in rs[1:]]),
            ROI_pixels=summary([pixel(r) for r in rs]),total_call_seconds=sum(r['seconds'] for r in rs))
    files=list(p for p in root.rglob('*') if p.is_file());size=lambda ps:sum(p.stat().st_size for p in ps)
    result=dict(status='completed',pipeline=acceptance['pipeline_status'],event_coverage=acceptance['event_coverage_status'],
        episode=recipe['episode_id'],people=len(recipe['people']),frames=count,source_timespan_seconds=recipe['selected_timeline'][-1]['episode_time_tick']*recipe['time_base'][0]/recipe['time_base'][1],
        successful_camera_calls=len(completed),failed_camera_calls=len(started)-len(completed),ROI_RGB_files=len(list(root.glob('frame_*/cam*/roi.png'))),
        packed_rows=acceptance['packed_rows'],actual_Dataset_reads=readback['actual_Dataset_reads'],observation_status_counts=acceptance['observation_status_counts'],
        Blender_pid=process['pid'],Blender_starts=wall['blender_start_count'],consumer_pid=read('consumer_worker_start.json')['pid'],
        assembly_combined_seconds=job['setup_seconds'],CPU_worker_startup_seconds=read('consumer_worker_start.json')['startup_seconds'],
        first_frame_loop_seconds=job['frames'][0]['seconds'],subsequent_frame_loop_seconds=summary([f['seconds'] for f in job['frames'][1:]]),
        geometry_update_seconds=summary([f['geometry_seconds'] for f in job['frames']]),ROI_planning_seconds=summary([f['roi_planning_seconds'] for f in render['frames']]),
        cameras=camera_stats,CPU_function_seconds=summary([c['seconds'] for c in consume]),CPU_roundtrip_seconds=summary([f['consumer_callback_seconds'] for f in job['frames']]),
        pack_seconds=readback['pack_seconds'],readback_and_all_frame_stats_seconds=readback['readback_seconds'],video_seconds=readback['video_seconds'],
        wall=wall,peak_GPU_MiB=process['sampled_peak_GPU_MiB'],peak_Blender_RSS_KiB=read('continuous_render_result.json')['peak_rss_kib'],
        peak_consumer_RSS_KiB=max(c['peak_rss_kib'] for c in consume),peak_finish_RSS_KiB=readback['peak_rss_kib'],
        sampled_peak_temporary_bytes=max(s['temporary_bytes'] for s in process['gpu_samples']),successful_temporary_bytes_cleaned=sum(c['source_bytes'] for c in consume),
        final_run_bytes_before_summary=size(files),ROI_RGB_bytes=size(root.glob('frame_*/cam*/roi.png')),final_EXRs=0,full_RGB_images=0,background_render_calls=0,
        statistic_units='point-times, final ERP pixels, seconds; camera render/compositor/write/sync combined, not GPU kernel time',
        per_person=statistics_by_person,events=acceptance['events'],video_time_base=readback['video_time_base'],last_frame_hold_seconds=readback['last_frame_display_hold_seconds'],
        supervision_unchanged=True,model_forwards=0,optimizer_updates=0,post_summary_seconds=time.monotonic()-begin)
    path=root.parent/'costs_and_acceptance.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(result_path=str(path),event_coverage=result['event_coverage'],total_wall=wall['total_generation_wall_seconds'],packed_rows=result['packed_rows']),indent=2))


if __name__=='__main__':main()
