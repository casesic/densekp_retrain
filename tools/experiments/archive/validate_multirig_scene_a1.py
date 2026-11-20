"""Bounded post-run checks for the two-rig single-person scene episode."""
import json
from pathlib import Path
import resource
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
from densekp_retrain.data.sequence_index import validate_sequence_index


def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def validate(root):
    root=Path(root).resolve();plan=read(root/'multirig.json');scene_id=plan['scene_episode_id']
    clearance=read(root/'camera_clearance.json')
    assert clearance['all_passed'] and len(clearance['records'])==4
    rigs={r['rig_id']:root/'rigs'/r['rig_id'] for r in plan['rigs']}
    old=ROOT/'outputs/experiments/second_environment_boundary_pass_a1_20261010'
    old_rig=read(old/'prepared_riverfront/rig.json')
    call_events=[json.loads(line) for line in (root/'render_calls.jsonl').read_text().splitlines()]
    started=[x for x in call_events if x['event']=='started'];completed=[x for x in call_events if x['event']=='completed']
    expected=[(k,r,c) for k in range(31) for r in ('rig_edge','rig_side') for c in ('cam2','cam4')]
    assert [(x['frame_index'],x['rig_id'],x['camera_id']) for x in started]==expected
    assert [(x['frame_index'],x['rig_id'],x['camera_id']) for x in completed]==expected
    assert len({(x['frame_index'],x['rig_id'],x['camera_id']) for x in started})==124
    assert read(root/'failure_attempt_1/diagnosis.json')['completed_consumed_units']==[dict(frame_index=0,rig_id='rig_edge')]
    first=read(root/'failure_attempt_1/single_process_setup.json')
    second=read(root/'single_process_setup.json')
    assert first['pid']!=second['pid'] and second['scene_load_count']==1
    assert read(root/'multirig_render_result.json')['normal_path_geometry_updates']==30
    world=np.load(root/'world_densekp.npy');assert world.shape==(31,1,138,3) and np.isfinite(world).all()
    assert np.array_equal(world,np.load(old/'prepared_riverfront/world_densekp.npy'))
    rig_results={}
    unit_states=[]
    timing_records={}
    relative=[]
    for rig_id,dir in rigs.items():
        recipe=read(dir/'recipe.json');rig=read(dir/'rig.json')
        assert recipe['scene_episode_id']==scene_id and recipe['rig_id']==rig_id
        assert np.array_equal(np.load(dir/'world_densekp.npy'),world)
        assert np.array_equal(np.load(dir/'scene_matrices.npy'),np.load(old/'prepared_riverfront/scene_matrices.npy'))
        poses={c['id']:np.asarray(c['camera_to_world']) for c in rig['cameras']}
        relative.append(np.linalg.inv(poses['cam2'])@poses['cam4'])
        if rig_id=='rig_edge':assert rig==old_rig
        pack=dir/'pack';ep=read(pack/'episode.json');table=read(pack/'sequence_index.json')
        validation=validate_sequence_index(ep,table,base_dir=pack,check_files=True,check_packed=True)
        assert ep['scene_episode_id']==scene_id and ep['rig_id']==rig_id
        assert all(b.get('rig_id')==rig_id for b in table['boxes'])
        assert all(o.get('rig_id')==rig_id for o in table['observations'])
        readback=read(pack/'readback.json');assert readback['actual_Dataset_reads']<=6
        in_view=[];box_count=0;roi_pixels=[];old_checks=0;old_xy_max=0.;old_in_view_equal=True
        timing_records[rig_id]=[]
        for k in range(31):
            frame=dir/f'frame_{k:03d}';report=read(frame/'report.json');assert report['frame_index']==k
            timing_records[rig_id].append(report['selected_time_record'])
            assert np.array_equal(np.load(frame/'shared_world_densekp.npy'),world[k])
            unit_states.append(dict(scene_episode_id=scene_id,frame_index=k,rig_id=rig_id,
                                    rendered=True,consumed=True,packed=True,
                                    model_rows=sum(x['observation_id'].startswith(f'f{k}_') for x in table['boxes'])))
            for camera in ('cam2','cam4'):
                folder=frame/camera;meta=read(folder/'raster_storage.json');box=read(folder/'boxes.json')[0]
                with Image.open(folder/'roi.png') as image:
                    x0,y0,x1,y1=meta['roi_xyxy'];assert image.size==(x1-x0,y1-y0)
                for mask_name in ('valid_mask.png','person_1_mask.png'):
                    with Image.open(folder/mask_name) as image:assert image.size==(x1-x0,y1-y0)
                assert meta['offset_xy']==[x0,y0] and meta['full_image_size_wh']==[5760,2880]
                roi_pixels.append((x1-x0)*(y1-y0));assert roi_pixels[-1]<=7_000_000
                with np.load(folder/'annotations.npz',allow_pickle=False) as ann:
                    assert str(ann['scene_episode_id'])==scene_id and str(ann['rig_id'])==rig_id
                    assert np.isfinite(ann['xy']).all() and np.isfinite(ann['world_densekp']).all()
                    in_view.append((k,camera,ann['in_view'][0].copy()))
                    if rig_id=='rig_edge':
                        prior=old/('probe_riverfront' if k<3 else 'riverfront_complete')/f'frame_{k:03d}'/camera/'annotations.npz'
                        with np.load(prior,allow_pickle=False) as prev:
                            old_xy_max=max(old_xy_max,float(np.max(np.abs(prev['xy']-ann['xy']))))
                            old_in_view_equal &= np.array_equal(prev['in_view'],ann['in_view'])
                            old_checks+=1
                if box['bbox_xyxy_exclusive'] is not None:box_count+=1
        common=[];edge=[]
        for k in range(31):
            a=next(v for frame,c,v in in_view if frame==k and c=='cam2')
            b=next(v for frame,c,v in in_view if frame==k and c=='cam4')
            both=int((a&b).sum());common.append(both)
            edge.append(bool(a.sum()>=8 and (~a).sum()>=5 and b.sum()>=8 and (~b).sum()>=5 and both>=8))
        if rig_id=='rig_edge':assert old_checks==62 and old_xy_max<=2e-5 and old_in_view_equal
        rig_results[rig_id]=dict(packed_rows=table['packed']['row_count'],valid_native_boxes=box_count,
            observations=len(table['observations']),Dataset_reads=readback['actual_Dataset_reads'],
            validation=validation,common_physical_points=common,common_at_least8_frames=sum(v>=8 for v in common),
            dual_edge_frames=sum(edge),max_roi_pixels=max(roi_pixels),old_edge_xy_max_abs_px=old_xy_max if rig_id=='rig_edge' else None,
            old_edge_in_view_exact=old_in_view_equal if rig_id=='rig_edge' else None,
            video_ref=str(pack/'dual_roi.mp4'))
    np.testing.assert_allclose(relative[0],relative[1],atol=1e-9,rtol=0)
    assert timing_records['rig_edge']==timing_records['rig_side']
    result=dict(status='passed',scene_episode_id=scene_id,normal_path_one_scene_load_and_person_update_per_frame=True,
        actual_blender_starts=2,actual_person_updates=31,failed_attempt_person_updates=1,
        successful_retry_person_updates=30,camera_calls=124,consumer_start_count=2,
        failed_attempt_retained=True,source_grouping='same_person_motion_trajectory_all_rigs',
        static_camera_center_clearance_passed=True,
        paired_relative_transform_max_abs=float(np.max(np.abs(relative[0]-relative[1]))),rigs=rig_results,
        peak_cpu_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(root/'unit_states.json',dict(schema='multirig_frame_rig_state_v1',units=unit_states,
          note='A person observation may be unavailable even when its frame-rig render and consumption succeeded'))
    write(root/'acceptance.json',result)
    return result


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();print(validate(args.root)['status'])
