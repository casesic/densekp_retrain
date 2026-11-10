"""Bounded postproduction check for the fixed three-cyclist, two-rig episode."""
import argparse
import json
from pathlib import Path
import resource
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src'), str(ROOT / 'assets_external/python_deps')]
from densekp_retrain.data.sequence_index import validate_sequence_index
from densekp_retrain.data.dynamic_scene import geometry_paths


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def longest(flags):
    run = best = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def validate(root):
    root = Path(root).resolve()
    spec = read(root / 'multirig.json')
    scene_id = spec['scene_episode_id']
    rig_ids = [r['rig_id'] for r in spec['rigs']]
    if rig_ids != ['rig_edge', 'rig_front_oblique']:
        raise ValueError('unexpected_rig_list')
    scene = read(root / 'recipe.json')
    if [p['person_id'] for p in scene['people']] != [1, 2, 3]:
        raise ValueError('scene_person_identity_conflict')
    provenance=scene['component_provenance']
    if (len({x['selection']['body_id'] for x in provenance})!=3 or
        len({x['selection']['motion_id'] for x in provenance})!=3 or
        len({x['source_group'] for x in provenance})!=3 or
        len({x['selection']['clothing_id'] for x in provenance})!=3):
        raise ValueError('three_distinct_riders_motion_sources_or_clothes_not_met')
    actual_motion=[]
    for person in scene['people']:
        asset=read(person['asset_manifest'])
        vertices=np.load(geometry_paths(asset)['body_vertices'],mmap_mode='r')
        frames=person['asset_frames']
        if len(set(frames))!=31 or not all(frames[k]<frames[k+1] for k in range(30)):
            raise ValueError('person_asset_frame_not_strictly_advancing')
        differences=[float(np.max(np.abs(vertices[frames[a]]-vertices[frames[b]])))
                     for a,b in ((0,15),(15,30))]
        if min(differences)<=1e-5:raise ValueError('static_or_repeated_body_motion')
        actual_motion.append(dict(person_id=person['person_id'],asset_frames=[frames[k] for k in (0,15,30)],
                                  max_vertex_changes_m=differences))
    source_world = np.load(root / 'world_densekp.npy')
    if source_world.shape != (31, 3, 138, 3) or not np.isfinite(source_world).all():
        raise ValueError('shared_world_axis_or_finite_conflict')
    if not np.array_equal(source_world, np.load(Path(scene['scene_matrices_ref']).parent / 'world_densekp.npy')):
        raise ValueError('shared_world_changed')
    calls = [json.loads(line) for line in (root / 'render_calls.jsonl').read_text().splitlines()]
    expected_order = [(k, r, c) for k in range(31) for r in rig_ids for c in ('cam2','cam4')]
    started = [(x['frame_index'], x['rig_id'], x['camera_id']) for x in calls if x['event']=='started']
    completed = [(x['frame_index'], x['rig_id'], x['camera_id']) for x in calls if x['event']=='completed']
    if started != expected_order or completed != expected_order:
        raise ValueError('frame_rig_camera_order_or_completion_conflict')
    render = read(root / 'multirig_render_result.json')
    if render['scene_load_count']!=1 or render['normal_path_geometry_updates']!=31 or render['consumer_process_count']!=1:
        raise ValueError('shared_scene_lifecycle_conflict')
    if not read(root/'camera_clearance.json')['all_passed']:
        raise ValueError('camera_static_clearance_conflict')
    matrices = np.load(Path(scene['scene_matrices_ref']))
    source_rig = read(Path(scene['scene_matrices_ref']).parent/'rig.json')
    results = {}
    timing = {}
    relative = []
    states = []
    for rig_id in rig_ids:
        folder = root/'rigs'/rig_id
        recipe = read(folder/'recipe.json')
        rig = read(folder/'rig.json')
        if recipe['scene_episode_id'] != scene_id or recipe['rig_id']!=rig_id:
            raise ValueError('recipe_rig_identity_conflict')
        if [p['person_id'] for p in recipe['people']] != [1,2,3] or recipe['people']!=scene['people']:
            raise ValueError('rig_person_plan_conflict')
        if not np.array_equal(np.load(folder/'world_densekp.npy'),source_world) or not np.array_equal(np.load(folder/'scene_matrices.npy'),matrices):
            raise ValueError('rig_world_or_matrices_changed')
        if rig_id=='rig_edge' and rig!=source_rig:
            raise ValueError('edge_rig_not_original')
        poses={x['id']:np.asarray(x['camera_to_world']) for x in rig['cameras']}
        relative.append(np.linalg.inv(poses['cam2']) @ poses['cam4'])
        pack=folder/'pack'
        episode=read(pack/'episode.json')
        index=read(pack/'sequence_index.json')
        validated=validate_sequence_index(episode,index,base_dir=pack,check_files=True,check_packed=True)
        if episode['scene_episode_id']!=scene_id or episode['rig_id']!=rig_id or len(index['observations'])!=186:
            raise ValueError('sequence_rig_or_observation_count_conflict')
        if any(x['rig_id']!=rig_id for x in index['observations']+index['boxes']):
            raise ValueError('cross_rig_index_row')
        readback=read(pack/'readback.json')
        planned=read(pack/'readback_plan.json')
        expected_slots={f'f{k}_{c}_p{p}' for k in (0,15) for c in ('cam2','cam4') for p in (1,2,3)}
        if set(planned['selected_observation_ids'])!=expected_slots or readback['actual_Dataset_reads']>12:
            raise ValueError('readback_plan_or_limit_conflict')
        boxes={o['observation_id'] for o in index['boxes']}
        in_view={};visible_points={};observed={};roi_pixels=[]
        timing[rig_id]=[]
        for k in range(31):
            frame=folder/f'frame_{k:03d}'
            report=read(frame/'report.json')
            timing[rig_id].append(report['selected_time_record'])
            if not np.array_equal(np.load(frame/'shared_world_densekp.npy'),source_world[k]):
                raise ValueError(f'world_points_changed:{rig_id}:{k}')
            states.append(dict(scene_episode_id=scene_id,rig_id=rig_id,frame_index=k,
                               rendered=True,consumed=True,packed=True,
                               model_rows=sum(f'f{k}_{c}_p{p}' in boxes for p in (1,2,3) for c in ('cam2','cam4'))))
            for ci,c in enumerate(('cam2','cam4')):
                camera=frame/c
                meta=read(camera/'raster_storage.json')
                x0,y0,x1,y1=meta['roi_xyxy'];size=(x1-x0,y1-y0)
                with Image.open(camera/'roi.png') as image:
                    if image.size!=size:raise ValueError('ROI_RGB_size_conflict')
                with Image.open(camera/'valid_mask.png') as image:
                    if image.size!=size:raise ValueError('ROI_valid_size_conflict')
                if meta['offset_xy']!=[x0,y0] or meta['full_image_size_wh']!=[5760,2880]:
                    raise ValueError('ROI_offset_or_full_size_conflict')
                area=size[0]*size[1]
                roi_pixels.append(area)
                if area>7_000_000:raise ValueError('production_ROI_area_exceeded')
                rows=read(camera/'boxes.json')
                if [x['person_id'] for x in rows]!=[1,2,3]:raise ValueError('box_person_axis_conflict')
                with np.load(camera/'annotations.npz',allow_pickle=False) as annotation:
                    if str(annotation['scene_episode_id'])!=scene_id or str(annotation['rig_id'])!=rig_id:
                        raise ValueError('annotation_rig_identity_conflict')
                    if annotation['xy'].shape!=(3,138,2) or annotation['in_view'].shape!=(3,138):
                        raise ValueError('annotation_person_axis_conflict')
                    if not np.isfinite(annotation['xy']).all() or not np.isfinite(annotation['world_densekp']).all():
                        raise ValueError('nonfinite_annotation')
                    if not np.array_equal(annotation['world_densekp'],source_world[k]):
                        raise ValueError('annotation_world_changed')
                    for j,p in enumerate((1,2,3)):
                        if not np.array_equal(annotation['in_view'][j],
                                              np.load(folder/'expected_in_view.npy',mmap_mode='r')[k,j,ci]):
                            raise ValueError(f'physical_projection_conflict:{rig_id}:{k}:{c}:p{p}')
                        in_view[(k,c,p)]=annotation['in_view'][j].copy()
                        visible_points[(k,c,p)]=int(annotation['visible'][j].sum())
                        native_box=rows[j]['bbox_xyxy_exclusive'] is not None
                        observed[(k,c,p)]=f'f{k}_{c}_p{p}' in boxes
                        if observed[(k,c,p)] and not native_box:
                            raise ValueError('packed_row_without_native_box')
                        if native_box and not observed[(k,c,p)]:
                            slot=next(x for x in index['observations'] if x['observation_id']==f'f{k}_{c}_p{p}')
                            if slot['status']=='observed' or not slot.get('reason'):
                                raise ValueError('unexplained_native_box_rejection')
                        with Image.open(camera/f'person_{p}_mask.png') as image:
                            if image.size!=size:raise ValueError('ROI_person_mask_size_conflict')
        common=[];edge=[];visible_runs={}
        for p in (1,2,3):
            for c in ('cam2','cam4'):
                flags=[observed[(k,c,p)] and visible_points[(k,c,p)]>=8 for k in range(31)]
                visible_runs[f'p{p}_{c}']=longest(flags)
        for k in range(31):
            a=in_view[(k,'cam2',1)];b=in_view[(k,'cam4',1)]
            common.append(int((a&b).sum()))
            edge.append(bool(a.sum()>=8 and (~a).sum()>=5 and b.sum()>=8 and (~b).sum()>=5 and common[-1]>=8))
        results[rig_id]=dict(packed_rows=index['packed']['row_count'],observations=len(index['observations']),
                             unavailable=len(index['observations'])-index['packed']['row_count'],
                             Dataset_reads=readback['actual_Dataset_reads'],validation=validated,
                             target_common_physical_at_least8=sum(x>=8 for x in common),
                             target_dual_edge_frames=sum(edge),max_roi_pixels=max(roi_pixels),
                             observed_runs=visible_runs,video_ref=str(pack/'dual_roi.mp4'))
    np.testing.assert_allclose(relative[0],relative[1],atol=1e-9,rtol=0)
    if timing['rig_edge']!=timing['rig_front_oblique']:
        raise ValueError('rig_source_time_divergence')
    front=read(root.parent/'prepared_front_oblique/geometry_preflight.json')['coverage']
    if not front['passed'] or results['rig_edge']['target_dual_edge_frames']<16:
        raise ValueError('coverage_goal_conflict')
    each_person={p:max(results[r]['observed_runs'][f'p{p}_{c}'] for r in rig_ids for c in ('cam2','cam4')) for p in (1,2,3)}
    result=dict(status='passed',scene_episode_id=scene_id,camera_calls=len(started),
                scene_loads=render['scene_load_count'],normal_path_geometry_updates=render['normal_path_geometry_updates'],
                consumer_process_count=render['consumer_process_count'],planned_observation_slots=372,
                Dataset_reads=sum(results[r]['Dataset_reads'] for r in rig_ids),
                each_person_longest_observed_run=each_person,three_person_content_passed=all(x>=6 for x in each_person.values()),
                paired_relative_transform_max_abs=float(np.max(np.abs(relative[0]-relative[1]))),
                shared_source_group='APD6526+participant_1_comfortable+participant_2_straight_all_rigs',
                actual_articulation_checks=actual_motion,
                rigs=results,peak_validation_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write(root/'unit_states.json',dict(schema='multirig_frame_rig_state_v1',units=states))
    write(root/'acceptance.json',result)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    print(validate(args.root)['status'])
