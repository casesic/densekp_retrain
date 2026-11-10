#!/usr/bin/env python3
"""Small manifest/annotation audit for the two frozen composed road episodes."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from densekp_retrain.data.dynamic_scene import frame_record


def read(path):
    return json.loads(Path(path).read_text())


def audit(run, preflight, pack_name='pack'):
    run=Path(run).resolve(); preflight=Path(preflight).resolve()
    recipe=read(run/'recipe.json')
    planned=read(preflight/'planned_regions.json')
    rig={camera['id']:np.asarray(camera['camera_to_world']) for camera in read(run/'rig.json')['cameras']}
    world=np.load(run/'world_densekp.npy',mmap_mode='r')
    pack=run/pack_name; validation=read(pack/'validation.json'); table=read(pack/'sequence_index.json')
    readback=read(run/'pack/readback.json'); movie=read(run/'pack/labeled_video.json')
    if validation['status']!='passed' or readback['status']!='passed' or movie['status']!='passed':
        raise ValueError('consumer_or_video_not_accepted')
    if readback['actual_Dataset_reads']!=6:raise ValueError('six_readback_contract')
    if table['packed']['row_count']!=62 or validation['observation_counts']!={'observed':62}:
        raise ValueError('expected_single_rider_rows')
    if world.shape!=(31,1,138,3) or not np.isfinite(world).all():raise ValueError('world_R138_shape_or_finite')
    observed=[];areas=[];source_ticks=[]
    for k in range(31):
        row=frame_record(recipe,k)
        source_ticks.append(row['person_records'][0]['source_time_tick'])
        frame=run/f'frame_{k:03d}'
        manifest=read(frame/'render_frame.json')
        source_manifest=manifest['source_manifest']
        if source_manifest['selected_time_record']!=row or not manifest['body_cloth_densekp_shared_transform'] or not all(source_manifest[x] for x in
                ('body_readback_exact','head_transform_once','groom_topology_fixed')):
            raise ValueError('renderer_person_time_or_transform_conflict')
        actual_matrix=np.asarray(source_manifest['people_state'][0]['scene_from_model'])
        if not np.allclose(actual_matrix,np.asarray(recipe['people'][0]['scene_from_model'][k]),atol=1e-9,rtol=0):
            raise ValueError('scene_matrix_conflict')
        for camera in ('cam2','cam4'):
            folder=frame/camera
            raster=read(folder/'raster_storage.json')
            rect=raster['roi_xyxy']
            if rect!=planned[k][camera]['roi_xyxy']:
                raise ValueError('preflight_actual_ROI_conflict')
            with Image.open(folder/'roi.png') as image:
                if list(image.size)!=raster['stored_size_wh']:
                    raise ValueError('ROI_RGB_storage_size_conflict')
            with Image.open(folder/'valid_mask.png') as mask:
                if list(mask.size)!=raster['stored_size_wh']:
                    raise ValueError('ROI_physical_mask_size_conflict')
            with Image.open(folder/'person_1_mask.png') as mask:
                if list(mask.size)!=raster['stored_size_wh']:
                    raise ValueError('ROI_person_mask_size_conflict')
            with np.load(folder/'annotations.npz',allow_pickle=False) as ann:
                if not np.allclose(ann['world_densekp'],world[k],atol=2e-6,rtol=0):
                    raise ValueError('world_R138_annotation_conflict')
                if not np.allclose(ann['camera_to_world_erp'],rig[camera],atol=1e-9,rtol=0):
                    raise ValueError('camera_matrix_annotation_conflict')
                if not all(np.isfinite(ann[key]).all() for key in ('xy','world_densekp')):
                    raise ValueError('nonfinite_annotation')
                if int(ann['person_asset_frame'][0])!=row['person_records'][0]['asset_frame'] or int(ann['person_source_frame'][0])!=row['person_records'][0]['source_frame']:
                    raise ValueError('person_source_frame_conflict')
            areas.append((rect[2]-rect[0])*(rect[3]-rect[1]))
            observed.append(dict(frame=k,camera=camera,ROI=rect))
    if any(b<=a for a,b in zip(source_ticks,source_ticks[1:])):
        raise ValueError('source_time_not_strictly_increasing')
    if list(run.rglob('*.exr')) or list(run.rglob('_temporary_ds_source')):
        raise ValueError('successful_temporary_source_not_cleaned')
    component=recipe['component_provenance']
    ep=read(pack/'episode.json')
    if ep['assets'][0]['component_ids']!=component['selection']:
        raise ValueError('packed_component_provenance_lost')
    return dict(status='passed',run=str(run),pack=str(pack),frames=31,ROI_camera_calls=62,
                valid_rows=62,Dataset_reads=6,source_first_last_ticks=[source_ticks[0],source_ticks[-1]],
                ROI_pixel_area_min_max=[min(areas),max(areas)],all_preflight_ROI_exact=True,
                all_camera_matrices_exact=True,all_world_R138_and_source_times_correct=True,
                body_cloth_head_transform_checks=31,temporary_EXR_count=0,
                selected_components=component['selection'],source_group=component['source_group'],
                source_origin_record=component['selected_source_origin_record'],
                labeled_video=movie['labeled_video'],video_frames=movie['frames'],
                video_pts_first_last=[movie['episode_pts_seconds'][0],movie['episode_pts_seconds'][-1]])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True);parser.add_argument('--pack-name',default='pack')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.run,args.preflight,args.pack_name)
    if args.output.exists():raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result))
