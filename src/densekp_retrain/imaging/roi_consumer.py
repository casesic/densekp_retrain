"""Production direct ERP+ROI consumer; no A/B gates or full-size source dependency."""
import json
from pathlib import Path
from types import SimpleNamespace
import time
import resource
import cv2
import numpy as np
from PIL import Image
from .source import read_ds_source
from .direct_erp import direct_annotations
from .regions import embed_region,region_mask,apply_physical_mask,require_crop_in_region
from .ds import save_ds_annotations,DSImages
from .output_modes import complete_output
from densekp_retrain.data.boxes import BoxRejected,crop_from_box,validate_box_context
from densekp_retrain.data.observation_policy import RECORD,policy_name,no_box_state,validate_native_roi,EXPECTED_UNAVAILABLE,pair_availability,enforce_crop_policy
from densekp_retrain.data.dynamic_scene import frame_record,person_matrices
from densekp.core.keypoint_supervision import make_targets
from densekp_retrain.data.raster_storage import roi_metadata,save_roi_observation,validate_roi_box


def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def consume_roi_frame(root,frame):
    root=Path(root);began=time.monotonic();config=json.loads((root/'run_config.json').read_text())
    recipe=json.loads((root/'recipe.json').read_text());out=root/f'frame_{frame:03d}'
    timing=frame_record(recipe,frame);policy=policy_name(recipe)
    roi_only=config.get('storage_mode','full_image')=='roi_only'
    source=out/'_temporary_ds_source/source';manifest=json.loads((source/'source_manifest.json').read_text())
    if ('selected_timeline' in recipe or 'person_timelines' in recipe) and manifest['selected_time_record']!=timing:raise ValueError('render_consumer_time_conflict')
    plan=json.loads((source/'roi_plan.json').read_text());world=np.load(root/'world_densekp.npy')[frame]
    ids=[p['person_id'] for p in recipe['people']];times=[];checks=[];size=(5760,2880)
    mamma_cfg=config.get('mamma_512_label')
    mamma_geometry=None
    if mamma_cfg:
        from densekp_retrain.data.mamma_labels import load_world_body,policy_record
        mamma_geometry=load_world_body(recipe,root,frame,mamma_cfg['template_path'])
        if mamma_geometry['person_ids'].tolist()!=ids:raise ValueError('mamma_person_identity_conflict')
        policy_path=root/'label_policy.json'
        if not policy_path.exists():write(policy_path,policy_record(mamma_cfg.get('sidecar_policy','seven_box_probe_v1')))
    edge_inputs={};edge_support={}
    for c,info in plan['cameras'].items():
        start=time.monotonic();roi=info['roi_xyxy'];x0,y0,x1,y1=roi
        local=read_ds_source(source/'roi',c)
        low=None if roi_only else np.asarray(Image.open(source/'background'/f'{c}_preview.png').convert('RGB'))
        assert local.rgb.shape==(y1-y0,x1-x0,3) and (roi_only or low.shape==(720,1440,3))
        read_seconds=time.monotonic()-start;t=time.monotonic()
        depth=embed_region(local.ray_range_m,roi,size,unknown_value=np.nan)
        instance=embed_region(local.instance,roi,size,unknown_value=np.nan)
        valid=np.asarray(Image.open(root/config['valid_masks'][c]))>0
        assert valid.shape==(2880,5760)
        data=direct_annotations(world,ids,np.array(info['camera_to_world']),c,valid,depth,instance,coverage_region=roi)
        if 'scene_episode_id' in recipe or 'rig_id' in recipe:
            if not recipe.get('scene_episode_id') or not recipe.get('rig_id'):
                raise ValueError('incomplete_scene_rig_identity')
            data.update(scene_episode_id=np.array(recipe['scene_episode_id']),
                        rig_id=np.array(recipe['rig_id']))
        if 'occlusion_metric' in recipe:
            from densekp_retrain.data.occlusion_events import occlusion_evidence
            # Exact geometry range; evidence is separate from existing visibility.
            distance=np.linalg.norm(world-np.array(info['camera_to_world'])[:3,3],axis=2)
            data.update(occlusion_evidence(data['xy'],distance,data['visible'],data['in_view'],ids,depth,instance))
        assert np.isfinite(data['xy']).all() and np.isfinite(data['world_densekp']).all()
        if 'selected_timeline' in recipe:
            data.update(frame_index=np.array(frame,dtype=np.int64),time_tick=np.array(timing['episode_time_tick'],dtype=np.int64),
                time_base=np.array(timing['time_base'],dtype=np.int64),source_time_tick=np.array(timing['source_time_tick'],dtype=np.int64),
                source_frame_index=np.array(timing['source_frame'],dtype=np.int64),asset_frame_index=np.array(timing['asset_frame'],dtype=np.int64))
        if 'person_timelines' in recipe:
            records=timing['person_records']
            data.update(frame_index=np.array(frame,dtype=np.int64),time_tick=np.array(timing['episode_time_tick'],dtype=np.int64),time_base=np.array(timing['time_base'],dtype=np.int64))
            for key in ['asset_frame','source_frame','source_time_tick','source_time_base','quantization_error_seconds']:
                data['person_'+key]=np.array([r[key] for r in records],dtype=np.int64)
            expected=np.load(recipe['expected_in_view_ref'])[frame,:,['cam2','cam4'].index(c)]
            np.testing.assert_array_equal(data['in_view'],expected,err_msg='actual physical domain differs from frozen CPU projection')
            edge_inputs[c]=data['in_view'][ids.index(recipe['target_person_id'])]
        label_seconds=time.monotonic()-t;t=time.monotonic()
        mamma_data=None;mamma_seconds=0.0
        if mamma_geometry is not None:
            from densekp_retrain.data.mamma_labels import build_evidence
            mt=time.monotonic()
            mamma_data=build_evidence(mamma_geometry['world'],mamma_geometry['full_body'],mamma_geometry['faces'],ids,
                np.array(info['camera_to_world']),c,valid,depth,instance,roi)
            mamma_data.update(template_id=np.array(mamma_cfg['template_id']),ordered_vertex_ids=mamma_geometry['vertex_ids'],
                sidecar_policy=np.array(mamma_cfg.get('sidecar_policy','seven_box_probe_v1')),
                asset_frame_index=mamma_geometry['asset_frames'],source_frame_index=mamma_geometry['source_frames'],
                frame_index=np.array(frame,np.int64),time_tick=np.array(timing['episode_time_tick'],np.int64),
                time_base=np.asarray(timing['time_base'],np.int64),render_coverage_xyxy=np.asarray(roi,np.int32),
                full_image_size_wh=np.asarray(size,np.int32),camera_id=np.array(c),rig_id=np.array(recipe.get('rig_id','')),
                scene_episode_id=np.array(recipe.get('scene_episode_id',recipe['episode_id'])))
            mamma_seconds=time.monotonic()-mt
        meta=roi_metadata(roi,size,(root/config['valid_masks'][c]).resolve()) if roi_only else None
        if roi_only:
            rgb=apply_physical_mask(local.rgb,valid[y0:y1,x0:x1]);compose_seconds=time.monotonic()-t;t=time.monotonic()
        else:
            rgb=cv2.resize(low,size,interpolation=cv2.INTER_LINEAR);rgb[y0:y1,x0:x1]=local.rgb
            rgb=apply_physical_mask(rgb,valid);compose_seconds=time.monotonic()-t;t=time.monotonic()
        dest=out/c;dest.mkdir(exist_ok=False)
        if roi_only:
            save_roi_observation(dest,rgb,local.instance,data,valid[y0:y1,x0:x1],meta)
        else:
            Image.fromarray(rgb).save(dest/'normal_erp.png');Image.fromarray(valid.astype(np.uint8)*255).save(dest/'valid_mask.png')
            save_ds_annotations(dest,data,ids,DSImages(None,rgb,instance),SimpleNamespace(valid_mask=valid),output_mode='compact')
            Image.fromarray(region_mask(roi,size).astype(np.uint8)*255).save(dest/'high_quality_region.png')
            write(dest/'render_coverage.json',dict(full_image_size_wh=list(size),high_quality_region_xyxy=roi,label_coverage_region_xyxy=roi,
                region_offset_xy=[x0,y0],local_shape_hw=[y1-y0,x1-x0],physical_valid_mask='valid_mask.png',
                outside_region_depth_and_ID='unknown; low-resolution buffers never used as GT',
                person_mask_zero_outside_region='unknown, not certified background',supported_person_ids=ids,
                crop_policy='deployment; every packed/new crop must pass require_crop_in_region',full_buffer_dependency=False))
        encode_seconds=time.monotonic()-t;t=time.monotonic()
        if mamma_data is not None:np.savez(dest/'mamma_512.npz',**mamma_data)
        boxes=json.loads((dest/'boxes.json').read_text())
        if [b['person_id'] for b in boxes]!=ids:raise ValueError('instance_box_identity_conflict')
        if mamma_data is not None and mamma_cfg.get('sidecar_policy')=='mask_native_v2':
            from densekp_retrain.data.mamma_crop import mask_native_support
            for box in boxes:
                mask=np.asarray(Image.open(dest/f"person_{box['person_id']}_mask.png"))
                support=mask_native_support(mask,roi,size)
                if (support is None)!=(box['bbox_xyxy_exclusive'] is None):
                    raise ValueError('mask_box_presence_conflict')
        if mamma_data is not None and mamma_cfg.get('sidecar_policy','seven_box_probe_v1')=='seven_box_probe_v1':
            from densekp_retrain.data.mamma_labels import seven_box_support
            box_support=[]
            for box in boxes:
                native=box['bbox_xyxy_exclusive']
                box_support.append(dict(person_id=box['person_id'],status='observed' if native is not None else 'absent',
                    reason=None if native is not None else 'no_visible_instance_box',native_box_xyxy=native,
                    variants=[] if native is None else seven_box_support(native,roi,size)))
            write(dest/'mamma_box_support.json',dict(schema='mamma_fixed_seven_box_support_v1',camera_id=c,frame_index=frame,
                full_image_size_wh=list(size),render_coverage_xyxy=roi,box_source='actual rendered instance mask',people=box_support))
        for j,box in enumerate(boxes):
            if policy==RECORD:
                pm=np.asarray(Image.open(dest/f'person_{ids[j]}_mask.png'))>0
                try:
                    center,scale=validate_native_roi(box['bbox_xyxy_exclusive'],box,ids[j],int(data['person_ids'][j]),meta,valid[y0:y1,x0:x1],pm)
                except BoxRejected as error:
                    if str(error) not in EXPECTED_UNAVAILABLE:raise
                    status,reason=no_box_state(data['in_view'][j],data['visible'][j],data.get('occlusion_evidence_code',None)[j] if 'occlusion_evidence_code' in data else None) if box['bbox_xyxy_exclusive'] is None else ('empty',str(error))
                    checks.append(dict(camera=c,person_id=ids[j],status=status,reason=reason,model_crop_available=False,geometry_GT_retained=True));continue
            if box['bbox_xyxy_exclusive'] is None:
                checks.append(dict(camera=c,person_id=ids[j],status='empty',reason='no_visible_instance_box'));continue
            pm=np.asarray(Image.open(dest/f'person_{ids[j]}_mask.png'))>0
            if roi_only and policy!=RECORD:
                center,scale=validate_roi_box(box['bbox_xyxy_exclusive'],box,ids[j],int(data['person_ids'][j]),meta,valid[y0:y1,x0:x1],pm)
            elif not roi_only:
                validate_box_context(box['bbox_xyxy_exclusive'],bbox_kind='visible_instance_tight',bbox_source=box['source'],
                    person_id=ids[j],mask_person_id=box['person_id'],annotation_person_id=int(data['person_ids'][j]),
                    image_size_wh=size,manifest_size_wh=size,valid_mask=valid,person_mask=pm,boundary_policy='erp_reject_seam',min_side_px=2)
                center,scale=crop_from_box(box['bbox_xyxy_exclusive'],image_size_wh=size,crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
                require_crop_in_region(center,scale,roi,size)
            targets=make_targets(data['xy'][j],data['in_view'][j],center,scale)
            assert np.isfinite(targets).all()
            if ids[j]==recipe.get('target_person_id'):edge_support[c]=targets[:,2]>0
            if policy==RECORD and not (targets[:,2]>0).any():
                checks.append(dict(camera=c,person_id=ids[j],status='empty',reason='no_GT_inside_full_crop',
                    legal_crop=True,training_row_available=False,geometry_GT_retained=True,supervised=0));continue
            checks.append(dict(camera=c,person_id=ids[j],status='passed',in_view=int(data['in_view'][j].sum()),
                supervised=int((targets[:,2]>0).sum()),supervised_occluded=int(((targets[:,2]>0)&~data['visible'][j]).sum()),center=center.tolist(),scale=float(scale)))
        # Real encoded final files are decoded before deleting this frame's buffers.
        np.testing.assert_array_equal(np.asarray(Image.open(dest/('roi.png' if roi_only else 'normal_erp.png'))),rgb)
        with np.load(dest/'annotations.npz') as saved:
            np.testing.assert_array_equal(saved['xy'],data['xy'])
        times.append(dict(camera=c,read_seconds=read_seconds,label_and_buffers_seconds=label_seconds,mamma_512_seconds=mamma_seconds,
            compose_seconds=compose_seconds,save_seconds=encode_seconds,checks_seconds=time.monotonic()-t,total_seconds=time.monotonic()-start))
    dual_edge=None
    if 'dual_edge_gate' in recipe:
        from densekp_retrain.data.mixed_road_pass import EDGE
        common=np.logical_and(*edge_inputs.values())
        simultaneous=all(v.sum()>=EDGE['min_inside_each'] and (~v).sum()>=EDGE['min_outside_each'] for v in edge_inputs.values()) and common.sum()>=EDGE['min_common_domain']
        availability=pair_availability(edge_support,minimum=EDGE['min_common_crop'])
        enforce_crop_policy(recipe,simultaneous,availability)
        dual_edge=dict(simultaneous=bool(simultaneous),common_physical_points=int(common.sum()),common_crop_points=(availability['common_points'] if policy==RECORD else (availability['common_points'] or 0)),model_input_availability=availability,observation_policy=policy)
    np.save(out/'shared_world_densekp.npy',world)
    retained=[]
    for record in manifest['sources']:
        r=dict(record);r['ephemeral_filename']=Path(r.pop('source')).name;r['retained']=False;retained.append(r)
    manifest['sources']=retained
    write(out/'render_frame.json',dict(source_manifest=manifest,plan=plan,run_config_ref=str(root/'run_config.json'),
        temporary_buffers_are_dependencies=False,body_cloth_densekp_shared_transform=True))
    write(out/'report.json',dict(final_erp_resolution=list(size),frame_index=frame,time_tick=timing['episode_time_tick'],time_base=timing['time_base'],selected_time_record=timing,
        imaging_route='direct_erp_roi',storage_mode=config.get('storage_mode','full_image'),output_mode='compact',checks=checks,camera_timings=times,
        dual_edge=dual_edge,consume_seconds=time.monotonic()-began,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
    raw_bytes=sum(p.stat().st_size for p in source.rglob('*') if p.is_file())
    retain=roi_only and frame in config.get('retain_reference_frames',[0,5,10])
    if not retain:
        complete_output(out,'compact',source,checks=dict(status='passed',all_final_files_decoded=True,all_native_crops_covered=True,
            discarded_temporary_bytes=raw_bytes,Dataset_checks='six selected rows after sequence pack'))
    write(out/'consume_timing.json',dict(seconds=time.monotonic()-began,cameras=times,source_bytes=raw_bytes,retained_for_same_buffer_comparison=retain,discarded_temporary_bytes=0 if retain else raw_bytes,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
