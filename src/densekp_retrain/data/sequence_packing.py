"""Pack one manifest-backed rendered sequence, keeping explicit missing slots.

Only compact final rasters are live Dataset dependencies. This does not merge
episodes, change the Dataset tuple or use visibility to exclude reliable GT.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, make_targets
from .boxes import BoxRejected, crop_from_box, validate_box_context, rejection_record
from .packing import pack_derivative_rows
from .sequence_index import SCHEMA, validate_sequence_index
from densekp_retrain.imaging.regions import require_crop_in_region
from densekp.core.datasets.image_layout import LAYOUT_FIELDS,parse_image_layout
from .raster_storage import validate_roi_box
from .dynamic_scene import frame_record,person_frame_record
from .frame_sources import frame_directory
from .observation_policy import RECORD,policy_name,no_box_state,validate_native_roi,EXPECTED_UNAVAILABLE,pair_availability


def optional_mamma_512_refs(camera_dir, *, annotation_person_ids, person_index,
                            frame_index, camera_id, label_policy_path, annotation=None,
                            rig_id=None, scene_episode_id=None):
    """Validate and expose optional MAMMA sidecars without changing old packs."""
    camera_dir=Path(camera_dir)
    label_path=camera_dir/'mamma_512.npz'
    support_path=camera_dir/'mamma_box_support.json'
    present=(label_path.is_file(),support_path.is_file())
    if not any(present):return {}
    if not present[0]:raise ValueError('incomplete_mamma_512_sidecars')
    label_policy_path=Path(label_policy_path)
    if not label_policy_path.is_file():raise ValueError('missing_mamma_512_label_policy')
    label_policy=json.loads(label_policy_path.read_text())
    person_ids=np.asarray(annotation_person_ids,dtype=np.int64)
    with np.load(label_path,allow_pickle=False) as labels:
        sidecar_policy=str(labels['sidecar_policy']) if 'sidecar_policy' in labels else 'seven_box_probe_v1'
        if sidecar_policy not in ('seven_box_probe_v1','mask_native_v2'):raise ValueError('unknown_mamma_sidecar_policy')
        if (sidecar_policy=='seven_box_probe_v1')!=present[1]:raise ValueError('incomplete_mamma_512_sidecars')
        if sidecar_policy=='mask_native_v2':
            if label_policy.get('sidecar_policy')!=sidecar_policy:raise ValueError('mamma_label_policy_conflict')
            if annotation is None or rig_id is None or scene_episode_id is None:
                raise ValueError('mamma_normal_source_context_missing')
            if (str(labels['rig_id'])!=rig_id or str(labels['scene_episode_id'])!=scene_episode_id or
                int(labels['time_tick'])!=int(annotation['time_tick']) or
                not np.array_equal(labels['time_base'],annotation['time_base']) or
                not np.array_equal(labels['asset_frame_index'],annotation['person_asset_frame']) or
                not np.array_equal(labels['source_frame_index'],annotation['person_source_frame'])):
                raise ValueError('mamma_normal_source_time_conflict')
        if str(labels['schema'])!='mamma_512_evidence_v1':raise ValueError('unsupported_mamma_512_schema')
        if not np.array_equal(labels['person_ids'],person_ids):raise ValueError('mamma_512_person_ids_conflict')
        if labels['xy_full_erp'].shape!=(len(person_ids),512,2):raise ValueError('invalid_mamma_512_xy_shape')
        if labels['visibility_known'].shape!=(len(person_ids),512):raise ValueError('invalid_mamma_512_visibility_shape')
        if labels['visibility_value'].shape!=(len(person_ids),512):raise ValueError('invalid_mamma_512_visibility_value_shape')
        if int(labels['frame_index'])!=int(frame_index):raise ValueError('mamma_512_frame_conflict')
        if str(labels['camera_id'])!=str(camera_id):raise ValueError('mamma_512_camera_conflict')
    if not 0<=int(person_index)<len(person_ids):raise ValueError('invalid_mamma_512_person_index')
    refs=dict(mamma_512_ref=str(label_path),mamma_512_person_index=int(person_index),
              mamma_label_policy_ref=str(label_policy_path),mamma_sidecar_policy=sidecar_policy)
    if present[1]:refs['mamma_box_support_ref']=str(support_path)
    return refs


def pack_rendered_sequence(directory, *, output_name):
    directory=Path(directory).resolve();recipe=json.loads((directory/'recipe.json').read_text())
    policy=policy_name(recipe)
    out=directory/output_name;out.mkdir(exist_ok=False)
    people=recipe['people'];assets={p['asset_id']:json.loads(Path(p['asset_manifest']).read_text()) for p in people}
    registry=[dict(person_id=p['person_id'],asset_id=p['asset_id'],source_ref=p['asset_manifest'],character_family_id=p['character_family'],identity_status='synthetic_instance') for p in people]
    ep=dict(schema=SCHEMA,episode_id=recipe['episode_id'],sequence_id=recipe['sequence_id'],data_kind='synthetic',temporal_kind='sequence',
        time_base=recipe['time_base'],time_semantics='synthetic_clock',time_origin=('selected source tick '+str(recipe['time_origin']['source_time_tick'])) if 'selected_timeline' in recipe else 'local episode zero; source time mapping retained per observation',
        timeline=dict(first_frame_index=0,frame_step=1,frame_count=recipe['frame_count'],expected_tick_step=None if 'selected_timeline' in recipe else 1),
        synchronization=dict(status='synthetic_same_state',basis='both cameras consume the exact same asset frame and scene matrix',evidence_ref=str(directory/'recipe.json')),
        source_refs=[str(directory/'recipe.json')],recipe_ref=str(directory/'recipe.json'),seed=recipe['child_seed'],generator_version='road_pass_scene_owned_v1' if 'selected_timeline' in recipe else 'road_pass_v1',
        integration_only=True,independent_validation=False,
        cameras=[dict(camera_id=c,rig_ref=str(directory/'rig.json'),calibration_ref=str(directory/'rig.json'),image_size_wh=[5760,2880]) for c in ['cam2','cam4']],
        assets=[dict(asset_id=k,asset_ref=next(p['asset_manifest'] for p in people if p['asset_id']==k),motion_source_ref=a['source_ref']) for k,a in assets.items()],people=registry)
    if 'scene_episode_id' in recipe or 'rig_id' in recipe:
        if not recipe.get('scene_episode_id') or not recipe.get('rig_id'):
            raise ValueError('incomplete_scene_rig_identity')
        ep.update(scene_episode_id=recipe['scene_episode_id'],rig_id=recipe['rig_id'],
                  rig_seed=recipe.get('rig_seed'),rig_plan_version=recipe.get('rig_plan_version'))
        for camera in ep['cameras']:camera['rig_id']=recipe['rig_id']
    if 'selected_timeline' in recipe:
        ep.update(time_unit_status=recipe['time_unit_status'],sampling_policy=recipe['sampling_policy'],unselected_asset_frames=recipe['unselected_asset_frames'],
            original_timeline_ref=recipe['original_timeline_ref'],selected_timeline_ref=recipe['selected_timeline_ref'],scene_matrices_ref=recipe['scene_matrices_ref'])
        ep['source_refs'] += [recipe['original_timeline_ref'],recipe['selected_timeline_ref'],recipe['scene_matrices_ref']]
        if 'scene_matrix_person_ids' in recipe:
            ep['scene_matrix_person_ids']=recipe['scene_matrix_person_ids']
            ep['generator_version']=recipe['generator_version']
    if 'person_timelines' in recipe:
        for key in ['time_mapping_schema','person_timeline_refs','source_timeline_refs','episode_timeline_ref','scene_matrices_ref','scene_matrix_person_ids','generator_version','time_unit_status']:
            ep[key]=recipe[key]
        ep['source_refs'] += [recipe['episode_timeline_ref'],recipe['scene_matrices_ref']]+list(recipe['person_timeline_refs'].values())+list(recipe['source_timeline_refs'].values())
        ep['synchronization']['basis']='both cameras consume same complete per-person mapped world state; synthetic10Hz, not exposure certification'
    if 'component_provenance' in recipe:
        provenance=recipe['component_provenance']
        entries=provenance if isinstance(provenance,list) else [dict(provenance,person_id=people[0]['person_id'])]
        if len({entry['person_id'] for entry in entries})!=len(entries):
            raise ValueError('duplicate_component_provenance_person')
        for entry in entries:
            person=next((p for p in people if p['person_id']==entry['person_id']),None)
            if person is None:raise ValueError('unknown_component_provenance_person')
            asset=assets[person['asset_id']]
            selection=entry.get('selection',entry.get('components'))
            if selection is None or ('selection' in entry and 'components' in entry and entry['selection']!=entry['components']):
                raise ValueError('invalid_component_provenance_selection')
            if (asset.get('components')!=selection or
                    person.get('component_ids')!=selection or
                    person.get('source_group')!=entry['source_group'] or
                    asset.get('source_group')!=entry['source_group']):
                raise ValueError('component_provenance_asset_conflict')
            delivery=entry.get('delivery_ref',person.get('composition_delivery_ref'))
            frozen=Path(delivery).parent/'frozen_recipe.json' if delivery else None
            component_versions=entry.get('versions',asset.get('component_versions'))
            if component_versions is None and frozen is not None and frozen.is_file():
                resolved=json.loads(frozen.read_text())
                component_versions={name:resolved['components'][name]['version'] for name in
                                    ('body','motion','clothing','hair')}
                if {name+'_id':resolved['components'][name]['id'] for name in component_versions}!=selection:
                    raise ValueError('component_dependency_version_conflict')
            asset_row=next(a for a in ep['assets'] if a['asset_id']==person['asset_id'])
            asset_row.update(component_ids=selection,
                component_versions=component_versions,
                composition_delivery_ref=delivery,
                source_group=entry['source_group'],
                technical_use_policy=entry.get('use_policy',asset.get('technical_use_policy')))
        ep['component_provenance']=provenance
    if policy==RECORD:
        ep['observation_policy']=policy
        resume_ref=directory/'resume_manifest.json'
        if recipe.get('parent_run_ref') and not resume_ref.is_file():
            raise ValueError('missing_resume_manifest_for_continuation')
        if resume_ref.is_file():
            ep['resume_manifest_ref']=str(resume_ref)
            ep['source_refs'].append(str(resume_ref))
    table=dict(schema=SCHEMA,episode_id=ep['episode_id'],sequence_id=ep['sequence_id'],frames=[],observations=[],boxes=[],packed=dict(path=str(out/'packed.npz'),row_count=0))
    if 'scene_episode_id' in ep:table.update(scene_episode_id=ep['scene_episode_id'],rig_id=ep['rig_id'])
    partial_path=directory/'partial_delivery.json'
    partial=json.loads(partial_path.read_text()) if partial_path.exists() else None
    if partial:
        ep.update(execution_status='partial_stopped',execution_stop_ref=str(partial_path));ep['source_refs'].append(str(partial_path))
    rows=[];rejects=[];coverage=[]
    for k in range(recipe['frame_count']):
        timing=frame_record(recipe,k)
        frame=frame_directory(directory,k);raster_report=frame/'report.json';acceptance_status='passed'
        if not raster_report.exists() and partial and k in partial['rejected_frame_indices']:
            raster_report=frame/'rejected_frame.json';rejected=json.loads(raster_report.read_text())
            if rejected.get('acceptance_status')!='rejected' or not rejected.get('reason'):raise ValueError('invalid_rejected_frame_record')
            acceptance_status='rejected'
        available=raster_report.is_file()
        reason=(rejected['reason'] if acceptance_status=='rejected' else '') if available else (partial['stop_reason'] if partial else 'not rendered within the completed stage; no frame duplication')
        table['frames'].append(dict(frame_index=k,time_tick=timing['episode_time_tick'],status='available' if available else 'missing',reason=reason,acceptance_status=acceptance_status if available else 'not_generated'))
        frame_support={}
        for c in ['cam2','cam4']:
            cd=frame/c
            if available:
                ann=dict(np.load(cd/'annotations.npz',allow_pickle=False));boxes=json.loads((cd/'boxes.json').read_text());valid=np.asarray(Image.open(cd/'valid_mask.png'))>0
                storage_path=cd/'raster_storage.json'
                meta=json.loads(storage_path.read_text()) if storage_path.exists() else None
                if recipe.get('storage_mode')=='roi_only' and meta is None:raise ValueError('missing_ROI_storage_metadata')
                layout=parse_image_layout(meta) if meta else None
                image_path=cd/('roi.png' if layout and layout.kind=='roi_image' else 'normal_erp.png')
                with Image.open(image_path) as im:stored_size=list(im.size)
                size=list(layout.full_size_wh) if layout else stored_size
                if layout and stored_size!=list(layout.stored_size_wh):raise ValueError('stored_image_size_conflict')
            for j,(p,r) in enumerate(zip(people,registry)):
                oid=f'f{k}_{c}_p{p["person_id"]}'
                obs=dict(observation_id=oid,frame_index=k,time_tick=timing['episode_time_tick'],person_id=p['person_id'],camera_id=c,
                    source_ref=r['source_ref'],character_family_id=r['character_family_id'],asset_id=p['asset_id'],
                    asset_frame_index=p['asset_frames'][k],asset_frame_status='known',source_frame_index=p['source_frames'][k],
                    source_time_tick=timing.get('source_time_tick',p['source_frames'][k]),source_time_base=timing['time_base'] if 'selected_timeline' in recipe else [1,30])
                if 'scene_episode_id' in ep:obs.update(scene_episode_id=ep['scene_episode_id'],rig_id=ep['rig_id'])
                if 'person_timelines' in recipe:
                    pr=person_frame_record(recipe,k,p)
                    for key in ['source_time_tick','source_time_base','source_record_index','source_origin_record','source_time_origin_tick','source_time_origin_base','mapping_policy','quantization_error_seconds','local_interval_seconds']:
                        obs[key]=pr[key]
                    obs.update(person_timeline_ref=recipe['person_timeline_refs'][str(p['person_id'])],timeline_row=k,scene_matrices_ref=recipe['scene_matrices_ref'],scene_matrix_row=k,scene_matrix_person_index=j)
                if 'original_100fps_mapping' in p:obs['original_100fps_mapping']=p['original_100fps_mapping']
                if 'selected_timeline' in recipe:
                    if timing['asset_frame']!=obs['asset_frame_index'] or timing['source_frame']!=obs['source_frame_index']:raise ValueError('timeline_person_frame_conflict')
                    obs.update(selected_timeline_ref=recipe['selected_timeline_ref'],timeline_row=k,scene_matrices_ref=recipe['scene_matrices_ref'],scene_matrix_row=k,source_time_unit_status=recipe['time_unit_status'])
                    if 'scene_matrix_person_ids' in recipe:
                        obs['scene_matrix_person_index']=recipe['scene_matrix_person_ids'].index(p['person_id'])
                if not available:
                    obs.update(status='camera_missing',reason=reason or 'scene time not rendered',image=None,annotations=None,valid_mask=None,person_mask=None,raster_manifest=None,annotation_person_index=None,image_size_wh=None)
                    table['observations'].append(obs);continue
                if 'selected_timeline' in recipe:
                    for key,value in [('time_tick',timing['episode_time_tick']),('source_time_tick',timing['source_time_tick']),('asset_frame_index',timing['asset_frame']),('source_frame_index',timing['source_frame']),('time_base',timing['time_base'])]:
                        if not np.array_equal(ann[key],value):raise ValueError('annotation_timeline_conflict:'+key)
                if 'person_timelines' in recipe:
                    for key in ['asset_frame','source_frame','source_time_tick','source_time_base','quantization_error_seconds']:
                        if not np.array_equal(ann['person_'+key][j],pr[key]):raise ValueError('annotation_person_timeline_conflict:'+key)
                    if not np.array_equal(ann['time_base'],timing['time_base']) or ann['time_tick']!=timing['episode_time_tick']:raise ValueError('annotation_episode_clock_conflict')
                entry=boxes[j];assert entry['person_id']==p['person_id']==int(ann['person_ids'][j])
                box=entry['bbox_xyxy_exclusive']
                obs.update(status='empty',reason='no valid packed box yet',image=str(image_path),annotations=str(cd/'annotations.npz'),
                    valid_mask=str(cd/'valid_mask.png'),person_mask=str(cd/f'person_{p["person_id"]}_mask.png'),raster_manifest=str(raster_report),annotation_person_index=j,image_size_wh=size)
                obs.update(optional_mamma_512_refs(cd,annotation_person_ids=ann['person_ids'],person_index=j,
                    frame_index=k,camera_id=c,label_policy_path=directory/'label_policy.json',annotation=ann,
                    rig_id=recipe.get('rig_id'),scene_episode_id=recipe.get('scene_episode_id')))
                obs['frame_acceptance_status']=acceptance_status
                if acceptance_status=='rejected':obs['frame_rejection_reason']=reason
                if meta:
                    obs.update({key:meta[key] for key in LAYOUT_FIELDS})
                    obs.update(raster_storage_ref=str(storage_path),physical_mask_ref=meta['physical_mask_ref'])
                    # An empty/fully occluded observation still has raster support.
                    # Keep this reference before the no-box rejection below.
                    if layout.kind=='roi_image':obs['render_coverage_ref']=str(cd/'render_coverage.json')
                try:
                    if policy==RECORD:
                        for value in ann.values():
                            if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('nonfinite_annotations')
                        pm=np.asarray(Image.open(obs['person_mask']))>0
                        center,scale=validate_native_roi(box,entry,p['person_id'],int(ann['person_ids'][j]),meta,valid,pm)
                    if box is None:raise BoxRejected('no_visible_instance_box')
                    pm=np.asarray(Image.open(obs['person_mask']))>0
                    if layout and layout.kind=='roi_image':
                        center,scale=validate_roi_box(box,entry,p['person_id'],int(ann['person_ids'][j]),meta,valid,pm)
                    else:
                        validate_box_context(box,bbox_kind='visible_instance_tight',bbox_source=entry['source'],person_id=p['person_id'],mask_person_id=entry['person_id'],annotation_person_id=int(ann['person_ids'][j]),image_size_wh=size,manifest_size_wh=json.loads((frame/'report.json').read_text())['final_erp_resolution'],valid_mask=valid,person_mask=pm,boundary_policy='erp_reject_seam',min_side_px=2)
                        center,scale=crop_from_box(box,image_size_wh=size,crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
                    coverage_path=cd/'render_coverage.json'
                    if recipe.get('imaging_route')=='direct_erp_roi' or coverage_path.exists():
                        if not coverage_path.is_file():raise BoxRejected('missing_render_coverage')
                        region=json.loads(coverage_path.read_text())['high_quality_region_xyxy']
                        try:require_crop_in_region(center,scale,region,size)
                        except ValueError as error:raise BoxRejected(str(error)) from error
                        obs['render_coverage_ref']=str(coverage_path)
                    targets=make_targets(ann['xy'][j],ann['in_view'][j],center,scale)
                    if not np.isfinite(targets).all():raise BoxRejected('nonfinite_GT')
                    support=targets[:,2]>0
                    # A legal crop has a defined support set, including empty.
                    if policy==RECORD:frame_support[(p['person_id'],c)]=support
                    if not support.any():raise BoxRejected('no_GT_inside_full_crop')
                    row=dict(imgname=obs['image'],center=center,scale=scale,proj_verts=targets,in_view=ann['in_view'][j],visible=ann['visible'][j],person_id=p['person_id'],camera_id=c)
                    if 'box_augmentation_policy' in recipe:
                        from .box_augmentation import validate_box_policy,require_full_box_family_support
                        validate_box_policy(recipe['box_augmentation_policy'])
                        row['parent_bbox_xyxy']=np.asarray(box,np.float32)
                        row['box_augmentation_reserved_policy']=json.dumps(recipe['box_augmentation_policy'],sort_keys=True)
                        try:
                            require_full_box_family_support(box,region,size,recipe['box_augmentation_policy'])
                            row['box_augmentation_family_supported']=True
                            row['box_augmentation_family_reason']=''
                        except ValueError as error:
                            # The native observation remains valid. An unsupported
                            # augmentation family blocks on/off study eligibility.
                            row['box_augmentation_family_supported']=False
                            row['box_augmentation_family_reason']=str(error)
                        obs['box_augmentation_family_supported']=row['box_augmentation_family_supported']
                        obs['box_augmentation_family_reason']=row['box_augmentation_family_reason']
                    if 'scene_episode_id' in ep:
                        if str(ann['scene_episode_id'])!=ep['scene_episode_id'] or str(ann['rig_id'])!=ep['rig_id']:
                            raise ValueError('annotation_scene_rig_identity_conflict')
                        row.update(scene_episode_id=ep['scene_episode_id'],rig_id=ep['rig_id'])
                    if meta:row.update({key:meta[key] for key in LAYOUT_FIELDS})
                    index=len(rows);rows.append(row)
                    table['boxes'].append(dict(box_id=oid+'_box',observation_id=oid,packed_row_id=index,bbox_xyxy=box,bbox_kind='visible_instance_tight',bbox_source=entry['source'],crop_policy='deployment',parent_box_id=None,origin_ref=str(cd/'boxes.json'),origin_row=j))
                    if 'scene_episode_id' in ep:table['boxes'][-1].update(scene_episode_id=ep['scene_episode_id'],rig_id=ep['rig_id'])
                    obs.update(status='observed',reason='')
                    frame_support[(p['person_id'],c)]=support
                    coverage.append(dict(frame_index=k,person_id=p['person_id'],camera_id=c,packed_row_id=index,
                        in_view=int(ann['in_view'][j].sum()),visible=int(ann['visible'][j].sum()),supervised=int(support.sum()),supervised_occluded=int((support&~ann['visible'][j]).sum()),
                        outside_domain=int((~ann['in_view'][j]).sum()),in_view_outside_crop=int((ann['in_view'][j]&~support).sum()),
                        target_mask_pixels=int(pm.sum())))
                except BoxRejected as error:
                    if policy==RECORD and str(error) not in EXPECTED_UNAVAILABLE:raise
                    obs['reason']=str(error);rejects.append(rejection_record(error,frame_index=k,person_id=p['person_id'],camera_id=c,box=box))
                    if policy==RECORD and box is None:
                        status,why=no_box_state(ann['in_view'][j],ann['visible'][j],ann['occlusion_evidence_code'][j] if 'occlusion_evidence_code' in ann else None)
                        obs.update(status=status,reason=why,geometry_GT_retained=True,model_crop_available=False)
                        if status=='person_absent':
                            obs.update(geometry_annotation_person_index=j,retained_person_mask_ref=obs['person_mask'],annotation_person_index=None,person_mask=None)
                    elif box is None and not ann['in_view'][j].any():
                        obs.update(status='person_absent',reason='zero_in_view_GT_and_no_instance_box',
                                   annotation_person_index=None,person_mask=None)
                table['observations'].append(obs)
        for p in people:
            a=frame_support.get((p['person_id'],'cam2'));b=frame_support.get((p['person_id'],'cam4'))
            if available:
                availability=pair_availability(dict(cam2=a,cam4=b))
                coverage.append(dict(frame_index=k,person_id=p['person_id'],paired=True,**availability))
    if rows:np.savez(out/'packed.npz',**pack_derivative_rows(rows),supervision_policy=SUPERVISION_POLICY,split=recipe['diagnostic_split'])
    else:table['packed']['path']=None
    table['packed']['row_count']=len(rows)
    for name,obj in [('episode',ep),('sequence_index',table),('coverage',coverage),('rejections',rejects)]:
        (out/f'{name}.json').write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
    validation=validate_sequence_index(ep,table,base_dir=out,check_image_headers=True,check_packed=True)
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    return out,ep,table,coverage
