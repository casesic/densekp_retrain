"""Explicit asset, integer timeline and scene-owned trajectory contracts.

Legacy walking recipes retain their fixed matrix and source-root behaviour.
No motion evaluation, projection, garment binding or Dataset implementation here.
"""
import json
from pathlib import Path
from fractions import Fraction
import numpy as np

SCHEMA='densekp_scene_asset_v2'
ROOT_MODE='fixed_pelvis_scene_owns_progression'
SUPPORT='fixed_scene_placement_height_no_pedestrian_foot_contact'
LEGACY_ROOT='source_root_relative_to_start_plus_fixed_scene_matrix'


def adapt_proposal(path):
    path=Path(path).resolve();p=json.loads(path.read_text())
    if p['schema']!='densekp_explicit_asset_combination_a2_proposal_v1':raise ValueError('unsupported_asset_proposal')
    if p['root_mode']!=ROOT_MODE or p['support_policy']!=SUPPORT:raise ValueError('unsupported_root_or_support')
    if p['garment']['material_kind']!='GarmentRest_continuous_shell':raise ValueError('unsupported_garment_material')
    if p['appearance']['kind']!='existing_character_skin_and_native_eye_shader':raise ValueError('unsupported_appearance')
    if p['groom']['representation']!='fixed_reference_curves_rigid_head':raise ValueError('unsupported_groom')
    authority=json.loads(Path(p['authority_manifest']).read_text())
    if authority['asset_id']!=p['asset_id']:raise ValueError('asset_authority_conflict')
    # Existing groom loader consumes an audited reference directory. Do not
    # silently substitute a different file for an explicit proposal reference.
    for key,file in [('points','points.npy'),('radius','radius.npy'),('curve_offsets','curve_offsets.npy'),('active_mask','active_curve_mask.npy')]:
        if Path(p['groom'][key]).resolve()!=(Path(p['groom']['reference'])/file).resolve():raise ValueError('unsupported_groom_file_layout')
    result=dict(schema=SCHEMA,asset_id=p['asset_id'],proposal_ref=str(path),authority_manifest=p['authority_manifest'],
        character_manifest=p['character_manifest'],character_family=p['character_family'],parameters=p['parameters'],
        source_ref=authority['source_ref'],source_group=p['source_group'],motion_source_id=p['motion_source_id'],
        body_vertices=p['geometry']['vertices'],body_faces=p['geometry']['faces'],densekp=p['geometry']['densekp'],
        cloth_vertices=p['garment']['vertices'],cloth_faces=p['garment']['faces'],head_transforms=p['groom']['head_transforms'],
        clothing_reference=str(Path(p['garment']['material_reference']).parent),groom_reference=p['groom']['reference'],
        timeline_ref=p['timeline'],nominal_source_rate=authority['nominal_source_rate'],time_base=p['time_base'],time_unit_status=p['time_unit_status'],frame_count=p['frame_count'],
        root_mode=p['root_mode'],support_policy=p['support_policy'],root_baked=p['root_baked'],
        regressor=p['regressor'],available_combinations_verified=True,integration_only=True,
        actual_supported_capabilities=['native_integer_timeline','explicit_geometry_paths','scene_owned_translation','fixed_height','existing_character_simple_materials'],
        asset_evidence=p['technical_evidence'],human_acceptance=p['human_acceptance'],independent_validation=False)
    for k in ['body_vertices','body_faces','densekp','cloth_vertices','cloth_faces','head_transforms','parameters','timeline_ref','regressor','character_manifest']:
        if not Path(result[k]).is_file():raise ValueError('missing_asset_file:'+k)
    return result


def geometry_paths(asset):
    if asset.get('schema')==SCHEMA:
        return {k:Path(asset[k]) for k in ['body_vertices','body_faces','cloth_vertices','cloth_faces','head_transforms','densekp']}
    g=Path(asset['geometry'])
    return dict(body_vertices=g/'vertices.npy',body_faces=g/'faces.npy',cloth_vertices=g/'clothing_vertices.npy',
                cloth_faces=Path(asset['clothing_reference'])/'clothing_faces.npy',head_transforms=g/'head_transforms.npy',densekp=g/'densekp_3d.npy')


def select_timeline(records, first, count, asset_count):
    if type(first) is not int or type(count) is not int or count<2 or first<0 or first+count>len(records):raise ValueError('invalid_timeline_selection')
    previous=None;seen=set();base=None
    for row in records:
        for key in ['asset_frame','source_frame','source_time_tick','episode_time_tick']:
            if type(row.get(key)) is not int:raise ValueError('missing_or_noninteger_time_or_frame')
        if not 0<=row['asset_frame']<asset_count or row['asset_frame'] in seen:raise ValueError('invalid_asset_frame')
        seen.add(row['asset_frame']);tb=row['time_base']
        if len(tb)!=2 or any(type(x)is not int or x<=0 for x in tb):raise ValueError('invalid_time_base')
        if base is None:base=tb
        if tb!=base or (previous is not None and row['source_time_tick']<=previous):raise ValueError('time_conflict')
        previous=row['source_time_tick']
    chosen=records[first:first+count];origin=chosen[0]['source_time_tick']
    return [dict(row,frame_index=k,asset_episode_time_tick=row['episode_time_tick'],episode_time_tick=row['source_time_tick']-origin) for k,row in enumerate(chosen)]


def scene_trajectory(records, start_xyz, direction_xy, speed_m_s, rotation):
    rotation=np.asarray(rotation,float);direction=np.asarray(direction_xy,float);start=np.asarray(start_xyz,float)
    if rotation.shape!=(3,3) or start.shape!=(3,) or direction.shape!=(2,) or not np.isfinite(np.r_[start,direction,speed_m_s]).all() or speed_m_s<0:raise ValueError('invalid_trajectory')
    if not np.isclose(np.linalg.norm(direction),1):raise ValueError('nonunit_path_direction')
    result=[]
    for row in records:
        seconds=float(Fraction(row['episode_time_tick'])*Fraction(*row['time_base']))
        M=np.eye(4);M[:3,:3]=rotation;M[:3,3]=start+np.r_[direction*speed_m_s*seconds,0]
        result.append(M)
    return checked_matrices(result,len(records))


def checked_matrices(value,count):
    a=np.asarray(value,float)
    if a.shape!=(count,4,4) or not np.isfinite(a).all():raise ValueError('invalid_scene_matrices')
    if not np.allclose(a[:,3,:],[0,0,0,1],atol=1e-8,rtol=0) or not np.allclose(a[:,:3,:3]@a[:,:3,:3].transpose(0,2,1),np.eye(3),atol=1e-6,rtol=0) or not np.allclose(np.linalg.det(a[:,:3,:3]),1,atol=1e-6,rtol=0):raise ValueError('nonrigid_scene_matrix')
    return a


def person_matrices(person,count):
    mode=person['root_mode']
    if mode==LEGACY_ROOT:
        if 'scene_from_model' in person:raise ValueError('double_root_ownership')
        return checked_matrices(np.repeat(np.array(person['model_to_world'])[None],count,axis=0),count)
    if mode==ROOT_MODE:
        if 'model_to_world' in person or 'source_root_relative' in person:raise ValueError('double_root_ownership')
        a=checked_matrices(person['scene_from_model'],count)
        if person['support_policy']!=SUPPORT or not np.allclose(a[:,2,3],person['placement_height_m'],atol=1e-8,rtol=0):raise ValueError('nonconstant_placement_height')
        return a
    raise ValueError('unknown_root_mode')


def frame_record(recipe,k):
    if not 0<=k<recipe['frame_count']:raise ValueError('frame_out_of_range')
    if 'person_timelines' in recipe:
        from .person_time import SCHEMA as TIME_SCHEMA,validate_record
        if recipe.get('time_mapping_schema')!=TIME_SCHEMA:raise ValueError('unknown_person_time_schema')
        row=recipe['episode_timeline'][k]
        if row!=dict(frame_index=k,episode_time_tick=k,time_base=recipe['time_base']):raise ValueError('mixed_episode_clock_conflict')
        records=[]
        if set(recipe['person_timelines'])!={str(p['person_id']) for p in recipe['people']}:raise ValueError('person_timeline_identity_conflict')
        for person in recipe['people']:
            timeline=recipe['person_timelines'][str(person['person_id'])]
            if len(timeline)!=recipe['frame_count']:raise ValueError('person_timeline_count_conflict')
            records.append(dict(validate_record(timeline[k],row,person),person_id=person['person_id']))
        return dict(row,person_records=records)
    if 'selected_timeline' not in recipe:
        return dict(frame_index=k,episode_time_tick=k,time_base=recipe['time_base'])
    row=recipe['selected_timeline'][k]
    if row['frame_index']!=k or row['time_base']!=recipe['time_base']:raise ValueError('selected_time_conflict')
    if any(type(row.get(key)) is not int for key in ['source_time_tick','episode_time_tick','asset_frame','source_frame']):raise ValueError('missing_native_time')
    if row['episode_time_tick']!=row['source_time_tick']-recipe['time_origin']['source_time_tick']:raise ValueError('native_time_origin_conflict')
    for person in recipe['people']:
        if person['asset_frames'][k]!=row['asset_frame'] or person['source_frames'][k]!=row['source_frame']:raise ValueError('person_selected_record_conflict')
    return row


def support_limits(person):
    mode=person.get('support_policy','pedestrian_source_root_constant_lift')
    if mode=='pedestrian_source_root_constant_lift' and person['root_mode']==LEGACY_ROOT:return (-.02,.12)
    if mode==SUPPORT and person['root_mode']==ROOT_MODE:
        lo,hi=person['support_checks']['body_min_clearance_m']
        if not 0<=lo<hi<=1:raise ValueError('invalid_cycling_support_limits')
        return lo,hi
    raise ValueError('unknown_support_policy')


def person_frame_record(recipe,k,person):
    """One observation's source clock; global frame_record stays backward compatible."""
    frame=frame_record(recipe,k)
    if 'person_records' in frame:return next(r for r in frame['person_records'] if r['person_id']==person['person_id'])
    if 'selected_timeline' in recipe:return dict(frame,source_time_base=frame['time_base'])
    return dict(frame,asset_frame=person['asset_frames'][k],source_frame=person['source_frames'][k],source_time_tick=person['source_frames'][k],source_time_base=[1,30])
