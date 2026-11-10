#!/usr/bin/env python3
"""Deliver one M01/BVH dressed asset and exercise A1's read-only adapter."""
import json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.data.dynamic_scene import adapt_proposal,geometry_paths,select_timeline

BASE=ROOT/'outputs/experiments/bvh_p2_dressed_asset_a2_20261010'
BODY=ROOT/'outputs/experiments/bvh_p2_smplx_pilot_a2_20261010'
INDEX=ROOT/'outputs/plans/dynamic_asset_handoff_a2_20261009/index.json'
MALE=ROOT/'outputs/plans/dynamic_asset_handoff_a2_20261009/cycling_m01.json'
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def path(p):return str(Path(p).resolve())

def main():
    recipe=json.loads((BASE/'frozen_recipe.json').read_text());v=json.loads((BASE/'validation.json').read_text())
    preview=json.loads((BASE/'preview/execution.json').read_text())
    assert preview['decoded_frames']==120 and (BASE/'preview/index.html').is_file()
    source=json.loads((BODY/'delivery_manifest.json').read_text());character=json.loads(Path(recipe['character_manifest']).read_text())
    assert source['body_dynamic_human_status']=='accepted_by_user' and v['frames']==120
    assert np.array_equal(np.load(recipe['source_authoritative_parameters'])['source_frame_ids'],np.arange(5700,5820))
    cloth=Path(recipe['source_clothing']);groom=Path(recipe['source_groom']);asset=recipe['asset_id']
    authority=dict(integration_only=True,asset_id=asset,character_family='cycling_M01',character_manifest=recipe['character_manifest'],
        parent_body_asset=source['asset_id'],parameters=recipe['source_authoritative_parameters'],source_ref=path(BODY/'delivery_manifest.json'),
        source_record='https://zenodo.org/records/10668611',source_credit='Kyriakou, Chrysanthou and Kyriakou, Dataset of Motion Capture of Cyclists',
        source_license='CC BY 4.0 in saved official Zenodo metadata; model and character resources separately licensed',
        source_group='zenodo_10668611_participant_2',session_id='2_RB_M27_20230809_PA',
        original_capture_person_identity='unknown beyond published participant/session ID',nominal_source_rate=[10000000,333333],
        source_frame_half_open=[5700,5820],available_source_frames=[5700,5819],asset_frame_half_open=[0,120],frame_count=120,
        time_base=[1,10000000],ticks_per_frame=333333,time_status='published BVH Frame Time nominal; actual exposure PTS unknown',
        timeline=path(BASE/'timeline.json'),time_mapping=path(BASE/'time_mapping.csv'),
        model=character['model_file'],gender='male',betas=character['betas'],model_vertex_count=10475,
        model_faces=path(BODY/'faces.npy'),body_vertices=recipe['body_cache'],body_joints=recipe['body_joints'],
        body_R138=recipe['densekp_cache'],regressor=path(ROOT/'assets/densekp_regressor_R.npy'),
        cloth_vertices=path(BASE/'full/clothing_vertices.npy'),cloth_faces=path(cloth/'clothing_faces.npy'),
        cloth_binding_indices=path(cloth/'body_barycentric_indices.npy'),cloth_binding_weights=path(cloth/'body_barycentric_weights.npy'),
        cloth_offset_m=.004,cloth_material_kind='GarmentRest_continuous_shell',cloth_material=path(cloth/'garment_material.json'),
        cloth_texture_kind='existing M01 procedural/simple material; no independent garment UV',
        groom_reference=path(groom),groom_points=path(groom/'points.npy'),groom_radius=path(groom/'radius.npy'),
        groom_curve_offsets=path(groom/'curve_offsets.npy'),groom_active_mask=path(groom/'active_curve_mask.npy'),
        head_transforms=path(BASE/'full/head_transforms.npy'),groom_reference_space='M01 shaped-rest pelvis-centered model',
        root_mode='fixed_pelvis_scene_owns_progression',root_baked='constant transl=-M01 shaped-rest pelvis; source global_orient in body/head once',
        scene_contract='body, cloth, R138: scene_from_model[t]; groom: scene_from_model[t] @ head[asset_frame]',
        scene_owns_road_progression=True,support_policy='fixed_scene_placement_height_no_pedestrian_foot_contact',
        fixed_scene_height_status='not selected or validated in this asset preview',
        body_bounds_model_m=v['body_bounds_model_m'],cloth_bounds_model_m=v['clothing_bounds_model_m'],body_lowpoint_y_range_m=v['body_lowpoint_y_range_m'],
        body_numeric_status='passed',body_human_status='accepted_by_user',cloth_groom_numeric_status='passed',
        dressed_human_status='pending_user',public_scene_status='not_consumed',split='integration_only',
        validation=path(BASE/'validation.json'),preview_html=path(BASE/'preview/index.html'),preview_mp4=path(BASE/'preview/comparison.mp4'),
        source_grouping='M01 also used with APD; different motion is not an independent character or automatic train/val boundary')
    write(BASE/'delivery_manifest.json',authority)
    # Proposal reuses A2's already exercised same-character schema; every motion/time/geometry field is replaced.
    p=json.loads(MALE.read_text());p.update(asset_id=asset,authority_manifest=path(BASE/'delivery_manifest.json'),
        parameters=recipe['source_authoritative_parameters'],source_group=authority['source_group'],
        motion_source_id='zenodo10668611/participant_2/2_RB_M27_20230809_PA/source5700_5820',
        source_parameters=recipe['source_authoritative_parameters'],frame_count=120,timeline=path(BASE/'timeline.json'),
        time_base=[1,10000000],time_unit_status=authority['time_status'],time_mapping=path(BASE/'time_mapping.csv'),
        technical_status='all120_body_cloth_groom_numeric_pass',technical_evidence=path(BASE/'validation.json'),
        human_acceptance='accepted_body_motion_by_user_dressed_pending_user',
        preview_html=path(BASE/'preview/index.html'),preview_mp4=path(BASE/'preview/comparison.mp4'),
        known_limits=['BVH source unit/axis adaptation is estimated; capture PTS unknown',
          'minor local simple clothing folds/intersections may occur; dynamic dressed review pending',
          'groom follows head rigidly, no hair physics','same M01 character family also appears with APD; no automatic split independence'],
        A1_schema_effective=False,public_consumer_tested=False)
    p['model']['semantics_evidence']=path(BODY/'readback_validation.json')
    p['geometry']={'vertices':recipe['body_cache'],'faces':recipe['faces'],'densekp':recipe['densekp_cache']}
    p['garment']['vertices']=authority['cloth_vertices'];p['groom']['head_transforms']=authority['head_transforms']
    write(BASE/'a1_proposal.json',p)
    adapted=adapt_proposal(BASE/'a1_proposal.json');paths=geometry_paths(adapted)
    for key,value in paths.items():assert value.is_file(),(key,value)
    rows=json.loads((BASE/'timeline.json').read_text());selected=select_timeline(rows,0,120,120)
    assert selected[0]['source_frame']==5700 and selected[-1]['source_frame']==5819
    assert selected[-1]['episode_time_tick']==119*333333
    write(BASE/'a1_adapted_asset.json',adapted)
    write(BASE/'a1_adapter_check.json',dict(integration_only=True,adapter='dynamic_scene.adapt_proposal + geometry_paths + select_timeline',
         result='passed_read_only',adapted_schema=adapted['schema'],frames=len(selected),first=selected[0],last=selected[-1],
         body_geometry_paths_exist=True,actual_scene_consumed=False,scene_images_or_packed_rows_created=False,
         remaining='A1 selects one fixed cycling placement height, explicit scene matrices and episode/person time mapping before road/ROI rendering'))
    # Append one concrete combination; preserve previous catalog bytes as a small separate record.
    before=BASE/'a2_catalog_before_append.json';assert not before.exists();before.write_bytes(INDEX.read_bytes())
    catalog=json.loads(INDEX.read_text());assert not any(x['asset_id']==asset for x in catalog['assets'])
    catalog['assets'].append(dict(asset_id=asset,manifest=path(BASE/'a1_proposal.json'),
        human_acceptance='accepted_body_motion_by_user_dressed_pending_user',densekp_status='compatible'))
    catalog['combination_count']=len(catalog['assets']);catalog['independent_motion_group_count']=4
    catalog['notes'].append('participant_2/BVH dressed M01 body accepted; garment/groom pending; M01 family remains linked to APD')
    write(INDEX,catalog)
    print(json.dumps(dict(asset_id=asset,adapter_schema=adapted['schema'],frames=len(selected),catalog_count=len(catalog['assets']))))

if __name__=='__main__':main()
