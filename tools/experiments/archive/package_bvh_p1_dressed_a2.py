#!/usr/bin/env python3
"""Package one M02 BVH dressed sequence and exercise A1 read-only adapters."""
import argparse,json,sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.data.dynamic_scene import adapt_proposal,geometry_paths,select_timeline
from densekp_retrain.data.person_time import map_timeline

TOP=ROOT/'outputs/experiments/bvh_p1_dressed_assets_a2_20261010'
TEMPLATE=ROOT/'outputs/plans/dynamic_asset_handoff_a2_20261009/cycling_m01.json'

def write(p,x):
    assert not p.exists(),f'preserve existing output: {p}'
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',choices=['p1_comfortable','p1_alternate'],required=True)
    args=ap.parse_args();base=TOP/args.case
    rec=json.loads((base/'frozen_recipe.json').read_text());v=json.loads((base/'validation.json').read_text())
    preview=json.loads((base/'preview/execution.json').read_text());lo,hi=rec['source_frame_half_open'];n=rec['asset_frames']
    assert n==96 and hi-lo==n and v['frames']==n and preview['decoded_frames']==n
    body=ROOT/'outputs/experiments/bvh_p1_smplx_pilot_a2_20261010'/args.case
    src=json.loads((body/'delivery_manifest.json').read_text());char=json.loads(Path(rec['character_manifest']).read_text())
    assert src['body_human_status']=='accepted_by_user' and char['character_id']=='cycling_M02_v3'
    p=np.load(rec['source_authoritative_parameters']);assert np.array_equal(p['source_frame_ids'],np.arange(lo,hi))
    assert np.array_equal(p['nominal_asset_time_ticks'],np.arange(n)*416667)
    cloth=Path(rec['source_clothing']);groom=Path(rec['source_groom']);asset=rec['asset_id']
    authority=dict(integration_only=True,asset_id=asset,character_family='cycling_M02',
      character_manifest=rec['character_manifest'],parent_body_asset=src['asset_id'],
      parameters=rec['source_authoritative_parameters'],source_ref=str(body/'delivery_manifest.json'),
      source_record='https://zenodo.org/records/10668611',
      source_credit='Kyriakou, Chrysanthou and Kyriakou, Dataset of Motion Capture of Cyclists',
      source_license='CC BY 4.0 in saved Zenodo metadata; SMPL-X and character resources separately licensed',
      source_group='zenodo_10668611_participant_1',session_id='1_RB_M42_20230719_PK',
      original_capture_person_identity='unknown beyond published participant/session ID',
      nominal_source_rate=[10000000,416667],source_frame_half_open=[lo,hi],available_source_frames=[lo,hi-1],
      asset_frame_half_open=[0,n],frame_count=n,time_base=[1,10000000],ticks_per_frame=416667,
      time_status='published BVH Frame Time nominal; actual exposure PTS unknown',
      timeline=str(base/'timeline.json'),time_mapping=rec['source_time_mapping'],
      model=char['model_file'],gender='male',betas=char['betas'],model_vertex_count=10475,
      model_faces=rec['faces'],body_vertices=rec['body_cache'],body_joints=rec['body_joints'],
      body_R138=rec['densekp_cache'],regressor=str(ROOT/'assets/densekp_regressor_R.npy'),
      cloth_vertices=str(base/'full/clothing_vertices.npy'),cloth_faces=str(cloth/'clothing_faces.npy'),
      cloth_binding_indices=str(cloth/'body_barycentric_indices.npy'),
      cloth_binding_weights=str(cloth/'body_barycentric_weights.npy'),cloth_offset_m=rec['garment_offset_m'],
      cloth_material_kind='GarmentRest_continuous_shell',cloth_material=str(cloth/'garment_material.json'),
      cloth_texture_kind='existing M02 procedural/simple material; no independent garment UV',
      groom_reference=str(groom),groom_points=str(groom/'points.npy'),groom_radius=str(groom/'radius.npy'),
      groom_curve_offsets=str(groom/'curve_offsets.npy'),groom_active_mask=str(groom/'active_curve_mask.npy'),
      head_transforms=str(base/'full/head_transforms.npy'),groom_reference_space=rec['groom_reference_space'],
      root_mode='fixed_pelvis_scene_owns_progression',
      root_baked='constant transl=-M02 shaped-rest pelvis; source global_orient in body/head once',
      scene_contract='body, cloth, R138: scene_from_model[t]; groom: scene_from_model[t] @ head[asset_frame]',
      scene_owns_road_progression=True,support_policy='fixed_scene_placement_height_no_pedestrian_foot_contact',
      fixed_scene_height_status='not selected or validated in studio preview',
      body_bounds_model_m=v['body_bounds_model_m'],cloth_bounds_model_m=v['clothing_bounds_model_m'],
      body_lowpoint_y_range_m=v['body_lowpoint_y_range_m'],
      body_numeric_status='passed',body_human_status='accepted_by_user',cloth_groom_numeric_status='passed',
      dressed_human_status='pending_user',public_scene_status='not_consumed',split='integration_only',
      validation=str(base/'validation.json'),preview_html=str(base/'preview/index.html'),
      preview_mp4=str(base/'preview/comparison.mp4'),
      source_grouping='both p1 segments share one participant and recording and M02 family; no independent train/val claim')
    write(base/'delivery_manifest.json',authority)
    q=json.loads(TEMPLATE.read_text());q.update(asset_id=asset,character_family='cycling_M02',
      character_manifest=rec['character_manifest'],authority_manifest=str(base/'delivery_manifest.json'),
      parameters=rec['source_authoritative_parameters'],source_group=authority['source_group'],
      motion_source_id=f'zenodo10668611/participant_1/1_RB_M42_20230719_PK/source{lo}_{hi}',
      source_parameters=rec['source_authoritative_parameters'],frame_count=n,timeline=str(base/'timeline.json'),
      time_base=[1,10000000],time_unit_status=authority['time_status'],time_mapping=rec['source_time_mapping'],
      technical_status='all96_body_cloth_groom_numeric_pass',technical_evidence=str(base/'validation.json'),
      human_acceptance='accepted_body_motion_by_user_dressed_pending_user',
      preview_html=authority['preview_html'],preview_mp4=authority['preview_mp4'],
      known_limits=['BVH unit and axis adaptation estimated; capture PTS unknown',
        'light local garment folds/intersections possible; dressed dynamic review pending',
        'groom follows head rigidly without hair physics',
        'both p1 windows share actor/session and M02 family; no independent validation claim'],
      A1_schema_effective=False,public_consumer_tested=False,
      provenance_refs=[str(body/'delivery_manifest.json'),str(rec['character_manifest']),
                       str(ROOT/'outputs/motion_inputs/cycling_fit_20260929/zenodo_record.json')],
      licenses=['Zenodo 10668611 CC BY 4.0','SMPL-X separate local model license',
                'M02 groom and skin textures retain their own terms'])
    q['model'].update(file=char['model_file'],gender='male',betas=char['betas'],num_betas=30,
                      semantics_evidence=str(body/'readback_validation.json'))
    q['geometry']={'vertices':rec['body_cache'],'faces':rec['faces'],'densekp':rec['densekp_cache']}
    q['garment'].update(vertices=authority['cloth_vertices'],faces=authority['cloth_faces'],
      binding_indices=authority['cloth_binding_indices'],binding_weights=authority['cloth_binding_weights'],
      offset_m=authority['cloth_offset_m'],materials=str(cloth/'materials.json'),
      material_reference=authority['cloth_material'],rest_coordinates=str(cloth/'clothing_rest_vertices.npy'))
    q['groom'].update(reference=str(groom),points=authority['groom_points'],radius=authority['groom_radius'],
      curve_offsets=authority['groom_curve_offsets'],active_mask=authority['groom_active_mask'],
      head_transforms=authority['head_transforms'])
    q['appearance']['character_manifest']=rec['character_manifest']
    write(base/'a1_proposal.json',q)
    adapted=adapt_proposal(base/'a1_proposal.json');paths=geometry_paths(adapted)
    assert all(x.is_file() for x in paths.values())
    rows=json.loads((base/'timeline.json').read_text());selected=select_timeline(rows,0,n,n)
    assert selected[0]['source_frame']==lo and selected[-1]['source_frame']==hi-1
    assert selected[-1]['episode_time_tick']==95*416667
    clocks=[dict(frame_index=i,episode_time_tick=i*416667,time_base=[1,10000000]) for i in range(n)]
    mapped=map_timeline(rows,clocks,origin_record=0)
    assert all(x['asset_frame']==i and x['quantization_error_seconds']==[0,1] for i,x in enumerate(mapped))
    write(base/'a1_adapted_asset.json',adapted)
    write(base/'a1_adapter_check.json',dict(integration_only=True,
       adapter='dynamic_scene.adapt_proposal + geometry_paths + select_timeline; person_time.map_timeline',
       result='passed_read_only',adapted_schema=adapted['schema'],frames=len(selected),
       first=selected[0],last=selected[-1],exact_person_time_count=len(mapped),
       all_geometry_paths_exist=True,actual_scene_consumed=False,scene_images_or_packed_rows_created=False,
       remaining='A1 chooses fixed cycling placement height, explicit scene matrices and episode/ROI timing'))
    print(json.dumps(dict(asset_id=asset,frames=n,adapter=adapted['schema'],mapped=len(mapped))))

if __name__=='__main__':main()
