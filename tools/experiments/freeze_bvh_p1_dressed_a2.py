#!/usr/bin/env python3
"""Freeze two accepted M02 body sequences for simple garment/groom review."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BODY=ROOT/'outputs/experiments/bvh_p1_smplx_pilot_a2_20261010'
OUT=ROOT/'outputs/experiments/bvh_p1_dressed_assets_a2_20261010'
CHAR=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_M02_v3/manifest.json'
CASES={'p1_comfortable':([7560,7656],[0,13,48,81,95]),'p1_alternate':([9600,9696],[0,48,55,84,95])}

def put(p,x):
    assert not p.exists(),p
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def main():
    assert not OUT.exists(),OUT
    char=json.loads(CHAR.read_text());assert char['character_id']=='cycling_M02_v3' and char['identity_family_id']=='cycling_M02'
    body_catalog=json.loads((BODY/'delivery_manifest.json').read_text());assert all(x['body_human_status']=='accepted_by_user' for x in body_catalog['items'])
    cloth=Path(char['reference']['clothing_geometry']);groom=Path(char['reference']['groom_geometry']);ref=Path(char['reference']['geometry'])
    cm=json.loads((cloth/'clothing_manifest.json').read_text());gm=json.loads((groom/'groom_manifest.json').read_text())
    assert cm['body_geometry']==str(ref) and gm['body_geometry']==str(ref)
    assert cm['character_id']=='cycling_M02_v3' and gm['character_id']=='cycling_M02_v2'
    assert '4mm' in cm['method'] and cm['max_vertex_offset_m']<.00401
    offset=.004
    assert np.load(cloth/'clothing_vertices.npy').shape[0]==1
    active=np.load(groom/'active_curve_mask.npy');offs=np.load(groom/'curve_offsets.npy')
    counts=dict(curves=int(active.sum()),points=int(np.diff(offs)[active].sum()))
    assert counts==dict(curves=80397,points=2559631)
    OUT.mkdir(parents=True)
    put(OUT/'frozen_manifest.json',dict(integration_only=True,character_manifest=str(CHAR),body_catalog=str(BODY/'delivery_manifest.json'),
        source_group='zenodo_10668611_participant_1',source_session='1_RB_M42_20230719_PK',
        cases={k:dict(source_frame_half_open=v[0],probe_asset_frames=v[1]) for k,v in CASES.items()},
        two_windows_same_participant_recording=True,body_accepted=True,dressed_review='pending_user',
        body_and_time_immutable=True,studio_samples=64,resolution_each=[512,512],denoising=False,
        total_compute_budget_seconds=1800,render_gpu_uuid='GPU-d481acc7-106f-29e5-1298-8c0d398261b2',
        no_scene_render=True))
    for case,(window,probe) in CASES.items():
        base=OUT/case;base.mkdir();source=BODY/case;old=json.loads((source/'delivery_manifest.json').read_text())
        p=dict(np.load(old['parameters'],allow_pickle=False));lo,hi=window
        assert old['body_human_status']=='accepted_by_user' and old['source_frame_half_open']==window
        assert np.array_equal(p['source_frame_ids'],np.arange(lo,hi)) and np.array_equal(p['nominal_asset_time_ticks'],np.arange(96)*416667)
        assert np.array_equal(p['betas'],np.asarray(char['betas'],np.float32))
        recipe=dict(integration_only=True,asset_id=f'cycling_M02_v3_zenodo10668611_{case}_{lo}_{hi}_simple_v1',
          character_manifest=str(CHAR),source_authoritative_parameters=old['parameters'],body_cache=old['body_vertices'],
          body_joints=old['body_joints'],densekp_cache=old['densekp_3d'],faces=old['faces'],
          source_clothing=str(cloth),source_groom=str(groom),source_time_mapping=old['time_mapping'],
          source_frame_half_open=window,asset_frames=96,time_base=[1,10000000],ticks_per_frame=416667,
          time_kind='BVH nominal Frame Time, actual exposure PTS unknown',source_group='zenodo_10668611_participant_1',
          character_family='cycling_M02',body_human_review='accepted_by_user',dressed_human_review='pending_user',
          probe_asset_frames=probe,probe_reason='first/middle/last and actual left/right knee flexion extrema in the frozen body',
          body_policy='unchanged cached M02 standard SMPL-X body; no source alignment or root reapplication',
          binding_policy='existing M02 simple_clothing_v1 body barycentric + 4mm normal; M02 groom_v5 fixed active curves and rigid head15',
          garment_offset_m=offset,groom_reference_space='M02 shaped-rest pelvis-centered model Y-up; head includes global orient once',
          groom_render_counts=counts,timeline=str(base/'timeline.json'),
          budget=dict(gpu_uuid='GPU-d481acc7-106f-29e5-1298-8c0d398261b2',render_engine='Cycles CUDA',
             resolution_each=[512,512],samples=64,denoising=False,seed=29,views=['front','side'],
             max_compute_seconds=1800,max_render_seconds=1800,reuse_probe=True),status='frozen_before_compute')
        put(base/'frozen_recipe.json',recipe)
        timeline=[dict(asset_frame=i,source_frame=lo+i,source_time_tick=(lo+i)*416667,episode_time_tick=i*416667,time_base=[1,10000000]) for i in range(96)]
        put(base/'timeline.json',timeline)
    print(OUT)

if __name__=='__main__':main()
