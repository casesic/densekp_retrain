#!/usr/bin/env python3
"""Freeze the two reviewed p1 BVH windows before any SMPL-X optimization."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiments/bvh_p1_smplx_pilot_a2_20261010'
SOURCE=Path('/mnt/data1/yy/home/yy/datasets/cycling_mocap/candidates/1_RB_M42_20230719_PK/1_RB_M42_20230719_PK.bvh')
CHAR=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930/characters/cycling_M02_v3/manifest.json'
CLIP=ROOT/'outputs/motion_inputs/cycling_fit_20260929/participant_1'
CASES={'p1_comfortable':(7560,7656),'p1_alternate':(9600,9696)}

def main():
    assert SOURCE.exists() and CHAR.exists() and (CLIP/'manifest.json').exists()
    char=json.loads(CHAR.read_text());cache=json.loads((CLIP/'manifest.json').read_text())
    assert char['character_id']=='cycling_M02_v3' and char['identity_family_id']=='cycling_M02'
    assert char['gender']=='male' and len(char['betas'])==30
    for key in ('model_file','parent_manifest'):
        assert Path(char[key]).exists(),key
    for key in ('geometry','clothing_geometry','groom_geometry'):
        assert Path(char['reference'][key]).exists(),key
    assert cache['source_bvh']==str(SOURCE)
    with SOURCE.open() as f:
        for line in f:
            if line.startswith('Frame Time:'):
                literal=line.split(':',1)[1].strip();break
        else:raise ValueError('Frame Time missing')
    assert literal=='0.0416667'
    assert not OUT.exists(),'output exists; preserve prior frozen version'
    OUT.mkdir(parents=True)
    root=dict(integration_only=True,source_family='zenodo_10668611_participant_1',
      source_session='1_RB_M42_20230719_PK',source_record='https://zenodo.org/records/10668611',
      license='CC BY 4.0 (Kyriakou, Chrysanthou and Kyriakou); model and character separate',
      windows={k:list(v) for k,v in CASES.items()},same_participant_and_recording=True,
      target_character='cycling_M02_v3',target_family='cycling_M02',character_manifest=str(CHAR),
      model_file=char['model_file'],fixed_betas=char['betas'],
      source_frame_time_literal=literal,time_base=[1,10000000],ticks_per_frame=416667,
      time_kind='published BVH nominal sample spacing, not exposure PTS',
      coordinate_policy='one shared inferred proper yaw and uniform scale for both windows; +Y up inferred',
      root_policy='source root retained as observation, target pelvis centered, scene owns travel',
      GPU='none; CPU only',compute_budget_seconds=1800,body_review='pending_user',
      clothing_groom='not_generated',public_scene='not_consumed')
    (OUT/'frozen_manifest.json').write_text(json.dumps(root,ensure_ascii=False,indent=2)+'\n')
    for case,(start,end) in CASES.items():
        sub=OUT/case;sub.mkdir()
        freeze=dict(integration_only=True,source_bvh=str(SOURCE),local_clip=str(CLIP),
          source_frame_half_open=[start,end],asset_frames=end-start,
          frame_time_original_decimal=literal,time_base=[1,10000000],ticks_per_frame=416667,
          time_kind='BVH nominal sample timing, not exposure PTS',
          target_character=char['character_id'],character_manifest=str(CHAR),
          model_file=char['model_file'],regressor=str(ROOT/'assets/densekp_regressor_R.npy'),
          shared_alignment=str(OUT/'shared_alignment.json'),
          coordinate_policy='one shared inferred +Y-up proper yaw plus one shared scale from both windows',
          root_policy='target pelvis centered using constant negative shaped-rest pelvis transl; scene owns travel and road yaw',
          budget=dict(probe_iterations=350,full_iterations=650,probe_wall_seconds=180,full_wall_seconds=1200,total_compute_seconds=1800),
          probe_asset_frames=[0,24,48,72,95],gpu_plan='CPU only; UUID null',
          source_group='zenodo_10668611_participant_1',source_session=root['source_session'],
          license=root['license'],status='frozen before optimization')
        (sub/'frozen_manifest.json').write_text(json.dumps(freeze,ensure_ascii=False,indent=2)+'\n')
    print(OUT)

if __name__=='__main__':main()
