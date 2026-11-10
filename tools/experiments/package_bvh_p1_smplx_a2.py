#!/usr/bin/env python3
"""Write body-only handoff for the two frozen p1 windows; no scene assets."""
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/bvh_p1_smplx_pilot_a2_20261010'
CASES={'p1_comfortable':'中度弯肘','p1_alternate':'较前倾'}

def write(path,data):
    assert not path.exists(),f'preserve existing result: {path}'
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    root=json.loads((BASE/'frozen_manifest.json').read_text());shared=json.loads((BASE/'shared_alignment.json').read_text())
    items=[]
    for key,label in CASES.items():
        base=BASE/key;fr=json.loads((base/'frozen_manifest.json').read_text());lo,hi=fr['source_frame_half_open']
        val=json.loads((base/'fit_attempt_1/validation.json').read_text())
        read=json.loads((base/'readback_validation.json').read_text())
        video=json.loads((base/'preview/execution.json').read_text())
        p=np.load(base/'fit_attempt_1/parameters.npz');assert hi-lo==96 and val['frames']==read['frames']==video['decoded_frames']==96
        assert np.array_equal(p['betas'],np.asarray(root['fixed_betas'],np.float32))
        assert read['all_finite'] and read['R138_full_readback_max_abs_m']<2e-6
        assert video['max_nominal_pts_error_seconds']<1e-5
        meta=dict(integration_only=True,asset_id=f'cycling_M02_v3_zenodo10668611_{key}_{lo}_{hi}_body_a2',
          representation='body_only_standard_SMPLX',source_record=root['source_record'],
          source_credit='Kyriakou, Chrysanthou and Kyriakou, Dataset of Motion Capture of Cyclists',
          source_license='CC BY 4.0 in local official Zenodo metadata; SMPL-X and character assets separately licensed',
          source_license_evidence=str(ROOT/'outputs/motion_inputs/cycling_fit_20260929/zenodo_record.json'),
          source_family=root['source_family'],session_id=root['source_session'],
          original_actor_identity='unknown beyond published participant/session ID',
          source_bvh=fr['source_bvh'],source_frame_half_open=[lo,hi],asset_frame_half_open=[0,96],
          frame_count=96,frame_time_original_decimal=fr['frame_time_original_decimal'],
          time_base=[1,10000000],ticks_per_frame=416667,
          time_status='published BVH nominal sample interval; actual exposure PTS unknown',
          time_mapping=str(base/'time_mapping.csv'),
          target_character='cycling_M02_v3',character_family='cycling_M02',
          character_manifest=fr['character_manifest'],model=fr['model_file'],gender='male',
          fixed_betas=root['fixed_betas'],model_topology='standard SMPL-X 10475 vertex M02 reference faces',
          parameters=str(base/'fit_attempt_1/parameters.npz'),body_vertices=str(base/'fit_attempt_1/vertices.npy'),
          body_joints=str(base/'fit_attempt_1/joints.npy'),densekp_3d=str(base/'fit_attempt_1/densekp_3d.npy'),
          faces=str(base/'faces.npy'),R=str(ROOT/'assets/densekp_regressor_R.npy'),
          alignment=str(BASE/'shared_alignment.json'),window_alignment_diagnostic=str(base/'alignment.json'),
          joint_mapping=str(base/'joint_mapping.json'),source_observations=str(base/'observations.npz'),
          axes='SMPL-X model +Y up; source +Y up inferred, one shared proper yaw applied',
          unit='target model metres; source unit unknown and scale inferred once for both windows',
          root_mode='pelvis_centered_scene_owns_progression_and_road_yaw',
          root_baked='constant transl=-M02 shaped-rest pelvis; no source root road displacement or foot ground snapping',
          body_numeric_status='passed',R138_status='passed_standard_M02_surface',
          body_human_status='pending_user',native_motion_human_status='accepted_by_user',
          clothing_groom_status='not_generated',public_scene_status='not_consumed',
          split='integration_only',validation=str(base/'fit_attempt_1/validation.json'),
          full_readback=str(base/'readback_validation.json'),preview_html=str(base/'preview/index.html'),
          preview_mp4=str(base/'preview/comparison.mp4'),
          source_grouping='p1 comfortable and alternate are same participant/session; M02 family stays linked across its versions; no independent validation claim')
        write(base/'delivery_manifest.json',meta)
        items.append(dict(case=key,label=label,asset_id=meta['asset_id'],source_frame_half_open=[lo,hi],
            manifest=str(base/'delivery_manifest.json'),preview=str(base/'preview/index.html'),
            numeric_status='passed',body_human_status='pending_user'))
    catalog=dict(integration_only=True,source_family=root['source_family'],same_participant_recording=True,
        shared_alignment=str(BASE/'shared_alignment.json'),target_character='cycling_M02_v3',
        items=items,public_scene_status='not_consumed',training_validation_claim='none')
    write(BASE/'delivery_manifest.json',catalog)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>participant_1 → M02 身体评审</title><style>body{font:16px sans-serif;max-width:1100px;margin:2em auto;padding:0 1em;background:#f4f6f9;color:#1b2638}video{width:100%;max-width:1000px}section{margin:2em 0;padding:1em;background:white;border-radius:8px}a{color:#155baf}</style><h1>participant_1 → cycling_M02_v3</h1><p>同一参与者/录制的两个独立 96 帧窗口，原 BVH 标称 0.0416667 秒间隔；同一固定 shape、轴和尺度。左侧原 BVH 骨架，右侧标准 SMPL-X 身体，各两个固定方向。仅为 integration_only 身体评审；两段拟合的动态观感待用户观看。</p>'''
    for item in items:
        key=item['case'];lo,hi=item['source_frame_half_open']
        html+=f'<section><h2>{item["label"]}：源帧 [{lo},{hi})</h2><p><a href="{key}/preview/index.html">逐帧 HTML</a> · <a href="{key}/preview/comparison.mp4" download="participant_1_{key}_body_comparison.mp4">下载原速 MP4</a> · <a href="{key}/delivery_manifest.json">交付清单</a></p><video controls preload="metadata" src="{key}/preview/comparison.mp4"></video></section>'
    html+='</html>'
    (BASE/'index.html').write_text(html)
    print(json.dumps(catalog,ensure_ascii=False))

if __name__=='__main__':main()
