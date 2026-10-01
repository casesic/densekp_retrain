#!/usr/bin/env python3
"""Summarize completed integration evidence and preserve intermediate versions."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def save(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('assets',type=Path);a=p.parse_args()
    a.assets=a.assets.resolve();final=a.assets/'acceptance_report.json'
    if final.exists():raise FileExistsError(final)
    numeric=json.loads((a.assets/'final_numeric_acceptance.json').read_text())
    ds=json.loads((a.assets/'ds_v1/report.json').read_text())
    visual=json.loads((a.assets/'visual_review_decision.json').read_text())
    ds_visual=json.loads((a.assets/'ds_visual_review.json').read_text())
    if any(d['status']!='passed' for d in [numeric,ds,visual,ds_visual]):raise ValueError('Incomplete acceptance')
    if len({(d['character_id'],d['pose_source_id']) for d in ds['coverage']})!=8:
        raise ValueError('DS coverage must contain all eight derivatives')
    tests=(a.assets/'tests_final.log').read_text()
    if 'Ran 16 tests' not in tests or not tests.rstrip().endswith('OK'):raise ValueError('Expected passing tests')
    assets=json.loads((a.assets/'report.json').read_text())
    save(a.assets/'report_before_final_acceptance.json',assets)
    timing=dict(geometry_seconds=assets['seconds'],clothing_refinement_seconds=json.loads((a.assets/'clothing_revision_v2.json').read_text())['seconds'],
                ds_pipeline_seconds=ds['seconds'],final_numeric_validation_seconds=numeric['seconds'])
    for version in ['review_v1','heads_material_v2','heads_material_v3','review_material_v3_clothing_v2']:
        timing[version+'_seconds']=json.loads((a.assets/version/'review_manifest.json').read_text())['seconds']
    timing['measured_stage_seconds_sum']=sum(timing.values())
    timing['scope']='Measured stages include rejected previews; exclude coding, review, approval waits and interruption. DS render times are included in ds_pipeline_seconds, not added twice.'
    now=datetime.now(timezone.utc)
    gpu_check=datetime.fromisoformat('2026-09-29T14:19:08+00:00')
    timing['elapsed_since_initial_gpu_check_seconds']=(now-gpu_check).total_seconds()
    timing['elapsed_scope']='Calendar interval including overnight interruption; not active work or GPU time.'
    original=a.assets.parent/'cycling_smplx_20260929'
    retained=[]
    for d in assets['derivatives']:
        source=original/d['pose_source_id']
        for p in [source/'canonical_pose.npz',source/'geometry/vertices.npy',source/'geometry/densekp_3d.npy']:
            if p.stat().st_mtime>gpu_check.timestamp():raise ValueError('Original reconstruction was modified during this task')
            record=dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
            if record not in retained:retained.append(record)
    pending=[dict(item='independent 3D hair and commuting clothing',status='next_priority_not_in_this_task'),
             dict(item='more independent public-image poses',status='next_priority_not_in_this_task'),
             dict(item='formal split and training scale',status='not_defined; all eight combinations remain integration_only'),
             dict(item='training/inference and datasets packaging',status='not_run; excluded by task')]
    result=dict(status='passed_for_integration_only',split='integration_only',completed_utc=now.isoformat(),
        counts=dict(characters=4,independent_pose_sources=2,derivatives=8,reference_bodies=4,
                    final_studio_views=48,ds_scenes=2,ds_camera_views=4,loader_rows=ds['loader_rows'],
                    supervised_points=ds['supervised_points'],supervised_occluded_points=ds['supervised_occluded_points']),
        timing=timing,thresholds=numeric['thresholds'],scene_validations=ds['scene_validations'],
        reviewed_failures=visual['rejected_versions'],pending=pending,
        ds_visual_review=ds_visual,
        limitations=visual['remaining_limits']+[visual['geometry_identity_limit']],
        retained_original_assets=retained,original_preservation_check='mtime predates task; original geometry never written by these tools',
        source_lineage=dict(photo_manifest=str(a.assets.parent.parent/'motion_inputs/cycling_images_20260929/manifest.json'),
                            original_smplx=str(original),character_manifests=assets['characters'],
                            derivative_pose_source='source_parameters and pose_source_id in each geometry_manifest.json'),
        tests=dict(count=16,status='passed',log='tests_final.log'),
        gpu_preflight=dict(time_local='2026-09-29 22:19:08 +08:00',gpu0_utilization_percent=0,gpu0_memory_mib=18,
                           existing_gpu1_blender_pid=2824895,render_device='CUDA_VISIBLE_DEVICES=0',
                           process_check='nvidia-smi and host ps; existing Blender kept running'),
        prohibited_actions=dict(downloads=0,reconstructions=0,densekp_inference_runs=0,training_runs=0,
                                commits=0,external_project_writes=0,sha256_calculations=0))
    save(final,result)
    assets['status']='passed_for_integration_only';assets['pending']=pending
    assets['acceptance_report']=str(final);save(a.assets/'report.json',assets)
    print(json.dumps(dict(output=str(final),counts=result['counts'],timing=timing),indent=2))


if __name__=='__main__':main()
