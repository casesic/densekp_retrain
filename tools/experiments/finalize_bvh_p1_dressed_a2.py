#!/usr/bin/env python3
"""Finalize two M02 dressed reviews after both complete, preserving A2 catalog."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/bvh_p1_dressed_assets_a2_20261010'
CATALOG=ROOT/'outputs/plans/dynamic_asset_handoff_a2_20261009/index.json'
CASES={'p1_comfortable':'中度弯肘','p1_alternate':'较前倾'}

def put(path,data):
    assert not path.exists(),path
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

def main():
    items=[];spent=0.
    for case,label in CASES.items():
        base=BASE/case;m=json.loads((base/'delivery_manifest.json').read_text())
        pre=json.loads((base/'validation.json').read_text());video=json.loads((base/'preview/execution.json').read_text())
        probe=json.loads((base/'probe/renders_gpu/manifest.json').read_text());full=json.loads((base/'full/renders_gpu/manifest.json').read_text())
        assert m['frame_count']==video['decoded_frames']==96 and pre['all_finite']
        assert len(probe['records'])==10 and len(full['records'])==182
        elapsed=pre['cpu_seconds']+probe['elapsed_seconds']+full['elapsed_seconds']+video['CPU_packaging_seconds']
        spent+=elapsed
        items.append(dict(case=case,label=label,asset_id=m['asset_id'],source_frame_half_open=m['source_frame_half_open'],
          manifest=str(base/'delivery_manifest.json'),preview=str(base/'preview/index.html'),
          mp4=str(base/'preview/comparison.mp4'),body_human_status='accepted_by_user',
          dressed_human_status='pending_user',A1_adapter_status='passed_read_only',public_scene_status='not_consumed',
          compute_seconds=elapsed))
    assert spent<1800,spent
    out=dict(integration_only=True,source_group='zenodo_10668611_participant_1',character_family='cycling_M02',
      same_participant_recording=True,items=items,body_human_status='accepted_by_user',
      dressed_human_status='pending_user',public_scene_status='not_consumed',independent_validation=False,
      summed_success_compute_seconds=spent,compute_budget_seconds=1800,
      failed_sandbox_probe_seconds_approx=0.6,total_including_failure_seconds_approx=spent+0.6)
    put(BASE/'delivery_manifest.json',out)
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>M02 两段骑行穿衣评审</title><style>body{font:16px system-ui;max-width:1100px;margin:2em auto;padding:0 1em;background:#f2f4f7;color:#1e293b}section{background:white;padding:1em;margin:2em 0;border-radius:8px}video{width:100%}a{color:#155cb6}</style><h1>cycling_M02_v3：两段骑行穿衣评审</h1><p>两段各96帧，原BVH标称间隔416667/10000000秒；身体动态已接受，新衣物与头发动态待人眼评审。两段同一原参与者和录制，全部integration_only；并非两个独立验证来源。</p>'''
    for x in items:
        case=x['case'];lo,hi=x['source_frame_half_open']
        html+=f'<section><h2>{x["label"]}：源帧 [{lo},{hi})</h2><p><a href="{case}/preview/index.html">逐帧页</a> · <a href="{case}/preview/comparison.mp4" download="participant_1_{case}_M02_dressed.mp4">下载原速 MP4</a> · <a href="{case}/delivery_manifest.json">交付清单</a></p><video controls preload="metadata" src="{case}/preview/comparison.mp4"></video></section>'
    (BASE/'index.html').write_text(html+'</html>')
    assert not (BASE/'catalog_before_append.json').exists()
    (BASE/'catalog_before_append.json').write_bytes(CATALOG.read_bytes())
    catalog=json.loads(CATALOG.read_text());known={x['asset_id'] for x in catalog['assets']}
    for x in items:
        assert x['asset_id'] not in known
        catalog['assets'].append(dict(asset_id=x['asset_id'],manifest=str(BASE/x['case']/'a1_proposal.json'),
          human_acceptance='accepted_body_motion_by_user_dressed_pending_user',densekp_status='compatible',
          source_group=out['source_group'],character_family=out['character_family'],
          A1_adapter_status='passed_read_only',public_scene_status='not_consumed'))
    catalog['combination_count']=len(catalog['assets']);catalog['independent_motion_group_count']=5
    catalog['notes'].append('participant_1 two M02 segments share actor/session and character family; body accepted, dressed pending, scene not consumed')
    CATALOG.write_text(json.dumps(catalog,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(items=len(items),success_seconds=spent,catalog_count=catalog['combination_count'])))

if __name__=='__main__':main()
