"""Small read-only geometric delivery for a bounded T-junction preflight."""
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiments/t_junction_multirig_a1_20261011'


def write(path,value):
    with Path(path).open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False,ensure_ascii=False)
        stream.write('\n')


def main():
    config=json.loads((ROOT/'configs/t_junction_multirig_a1_20261011.json').read_text())
    values={}
    for view in ('back','front','side'):
        check=json.loads((OUT/f'light_{view}_checks.json').read_text())
        best=max(check['attempts'],key=lambda row:(row['view_window'],row['smaller_camera_R138_height_median_px']))
        values[view]=dict(candidates=check['attempt_count'],light_passed=check['passed_count'],
                          maximum_view_window=max(row['view_window'] for row in check['attempts']),
                          maximum_dual_edge_window=max(row['edge_window'] for row in check['attempts']),
                          maximum_smaller_camera_proxy_height_px=max(row['smaller_camera_R138_height_median_px'] for row in check['attempts']),
                          frozen_minimum_proxy_height_px=config['minimum_target_proxy_height_px'][view],
                          representative_attempt=best['attempt'],
                          representative_origin_m=best['origin_m'],
                          representative_camera_seed=best['camera_child_seed'],
                          representative_frame_0_15_30=[best['frames'][i] for i in (0,15,30)])
    all_edge=max(row['edge_window'] for view in ('back','front','side')
                 for row in json.loads((OUT/f'light_{view}_checks.json').read_text())['attempts'])
    write(OUT/'candidate_summary.json',dict(schema='t_junction_candidate_summary_v1',
        attempted_camera_candidates=48,maximum_any_rig_dual_edge_run=all_edge,
        all_three_rigs_ready=False,render_not_started_reason='frozen_camera_gates_not_satisfied',
        no_new_candidates_after_48=True,views=values,
        accepted_front_repair_reference_native_cam4_median_px=438,
        note='R138 heights are projected geometry proxies; no visible-instance box or RGB exists for this task.'))
    matrices=np.load(OUT/'prepared_scene/scene_matrices.npy')
    def xy(x,y):return round(95+(x+13)*30,1),round(35+(94-y)*30,1)
    def rect(x0,y0,x1,y1,fill):
        x,y=xy(x0,y1);return f'<rect x="{x}" y="{y}" width="{(x1-x0)*30}" height="{(y1-y0)*30}" fill="{fill}"/>'
    elements=[rect(-13,72,13,94,'#deded8'),rect(-3.5,72,3.5,80,'#777777'),
              rect(-13,80,13,88,'#777777'),rect(-7.5,72,-3.5,80,'#cfc8b4'),
              rect(3.5,72,7.5,80,'#cfc8b4'),rect(-13,88,13,92,'#cfc8b4')]
    colors=['#ef5350','#2576c1','#2e9a61']
    for person in range(3):
        points=[xy(*matrices[k,person,:2,3]) for k in range(31)]
        poly=' '.join(f'{x},{y}' for x,y in points)
        elements.append(f'<polyline points="{poly}" fill="none" stroke="{colors[person]}" stroke-width="5"/>')
        for label,k in [('0',0),('15',15),('30',30)]:
            x,y=points[k]
            elements.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{colors[person]}"/>')
            elements.append(f'<text x="{x+7}" y="{y-7}" font-size="12" fill="#111">p{person+1} f{label}</text>')
    for view,color in [('back','#202060'),('front','#cc7e16'),('side','#78218b')]:
        x,y=xy(*values[view]['representative_origin_m'][:2])
        elements.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}" stroke="#fff" stroke-width="2"/>')
        elements.append(f'<text x="{x+10}" y="{y+5}" font-size="13" fill="{color}">{view} candidate</text>')
    elements.append('<text x="20" y="25" font-size="17">Positive-Y city T junction · road and frozen candidate geometry</text>')
    elements.append('<text x="20" y="680" font-size="12">Road top Z=0; paths sampled at 10 Hz. Camera markers are rejected candidates, not rendered rigs.</text>')
    svg='<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 700">'+''.join(elements)+'</svg>'
    (OUT/'junction_topdown.svg').write_text(svg)
    html='''<!doctype html><meta charset="utf-8"><title>T-junction preflight</title>
    <h1>城市 T 形路口：有界几何预检</h1><p>三人轨迹通过道路检查；三组相机候选没有同时满足冻结的视角、尺寸及双边缘条件，因此未启动渲染，也没有新视频。</p>
    <img src="junction_topdown.svg" alt="Road and three rider paths" style="max-width:100%">
    <p><a href="candidate_summary.json">48个相机候选的摘要</a> · <a href="trajectory_checks.json">轨迹检查</a> · <a href="junction_inspection.json">原.blend路面射线结果</a></p>'''
    (OUT/'index.html').write_text(html)


if __name__=='__main__':main()
