#!/usr/bin/env python3
"""Package existing fixed-view renders using the frozen APD timestamps; no geometry changes."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import csv
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tools/visualize'))
from preview_apd_6526_pilot import encode
BASE=ROOT/'outputs/experiments/apd_6526_dressed_pilot_20261008'


def main():
    global BASE
    parser=argparse.ArgumentParser();parser.add_argument('--asset-root',type=Path);args=parser.parse_args()
    if args.asset_root:BASE=args.asset_root.resolve()
    began=time.perf_counter();recipe=json.loads((BASE/'frozen_recipe.json').read_text());decision=json.loads((BASE/'probe_decision.json').read_text())
    character=recipe.get('character','cycling_M01_v2');samples=recipe['budget']['samples']
    probe=BASE/decision['reuse_probe_render_folder'];full=BASE/decision['full_render_folder']
    pm=json.loads((probe/'manifest.json').read_text());fm=json.loads((full/'manifest.json').read_text())
    records=pm['records']+fm['records'];assert len(records)==152
    for view in ['front','side']:
        rows=[r for r in records if r['view']==view]
        assert sorted(r['asset_frame'] for r in rows)==list(range(76))
        assert all(r['camera_from_model']==rows[0]['camera_from_model'] for r in rows)
        assert all(r['body_readback_exact'] and r['groom_constant_topology'] and r['head_matrix_applied_once'] for r in rows)
    p=np.load(recipe['authoritative_body']);times=p['clip_seconds'];mapping=list(csv.DictReader((BASE/'time_mapping.csv').open()))
    preview=BASE/'preview';preview.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    files=[]
    for n in range(76):
        folder=probe if n in recipe['probe_asset_frames'] else full
        im=Image.new('RGB',(1024,640),'#eef1f5');d=ImageDraw.Draw(im)
        d.text((12,8),f'APD 6526 / {character} | asset {n:02d} | source {n+30} | clip {times[n]:.6f}s',font=font,fill='#142033')
        d.text((12,34),f'Source time {mapping[n]["source_relative_seconds"]}s | frozen body + existing clothes + rigid-head groom | INTEGRATION ONLY',font=small,fill='#384458')
        for k,view in enumerate(['front','side']):
            raw=Image.open(folder/f'frame_{n:03d}_{view}.png');assert raw.size==(512,512)
            im.paste(raw,(k*512,64));d.text((12+k*512,70),view+' / fixed camera',font=small,fill='white')
        d.text((12,586),'Same source timestamp, fixed target pelvis, no per-frame floor shift. Target dressed motion awaits user review.',font=small,fill='#384458')
        d.text((12,609),f'{samples} samples. APD: Kress et al., Zenodo 5724486 / CC BY 4.0. Separate character/model licenses apply.',font=small,fill='#384458')
        name=f'frame_{n:03d}.png';im.save(preview/name);files.append(name)
    contacts=[]
    for start in range(0,76,8):
        count=min(8,76-start);sheet=Image.new('RGB',(1024,320*((count+1)//2)),'white')
        for k in range(count):
            with Image.open(preview/files[start+k]) as im:sheet.paste(im.resize((512,320)),((k%2)*512,(k//2)*320))
        name=f'contact_{start:03d}.jpg';sheet.save(preview/name,quality=93);contacts.append(name)
    encode(preview,times)
    data=dict(files=files,times=times.tolist(),source_ids=p['source_frame_ids'].tolist(),source_seconds=[float(r['source_relative_seconds']) for r in mapping])
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>APD 6526 · 穿衣动态评审</title><style>body{font:16px system-ui;max-width:1100px;margin:24px auto;padding:0 16px;background:#eef1f5;color:#142033}img,video{width:100%;background:#505050}button{padding:9px;margin:4px}input{width:60%}.note{background:white;padding:16px;border-radius:8px}a{color:#0757b0}</style>
<h1>同一 76 帧：衣物与头发动态评审</h1><p class="note">cycling_M01_v2；APD 6526 原帧 30–105，首末跨度 3 秒。身体已按用户决定冻结，左臂快摆保持不变。<b>本轮穿衣动态仍待用户确认，integration_only。</b> 请重点看源 35–48 帧袖口/腋下、躯干条带和头发是否稳定随头运动。允许轻微局部穿模。</p>
<video controls preload="metadata" poster="frame_000.png" src="comparison.mp4"></video><p><a href="comparison.mp4" download="__DOWNLOAD_NAME__">下载 MP4</a> · <a href="../time_mapping.csv">原始时间映射</a> · <a href="../ATTRIBUTION.md">来源与修改说明</a> · <a href="../diagnostics/summary.json">数值诊断</a></p>
<p>正/侧两个固定方向，同一展示尺度；没有逐帧贴地。8 samples 低成本预览有可见颗粒，不是高质量外观认证。局部衣物翻面、毫米级衣物/头皮穿插见报告；数值通过不等于动态观感通过。</p>
<h2>按源时间播放 / 逐帧定位</h2><button id="play">播放 / 暂停</button><button id="restart">回到开头</button><input id="seek" type="range" min="0" max="75" value="0"><p id="status"></p><img id="frame" alt="双方向穿衣预览">
<p>播放保留原始时间差，不循环、不变速、不插值。MP4 PTS 与 76 个源时刻逐一核验；最后 40ms 为末帧展示时间，不增加观察。头发仅随头刚性运动，没有毛发动力学。尚无 DS 图像/标签打包，也没有独立 train/val。</p>
<p>APD: Kress et al., <a href="https://doi.org/10.5281/zenodo.5724486">Aschaffenburg Pose Dataset v1.0</a>，CC BY4.0。身体、shape、皮肤和 groom 沿用各自来源许可。</p>
<script>const D=__DATA__;D.files.forEach(f=>{const im=new Image();im.src=f;});let i=0,active=false,epoch=0;const slider=document.getElementById('seek');
function show(j){i=j;slider.value=j;document.getElementById('frame').src=D.files[j];document.getElementById('status').textContent='asset '+j+' / 源帧 '+D.source_ids[j]+' / 源 '+D.source_seconds[j].toFixed(6)+' s / 片段 '+D.times[j].toFixed(6)+' s';}
function tick(now){if(!active)return;const us=Math.round((now-epoch)*1000);let j=0;while(j<D.times.length-1&&Math.round(D.times[j+1]*1e6)<=us)j++;show(j);if(us>=Math.round(D.times.at(-1)*1e6)){active=false;return;}requestAnimationFrame(tick);}
document.getElementById('play').onclick=()=>{active=!active;if(active){epoch=performance.now()-D.times[i]*1000;requestAnimationFrame(tick);}};document.getElementById('restart').onclick=()=>{active=false;show(0);};slider.oninput=()=>{active=false;show(+slider.value);};show(0);</script></html>'''
    if args.asset_root:
        html=html.replace('cycling_M01_v2',character).replace('8 samples',f'{samples} samples')
        html=html.replace('身体已按用户决定冻结','沿用已接受APD姿态旋转，女性shape与固定pelvis另行求值')
        html=html.replace('../diagnostics/summary.json','../validation.json')
    download_name='APD6526_F02_dressed.mp4' if args.asset_root else 'APD6526_M01_dressed.mp4'
    (preview/'index.html').write_text(html.replace('__DATA__',json.dumps(data)).replace('__DOWNLOAD_NAME__',download_name))
    (BASE/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>APD6526 dressed pilot</title><h1>APD6526 穿衣动态评审</h1><p><a href="preview/index.html">打开双视角播放与逐帧评审页</a></p><p><a href="preview/comparison.mp4">MP4</a> · <a href="delivery_manifest.json">交付清单</a></p><p>integration_only；身体冻结，衣物与头发动态待用户确认。</p></html>')
    (preview/'manifest.json').write_text(json.dumps(dict(integration_only=True,frames=76,views=['front','side'],source_frames=[30,106],
        body_parameter_authority=recipe['authoritative_body'],all_view_frame_records_verified=True,common_fixed_scale_m=2.3,
        files=files,contacts=contacts,original_timing=True,interpolation=False,loop=False,independent_auto_scaling=False,
        reused_probe_asset_frames=recipe['probe_asset_frames'],exact_pts_metadata='mp4_metadata.json',
        human_dynamic_review='pending_user',elapsed_seconds=time.perf_counter()-began),indent=2)+'\n')
    print('Packaged 76 fixed-time frames; human dynamic review pending.')


if __name__=='__main__':main()
