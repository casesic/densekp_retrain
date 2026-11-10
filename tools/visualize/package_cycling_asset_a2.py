"""Review entry: accepted continuous preview plus explicitly discrete failed probes."""
import csv
import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiments/continuous_cycling_asset_a2_20261009'

def main():
    out=OUT/'review_for_user';out.mkdir(exist_ok=False)
    rows=list(csv.DictReader((OUT/'time_mapping.csv').open()));font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    frames=[0,6,16,38,75];records=[]
    for i in frames:
        old=Image.open(ROOT/f'outputs/experiments/apd_groom_flicker_20261009/full_preview/frames/frame_{i:03d}.png').convert('RGB')
        canvas=Image.new('RGB',(1536,1100),'#eef1f5');draw=ImageDraw.Draw(canvas)
        draw.text((10,8),f'APD6526 source {i+30} / asset {i} / source time {rows[i]["source_relative_seconds"]} s',font=font,fill='black')
        for col,title in enumerate(['Accepted simple clothes','BEDLAM initial: USER REVIEW','BEDLAM local correction: USER REVIEW']):draw.text((col*512+10,34),title,font=font,fill='black')
        for j,view in enumerate(['front','side']):
            canvas.paste(old.crop((j*512,64,(j+1)*512,576)),(0,60+j*512))
            for n in [1,2]:
                source=OUT/f'bedlam_attempt_{n}/probe_renders/frame_{i:03d}_{view}.png'
                with Image.open(source) as im:canvas.paste(im.convert('RGB'),(n*512,60+j*512))
        canvas.save(out/f'probe_{i:03d}.png');records.append(dict(asset_frame=i,source_frame=i+30,source_time_tick=int(rows[i]['timestamp_original']),clip_seconds=float(rows[i]['clip_seconds']),file=f'probe_{i:03d}.png'))
    (out/'probe_mapping.json').write_text(json.dumps(records,indent=2)+'\n')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>A2 骑行资产交付</title>
<style>body{font:16px sans-serif;margin:24px;background:#edf1f4;color:#182431}img{max-width:100%;height:auto}button{padding:8px;margin:4px}a{color:#165cbb}</style>
<h1>APD6526 / cycling_M01_v2 — integration_only</h1>
<p><a href="../../apd_groom_flicker_20261009/full_preview/frames/index.html">已接受的连续76帧：原64 samples双视角播放页</a> ·
<a href="../../apd_groom_flicker_20261009/full_preview/frames/lossless.mp4">原连续MP4</a> · <a href="../accepted/manifest.json">冻结资产manifest</a></p>
<h2>BEDLAM 实际适配探针：以用户观感判断</h2>
<p>以下仅5个不连续姿态，不能代表连续播放质量。左：已接受简单衣物；中：初始BEDLAM绑定；右：唯一一次局部修正。
两版的袖肘局部拉伸/折转是诊断提示，不以边长比单独否决。等待用户选择哪版进入完整76帧预览；目前不替换已接受版。</p>
<p>相同冻结身体、groom、材质外观（衣物本身除外）、相机、灯光、64 samples；每个面板原生512×512。上排正面，下排侧面。</p>
<div id="buttons"></div><p id="label"></p><img id="frame" alt="三个衣物版本同姿态同视角">
<p><a href="../ATTRIBUTION.md">APD、BEDLAM和角色各自来源/许可</a>。不是高精度重建、正式数据集或骑行双路生成验收。</p>
<script>const D='''+json.dumps(records)+''';function show(i){const d=D[i];document.getElementById('frame').src=d.file;document.getElementById('label').textContent=`asset ${d.asset_frame} / source ${d.source_frame} / clip ${d.clip_seconds.toFixed(6)}s（离散探针）`;}
D.forEach((d,i)=>{const b=document.createElement('button');b.textContent='源帧 '+d.source_frame;b.onclick=()=>show(i);document.getElementById('buttons').appendChild(b);});show(0);</script>''')
    print(out/'index.html')
if __name__=='__main__':main()
