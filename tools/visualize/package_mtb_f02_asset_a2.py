"""Package the original-PTS F02 studio renders, without motion evaluation."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from fractions import Fraction
import json
import subprocess
import time
import av
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/mtb_f02_dressed_asset_a2_20261009'

def main():
    began=time.monotonic()
    recipe=json.loads((BASE/'frozen_recipe.json').read_text());timeline=json.loads((BASE/'timeline.json').read_text())
    probe=BASE/'probe/renders';full=BASE/'full/renders'
    manifests=[json.loads((p/'manifest.json').read_text()) for p in [probe,full]]
    records=sum([m['records'] for m in manifests],[])
    assert len(records)==240 and len(timeline)==120
    for view in ['front','side']:
        rows=[r for r in records if r['view']==view]
        assert sorted(r['asset_frame'] for r in rows)==list(range(120))
        assert all(r['camera_from_model']==rows[0]['camera_from_model'] for r in rows)
    assert all(r['source_frame']==timeline[r['asset_frame']]['source_frame'] for r in records)
    assert all(r['body_readback_exact'] and r['head_matrix_applied_once'] and r['groom_constant_topology'] for r in records)
    preview=BASE/'preview';preview.mkdir(exist_ok=False)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    for row in timeline:
        i=row['asset_frame'];folder=probe if i in recipe['probe_asset_frames'] else full
        im=Image.new('RGB',(1024,624),'#eef1f5');draw=ImageDraw.Draw(im)
        draw.text((12,8),f'F02 / MTB | asset {i:03d} | source {row["source_frame"]} | PTS {row["source_time_tick"]}/30000',font=font,fill='#142033')
        draw.text((12,35),f'Clip {row["episode_time_tick"]/30000:.6f}s | original 30000/1001 fps | fixed cameras / 64 samples',font=small,fill='#142033')
        for k,view in enumerate(['front','side']):
            with Image.open(folder/f'frame_{i:03d}_{view}.png') as raw:
                assert raw.size==(512,512);im.paste(raw,(512*k,64))
        draw.text((12,584),'INTEGRATION ONLY | Source body accepted; F02 body/cloth/groom await user review. Source license unknown.',font=small,fill='#142033')
        im.save(preview/f'frame_{i:03d}.png')
    # Published clock is exactly 1001 ticks; no float accumulation, duplicated end image or interpolation.
    ticks=[r['episode_time_tick'] for r in timeline]
    assert ticks==[1001*i for i in range(120)]
    cmd=['ffmpeg','-nostdin','-v','error','-framerate','30000/1001','-i',str(preview/'frame_%03d.png'),'-frames:v','120','-c:v','libx264','-crf','18','-preset','fast','-threads','2','-pix_fmt','yuv420p','-video_track_timescale','30000','-movflags','+faststart',str(preview/'comparison.mp4')]
    with (preview/'encode.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=log,check=True)
    with av.open(str(preview/'comparison.mp4')) as video:
        decoded=[Fraction(f.pts)*f.time_base for f in video.decode(video=0)]
    assert decoded==[Fraction(t,30000) for t in ticks]
    validation=dict(integration_only=True,frames=120,decoded_frame_count=len(decoded),exact_decoded_pts=True,time_base=[1,30000],relative_pts=ticks,source_first_pts=8670662,rate=[30000,1001],first_last_seconds=float(decoded[-1]),display_duration_seconds=120*1001/30000,interpolation=False,duplication=False,source_mapping='../time_mapping.csv',all_render_records_verified=True,command=cmd,elapsed_seconds=time.monotonic()-began)
    (preview/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    data=json.dumps(timeline)
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>F02 · 山地车穿衣动态</title>
<style>body{font:17px system-ui;max-width:1080px;margin:24px auto;padding:0 16px;background:#eef1f5;color:#142033}video,img{width:100%}button{padding:8px}input{width:65%}a{color:#0757b0}</style>
<h1>F02 · 新山地车动作 · 120帧</h1><p>源身体动作已获用户接受；此处是 F02 自身体型、原简单衣物和短波波头的新组合，<b>完整动态观感仍待你决定</b>。保留原前倾、双臂、交替屈膝及头颈估计；轻微局部穿模允许。</p>
<video controls preload="metadata" poster="preview/frame_000.png" src="preview/comparison.mp4"></video>
<p><a href="preview/comparison.mp4" download="bikefit_F02_dressed.mp4">下载 MP4</a> · <a href="../mtb_video_smplx_pilot_a2_20261009/index.html">已接受源身体与视频对照</a> · <a href="delivery_manifest.json">资产manifest</a> · <a href="timeline.json">整数时间表</a> · <a href="readback.json">参数/R138回读</a></p>
<p>两个固定方向（沿用studio的0°/90°相机，并非角色解剖正/侧面），每方向512×512、64 samples，共同固定2.3m视野。原展示帧8662–8781，PTS8670662–8789781、time_base=1/30000，30000/1001 fps。首末3.9706333秒，视频4.004秒含末帧展示；不插值、复制或循环。</p>
<p>远侧手臂与屈膝腿有投影重叠，遮挡头部仍为模型估计；没有新拟合或衣物物理。头发刚性随头，顶部亮度不作本轮调整。助理只检查关键帧与文件/数值，未将其称为动态人工验收。</p>
<h2>逐帧定位（原帧，不补帧）</h2><input id="seek" type="range" min="0" max="119" value="0"><p id="status"></p><img id="frame" src="preview/frame_000.png">
<p>integration_only。Bike Fit Adviser / <a href="https://www.youtube.com/watch?v=XBLMidr4dps">原视频</a>；源视频许可unknown，仅内部可行性评审，不公开发布素材/派生数据。F02外观不是源演员身份，APD与本动作上的F02共用角色家族；不独立拆分train/val。</p>
<script>const D=__DATA__;const slider=document.getElementById('seek');function show(i){const r=D[i];document.getElementById('frame').src='preview/frame_'+String(i).padStart(3,'0')+'.png';document.getElementById('status').textContent='asset '+i+' / source '+r.source_frame+' / PTS '+r.source_time_tick+'/30000 / clip '+(r.episode_time_tick/30000).toFixed(6)+' s';}slider.oninput=()=>show(+slider.value);show(0);</script></html>'''
    (BASE/'index.html').write_text(page.replace('__DATA__',data))
    (preview/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=../index.html"><a href="../index.html">F02完整播放页</a>')
    print(json.dumps(validation))

if __name__=='__main__':main()
