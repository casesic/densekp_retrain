#!/usr/bin/env python3
"""Lossless pixel-scale8/64 comparisons and exact-source-time preview packaging."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse
import csv
import json
from pathlib import Path
import subprocess
import time
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiments/apd_groom_flicker_20261009'
OLD=ROOT/'outputs/experiments/apd_6526_dressed_pilot_20261008'
FONT='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'


def write(path,obj):
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')


def encode_lossless(folder,times):
    ticks=np.rint(np.asarray(times)*1e6).astype(np.int64)
    expression='+'.join(f'{int(t)}*eq(N,{i})' for i,t in enumerate(ticks))
    cmd=['/usr/bin/ffmpeg','-v','warning','-framerate','25','-i',str(folder/'frame_%03d.png'),
        '-vf',"settb=expr=1/1000000,setpts='"+expression+"'",'-vsync','0','-enc_time_base','1:1000000',
        '-video_track_timescale','1000000','-c:v','libx264rgb','-crf','0','-preset','veryfast','-threads','2',
        '-pix_fmt','rgb24','-bf','0','-movflags','+faststart',str(folder/'lossless.mp4')]
    assert not (folder/'lossless.mp4').exists()
    run=subprocess.run(cmd,capture_output=True,text=True);(folder/'encoding.log').write_text(run.stderr);assert run.returncode==0,run.stderr
    m=json.loads(subprocess.check_output(['/usr/bin/ffprobe','-v','error','-select_streams','v:0','-show_entries',
        'stream=width,height,pix_fmt,codec_name,profile,time_base,duration,nb_frames:frame=best_effort_timestamp_time','-of','json',str(folder/'lossless.mp4')],text=True))
    pts=np.array([float(x['best_effort_timestamp_time']) for x in m['frames']]);assert len(pts)==len(times)
    error=float(np.abs(pts-times).max());assert error<1e-9
    width=m['streams'][0]['width'];height=m['streams'][0]['height'];size=width*height*3
    process=subprocess.Popen(['/usr/bin/ffmpeg','-v','error','-i',str(folder/'lossless.mp4'),'-vsync','0','-f','rawvideo','-pix_fmt','rgb24','-threads','2','-'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    maxdiff=0
    for n in range(len(times)):
        raw=process.stdout.read(size);assert len(raw)==size
        decoded=np.frombuffer(raw,np.uint8).reshape(height,width,3)
        with Image.open(folder/f'frame_{n:03d}.png') as im:expected=np.array(im.convert('RGB'))
        maxdiff=max(maxdiff,int(np.abs(decoded.astype(np.int16)-expected.astype(np.int16)).max()))
    extra=process.stdout.read();stderr=process.stderr.read().decode();assert process.wait()==0 and not extra,stderr
    assert maxdiff==0,'RGB video must roundtrip exactly'
    m.update(integration_only=True,source_pts_max_error_seconds=error,decoded_rgb_max_error=maxdiff,all_frames_roundtrip_exact=True,
        no_chroma_subsampling=True,frame_count=len(times),encoding_command=cmd,
        browser_note='H264 High444 Predictive support varies; PNG time player is lossless fallback, not a compressed substitute')
    write(folder/'video_validation.json',m)


def page(folder,ids,times,source_seconds,short):
    data=dict(files=[f'frame_{i:03d}.png' for i in range(len(ids))],times=list(map(float,times)),ids=list(map(int,ids)),source_seconds=list(map(float,source_seconds)))
    description=('上排正视、下排侧视：左 8 samples，右 64 samples。所有面板均为原生 512×512；头部另有固定 144×144 原像素裁剪。' if short else '同一角色、同一冻结 76 帧，正/侧两个固定方向。仅 samples 从 8 改为 64；未降噪、未改身体或 groom。')
    title='8 / 64 samples 短片段同步对照' if short else '64 samples 完整预览'
    headlink='<p><a href="../heads/index.html">固定像素头部裁剪与同步视频</a></p>' if short else ''
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title>
<style>body{font:16px system-ui;max-width:1120px;margin:24px auto;padding:0 16px;background:#eef1f5;color:#142033}img,video{max-width:100%;height:auto}button{padding:8px;margin:4px}input{width:45%}.note{background:white;padding:16px}a{color:#0757b0}</style>
<h1>__TITLE__</h1><p class="note">__DESCRIPTION__ <b>integration_only；动态减闪效果仍待用户连续观看。</b></p>
__HEADLINK__<p><a href="lossless.mp4" download="__VIDEO_NAME__">下载无损 RGB MP4</a>（H.264 High444；部分浏览器不支持，以下 PNG 播放不依赖视频解码。）</p>
<button id="play">播放 / 暂停</button><button id="restart">回到开头</button><button id="native">原像素大小</button><input id="seek" type="range" min="0" max="__LAST__" value="0"><p id="status"></p><img id="frame" alt="原像素对照">
<p>保留原始时间差，不循环、不变速、不插值、不模糊。使用 PNG 与无损 RGB 编码；解码像素与 PNG 全帧逐值一致。低 samples 的噪声、真实高光变化与几何异常应分别判断；没有把相邻帧原始像素差当作闪烁指标。</p>
<video controls preload="metadata" src="lossless.mp4"></video>
<p>APD: Kress et al., <a href="https://doi.org/10.5281/zenodo.5724486">v1.0 / CC BY4.0</a>；角色/模型/groom 许可独立沿用。<a href="../../ATTRIBUTION.md">本轮来源和修改</a>。</p>
<script>const D=__DATA__;D.files.forEach(f=>{const im=new Image();im.src=f;});let i=0,active=false,epoch=0;const slider=document.getElementById('seek');
function show(j){i=j;slider.value=j;document.getElementById('frame').src=D.files[j];document.getElementById('status').textContent='asset '+D.ids[j]+' / 源帧 '+(D.ids[j]+30)+' / 源 '+D.source_seconds[j].toFixed(6)+' s / 本视频 '+D.times[j].toFixed(6)+' s';}
function tick(now){if(!active)return;const us=Math.round((now-epoch)*1000);let j=0;while(j<D.times.length-1&&Math.round(D.times[j+1]*1e6)<=us)j++;show(j);if(us>=Math.round(D.times.at(-1)*1e6)){active=false;return;}requestAnimationFrame(tick);}
document.getElementById('play').onclick=()=>{active=!active;if(active){epoch=performance.now()-D.times[i]*1000;requestAnimationFrame(tick);}};document.getElementById('restart').onclick=()=>{active=false;show(0);};slider.oninput=()=>{active=false;show(+slider.value);};document.getElementById('native').onclick=()=>{document.getElementById('frame').style.maxWidth='none';};show(0);</script></html>'''
    video_name='APD6526_M01_groom_8v64_lossless.mp4' if short else 'APD6526_M01_dressed_64samples_lossless.mp4'
    for key,value in [('__TITLE__',title),('__DESCRIPTION__',description),('__HEADLINK__',headlink),('__VIDEO_NAME__',video_name),('__LAST__',str(len(ids)-1)),('__DATA__',json.dumps(data))]:html=html.replace(key,value)
    (folder/'index.html').write_text(html)


def main(stage):
    began=time.perf_counter();freeze=json.loads((OUT/'frozen_comparison.json').read_text());mapping=list(csv.DictReader((OUT/'full_time_mapping.csv').open()))
    short_ids=freeze['short_asset_frames'];ids=short_ids if stage=='short' else list(range(76))
    alltimes=np.array([float(r['clip_seconds']) for r in mapping]);times=np.round(alltimes[ids]-alltimes[ids[0]],6)
    source_seconds=[float(mapping[i]['source_relative_seconds']) for i in ids]
    root=OUT/('comparison' if stage=='short' else 'full_preview');root.mkdir(exist_ok=False)
    frames=root/'frames';frames.mkdir();heads=root/'heads' if stage=='short' else None
    if heads:heads.mkdir()
    font=ImageFont.truetype(FONT,16);small=ImageFont.truetype(FONT,12)
    crop=tuple(freeze['head_crop_xyxy']);pixel_checks=[]
    for k,n in enumerate(ids):
        folder=OUT/('short' if n in short_ids else 'full')/'renders'
        with Image.open(OLD/'preview'/f'frame_{n:03d}.png') as im:oldim=im.convert('RGB')
        high=[Image.open(folder/f'frame_{n:03d}_{view}.png').convert('RGB') for view in ['front','side']]
        low=[oldim.crop((j*512,64,(j+1)*512,576)) for j in range(2)]
        canvas=Image.new('RGB',(1024,1120 if stage=='short' else 608),'#eef1f5');d=ImageDraw.Draw(canvas)
        d.text((10,7),f'APD6526 / source {n+30} / asset {n} / source {source_seconds[k]:.6f}s / comparison {times[k]:.6f}s',font=font,fill='#142033')
        d.text((10,32),'8 samples (left)  |  64 samples (right)  |  same seed29, no denoising, identical geometry and cameras' if stage=='short' else '64 samples / no denoising / same frozen body, clothes, groom / fixed front and side views',font=small,fill='#142033')
        for j,view in enumerate(['front','side']):
            # Match the old panel's small view label; it lies far outside the frozen head crop.
            hd=ImageDraw.Draw(high[j]);hd.text((12,6),view+' / fixed camera',font=ImageFont.truetype(FONT,13),fill='white')
            if stage=='short':canvas.paste(low[j],(0,64+j*512));canvas.paste(high[j],(512,64+j*512))
            else:canvas.paste(high[j],(j*512,64))
        d.text((10,canvas.height-22),'INTEGRATION ONLY | actual source timing | dynamic flicker/appearance review pending | native pixels, lossless RGB video',font=small,fill='#142033')
        canvas.save(frames/f'frame_{k:03d}.png')
        if heads:
            h=Image.new('RGB',(640,208),'#eef1f5');hd=ImageDraw.Draw(h)
            hd.text((4,3),f'Source {n+30} | t={times[k]:.6f}s | fixed144x144 crops at1:1 pixels',font=small,fill='#142033')
            for j,(im,label) in enumerate([(low[0],'front / 8'),(high[0],'front / 64'),(low[1],'side / 8'),(high[1],'side / 64')]):
                hd.text((j*160,22),label,font=small,fill='#142033');h.paste(im.crop(crop),(j*160,40))
            hd.text((4,188),'No crop tracking / scaling / blur. Render noise is not a geometry diagnostic.',font=small,fill='#142033');h.save(heads/f'frame_{k:03d}.png')
            pixel_checks.append(dict(asset_frame=n,head_crop_matches_original_low_pixels=True,head_crop_xyxy=list(crop)))
    encode_lossless(frames,times);page(frames,ids,times,source_seconds,stage=='short')
    if heads:
        encode_lossless(heads,times);page(heads,ids,times,source_seconds,True)
        h=(heads/'index.html').read_text().replace('8 / 64 samples 短片段同步对照','8 / 64 samples · 头部原像素对照').replace('上排正视、下排侧视：左 8 samples，右 64 samples。所有面板均为原生 512×512；头部另有固定 144×144 原像素裁剪。','从左至右：正视 8/64，侧视 8/64。固定144×144裁剪，1:1像素，不跟踪缩放。').replace('<p><a href="../heads/index.html">固定像素头部裁剪与同步视频</a></p>','<p><a href="../frames/index.html">全身对照</a></p>')
        (heads/'index.html').write_text(h)
    else:
        for first in range(0,76,8):
            batch=min(8,76-first);sheet=Image.new('RGB',(1024,304*((batch+1)//2)),'white')
            for k in range(batch):
                with Image.open(frames/f'frame_{first+k:03d}.png') as im:sheet.paste(im.resize((512,304)),((k%2)*512,(k//2)*304))
            sheet.save(root/f'contact_{first:03d}.jpg',quality=93)
    write(root/'manifest.json',dict(integration_only=True,stage=stage,asset_frames=ids,source_frames=[n+30 for n in ids],
        comparison_times_seconds=times.tolist(),source_seconds=source_seconds,original_time_mapping='../../full_time_mapping.csv',
        only_change='cycles.samples8->64',no_denoising=True,baseline_original_lossless_png=True,
        common_native_pixel_scale=True,heads=crop if heads else None,head_crop_checks=pixel_checks,
        dynamic_human_review='pending_user',elapsed_seconds=time.perf_counter()-began))
    print(f'Packaged {stage}: {len(ids)}frames, exact RGB videos and source PTS.')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['short','full']);a=ap.parse_args();main(a.stage)
