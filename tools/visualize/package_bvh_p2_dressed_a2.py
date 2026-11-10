#!/usr/bin/env python3
"""Package two-view studio PNGs at the published BVH nominal sample interval."""
import argparse,json,subprocess,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'outputs/experiments/bvh_p2_dressed_asset_a2_20261010'
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--asset-root',type=Path,default=BASE)
    ap.add_argument('--probe-folder',default='renders_gpu_retry');ap.add_argument('--full-folder',default='renders_gpu')
    opts=ap.parse_args();base=opts.asset_root.resolve();start=time.monotonic();recipe=json.loads((base/'frozen_recipe.json').read_text())
    probe=base/'probe'/opts.probe_folder;full=base/'full'/opts.full_folder;out=base/'preview';out.mkdir(exist_ok=False)
    n=recipe['asset_frames'];lo,hi=recipe['source_frame_half_open'];tick=recipe['ticks_per_frame'];name=json.loads(Path(recipe['character_manifest']).read_text())['character_id']
    assert hi-lo==n
    chosen=set(recipe['probe_asset_frames']);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    for k in range(n):
        stage=probe if k in chosen else full
        front=stage/f'frame_{k:03d}_front.png';side=stage/f'frame_{k:03d}_side.png'
        assert front.is_file() and side.is_file(),(k,stage)
        with Image.open(front) as a,Image.open(side) as b:
            assert a.size==b.size==(512,512)
            canvas=Image.new('RGB',(1024,560),'#edf0f5');canvas.paste(a,(0,42));canvas.paste(b,(512,42))
        d=ImageDraw.Draw(canvas);t=k*tick/10000000
        d.text((10,9),f'{name} | asset {k} | BVH source {lo+k} | nominal t={t:.7f}s',font=font,fill='#17253a')
        d.text((10,537),'Front and side fixed studio views · 64 samples · source exposure PTS unknown · integration_only',font=font,fill='#354257')
        canvas.save(out/f'frame_{k:04d}.jpg',quality=91)
    rate=f'10000000/{tick}';mp4=out/'comparison.mp4'
    cmd=['ffmpeg','-nostdin','-v','error','-n','-framerate',rate,'-i',str(out/'frame_%04d.jpg'),'-frames:v','120',
         '-c:v','libx264','-preset','veryfast','-threads','2','-crf','20','-pix_fmt','yuv420p','-video_track_timescale','10000000','-movflags','+faststart',str(mp4)]
    cmd[cmd.index('-frames:v')+1]=str(n)
    subprocess.run(cmd,check=True)
    probe=subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_entries','frame=best_effort_timestamp_time','-of','csv=p=0',str(mp4)],capture_output=True,text=True,check=True)
    pts=np.array([float(q.strip()) for q in probe.stdout.splitlines() if q.strip()]);assert len(pts)==n
    diff=float(np.max(np.abs(pts-np.arange(n)*tick/10000000)));assert diff<1e-5
    html='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CHAR dressed review</title><style>body{font-family:system-ui;max-width:1100px;margin:2em auto;background:#f2f4f7;color:#1e293b}video,img,input{width:100%}a{color:#155cb6}</style><h1>CHAR：穿衣连续预览</h1><p>N帧源BVH [LO,HI)，标称间隔TICK/10000000秒。左右为固定studio观察方向；身体已获用户接受，本次衣物与头发动态待用户观看。原BVH曝光时刻未知，播放时间来自发布的BVH Frame Time，不是拍摄PTS。全部integration_only。</p><p><a href="BODYLINK">已接受身体对照</a></p><video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="__DOWNLOAD_NAME__">下载 MP4</a></p><p>asset帧 <span id="asset">0</span> · BVH源帧 <span id="source">LO</span> · 标称片段时间 <span id="seconds">0</span>s</p><input type="range" min="0" max="LAST" value="0" id="seek"><img src="frame_0000.jpg" id="frame"><script>let s=document.getElementById('seek');s.oninput=()=>{let i=+s.value;document.getElementById('asset').textContent=i;document.getElementById('source').textContent=LO+i;document.getElementById('seconds').textContent=(i*DT).toFixed(7);document.getElementById('frame').src='frame_'+String(i).padStart(4,'0')+'.jpg'};</script></html>'''
    body_link='../../../bvh_p1_smplx_pilot_a2_20261010/'+base.name+'/preview/index.html' if opts.asset_root.resolve()!=BASE else '../../bvh_p2_smplx_pilot_a2_20261010/preview/index.html'
    for key,value in [('BODYLINK',body_link),('CHAR',name),('TICK',str(tick)),('LAST',str(n-1)),('LO',str(lo)),('HI',str(hi)),('DT',f'{tick/10000000:.7f}'),('N',str(n))]:
        html=html.replace(key,value)
    (out/'index.html').write_text(html.replace('__DOWNLOAD_NAME__',f'{name}_dressed.mp4'))
    result=dict(integration_only=True,frames=n,decoded_frames=len(pts),video_rate=rate,video_time_base=[1,10000000],
        max_display_PTS_error_s=diff,nominal_interval=f'{tick}/10000000 seconds from BVH header; actual capture PTS unknown',
        views=['front label: studio camera at display negative Y','side label: studio camera at display positive X'],
        probe_frames_reused=sorted(chosen),full_frames_rendered=n-len(chosen),dynamic_human_review='pending_user',
        CPU_packaging_seconds=time.monotonic()-start)
    (out/'execution.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
