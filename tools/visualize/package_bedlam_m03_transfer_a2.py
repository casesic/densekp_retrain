"""Package native BEDLAM2 garment/M03 comparison at the frozen 30 Hz clock."""
import argparse
import json
from pathlib import Path
import subprocess

from package_composed_asset_a2 import package


def packets(path):
    raw = subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0',
        '-show_packets','-show_entries','packet=pts','-of','json',str(path)])
    return sorted(int(p['pts']) for p in json.loads(raw)['packets'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-root', type=Path, required=True)
    ap.add_argument('--accepted-video', type=Path, required=True)
    a = ap.parse_args()
    out = a.output_root.resolve()
    name = 'M03_KIT11_BEDLAM_native'
    result = package(out, name, 'preview_full')
    new = out/'builds'/name/'preview_full/comparison.mp4'
    old = a.accepted_video.resolve()
    if packets(old) != list(range(120)) or packets(new) != list(range(120)):
        raise ValueError('source_video_timeline_mismatch')
    combined = out/'M03_simple_vs_BEDLAM_transferred.mp4'
    if combined.exists(): raise FileExistsError(combined)
    subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-i',str(old),'-i',str(new),
        '-filter_complex','[0:v][1:v]hstack=inputs=2[v]','-map','[v]',
        '-c:v','libx264','-preset','veryfast','-crf','20','-pix_fmt','yuv420p',
        '-video_track_timescale','30','-vsync','0',str(combined)],check=True)
    pts = packets(combined)
    if pts != list(range(120)): raise ValueError('combined_video_timeline_mismatch')
    page = out/'index.html'
    if page.exists(): raise FileExistsError(page)
    page.write_text('''<!doctype html><html lang="zh"><meta charset="utf-8"><title>M03衣物对照</title>
<style>body{font:16px sans-serif;background:#181818;color:#eee;max-width:2100px;margin:2em auto;line-height:1.5}a{color:#9dd}video{width:100%}</style>
<h1>同一M03身体与行走：贴身衣物 / BEDLAM2原配衣物转移</h1>
<p>左半为已接受的M03贴身上衣和短裤，右半为BEDLAM2 gr_eric_033_3XL原衣物仿真缓存转移到M03。两半各有正面、侧面固定镜头；120个发布帧，30Hz同一时间表，无改身体或动作。转移不是M03布料重仿真；轻微局部穿模可接受，连续观感待用户评审。</p>
<video controls preload="metadata" src="M03_simple_vs_BEDLAM_transferred.mp4"></video>
<p><a href="builds/M03_KIT11_BEDLAM_native/preview_full/index.html">转移版逐帧</a> · <a href="builds/M03_KIT11_BEDLAM_native/delivery_manifest.json">来源与绑定</a> · <a href="probe_contact_gray.png">无贴图几何探针</a></p>
<p>只供内部integration_only；没有公共道路/训练数据验收。</p></html>''')
    delivery = dict(status='complete_pending_user_visual_review',frames=120,time_base=[1,30],
        matched_display_pts=True,combined_mp4=str(combined),new_preview=result,
        original_baseline=str(old),native_outfit='gr_eric_033_3XL',
        original_garment_cache_preserved=True,target_body_unchanged=True,
        original_diffuse_and_normal_applied=True,public_scene_consumed=False)
    (out/'delivery.json').write_text(json.dumps(delivery,indent=2)+'\n')
    print(json.dumps(dict(index=str(page),frames=120,pts_exact=True)))


if __name__=='__main__': main()
