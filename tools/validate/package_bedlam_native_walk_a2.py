"""Validate and package the frozen native walking preview, without rerendering."""
import csv
import argparse
import json
from pathlib import Path
import subprocess
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/experiments/bedlam_native_walk_a2_20261009'
ASSET = OUT / 'native_attempt_2'


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    global ASSET
    parser = argparse.ArgumentParser()
    parser.add_argument('--appearance-overlay', type=Path)
    options = parser.parse_args()
    source_asset = ASSET
    if options.appearance_overlay:
        ASSET = options.appearance_overlay.resolve()
    started = time.monotonic()
    recipe = json.loads((source_asset / 'frozen_recipe.json').read_text())
    rows = list(csv.DictReader((source_asset / 'time_mapping.csv').open()))
    assert len(rows) == 91
    for i, row in enumerate(rows):
        assert all(int(row[k]) == i for k in ('asset_frame', 'body_frame', 'cloth_frame', 'time_tick'))
        assert int(row['cloth_simulation_frame']) == i + 101
        assert (int(row['time_base_num']), int(row['time_base_den'])) == (1, 30)
    g = source_asset / 'geometry'
    body = np.load(g / 'vertices.npy')
    cloth = np.load(g / 'clothing_vertices.npy')
    faces = np.load(g / 'faces.npy')
    cf = np.load(g / 'clothing_faces.npy')
    uv = np.load(g / 'clothing_vertices_uv.npy')
    uf = np.load(g / 'clothing_faces_uv.npy')
    labels = np.load(g / 'densekp_3d_pending.npy')
    for x in (body, cloth, uv, labels):
        assert np.isfinite(x).all()
    assert body.shape == (91, 10475, 3) and cloth.shape == (91, 19639, 3)
    assert cf.shape == uf.shape == (38861, 3)
    assert cf.min() >= 0 and cf.max() < cloth.shape[1]
    assert uf.min() >= 0 and uf.max() < len(uv)
    with np.load(recipe['clothing_source'], allow_pickle=False) as z:
        np.testing.assert_array_equal(cloth, z['vertices_seq'][:91].astype(np.float32))
        for key, value in [('faces', cf), ('vertices_uv', uv), ('faces_uv', uf)]:
            np.testing.assert_array_equal(value, z[key])
    R = np.load(ROOT / 'assets/densekp_regressor_R.npy')
    expected = np.einsum('kv,tvc->tkc', R, body, optimize=True)
    error = float(np.max(np.abs(expected - labels)))
    assert error < 2e-6
    continuity = {}
    for name, value in [('body', body), ('clothing', cloth)]:
        speed = np.linalg.norm(np.diff(value, axis=0), axis=-1)
        continuity[name] = dict(per_step_median_m=np.median(speed, axis=1).tolist(),
                                per_step_p99_m=np.quantile(speed, .99, axis=1).tolist(),
                                max_vertex_step_m=float(speed.max()),
                                interpretation='includes original root; diagnostic, not a collision or perceptual acceptance test')
    textures = {}
    for name in ('diffuse', 'normal'):
        with Image.open(recipe[name]) as im:
            textures[name] = dict(size=im.size, mode=im.mode)
            im.verify()
    for i in range(91):
        for view in ('front', 'side'):
            with Image.open(ASSET / 'renders' / f'frame_{i:03d}_{view}.png') as im:
                assert im.size == (512, 512)
                im.verify()
    video = ASSET / 'native_walk_two_views.mp4'
    assert not video.exists(), 'Preserve existing video; do not silently overwrite'
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-n', '-f', 'rawvideo',
               '-pixel_format', 'rgb24', '-video_size', '1024x568', '-framerate', '30',
               '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
               '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)]
    encoding_started = time.monotonic()
    with (ASSET / 'encoding.log').open('wb') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=log)
        for i in range(91):
            canvas = Image.new('RGB', (1024, 568), (25, 25, 25))
            for j, view in enumerate(('front', 'side')):
                with Image.open(ASSET / 'renders' / f'frame_{i:03d}_{view}.png') as im:
                    canvas.paste(im.convert('RGB'), (512*j, 0))
            d = ImageDraw.Draw(canvas)
            d.text((12, 516), f'FRONT                                      SIDE     |     original root / fixed cameras', font=font, fill='white')
            d.text((12, 540), f'asset/body/cache {i:03d} | cloth simulation {101+i} | t={i}/30 = {i/30:.3f}s | 30 Hz', font=font, fill='white')
            if i == 45:
                canvas.save(ASSET / 'poster.jpg', quality=95)
            proc.stdin.write(canvas.tobytes())
        proc.stdin.close()
        assert proc.wait() == 0
    encoding_seconds = time.monotonic() - encoding_started
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height,r_frame_rate,nb_frames,duration',
                        '-of', 'json', str(video)], text=True))
    stream = probe['streams'][0]
    assert int(stream['nb_frames']) == 91 and stream['r_frame_rate'] == '30/1'
    write(ASSET / 'video_validation.json', dict(ffprobe=probe, input_images=182,
          encoding_seconds=encoding_seconds, frame_mapping='video n = asset n = body n = cloth cache n; simulation n+101',
          first_last_span_seconds=3, encoded_duration_seconds=91/30, resampling=False,
          repeated_or_dropped_frames=False, command=command))
    page = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>BEDLAM2 原配行走预览</title><style>body{font:17px sans-serif;background:#202124;color:#eee;max-width:1120px;margin:30px auto;line-height:1.6}video{width:100%;max-width:1024px}a{color:#9ed1ff}</style>
<h1>BEDLAM2 原身体＋原动作＋原配衣物</h1>
<p>us_2495_L / 2200 · nt_belle_007 texture_01 · integration_only · <b>待用户动态观感评审</b></p>
<video controls preload="metadata" poster="poster.jpg" src="native_walk_two_views.mp4"></video>
<p>91个原时刻，30 Hz，源帧0–90，首末跨度3秒；视频占时91/30秒。左右为固定正面/侧面，同一时刻、同一比例，保留原行进位移，没有补帧或变速。播放器不自动循环。</p>
<p>请重点观看走路是否自然、衣物是否有明显错位/断裂或持续闪烁。轻微局部穿模不自动否决。没有加头发或鞋；身体使用中性预览材质，衣物使用原UV及diffuse/normal贴图。64 samples，无降噪。</p>
<p>技术回读通过不能替代动态人眼验收。R138数值回读通过，但locked-head模板上的语义兼容仍待确认；尚未接入道路或DS生成。</p>
<p><a href="native_walk_two_views.mp4" download="KIT425_BEDLAM_native_walk.mp4">下载MP4</a> · <a href="manifest.json">资产manifest</a> · <a href="time_mapping.csv">原时间映射</a> · <a href="validation.json">技术检查</a></p></html>'''
    if options.appearance_overlay:
        page = page.replace('BEDLAM2 原身体＋原动作＋原配衣物', 'BEDLAM2 已接受行走＋新增眼睛和短波波头')
        page = page.replace('没有加头发或鞋；身体使用中性预览材质，衣物使用原UV及diffuse/normal贴图。',
            '身体、衣物和动作已经用户接受且保持原样；新增原生眼球材质与本地BEDLAM2短波波头。发型为本次选择，不宣称原演员配套发型。没有毛发物理或新增鞋。')
        page = page.replace('<p>技术回读通过', '<p><a href="head_detail_frame045.png">查看第45帧头部细节（静帧）</a>。请重点连续观看眼睛是否自然、头发是否随头且无脱体/几何跳动；64 samples无降噪仍可能有采样颗粒。</p><p>技术回读通过')
        (ASSET/'time_mapping.csv').write_text((source_asset/'time_mapping.csv').read_text())
    (ASSET / 'preview.html').write_text(page)
    write(ASSET / 'validation.json', dict(integration_only=True, finite=True,
          frames=91, shared_fixed_topology=True, original_cloth_exact_float32=True,
          original_uv_and_faces_exact=True, original_16_betas=True,
          R_max_readback_abs_error=error, R_numeric_pass=True,
          densekp_label_ready='pending_locked_head_template_semantics',
          body_vertices=10475, body_faces=len(faces), garment_vertices=19639,
          garment_faces=len(cf), textures=textures, continuity=continuity,
          images_checked=182, video_frames_checked=91,
          dynamic_human_review='pending_user', per_frame_grounding=False,
          per_frame_scaling=False, garment_rebinding=False,
          package_and_validation_seconds=time.monotonic()-started))
    print(json.dumps(dict(video=str(video), R_error=error, encoding_seconds=encoding_seconds)))


if __name__ == '__main__':
    main()
