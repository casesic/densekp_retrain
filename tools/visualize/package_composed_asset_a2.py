#!/usr/bin/env python3
"""Package a configured studio check at its published source clock."""
import argparse
import html
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[2]
NAMES = ("A_f02_p2", "B_f02_p2_m02_cloth", "C_f02_p2_m02_cloth_hair")


def read(path):
    return json.loads(path.read_text())


def package(out, name, folder_name="preview_v3"):
    began = time.monotonic()
    base = out / "builds" / name
    manifest = read(base / "delivery_manifest.json")
    recipe = read(base / "preview_recipe.json")
    frames = int(recipe["preview_window"][1])
    rows = read(base / "timeline.json")[:frames]
    probe = read(base / "probe_decision.json")
    source = base / "probe" / probe["probe_render_folder"]
    rendered = base / "full/renders"
    if not (rendered / "manifest.json").is_file():
        raise ValueError("incomplete_full_render:" + name)
    # A first encoding attempt used a newer FFmpeg option unavailable here.
    # Keep that failure intact and publish successful work in a new directory.
    folder = base / folder_name
    if folder.exists():
        raise FileExistsError(folder)
    images = folder / "frames"
    images.mkdir(parents=True)
    for i, row in enumerate(rows):
        sides = []
        for view in ("front", "side"):
            image = source / f"frame_{i:03d}_{view}.png" if i in recipe["probe_asset_frames"] else rendered / f"frame_{i:03d}_{view}.png"
            if not image.is_file():
                raise FileNotFoundError(image)
            sides.append(Image.open(image).convert("RGB"))
        frame = Image.new("RGB", (1024, 544), (40, 40, 40))
        frame.paste(sides[0], (0, 32))
        frame.paste(sides[1], (512, 32))
        ImageDraw.Draw(frame).text((8, 8), f'{name}  asset {i:03d}  source {row["source_frame"]}  t={row["episode_time_tick"]*row["time_base"][0]/row["time_base"][1]:.6f}s', fill="white")
        frame.save(images / f"frame_{i:03d}.png")
    ticks = np.diff([x["source_time_tick"] for x in rows])
    if len(set(ticks.tolist())) != 1:
        raise ValueError("MP4_uniform_native_time_only; HTML frame review remains available")
    tick = int(ticks[0])
    tb = rows[0]["time_base"]
    rate = f'{tb[1]}/{tb[0]*tick}'
    video = folder / "comparison.mp4"
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-framerate", rate,
                    "-i", str(images / "frame_%03d.png"), "-frames:v", str(frames), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "-video_track_timescale", str(tb[1]),
                    "-vsync", "0", str(video)], check=True)
    packet = json.loads(subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_packets",
                                                 "-show_entries", "packet=pts,duration", "-of", "json", str(video)]))
    # H.264 B-frames make packet order differ from display order; PTS is the
    # display clock and must be checked in ascending order.
    pts = sorted(int(p["pts"]) for p in packet["packets"])
    if len(pts) != frames or max(abs(a-b) for a, b in zip(pts, range(0, frames*tick, tick))) > 1:
        raise ValueError("encoded_frame_count_or_time_mismatch")
    frame_links = " ".join(f'<a href="frames/frame_{i:03d}.png">{i}</a>' for i in range(frames))
    title = html.escape(name)
    (folder / "index.html").write_text(f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>{title}</title>
<style>body{{font:16px sans-serif;background:#181818;color:#eee;margin:2em}}a{{color:#9dd}}video{{max-width:100%}}</style>
<h1>{title}</h1><p>{html.escape(manifest["character_family"])} + {html.escape(manifest["motion_source_id"])} 原 {frames} 个连续时刻；source {rows[0]["source_frame"]}–{rows[-1]["source_frame"]}，发布标称 tick {tick}/{tb[1]} 秒。正面/侧面固定相机；人工动态观感待评审。</p>
<video controls preload="metadata" src="comparison.mp4"></video><p><a href="comparison.mp4" download="{title}_comparison.mp4">下载本页视频 MP4</a></p><p>逐帧：{frame_links}</p>
<p><a href="../delivery_manifest.json">asset manifest</a> · <a href="../validation.json">全段数值检查</a> · <a href="../a1_adapter_check.json">A1只读检查</a></p></html>''')
    info = {"status": "complete_pending_user_visual_review", "frames": frames, "source_frames": [rows[0]["source_frame"], rows[-1]["source_frame"]],
            "time_base": tb, "ticks_per_frame": tick, "rate": rate, "mp4_packet_count": len(pts), "max_pts_error_ticks": max(abs(a-b) for a, b in zip(pts, range(0, frames*tick, tick))),
            "views": ["front", "side"], "samples": 64, "cpu_package_seconds": time.monotonic()-began,
            "source_group": manifest["source_group"], "integration_only": True}
    (folder / "delivery.json").write_text(json.dumps(info, indent=2) + "\n")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-root", type=Path, default=ROOT / "outputs/experiments/asset_composition_pipeline_a2_20261010")
    ap.add_argument("--name")
    ap.add_argument("--folder", default="preview_v3")
    args = ap.parse_args()
    out = args.output_root.resolve()
    if args.name:
        print(json.dumps(package(out, args.name, args.folder)))
        return
    for name in NAMES:
        if not (out / "builds" / name / "preview_v3/index.html").is_file():
            raise FileNotFoundError("package each case before root index:" + name)
    tiles = []
    for name, description in ((NAMES[0], "F02 原衣物 + 原短波波头"), (NAMES[1], "换 M02 简单衣型"), (NAMES[2], "再换 M02 卷发")):
        tiles.append(f'<section><h2>{html.escape(name)}：{description}</h2><video controls preload="metadata" src="builds/{name}/preview_v3/comparison.mp4"></video><p><a href="builds/{name}/preview_v3/index.html">逐帧与来源</a></p></section>')
    page = out / "index.html"
    if page.exists():
        raise FileExistsError(page)
    page.write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>A2 资产组合预览</title><style>body{font:16px sans-serif;background:#181818;color:#eee;margin:2em}a{color:#9dd}video{width:100%;max-width:1024px}section{margin:3em 0}</style><h1>F02 + participant_2：逐项换件</h1><p>三段均为同一源动作前60帧，固定视角、64 samples、原BVH标称时间；120帧资产数值已检查。人眼动态观感待用户评审，未进入公共场景。</p>' + ''.join(tiles) + '</html>')
    print(page)


if __name__ == "__main__":
    main()
