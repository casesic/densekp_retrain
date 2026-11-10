"""Summarize the frozen junction run and link its existing ROI videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess

import numpy as np


RIGS = ("rig_back", "rig_front", "rig_side")


def read(path: Path):
    return json.loads(path.read_text())


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def stats(values):
    return dict(count=len(values), total_seconds=sum(values), mean_seconds=statistics.mean(values),
                median_seconds=statistics.median(values), range_seconds=[min(values), max(values)])


def video_check(video: Path):
    result = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                             "-show_entries", "frame=best_effort_timestamp_time", "-of", "json", str(video)],
                            check=True, capture_output=True, text=True)
    pts = [float(row["best_effort_timestamp_time"]) for row in json.loads(result.stdout)["frames"]]
    if len(pts) != 31 or any(abs(value - k / 10) > 1e-6 for k, value in enumerate(pts)):
        raise ValueError(f"video_PTS_or_frame_count_conflict:{video}")
    return dict(frames=len(pts), first_pts_seconds=pts[0], last_pts_seconds=pts[-1],
                final_display_hold_seconds=.1, file_bytes=video.stat().st_size)


def topdown(source: Path, selection: dict, output: Path):
    matrices = np.load(source / "scene_matrices.npy")
    def xy(x, y):
        return round(95 + (x + 13) * 30, 1), round(35 + (94 - y) * 30, 1)
    def rect(x0, y0, x1, y1, fill):
        x, y = xy(x0, y1)
        return f'<rect x="{x}" y="{y}" width="{(x1-x0)*30}" height="{(y1-y0)*30}" fill="{fill}"/>'
    items = [rect(-13, 72, 13, 94, "#deded8"), rect(-3.5, 72, 3.5, 80, "#777777"),
             rect(-13, 80, 13, 88, "#777777"), rect(-7.5, 72, -3.5, 80, "#cfc8b4"),
             rect(3.5, 72, 7.5, 80, "#cfc8b4"), rect(-13, 88, 13, 92, "#cfc8b4")]
    for person, color in enumerate(("#ef5350", "#2576c1", "#2e9a61")):
        points = [xy(*matrices[k, person, :2, 3]) for k in range(31)]
        items.append(f'<polyline points="{" ".join(f"{x},{y}" for x,y in points)}" '
                     f'fill="none" stroke="{color}" stroke-width="5"/>')
        for k in (0, 15, 30):
            x, y = points[k]
            items.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{color}"/>')
            items.append(f'<text x="{x+7}" y="{y-7}" font-size="12">p{person+1} f{k}</text>')
    for view, color in (("back", "#202060"), ("front", "#cc7e16"), ("side", "#78218b")):
        picked = selection["selected"][view]
        cameras = read(Path(picked["prepared_ref"]) / "rig.json")["cameras"]
        positions = [np.asarray(camera["camera_to_world"])[:2, 3] for camera in cameras]
        x, y = xy(*np.mean(positions, axis=0))
        rotation = np.asarray(cameras[0]["camera_to_world"])
        yaw = np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])) % 360
        items.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{color}" stroke="#fff" stroke-width="2"/>')
        items.append(f'<text x="{x+10}" y="{y+5}" font-size="13" fill="{color}">'
                     f'{view} rig · yaw {yaw:.0f}°</text>')
    items.append('<text x="20" y="25" font-size="17">城市正端 T 形路口 · 三人冻结轨迹与实渲染 rig</text>')
    items.append('<text x="20" y="680" font-size="12">平面示意；彩点为整组 rig 原点，双机相对基线 0.18 m 为合成设置。物理有效域以逐点投影为准。</text>')
    output.write_text('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 700">' +
                      "".join(items) + '</svg>\n')


def finalize(root: Path):
    root = root.resolve()
    task = root.parent
    accepted = read(root / "junction_acceptance.json")
    if accepted["status"] != "technical_passed" or accepted["camera_calls"] != 186:
        raise ValueError("junction_not_fully_accepted")
    selected = read(task / "rig_selection.json")
    frozen = read(task / "frozen_execution.json")
    calls = [json.loads(line) for line in (root / "render_calls.jsonl").read_text().splitlines()]
    camera = {}
    consumers = {}
    videos = {}
    for rig in RIGS:
        camera[rig] = {c: stats([float(row["seconds"]) for row in calls
                                 if row["event"] == "completed" and row["rig_id"] == rig
                                 and row["camera_id"] == c]) for c in ("cam2", "cam4")}
        consumers[rig] = stats([float(read(root / "rigs" / rig / f"frame_{k:03d}" /
                                            "consume_timing.json")["seconds"]) for k in range(31)])
        videos[rig] = video_check(root / "rigs" / rig / "pack" / "dual_roi.mp4")
    frames = [float(read(root / "rigs" / "rig_back" / f"frame_{k:03d}" /
                         "render_frame.json")["source_manifest"]["render_seconds"]) for k in range(31)]
    wall = read(root / "task_wall.json")
    costs = dict(schema="junction_local_delivery_cost_v1", assembly_seconds=read(root / "single_process_setup.json")["setup_callback_seconds"],
                 first_scene_frame_render_seconds=frames[0], later_scene_frame_render=stats(frames[1:]),
                 paired_camera_render_seconds=camera, synchronous_consumer_seconds=consumers,
                 one_blender_run_seconds=read(root / "multirig_render_result.json")["elapsed_seconds"],
                 pack_readback_video_seconds={rig: {key: read(root / "rigs" / rig / "pack" / "readback.json")[key]
                      for key in ("pack_seconds", "readback_seconds", "video_seconds", "elapsed_seconds")}
                      for rig in RIGS},
                 final_success_wall_seconds=wall["elapsed_seconds"], prior_failed_blender_seconds=[
                     read(task / failed / "failure.json")["elapsed_seconds"]
                     for failed in ("production", "production_retry1", "production_retry2")],
                 all_four_blender_starts_wall_seconds=wall["elapsed_seconds"] + sum(
                     read(task / failed / "failure.json")["elapsed_seconds"]
                     for failed in ("production", "production_retry1", "production_retry2")),
                 final_disk_bytes=wall["disk"]["disk_bytes"],
                 owned_temporary_bytes_remaining=wall["disk"]["temporary_bytes"],
                 blender_peak_rss_kib=read(root / "multirig_render_result.json")["peak_rss_kib"],
                 videos=videos)
    write(task / "delivery_cost.json", costs)
    topdown(Path(frozen["source_prepared"]), selected, task / "junction_topdown.svg")
    html = ['<!doctype html><html lang="zh"><meta charset="utf-8"><title>城市 T 形路口 · 三组双机</title>',
            '<style>body{font:16px system-ui;background:#171b21;color:#eee;margin:2rem}'
            'main{max-width:1260px;margin:auto}video{width:100%;background:#111}'
            'article{padding:1rem;margin:1rem 0;background:#242b34;border-radius:8px}'
            'a{color:#8dd0ff}.download{display:inline-block;padding:.4rem .7rem;background:#355b76;'
            'border-radius:5px;color:#fff;text-decoration:none}.download:hover{background:#467694}'
            'img{width:100%;max-width:900px}</style><main>',
            '<h1>城市 T 形路口 · 正／背／侧三组双机</h1>',
            '<p>同一段三人31时刻场景，每时刻更新一次人物后依次渲染三组。'
            '本批是转弯与多视角样例；双路边缘连续6时刻条件未通过，不计入边缘配额。'
            '灰色区域未渲染，不代表真实背景。每条视频31帧、10 Hz、PTS 0–3.0秒，末帧另展示0.1秒。</p>']
    for rig, title in (("rig_back", "背面"), ("rig_front", "正面"), ("rig_side", "侧面")):
        result = accepted["rigs"][rig]
        html.append(f'<article><h2>{title} · {rig}</h2><p>p1双机同类+共同点：'
                    f'{result["view_common_frames"]}/31时刻，最长{result["longest_view_common_run"]}时刻；'
                    f'双路边缘：{result["target_dual_edge_frames"]}/31时刻。</p>'
                    f'<video controls preload="metadata" src="production_retry3/rigs/{rig}/pack/dual_roi.mp4"></video>'
                    f'<p><a href="production_retry3/rigs/{rig}/pack/dual_roi.mp4">直接打开视频</a> · '
                    f'<a class="download" href="production_retry3/rigs/{rig}/pack/dual_roi.mp4" '
                    f'download="t_junction_{rig}.mp4">下载 MP4</a> · '
                    f'<a href="production_retry3/rigs/{rig}/pack/sequence_index.json">序列索引</a></p></article>')
    html.append('<article><h2>道路与轨迹</h2><img src="junction_topdown.svg" alt="三人轨迹和三组 rig 的俯视图">'
                '<p><a href="production_retry3/junction_acceptance.json">实际覆盖与观察统计</a> · '
                '<a href="delivery_cost.json">实测成本</a> · '
                '<a href="rig_selection.json">冻结机位与拒绝</a></p></article></main></html>')
    (task / "index.html").write_text("\n".join(html) + "\n")
    return costs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(finalize(args.root), indent=2))
