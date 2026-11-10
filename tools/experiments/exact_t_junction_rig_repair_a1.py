"""Run at most three already frozen exact support candidates per view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from tools.experiments.prepare_urban_three_multirig_a1 import preflight

CONFIG = ROOT / "configs/t_junction_rig_repair_a1_20261011.json"
OUTPUT = ROOT / "outputs/experiments/t_junction_rig_repair_a1_20261011"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def run(output):
    start = time.monotonic()
    config = json.loads(CONFIG.read_text())
    frozen = json.loads((output / "frozen_exact_order.json").read_text())
    clear = {(r["view"], r["attempt"]): r["passed"] for r in
             json.loads((output / "static_clearance.json").read_text())["checks"]}
    if len(frozen["candidates"]) > 9:
        raise ValueError("exact_candidate_limit")
    selected = {}
    checks = []
    for view in config["views"]:
        candidates = [r for r in frozen["candidates"] if r["view"] == view]
        if len(candidates) > 3:
            raise ValueError("exact_candidate_limit_per_view")
        for candidate in candidates:
            name = f"exact_{view}_{candidate['attempt']}"
            row = dict(view=view, attempt=candidate["attempt"], static_clearance=clear[(view, candidate["attempt"])])
            if not row["static_clearance"]:
                row["status"] = "static_obstacle_rejected"
                checks.append(row)
                continue
            exact_config = dict(schema="t_junction_multirig_camera_v1",
                source_prepared=str((ROOT / config["source_prepared"]).resolve()),
                scene_episode_id=config["scene_episode_id"], view_goal=view, rig_id=f"rig_{view}",
                coverage_goal=f"{view}_junction_view_control",
                minimum_common_points=8, minimum_common_physical_frames=6,
                minimum_view_common_run=6, minimum_view_common_frames=6,
                minimum_dual_edge_frames=0, minimum_dual_edge_run=0,
                require_road_cross_section=False,
                minimum_camera_to_person_root_m=config["minimum_camera_to_person_root_m"],
                minimum_camera_road_x_abs_m=None,
                maximum_union_roi_pixels=config["maximum_union_roi_pixels"],
                maximum_exact_support_candidates=1)
            write(output / f"{name}_config.json", exact_config)
            write(output / f"{name}_candidate.json", dict(schema=exact_config["schema"],
                source_prepared=exact_config["source_prepared"], scene_episode_id=config["scene_episode_id"],
                candidates=[{key: candidate[key] for key in
                    ("attempt", "camera_child_seed", "origin_m", "yaw_deg")}]))
            t = time.monotonic()
            result = preflight(output / f"{name}_config.json", output / f"{name}_candidate.json",
                               output / f"prepared_{name}", time_limit_seconds=180)
            row.update(status="passed" if result is not None else "exact_rejected",
                       elapsed_seconds=time.monotonic()-t,
                       geometry_preflight_ref=str(output / f"prepared_{name}" / "geometry_preflight.json"))
            checks.append(row)
            print(view, candidate["attempt"], row["status"], row["elapsed_seconds"], flush=True)
            if result is not None:
                # The source/person arrays must be bitwise equal to the frozen scene.
                for array in ("world_densekp.npy", "scene_matrices.npy"):
                    if not np.array_equal(np.load(output / f"prepared_{name}" / array),
                                          np.load(ROOT / config["source_prepared"] / array)):
                        raise ValueError("frozen_world_scene_changed")
                selected[view] = candidate
                break
        if view not in selected:
            break
    joint = len(selected) == 3 and any(
        json.loads((output / f"light_{v}.json").read_text())["attempts"][c["attempt"]]
        ["edge_and_view_scale_passed"] for v, c in selected.items())
    write(output / "exact_summary.json", dict(elapsed_seconds=time.monotonic()-start,
        attempts=checks, selected=selected, joint_passed=joint,
        reason=None if joint else "no_joint_exact_supported_three_rig_set_with_edge"))
    return joint


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, default=OUTPUT)
    a = p.parse_args()
    if not run(a.output):
        raise SystemExit(2)
