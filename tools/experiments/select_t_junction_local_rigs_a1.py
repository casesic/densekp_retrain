"""Apply frozen edge-first/ordinary-fallback exact support policy."""
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

OUT = ROOT / "outputs/experiments/t_junction_local_rig_delivery_a1_20261011"
CONFIG = ROOT / "configs/t_junction_local_rig_delivery_a1_20261011.json"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write("\n")


def exact_candidate(candidate, config, out):
    name = candidate["key"]
    source = str((ROOT / config["source_prepared"]).resolve())
    exact_config = dict(schema="t_junction_multirig_camera_v1",
        source_prepared=source, scene_episode_id=config["scene_episode_id"],
        view_goal=candidate["view"], rig_id=f"rig_{candidate['view']}",
        coverage_goal=f"{candidate['view']}_junction_view_control",
        minimum_common_points=8, minimum_common_physical_frames=6,
        minimum_view_common_run=6, minimum_view_common_frames=6,
        # Group-level edge decision is made only after exact support.
        minimum_dual_edge_frames=0, minimum_dual_edge_run=0,
        require_road_cross_section=False,
        minimum_camera_to_person_root_m=config["minimum_camera_to_person_root_m"],
        minimum_camera_road_x_abs_m=None,
        maximum_union_roi_pixels=config["maximum_union_roi_pixels"],
        maximum_exact_support_candidates=1)
    write(out / f"{name}_config.json", exact_config)
    write(out / f"{name}_frozen.json", dict(schema=exact_config["schema"],
          source_prepared=source, scene_episode_id=config["scene_episode_id"],
          candidates=[{k: candidate[k] for k in
            ("attempt", "origin_m", "yaw_deg", "camera_child_seed")}]))
    dest = out / f"prepared_{name}"
    t0 = time.monotonic()
    result = preflight(out / f"{name}_config.json", out / f"{name}_frozen.json",
                       dest, time_limit_seconds=180)
    row = dict(key=name, tier=candidate["tier"], view=candidate["view"],
               attempt=candidate["attempt"], elapsed_seconds=time.monotonic()-t0,
               status="passed" if result else "exact_support_rejected",
               prepared_ref=str(dest.resolve()),
               geometry_preflight_ref=str((dest / "geometry_preflight.json").resolve()))
    if result is not None:
        src = ROOT / config["source_prepared"]
        for array in ("scene_matrices.npy", "world_densekp.npy"):
            if not np.array_equal(np.load(dest / array), np.load(src / array)):
                raise ValueError("frozen_scene_person_geometry_changed:" + array)
        selected = json.loads((dest / "selected_projection.json").read_text())
        if (selected["coverage"]["longest_view_common_run"] < 6 or
            candidate["proxy_height_px"] < config["minimum_target_proxy_height_px"][candidate["view"]]):
            raise ValueError("exact_view_scale_inconsistent_with_light")
        rows = json.loads((dest / "planned_regions.json").read_text())
        row["maximum_roi_pixels"] = max((region["roi_xyxy"][2]-region["roi_xyxy"][0]) *
                                         (region["roi_xyxy"][3]-region["roi_xyxy"][1])
                                         for frame in rows for region in frame.values())
        row["edge_window_frames"] = selected["coverage"]["longest_dual_edge_run"]
        row["view_window_frames"] = selected["coverage"]["longest_view_common_run"]
    else:
        data = json.loads((dest / "candidate_checks.json").read_text())["attempts"][0]
        row["rejections"] = data["rejections"]
    return row


def run(out):
    t0 = time.monotonic()
    out = Path(out)
    config = json.loads(CONFIG.read_text())
    frozen = json.loads((out / "frozen_exact_candidates.json").read_text())
    clearance = {(r["view"], r["attempt"]): r["passed"] for r in
                 json.loads((out / "static_clearance.json").read_text())["checks"]}
    candidates = frozen["candidates"]
    if (len(candidates) > 12 or sum(r["tier"] == "edge" for r in candidates) > 3 or
        any(sum(r["tier"] == "ordinary" and r["view"] == view for r in candidates) > 3
            for view in ("back", "front", "side"))):
        raise ValueError("exact_candidate_budget")
    attempts = []
    chosen = {}
    # Edge is an explicit preference, never silently required for ordinary production.
    for candidate in (r for r in candidates if r["tier"] == "edge"):
        if not clearance[(candidate["view"], candidate["attempt"])]:
            attempts.append(dict(key=candidate["key"], status="native_static_clearance_rejected"))
            continue
        row = exact_candidate(candidate, config, out)
        attempts.append(row)
        print(row["key"], row["status"], row.get("rejections"), flush=True)
        if row["status"] == "passed" and row["edge_window_frames"] >= 6:
            chosen[candidate["view"]] = row
            break
    edge_view = next(iter(chosen), None)
    for view in ("back", "front", "side"):
        if view in chosen:
            continue
        for candidate in (r for r in candidates if r["tier"] == "ordinary" and r["view"] == view):
            if not clearance[(candidate["view"], candidate["attempt"])]:
                attempts.append(dict(key=candidate["key"], status="native_static_clearance_rejected"))
                continue
            row = exact_candidate(candidate, config, out)
            attempts.append(row)
            print(row["key"], row["status"], row.get("rejections"), flush=True)
            if row["status"] == "passed":
                chosen[view] = row
                break
    status = ("ready_preferred_edge" if len(chosen) == 3 and edge_view else
              "ready_ordinary_no_edge_quota" if len(chosen) == 3 else "rejected_missing_exact_rig")
    result = dict(schema="junction_local_rig_selection_v1", status=status,
                  edge_requirement=config["edge_requirement"], edge_selected_rig_id=
                  f"rig_{edge_view}" if edge_view else None,
                  scene_episode_id=config["scene_episode_id"], exact_attempts=attempts,
                  exact_attempt_count=len(attempts), selected=chosen,
                  elapsed_seconds=time.monotonic()-t0,
                  ordinary_note="If no edge exact-pass rig, junction/multiview sample only; no edge quota.")
    write(out / "rig_selection.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    value = run(args.output)
    if value["status"] == "rejected_missing_exact_rig":
        raise SystemExit(2)
