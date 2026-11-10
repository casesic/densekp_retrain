"""Bounded, replayable position-and-physical-domain search for a fixed junction scene."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.data.viewpoint import classify_view, view_angle
from tools.experiments.plan_t_junction_rigs_a1 import headings, longest

CONFIG = ROOT / "configs/t_junction_rig_repair_a1_20261011.json"
OUTPUT = ROOT / "outputs/experiments/t_junction_rig_repair_a1_20261011"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def source_data(config):
    src = ROOT / config["source_prepared"]
    recipe = json.loads((src / "recipe.json").read_text())
    world = np.load(src / "world_densekp.npy")
    matrices = np.load(src / "scene_matrices.npy")
    assert recipe["episode_id"] == config["scene_episode_id"]
    assert world.shape == (31, 3, 138, 3)
    assert matrices.shape == (31, 3, 4, 4)
    return src, recipe, world, matrices


def anchor_positions(config, view, matrices, forward):
    """Cheap position/angle stage; no 31-frame physical projection here."""
    rng = random.Random(f"junction_repair_positions_v1|{config['camera_master_seed']}|{view}")
    positions = []
    for x in config["position_grid_x_m"]:
        for y in config["position_grid_y_m"]:
            seed = rng.getrandbits(63)
            local = random.Random(seed)
            origin = [x + local.uniform(-config["position_jitter_m"], config["position_jitter_m"]),
                      y + local.uniform(-config["position_jitter_m"], config["position_jitter_m"]), 0.]
            center = np.array(origin) + [0., 0., 1.15]
            roots = matrices[:, :, :3, 3]
            nearest = float(np.min(np.linalg.norm(roots[:, :, :2] - center[:2], axis=2)))
            target_dist = np.linalg.norm(roots[:, 0, :2] - center[:2], axis=1)
            angles = [view_angle(roots[k, 0], forward[k], center) for k in range(31)]
            flags = [classify_view(a) == view for a in angles]
            run = longest(flags)
            # Reject close-to-person positions before the 512 full projections.
            eligible = nearest >= config["minimum_camera_to_person_root_m"] and run >= 6
            positions.append(dict(cell_xy_m=[x, y], origin_m=origin, position_child_seed=seed,
                                  nearest_person_root_m=nearest,
                                  target_median_distance_m=float(np.median(target_dist)),
                                  matching_view_run=run, eligible=eligible))
    # Balance long view windows, image scale, and spatial coverage. The
    # original shuffled seed resolves equal-ranked cells reproducibly.
    eligible = [p for p in positions if p["eligible"]]
    rng.shuffle(eligible)
    eligible.sort(key=lambda p: (-min(p["matching_view_run"], 14),
                                 abs(p["target_median_distance_m"] - (3.4 if view == "front" else 4.3))))
    chosen = []
    for p in eligible:
        if all(np.linalg.norm(np.subtract(p["cell_xy_m"], q["cell_xy_m"])) >= 2 for q in chosen):
            chosen.append(p)
        if len(chosen) == config["positions_per_view"]:
            break
    return positions, chosen


def light_candidate(candidate, view, world, matrices, forward, masks, baseline, config):
    poses, _, _ = rig_poses(candidate["origin_m"], candidate["yaw_deg"], baseline)
    minimum_root_distance = min(float(np.linalg.norm(
        matrices[:, :, :2, 3] - pose[:2, 3], axis=2).min()) for pose in poses.values())
    placement_passed = minimum_root_distance >= config["minimum_camera_to_person_root_m"]
    frames = []
    for k in range(31):
        per = {}
        inside = {}
        for camera, pose in poses.items():
            xy, _, _, _, valid = point_in_view(world[k, 0], pose, camera, masks[camera])
            inside[camera] = valid
            angle = view_angle(matrices[k, 0, :3, 3], forward[k], pose[:3, 3])
            per[camera] = dict(angle_deg=float(angle), view_class=classify_view(angle),
                               in_domain=int(valid.sum()), out_domain=int((~valid).sum()),
                               R138_projected_height_px=float(np.ptp(xy[valid, 1])) if valid.sum() >= 8 else 0.)
        common = int((inside["cam2"] & inside["cam4"]).sum())
        both = common >= 8 and all(per[c]["view_class"] == view for c in per)
        edge = common >= 8 and all(per[c]["in_domain"] >= 8 and per[c]["out_domain"] >= 5 for c in per)
        # Useful when no coarse yaw lands exactly in the narrow overlap.
        near_edge = min(min(per[c]["in_domain"], per[c]["out_domain"]) for c in per)
        frames.append(dict(frame_index=k, cameras=per, common_physical_points=common,
                           matching_view_and_common=both, dual_edge=edge, near_edge_points=near_edge))
    view_flags = [r["matching_view_and_common"] for r in frames]
    edge_flags = [r["dual_edge"] for r in frames]
    matching = [r for r in frames if r["matching_view_and_common"]]
    height = min(float(np.median([r["cameras"][c]["R138_projected_height_px"] for r in matching]))
                 for c in ("cam2", "cam4")) if matching else 0.
    view_run = longest(view_flags)
    edge_run = longest(edge_flags)
    view_scale = (placement_passed and view_run >= 6 and
                  height >= config["minimum_target_proxy_height_px"][view])
    # Keep both the view/scale pool and the edge-bearing subset.
    return dict(**candidate, view_window_frames=view_run, view_count=sum(view_flags),
                edge_window_frames=edge_run, edge_count=sum(edge_flags),
                common_count=sum(r["common_physical_points"] >= 8 for r in frames),
                near_edge_score=max(r["near_edge_points"] for r in frames),
                minimum_camera_to_person_root_m=minimum_root_distance,
                placement_passed=placement_passed,
                smaller_camera_R138_height_median_px=height,
                view_and_scale_passed=view_scale,
                edge_and_view_scale_passed=view_scale and edge_run >= 6,
                frames=frames)


def run(config_path, output):
    started = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    if config["schema"] != "t_junction_rig_repair_v1":
        raise ValueError("invalid_search_schema")
    if (config["positions_per_view"] * (360 // config["coarse_yaw_step_deg"] +
                                        len(config["local_yaw_offsets_deg"])) > config["maximum_light_candidates_per_view"]):
        raise ValueError("candidate_budget_exceeded")
    src, recipe, world, matrices = source_data(config)
    forward = headings(src)
    masks = {c: np.asarray(Image.open(src / "calibration" / f"{c}_valid_mask.png")) > 0
             for c in ("cam2", "cam4")}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write(output / "frozen_search.json", dict(config=config, source_prepared=str(src.resolve()),
          scene_episode_id=recipe["episode_id"], parent_search=str(ROOT / config["parent_search"]),
          algorithm="position_grid_then_full_yaw_15deg_then_local_yaw_offsets_v1",
          limit_per_view=512, exact_limit_per_view=3, selection="first joint exact-pass triplet with any rig edge_run>=6"))
    summary = {}
    for view in config["views"]:
        anchors, chosen = anchor_positions(config, view, matrices, forward)
        write(output / f"anchor_{view}.json", dict(calls=len(anchors), elapsed_seconds=time.monotonic()-started,
              grid=anchors, chosen=chosen, position_rule="target body-forward angle, all-person root distance, median target range"))
        rows = []
        seen = set()
        for pos_index, position in enumerate(chosen):
            start = len(rows)
            for step in range(0, 360, config["coarse_yaw_step_deg"]):
                yaw = float(step)
                key = (pos_index, yaw)
                if key in seen:
                    continue
                seen.add(key)
                candidate = dict(attempt=len(rows), stage="coarse", position_index=pos_index,
                    camera_child_seed=position["position_child_seed"], origin_m=position["origin_m"],
                    yaw_deg=yaw)
                rows.append(light_candidate(candidate, view, world, matrices, forward, masks,
                                            recipe["rig"]["baseline_m"], config))
            coarse = rows[start:]
            best = max(coarse, key=lambda r: (r["edge_window_frames"], r["edge_count"],
                        r["near_edge_score"], r["view_window_frames"], r["common_count"]))
            for offset in config["local_yaw_offsets_deg"]:
                yaw = (best["yaw_deg"] + offset) % 360
                key = (pos_index, yaw)
                if key in seen:
                    continue
                seen.add(key)
                candidate = dict(attempt=len(rows), stage="local", position_index=pos_index,
                    camera_child_seed=position["position_child_seed"], origin_m=position["origin_m"],
                    yaw_deg=yaw, coarse_anchor_attempt=best["attempt"])
                rows.append(light_candidate(candidate, view, world, matrices, forward, masks,
                                            recipe["rig"]["baseline_m"], config))
        if len(rows) > config["maximum_light_candidates_per_view"]:
            raise ValueError("light_candidate_limit_exceeded")
        passing = [r for r in rows if r["view_and_scale_passed"]]
        edge = [r for r in passing if r["edge_and_view_scale_passed"]]
        rank = lambda r: (-r["edge_window_frames"], -r["view_window_frames"],
                          -r["smaller_camera_R138_height_median_px"], r["attempt"])
        passing.sort(key=rank)
        edge.sort(key=rank)
        summary[view] = dict(anchor_calls=len(anchors), anchor_eligible=sum(r["eligible"] for r in anchors),
                             chosen_positions=len(chosen), light_candidates=len(rows),
                             view_scale_count=len(passing), edge_eligible_count=len(edge),
                             top_view_attempts=[r["attempt"] for r in passing[:10]],
                             top_edge_attempts=[r["attempt"] for r in edge[:10]],
                             maximum_view_run=max((r["view_window_frames"] for r in rows), default=0),
                             maximum_edge_run=max((r["edge_window_frames"] for r in rows), default=0),
                             maximum_height_px=max((r["smaller_camera_R138_height_median_px"] for r in rows), default=0))
        write(output / f"light_{view}.json", dict(view=view, candidate_count=len(rows),
              view_scale_pool=[r["attempt"] for r in passing],
              edge_capable_pool=[r["attempt"] for r in edge], attempts=rows))
        print(view, summary[view], flush=True)
    write(output / "light_summary.json", dict(schema=config["schema"], elapsed_seconds=time.monotonic()-started,
          views=summary, joint_light_feasible=all(summary[v]["view_scale_count"] for v in config["views"])
          and any(summary[v]["edge_eligible_count"] for v in config["views"])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    run(args.config, args.output)
