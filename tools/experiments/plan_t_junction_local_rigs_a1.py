"""Freeze and evaluate bounded local camera translations plus rig yaw."""
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
from densekp_retrain.camera.rig import rig_poses
from tools.experiments.plan_t_junction_rigs_a1 import headings
from tools.experiments.repair_t_junction_rigs_a1 import light_candidate

CONFIG = ROOT / "configs/t_junction_local_rig_delivery_a1_20261011.json"
OUT = ROOT / "outputs/experiments/t_junction_local_rig_delivery_a1_20261011"


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def root_clearance(origin, yaw, matrices, baseline):
    poses, _, _ = rig_poses(origin, yaw, baseline)
    distances = []
    for name, pose in poses.items():
        d = np.linalg.norm(matrices[:, :, :2, 3] - pose[:2, 3], axis=2)
        k, j = np.unravel_index(np.argmin(d), d.shape)
        distances.append(dict(camera=name, distance_m=float(d[k, j]), frame_index=int(k),
                              person_id=int(j+1), center_xy=pose[:2, 3].tolist(),
                              person_root_xy=matrices[k, j, :2, 3].tolist()))
    return min(distances, key=lambda d: d["distance_m"]), distances


def freeze(config, matrices, out):
    old = Path(ROOT / config["prior_search"])
    rng = random.Random(f"t_junction_local_v1|{config['camera_master_seed']}")
    proposals = []
    anchors = []
    for item in config["local_anchors"]:
        view = item["view"]
        attempt = item["attempt"]
        source = json.loads((old / f"light_{view}.json").read_text())["attempts"][attempt]
        assert source["attempt"] == attempt
        bad, all_cameras = root_clearance(source["origin_m"], source["yaw_deg"], matrices, .18)
        away = np.subtract(bad["center_xy"], bad["person_root_xy"])
        away = away / np.linalg.norm(away)
        lateral = np.array([-away[1], away[0]])
        shifts = [(f, s) for f in config["shift_forward_m"] for s in config["shift_lateral_m"]
                  if math.hypot(f, s) <= config["maximum_shift_radius_m"]]
        rng.shuffle(shifts)
        shifts.sort(key=lambda v: math.hypot(*v))
        shifts = shifts[:config["positions_per_anchor"]]
        anchors.append(dict(view=view, prior_attempt=attempt, origin_m=source["origin_m"],
                            yaw_deg=source["yaw_deg"], nearest_root=bad,
                            away_unit_xy=away.tolist(), shifts=shifts))
        for pos_index, (forward_shift, lateral_shift) in enumerate(shifts):
            xy = np.asarray(source["origin_m"][:2]) + forward_shift*away + lateral_shift*lateral
            origin = [float(xy[0]), float(xy[1]), 0.]
            for offset in config["yaw_offsets_deg"]:
                candidate = dict(proposal_id=len(proposals), view=view, anchor_attempt=attempt,
                    position_index=pos_index, forward_shift_m=forward_shift,
                    lateral_shift_m=lateral_shift, origin_m=origin,
                    yaw_deg=float((source["yaw_deg"]+offset) % 360),
                    yaw_offset_deg=offset, camera_child_seed=rng.getrandbits(63))
                proposals.append(candidate)
    keys = [(round(r["origin_m"][0], 8), round(r["origin_m"][1], 8), round(r["yaw_deg"], 8))
            for r in proposals]
    if len(proposals) > config["maximum_local_proposals"] or len(set(keys)) != len(keys):
        raise ValueError("local_proposal_budget_or_duplicate")
    write(out / "frozen_local_proposals.json", dict(schema=config["schema"],
          camera_master_seed=config["camera_master_seed"], source_prepared=str(ROOT / config["source_prepared"]),
          prior_search=str(old), anchors=anchors, proposals=proposals,
          order="smaller_shift_radius_first; seeded_tie_order; config_yaw_order",
          scope="each proposal includes cheap rejection; at most 512 unique origins/yaws"))
    return proposals


def run(config_path, out):
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    if config["schema"] != "t_junction_local_rig_delivery_v1":
        raise ValueError("invalid_local_schema")
    source = ROOT / config["source_prepared"]
    recipe = json.loads((source / "recipe.json").read_text())
    if recipe["episode_id"] != config["scene_episode_id"] or len(recipe["people"]) != 3:
        raise ValueError("frozen_scene_mismatch")
    matrices = np.load(source / "scene_matrices.npy")
    world = np.load(source / "world_densekp.npy")
    assert matrices.shape == (31, 3, 4, 4) and world.shape == (31, 3, 138, 3)
    forward = headings(source)
    masks = {c: np.asarray(Image.open(source / "calibration" / f"{c}_valid_mask.png")) > 0
             for c in ("cam2", "cam4")}
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    write(out / "frozen_config.json", config)
    proposals = freeze(config, matrices, out)
    rows = []
    projected = 0
    for proposal in proposals:
        bad, cameras = root_clearance(proposal["origin_m"], proposal["yaw_deg"], matrices,
                                      recipe["rig"]["baseline_m"])
        item = dict(**proposal, minimum_root_distance_m=bad["distance_m"],
                    nearest_root=bad, per_camera_nearest_root=cameras)
        if bad["distance_m"] < config["minimum_camera_to_person_root_m"]:
            item.update(status="cheap_root_rejected", rejection="camera_near_person_root")
        else:
            candidate = dict(attempt=proposal["proposal_id"], origin_m=proposal["origin_m"],
                             yaw_deg=proposal["yaw_deg"], camera_child_seed=proposal["camera_child_seed"])
            screen = light_candidate(candidate, proposal["view"], world, matrices, forward, masks,
                                     recipe["rig"]["baseline_m"], config)
            projected += 1
            item.update(status="projected", view_window_frames=screen["view_window_frames"],
                        view_count=screen["view_count"], edge_window_frames=screen["edge_window_frames"],
                        edge_count=screen["edge_count"], common_count=screen["common_count"],
                        smaller_camera_R138_height_median_px=screen["smaller_camera_R138_height_median_px"],
                        view_and_scale_passed=screen["view_and_scale_passed"],
                        edge_and_view_scale_passed=screen["edge_and_view_scale_passed"],
                        frames=screen["frames"])
        rows.append(item)
    write(out / "local_checks.json", dict(schema=config["schema"], proposal_count=len(rows),
          cheap_root_rejected=sum(r["status"] == "cheap_root_rejected" for r in rows),
          fully_projected=projected, elapsed_seconds=time.monotonic()-t0, attempts=rows))
    print("local", len(rows), "cheap_rejected", len(rows)-projected, "projected", projected,
          "edge_eligible", sum(r.get("edge_and_view_scale_passed", False) for r in rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--output", type=Path, default=OUT)
    args = p.parse_args()
    run(args.config, args.output)
