"""Validate one frozen three-rig junction delivery without model inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "assets_external/python_deps")]
from densekp_retrain.data.viewpoint import classify_view, view_angle
from tools.experiments.validate_dispersed_cyclists_a1 import validate as validate_slots

RIGS = ("rig_back", "rig_front", "rig_side")
LIMITS = {"back": 180, "front": 280, "side": 180}


def read(path):
    return json.loads(Path(path).read_text())


def longest(flags):
    best = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def validate(root):
    root = Path(root).resolve()
    parent = root.parent
    frozen = read(parent / "frozen_execution.json")
    selection = read(parent / "rig_selection.json")
    config = read(ROOT / "configs/t_junction_local_rig_delivery_a1_20261011.json")
    prior_search = ROOT / config["prior_search"]
    if selection["status"] != "ready_ordinary_no_edge_quota":
        raise ValueError("unexpected_edge_policy")
    source = Path(frozen["source_prepared"])
    base = validate_slots(root, expected_rigs=list(RIGS))
    scene = read(root / "recipe.json")
    if scene["scene_episode_id"] != frozen["scene_episode_id"]:
        raise ValueError("junction_scene_id_changed")
    for array in ("world_densekp.npy", "scene_matrices.npy"):
        if not np.array_equal(np.load(source / array), np.load(root / array if array == "world_densekp.npy"
                                                               else root / "rigs" / RIGS[0] / array)):
            raise ValueError("frozen_scene_array_changed:" + array)
    matrices = np.load(source / "scene_matrices.npy")
    asset = read(scene["people"][0]["asset_manifest"])
    with np.load(asset["parameters"]) as params:
        orient = params["global_orient"][scene["people"][0]["asset_frames"]]
    forward = [matrices[k, 0, :3, :3] @ cv2.Rodrigues(orient[k])[0] @ [0., 0., 1.]
               for k in range(31)]
    rig_results = {}
    for rig_id in RIGS:
        view = rig_id.removeprefix("rig_")
        folder = root / "rigs" / rig_id
        rig = read(folder / "rig.json")
        poses = {x["id"]: np.asarray(x["camera_to_world"]) for x in rig["cameras"]}
        index = read(folder / "pack" / "sequence_index.json")
        legal_boxes = {x["observation_id"] for x in index["boxes"]}
        view_flags, edge_flags, actual_dual_inputs = [], [], []
        native_heights = {"cam2": [], "cam4": []}
        frames = []
        for k in range(31):
            per = {}
            physical = {}
            boxes = {}
            for camera in ("cam2", "cam4"):
                with np.load(folder / f"frame_{k:03d}" / camera / "annotations.npz",
                             allow_pickle=False) as annotation:
                    physical[camera] = annotation["in_view"][0].copy()
                angle = view_angle(matrices[k, 0, :3, 3], forward[k], poses[camera][:3, 3])
                rows = read(folder / f"frame_{k:03d}" / camera / "boxes.json")
                box = rows[0]["bbox_xyxy_exclusive"]
                if box is not None:
                    native_heights[camera].append(float(box[3] - box[1]))
                boxes[camera] = f"f{k}_{camera}_p1" in legal_boxes
                per[camera] = dict(angle_deg=angle, view_class=classify_view(angle),
                                   physical_in=int(physical[camera].sum()),
                                   physical_out=int((~physical[camera]).sum()),
                                   native_box_height_px=float(box[3] - box[1]) if box else None,
                                   legal_model_input=boxes[camera])
            common = int((physical["cam2"] & physical["cam4"]).sum())
            view_flag = common >= 8 and all(per[c]["view_class"] == view for c in per)
            edge_flag = common >= 8 and all(per[c]["physical_in"] >= 8 and
                                              per[c]["physical_out"] >= 5 for c in per)
            view_flags.append(view_flag)
            edge_flags.append(edge_flag)
            actual_dual_inputs.append(view_flag and all(boxes.values()))
            frames.append(dict(frame_index=k, cameras=per, common_physical_points=common,
                               matching_view_and_common=view_flag, dual_edge=edge_flag))
        if longest(view_flags) < 6:
            raise ValueError("actual_view_common_window_failed:" + rig_id)
        selected = selection["selected"][view]
        proxy_height = selected.get("proxy_height_px")
        if selected["key"].startswith(f"prior_{view}_"):
            proxy_height = read(prior_search / f"light_{view}.json")["attempts"][selected["attempt"]][
                "smaller_camera_R138_height_median_px"]
        if selected["view_window_frames"] < 6:
            raise ValueError("selected_exact_view_window_conflict")
        if selected["maximum_roi_pixels"] > 7_000_000:
            raise ValueError("selected_roi_budget_conflict")
        rig_results[rig_id] = dict(
            view_goal=view, view_common_frames=sum(view_flags),
            longest_view_common_run=longest(view_flags),
            actual_dual_model_input_view_frames=sum(actual_dual_inputs),
            longest_actual_dual_model_input_view_run=longest(actual_dual_inputs),
            target_dual_edge_frames=sum(edge_flags),
            longest_target_dual_edge_run=longest(edge_flags),
            smaller_camera_proxy_R138_median_px=proxy_height,
            native_box_height_px={camera: dict(
                count=len(values), median=float(np.median(values)) if values else None,
                minimum=min(values) if values else None, maximum=max(values) if values else None)
                for camera, values in native_heights.items()},
            observed_slots=base["rigs"][rig_id]["packed_rows"],
            unavailable_slots=base["rigs"][rig_id]["unavailable_slots"],
            neighbor_event=base["rigs"][rig_id]["target_neighbor_three_stage_event"],
            neighbor_evidence_available=base["rigs"][rig_id]["target_neighbor_evidence_available"],
            video_ref=base["rigs"][rig_id]["video"],
            frames=frames)
    decision = dict(schema="junction_local_delivery_acceptance_v1",
        status="technical_passed" if base["status"] == "technical_passed" else base["status"],
        coverage_goal="junction_turn_and_three_views_ordinary",
        edge_requirement="prefer_edge_then_ordinary_v1",
        edge_quota_passed=any(r["longest_target_dual_edge_run"] >= 6
                              for r in rig_results.values()),
        edge_quota_claimed=False,
        scene_episode_id=frozen["scene_episode_id"],
        source_grouping="three_motions_all_rigs_one_shared_source_diagnostic",
        camera_calls=base["actual_completed_camera_calls"],
        planned_slots=base["planned_observation_slots"],
        packed_rows=base["packed_rows"],
        Dataset_reads=base["actual_Dataset_reads"],
        all_three_view_windows_passed=all(r["longest_view_common_run"] >= 6
                                          for r in rig_results.values()),
        all_people_content_passed=base["all_people_content_passed"],
        each_person_longest_legal_visible_run=base["each_person_longest_legal_visible_run"],
        relative_pair_max_abs=base["rig_pair_relative_transform_max_abs"],
        rigs=rig_results)
    (root / "junction_acceptance.json").write_text(json.dumps(decision, indent=2, allow_nan=False) + "\n")
    return decision


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    print(validate(args.root)["status"])
