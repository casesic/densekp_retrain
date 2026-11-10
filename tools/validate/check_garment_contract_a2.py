#!/usr/bin/env python3
"""Bounded CPU readback of existing M03 garment views and A1 time/root helpers."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from densekp_retrain.assets.garment_contract import read_json, read_view_frame, validate_description
from densekp_retrain.data.dynamic_scene import SCHEMA, geometry_paths, person_matrices, select_timeline
from densekp_retrain.data.person_time import map_timeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", type=Path, required=True)
    args = ap.parse_args()
    began = time.monotonic()
    folder = args.folder.resolve()
    fixed = validate_description(read_json(folder / "M03_BEDLAM.garment.json"))
    simple = validate_description(read_json(folder / "M03_simple.garment.json"))
    views = {name: read_json(folder / "views" / (name + ".json")) for name in
             ("M03_BEDLAM_full", "M03_BEDLAM_15_106", "M03_simple_simple_15_106")}
    R = np.load(ROOT / "assets/densekp_regressor_R.npy")
    read_count = 0
    max_r = 0.0
    for name, view in views.items():
        geometry = view["geometry"]
        a1_asset = dict(schema=SCHEMA, **{key: geometry[key] for key in
            ("body_vertices", "body_faces", "cloth_vertices", "cloth_faces", "head_transforms")},
            densekp=geometry["densekp"])
        paths = geometry_paths(a1_asset)
        if any(not path.is_file() for path in paths.values()):
            raise FileNotFoundError("A1_geometry_paths_missing")
        for local, original in enumerate(view["source_asset_frame_indices"]):
            values = read_view_frame(view, local)
            if not all(np.isfinite(array).all() for array in values.values()):
                raise ValueError("nonfinite_garment_view")
            if values["head_transforms"].shape != (4, 4):
                raise ValueError("head_shape_conflict")
            predicted = R @ values["body_vertices"]
            max_r = max(max_r, float(np.max(np.abs(predicted - values["densekp"]))))
            for key in ("body_vertices", "cloth_vertices", "head_transforms", "densekp"):
                if not np.array_equal(values[key], np.load(geometry[key], mmap_mode="r")[original]):
                    raise ValueError("view_component_original_frame_mismatch:" + key)
            read_count += 1
        source = read_json(view["source_timeline_ref"])
        selected = select_timeline(source, view["selection"][0], view["frame_count"], len(source))
        if [r["asset_frame"] for r in selected] != view["source_asset_frame_indices"]:
            raise ValueError("A1_selected_asset_frame_conflict")
    if max_r > 3e-5:
        raise ValueError("R138_readback_conflict")
    sub = views["M03_BEDLAM_15_106"]
    source = read_json(sub["source_timeline_ref"])
    chosen = source[15:106]
    clock = [dict(frame_index=k, episode_time_tick=k, time_base=[1, 10]) for k in range(31)]
    mapped = map_timeline(chosen, clock, origin_record=0)
    if [x["asset_frame"] for x in mapped] != list(range(15, 106, 3)):
        raise ValueError("A1_10Hz_original_mapping_conflict")
    with np.load(sub["geometry"]["parameters"], allow_pickle=False) as z:
        relative = np.asarray(z["source_root_relative"])
    fixed_scene = np.eye(4)
    anchor = np.array([2.0, 0.85, 4.0])
    fixed_scene[:3, 3] = anchor - relative[15]
    matrices = person_matrices(dict(root_mode=sub["root_mode"], model_to_world=fixed_scene.tolist()), 91)
    algebra = []
    for local in (0, 45, 90):
        frame = 15 + local
        values = read_view_frame(sub, local)
        actual = (matrices[local] @ np.r_[relative[frame], 1.])[:3]
        expected = anchor + relative[frame] - relative[15]
        if not np.allclose(actual, expected, atol=2e-6, rtol=0):
            raise ValueError("source_root_removed_more_than_once")
        head_world = matrices[local] @ values["head_transforms"]
        if not np.allclose(head_world[:3, 3], values["head_transforms"][:3, 3] + fixed_scene[:3, 3], atol=2e-6, rtol=0):
            raise ValueError("scene_head_transform_conflict")
        algebra.append(dict(view_frame=local, original_asset_frame=frame, source_root_m=relative[frame].tolist(),
                            world_root_m=actual.tolist(), expected_world_root_m=expected.tolist()))
    result = dict(status="passed_readonly_asset_view", integration_only=True,
        view_frame_reads=read_count, R138_readback_max_m=max_r,
        full_frame_count=120, subwindow_frame_count=91,
        A1_10Hz_scene_records=len(mapped), A1_asset_frames=[x["asset_frame"] for x in mapped],
        A1_scene_consumer_tested=False, source_root_reference_asset_frame=0,
        root_algebra=algebra, seconds=time.monotonic()-began)
    target = folder / "view_validation.json"
    if target.exists():
        raise FileExistsError(target)
    target.write_text(json.dumps(result, indent=2) + "\n")
    (folder / "a1_person_time_31.json").write_text(json.dumps(mapped, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "frames_read": read_count,
                      "r138_max_m": max_r, "seconds": result["seconds"]}))


if __name__ == "__main__":
    main()
