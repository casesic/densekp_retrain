#!/usr/bin/env python3
"""Independent old-authority replay and new-component differentiation checks."""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/experiments/asset_composition_pipeline_a2_20261010"


def read(path):
    return json.loads(Path(path).read_text())


def delta(a, b):
    x, y = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
    if x.shape != y.shape:
        raise ValueError("shape_mismatch:" + str((a, b, x.shape, y.shape)))
    return float(np.max(np.abs(x - y)))


def main():
    old = {
        "baseline_m02_p1": ROOT / "outputs/experiments/bvh_p1_dressed_assets_a2_20261010/p1_comfortable/delivery_manifest.json",
        "baseline_f02_bikefit": ROOT / "outputs/experiments/mtb_f02_dressed_asset_a2_20261009/delivery_manifest.json",
    }
    baselines = {}
    for name, path in old.items():
        parent = read(path)
        new = read(OUT / "builds" / name / "delivery_manifest.json")
        if name == "baseline_m02_p1":
            authority = {"body": parent["body_vertices"], "cloth": parent["cloth_vertices"],
                         "head": parent["head_transforms"], "R138": parent["body_R138"], "faces": parent["model_faces"]}
        else:
            authority = {"body": str(path.parent / "full/vertices.npy"), "cloth": str(path.parent / "full/clothing_vertices.npy"),
                         "head": str(path.parent / "full/head_transforms.npy"), "R138": str(path.parent / "full/densekp_3d.npy"),
                         "faces": str(path.parent / "faces.npy")}
        actual = {"body": new["body_vertices"], "cloth": new["cloth_vertices"], "head": new["head_transforms"],
                  "R138": new["body_R138"], "faces": new["body_faces"]}
        differences = {k: delta(actual[k], authority[k]) for k in authority}
        assert differences["body"] < 2e-6 and differences["cloth"] < 2e-6 and differences["head"] < 2e-6 and differences["R138"] < 2e-6
        assert differences["faces"] == 0
        source_rows, rows = read(parent["timeline"]), read(new["timeline"])
        for x, y in zip(source_rows, rows):
            assert all(x[k] == y[k] for k in ("source_frame", "source_time_tick", "time_base"))
        oldp = np.load(parent["parameters"], allow_pickle=False)
        newp = np.load(new["parameters"], allow_pickle=False)
        assert all(np.array_equal(oldp[k], newp[k]) for k in ("global_orient", "body_pose", "jaw_pose", "leye_pose", "reye_pose", "left_hand_pose", "right_hand_pose", "expression", "betas", "transl"))
        baselines[name] = {"max_differences": differences, "source_time_exact": True, "pose_parameters_exact": True,
                           "old_human_acceptance_inherited_only_after_replay": True}
    names = ("A_f02_p2", "B_f02_p2_m02_cloth", "C_f02_p2_m02_cloth_hair")
    m = [read(OUT / "builds" / n / "delivery_manifest.json") for n in names]
    assert len({x["body_vertices"] for x in m}) == 1
    assert m[0]["cloth_vertices"] != m[1]["cloth_vertices"] and m[1]["cloth_vertices"] == m[2]["cloth_vertices"]
    assert m[0]["groom_reference"] == m[1]["groom_reference"] and m[1]["groom_reference"] != m[2]["groom_reference"]
    assert m[0]["head_transforms"] == m[1]["head_transforms"] == m[2]["head_transforms"]
    assert all(x["frame_count"] == 120 and x["public_scene_status"] == "not_consumed" for x in m)
    cf_a = np.load(m[0]["cloth_faces"])
    cf_b = np.load(m[1]["cloth_faces"])
    assert len(cf_a) != len(cf_b)
    pts = np.load(Path(m[2]["groom_reference"]) / "points.npy", mmap_mode="r")
    active = np.load(Path(m[2]["groom_reference"]) / "active_curve_mask.npy")
    assert np.isfinite(pts).all() and active.any()
    result = {"status": "passed", "baselines": baselines,
              "new_combinations": {"frames_each": 120, "shared_body_cache": m[0]["body_vertices"],
                                   "A_cloth_faces": len(cf_a), "B_C_cloth_faces": len(cf_b),
                                   "A_B_hair_same": True, "C_hair_transferred": True,
                                   "C_hair_active_curves": int(active.sum()), "human_visual_state": "pending_user",
                                   "public_scene_state": "not_consumed"},
              "no_dataset_or_scene_generated": True}
    dest = OUT / "cross_validation.json"
    if dest.exists():
        raise FileExistsError(dest)
    dest.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
