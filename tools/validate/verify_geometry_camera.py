"""Freeze/replay selected CPU SMPL-X, clothing and camera arrays. Never render."""

import argparse
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import torch
from tests.geometry_camera_fixture import api, synthetic, Capture
from densekp_retrain.body.geometry import load_smplx_model, compute_densekp_3d


def compare(before, after):
    old = {p.relative_to(before) for p in before.rglob("*") if p.is_file()}
    new = {p.relative_to(after) for p in after.rglob("*") if p.is_file()}
    assert old == new, (old - new, new - old)
    arrays = jsons = 0
    for name in sorted(old):
        left, right = before / name, after / name
        if name.suffix == ".json":
            l = left.read_text().replace(str(before), "<RUN>")
            r = right.read_text().replace(str(after), "<RUN>")
            assert json.dumps(json.loads(l)) == json.dumps(json.loads(r)), str(name)
            jsons += 1
        elif name.suffix == ".npz":
            with np.load(left, allow_pickle=False) as x, np.load(right, allow_pickle=False) as y:
                assert x.files == y.files, str(name)
                for key in x.files:
                    assert x[key].dtype == y[key].dtype and x[key].shape == y[key].shape, (
                        str(name),
                        key,
                    )
                    np.testing.assert_array_equal(x[key], y[key], err_msg=str(name) + "/" + key)
                    arrays += 1
        elif name.suffix == ".npy":
            x, y = np.load(left, allow_pickle=False), np.load(right, allow_pickle=False)
            assert x.dtype == y.dtype and x.shape == y.shape, str(name)
            np.testing.assert_array_equal(x, y, err_msg=str(name))
            arrays += 1
        else:
            assert left.read_bytes() == right.read_bytes(), str(name)
    return dict(
        status="passed",
        arrays=arrays,
        json=jsons,
        comparison="exact values, shape, dtype, keys/order; only run-output path prefix normalized",
        gpu=False,
        blender=False,
    )


def run(out, original):
    f = api(original)
    if original and not f["evaluate"].__module__.startswith("tools."):
        raise RuntimeError("Capture requires independent pre-migration implementation")
    torch.set_num_threads(2)
    torch.manual_seed(3102)
    np.random.seed(3102)
    random.seed(3102)
    synthetic(f).save(out / "synthetic")
    source = ROOT / "outputs/experiments/cycling_four_characters_20260929"
    regressor = np.load(ROOT / "assets/densekp_regressor_R.npy")
    labels = np.load(ROOT / "outputs/body_texture_audit/body_uv.npz")["component_labels"]
    records = []
    for character in ("cycling_F01_v1", "cycling_M01_v1"):
        manifest = json.loads((source / "characters" / character / "manifest.json").read_text())
        model = load_smplx_model(
            Path(manifest["model_dir"]), gender=manifest["gender"], num_betas=10, batch_size=1
        ).eval()
        for pose in ("coco_292226_person_1330527", "coco_306800_person_225160"):
            geometry = source / "derivatives" / character / pose / "geometry"
            with np.load(geometry / "motion_params.npz") as z:
                params = {k: z[k].copy() for k in f["PARAMETERS"]}
            c = Capture()
            c.call("params", lambda: params)
            body, joints = c.call("evaluate", f["evaluate"], model, params)
            c.call("faces", lambda: model.faces)
            dense = c.call("densekp", compute_densekp_3d, regressor, body)
            rest, rj, pelvis = c.call(
                "rest", f["rest_body"], model, np.asarray(params["betas"]).reshape(-1)
            )
            c.call("zero_pose", f["zero_pose"], params["betas"], pelvis)
            measure_rows = c.call(
                "measurement_rows",
                f["measurement_rows"],
                rest,
                rj,
                labels,
                model.lbs_weights[:, 15].numpy(),
            )
            if measure_rows is not None:
                c.call("measure", f["measure"], rest, measure_rows)
            selected, material = c.call("regions", f["outfit_regions"], rest, rj, model.faces)
            c.call("shell", f["shell"], body[0], model.faces, selected, 0.004)
            clipped = c.call("clipped_shell", f["clipped_shell"], rest, rj, body[0], model.faces)
            if clipped:
                arrays, _ = clipped
                oldcloth = geometry.parent / "clothing_v2"
                comparisons = {}
                for key, value in arrays.items():
                    old = np.load(oldcloth / (key + ".npy"))
                    comparisons[key] = dict(
                        shape=list(old.shape),
                        dtype=str(old.dtype),
                        max_abs=float(np.max(np.abs(value - old))),
                    )
                c.call("historical_clothing", lambda: comparisons)
            c.call("pose_rotations", f["pose_rotations"], model, params, rj)
            probes = rest[np.arange(0, len(rest), 211)] + np.array([0.001, 0.002, 0.003])
            c.call("closest", f["closest_surface"], probes, rest, model.faces)
            c.call("binding", f["bind_new_points"], probes, rest, model.faces)
            c.call(
                "clip_surface",
                f["clip_surface"],
                rest,
                model.faces,
                [(0, 1, 0.2), (0, -1, 0.2), (1, 1, 0.3), (1, -1, 0.5)],
            )
            c.call("triangle_frame", f["triangle_frame"], body[0], model.faces[:150])
            person = dict(yaw_deg=90, location=[1.8, 0.3, 0.0])
            c.call("world_body", f["world_points"], body[0], body[0], person)
            c.call("world_dense", f["world_points"], dense[0], body[0], person)
            if clipped:
                c.call(
                    "world_clothing",
                    f["world_points"],
                    clipped[0]["clothing_vertices"][0],
                    body[0],
                    person,
                )
            export = out / "exports" / character / pose
            c.call(
                "save_geometry",
                f["save_geometry"],
                export,
                model,
                params,
                dict(character_id=character, pose=pose),
                regressor,
            )
            maxbody = float(np.max(np.abs(body - np.load(geometry / "vertices.npy"))))
            maxdense = float(np.max(np.abs(dense - np.load(geometry / "densekp_3d.npy"))))
            assert maxbody <= 2e-5 and maxdense <= 2e-5
            np.testing.assert_array_equal(model.faces, np.load(geometry / "faces.npy"))
            c.call(
                "historical_body",
                lambda: dict(
                    max_body_error_m=maxbody, max_dense_error_m=maxdense, faces_exact=True
                ),
            )
            c.save(out / "combinations" / character / pose)
            records.append(
                dict(
                    character=character,
                    gender=manifest["gender"],
                    pose=pose,
                    geometry=str(geometry),
                    model_file=manifest["model_file"],
                    num_betas=10,
                    errors={k: v["error"] for k, v in c.records.items() if v["error"]},
                )
            )
    # Replay already-rendered arrays only, without calling process_scene/warp_case.
    scene = ROOT / "outputs/experiments/source_isolated_pilot_20260930/layouts/val_000"
    rig = json.loads((scene / "rig.json").read_text())
    c = Capture()
    for camera_record in rig["cameras"]:
        name = camera_record["id"]
        matrix = np.asarray(camera_record["camera_to_world"])
        with np.load(scene / name / "annotations.npz") as z:
            annotations = dict(z)
        with np.load(scene / name / "rendered_buffers.npz") as z:
            buffers = dict(z)
        points = annotations["world_densekp"].reshape(-1, 3)
        mask = buffers["valid_mask"]
        xy, distance = c.call(
            name + "_erp", f["project_erp"], points, matrix, mask.shape[1], mask.shape[0]
        )
        c.call(name + "_view", f["point_in_view"], points, matrix, name, mask)
        c.call(
            name + "_visibility",
            f["erp_visibility"],
            xy,
            distance,
            buffers["depth_range_m"],
            buffers["instance_id"],
            annotations["person_ids"].tolist(),
        )
        c.call(
            name + "_saved_labels",
            lambda: dict(
                xy=annotations["xy"], in_view=annotations["in_view"], visible=annotations["visible"]
            ),
        )
    c.save(out / "saved_scene")
    state = Capture()
    state.call("torch", lambda: torch.get_rng_state().numpy())
    state.call("numpy", np.random.get_state)
    state.call("python", random.getstate)
    state.save(out / "rng")
    (out / "selection.json").write_text(json.dumps(records, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--before", type=Path)
    a = p.parse_args()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run(out, a.before is None)
    if a.before:
        result = compare(a.before.resolve(), out)
        (out.parent / (out.name + "_comparison.json")).write_text(
            json.dumps(result, indent=2) + "\n"
        )
        print(result)


if __name__ == "__main__":
    main()
