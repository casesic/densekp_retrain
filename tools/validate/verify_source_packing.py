"""CPU-only independent source/packing/box acceptance; writes fresh output only."""

import argparse
import copy
import importlib
import json
import random
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import torch
from PIL import Image

DATA = ROOT / "outputs/experiments/source_isolated_pilot_20260930"
TRIAL = ROOT / "outputs/experiments/source_isolated_trial_400_gpu1_20261001"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=True) + "\n")


def outcome(function, *args, **kwargs):
    try:
        return {"value": function(*args, **kwargs), "error": None}
    except Exception as error:
        return {"value": None, "error": {"type": type(error).__name__, "message": str(error)}}


def normalize(value, root):
    if isinstance(value, str):
        return value.replace(str(root), "<RUN>")
    if isinstance(value, list):
        return [normalize(v, root) for v in value]
    if isinstance(value, dict):
        return {k: normalize(v, root) for k, v in value.items()}
    return value


def source_boundaries(root, validate):
    """Every rendered person is audited, even if absent from index target rows."""
    folder = root / "source_boundaries"
    folder.mkdir()
    audit = folder / "audit.json"
    write(audit, {"conservative_possible_source_groups": {"x": ["F01"], "y": ["F02"]}})
    characters = [
        dict(
            character_id=n,
            identity_family_id="cycling_" + n,
            partition=s,
            gender=g,
            face_source_group=n,
            shape_source_group=n,
            all_texture_members=[n + "_texture"],
            ancestry=[{"asset_id": n + "_asset"}],
        )
        for n, s, g in [
            ("F01", "train", "female"),
            ("M01", "train", "male"),
            ("F02", "val", "female"),
            ("M02", "val", "male"),
        ]
    ]
    split = dict(
        status="frozen_before_new_combinations",
        source_audit=str(audit),
        characters=characters,
        excluded_families=[],
        source_groups={
            s: dict(cycling=s + "photo", pedestrian=s + "clip", pedestrian_person=s + "actor")
            for s in ("train", "val")
        },
    )
    scenes = {}
    for side, target, occluder in [("train", "F01", "M01"), ("val", "F02", "M02")]:
        scene = folder / side
        scene.mkdir()
        scenes[side] = scene
        write(
            scene / "people.json",
            {
                "people": [
                    dict(character_id=target, action="cycling", source_group=side + "photo"),
                    dict(character_id=occluder, action="walking", source_group=side + "clip"),
                ]
            },
        )
    indices = {
        s: dict(
            split=s,
            frozen_utc="fixed",
            rows=[
                dict(
                    quality_state="candidate_accepted",
                    split=s,
                    scene=str(scenes[s]),
                    image=str(scenes[s] / "image.png"),
                    camera="cam2",
                    character_id="F01" if s == "train" else "F02",
                    pose_source_id=s + "pose",
                    given_bbox_xyxy=[0, 0, 10, 20],
                    affine_flat=[1, 0, 0, 0, 1, 0],
                    target_object_to_world_flat=[1],
                    camera_to_world_flat=[1],
                )
            ],
        )
        for s in scenes
    }
    results = {}

    def check(name, split_value=None, index_value=None, **kw):
        results[name] = outcome(
            validate,
            split if split_value is None else split_value,
            indices if index_value is None else index_value,
            **kw,
        )

    check("valid_with_unindexed_occluder")
    check("empty_indices", index_value={})
    for field in (
        "identity_family_id",
        "shape_source_group",
        "face_source_group",
        "all_texture_members",
        "ancestry",
    ):
        bad = copy.deepcopy(split)
        bad["characters"][2][field] = copy.deepcopy(bad["characters"][0][field])
        check(field + "_leak", bad)
    for field in ("cycling", "pedestrian", "pedestrian_person"):
        bad = copy.deepcopy(split)
        bad["source_groups"]["val"][field] = bad["source_groups"]["train"][field]
        check(field + "_leak", bad)
    for status in ("unreviewed", "proposal_awaiting_scope_decision"):
        bad = dict(split, status=status)
        check(status, bad)
    check(
        "allowed_proposal",
        dict(split, status="proposal_awaiting_scope_decision"),
        allow_proposal=True,
    )
    bad = copy.deepcopy(split)
    bad.pop("source_audit")
    check("missing_audit", bad)
    bad = copy.deepcopy(split)
    bad["characters"][0].pop("shape_source_group")
    check("missing_source_field", bad)
    bad = copy.deepcopy(split)
    bad["excluded_families"] = [{"identity_family_id": "cycling_F01"}]
    check("excluded_family", bad)
    bad = copy.deepcopy(split)
    bad["characters"][1]["gender"] = "female"
    check("missing_gender", bad)
    write(audit, {"conservative_possible_source_groups": {"bridge": ["F01", "F02"]}})
    check("possible_facial_source_leak")
    write(audit, {"conservative_possible_source_groups": {"x": ["F01"], "y": ["F02"]}})
    for name, mutate in [
        ("unfrozen_index", lambda x: x["train"].pop("frozen_utc")),
        ("index_side", lambda x: x["train"].update(split="val")),
        ("unaccepted_row", lambda x: x["train"]["rows"][0].update(quality_state="pending")),
        ("wrong_row_side", lambda x: x["train"]["rows"][0].update(split="val")),
        ("missing_image", lambda x: x["train"]["rows"][0].pop("image")),
        ("same_image", lambda x: x["val"]["rows"][0].update(image=x["train"]["rows"][0]["image"])),
        ("same_scene", lambda x: x["val"]["rows"][0].update(scene=x["train"]["rows"][0]["scene"])),
    ]:
        bad = copy.deepcopy(indices)
        mutate(bad)
        check(name, index_value=bad)
    check("unknown_partition", index_value={"test": indices["train"]})
    people = json.loads((scenes["train"] / "people.json").read_text())
    for name, change in [
        ("occluder_identity", dict(character_id="M02")),
        ("occluder_unknown", dict(character_id="unknown")),
        ("occluder_pose", dict(source_group="valclip")),
        ("target_pose", dict(source_group="valphoto")),
    ]:
        bad = copy.deepcopy(people)
        bad["people"][0 if name == "target_pose" else 1].update(change)
        write(scenes["train"] / "people.json", bad)
        check(name)
    write(scenes["train"] / "people.json", people)
    paired = copy.deepcopy(indices)
    row = paired["train"]["rows"][0]
    row["occlusion_pair_id"] = "pair"
    paired["train"]["rows"].append(copy.deepcopy(row))
    check("valid_pair", index_value=paired)
    for field in (
        "character_id",
        "pose_source_id",
        "given_bbox_xyxy",
        "affine_flat",
        "target_object_to_world_flat",
        "camera_to_world_flat",
    ):
        bad = copy.deepcopy(paired)
        bad["train"]["rows"][1][field] = [2] if isinstance(row[field], list) else "other"
        check("pair_" + field, index_value=bad)
    return normalize(results, root)


def contract_cases(root):
    from densekp.core.keypoint_contract import validate_annotations
    from densekp.core.keypoint_supervision import make_targets, points_in_mask

    arrays = {}
    errors = {}
    points = np.column_stack((np.linspace(-1, 16, 138), np.linspace(0, 16, 138))).astype(np.float32)
    mask = np.ones((16, 16), bool)
    mask[:, 8:] = False
    arrays["explicit_mask"] = points_in_mask(points, mask)
    arrays["new_targets"] = make_targets(points, arrays["explicit_mask"], [8, 8], 0.08)
    data = dict(
        imgname=np.array(["image.png"]),
        center=np.array([[8, 8]], np.float32),
        scale=np.array([0.08], np.float32),
        proj_verts=np.column_stack((points, np.linspace(0, 1, 138))).astype(np.float32)[None],
    )
    errors["legacy_weights"] = outcome(validate_annotations, data, "supervision_weight")
    for field in ("imgname", "center", "scale", "proj_verts"):
        bad = copy.deepcopy(data)
        bad.pop(field)
        errors["missing_" + field] = outcome(validate_annotations, bad, "supervision_weight")
    errors["undeclared"] = outcome(validate_annotations, data, None)
    empty = copy.deepcopy(data)
    empty["proj_verts"][..., 2] = 0
    errors["empty_supervision"] = outcome(validate_annotations, empty, "supervision_weight")
    empty["proj_verts"][..., :2] = np.nan
    errors["undefined_zero_weight_xy"] = outcome(validate_annotations, empty, "supervision_weight")
    new = copy.deepcopy(data)
    new.update(
        supervision_policy=np.array("in_view_full_crop"),
        in_view=np.ones((1, 138), bool),
        visible=np.zeros((1, 138), bool),
    )
    errors["occluded_reliable"] = outcome(validate_annotations, new, "supervision_weight")
    for field in ("in_view", "visible"):
        bad = copy.deepcopy(new)
        bad.pop(field)
        errors["missing_" + field] = outcome(validate_annotations, bad, "supervision_weight")
    bad = copy.deepcopy(new)
    bad["visible"][:] = True
    bad["in_view"][:] = False
    errors["visible_outside_view"] = outcome(validate_annotations, bad, "supervision_weight")
    for name, value in [("negative", -1), ("nan", np.nan), ("too_large", 2)]:
        bad = copy.deepcopy(data)
        bad["proj_verts"][0, 0, 2] = value
        errors["weight_" + name] = outcome(validate_annotations, bad, "supervision_weight")
    np.savez(root / "contract_arrays.npz", **arrays)
    return errors


def pure_boxes(root, module, official):
    arrays = {}
    metadata = {}
    cases = [
        {},
        dict(dx=0.15),
        dict(dy=-0.2),
        dict(factor=1 / 1.2),
        dict(factor=0),
        dict(factor=np.nan),
        *[dict(edge=i, fraction=0.35) for i in range(4)],
        dict(edge=4, fraction=0.2),
    ]
    for b, box in enumerate(
        ([100, 100, 200, 300], [-40, 0, 1700, 900], [0, 0, 1, 1], [0, 0, np.nan, 4])
    ):
        for i, kw in enumerate(cases):
            key = f"change_{b}_{i}"
            result = outcome(module.change_box, box, **kw)
            if result["error"]:
                metadata[key] = result["error"]
            else:
                arrays[key] = result["value"]
    joints = np.column_stack((np.arange(22) * 2, np.arange(22) * 3, np.ones(22)))
    for name, j in [
        ("valid", joints),
        ("too_few", joints[:7]),
        ("degenerate", np.ones((22, 3))),
        ("nonfinite", joints * np.nan),
        ("bounds", joints - 10),
    ]:
        value = official(j, 64, 48)
        metadata["official_" + name] = value is None
        if value is not None:
            for i, v in enumerate(value):
                arrays[f"official_{name}_{i}"] = np.asarray(v)
    np.savez(root / "boxes.npz", **arrays)
    return metadata


class TracedGenerator:
    def __init__(self, generator, trace):
        self.generator = generator
        self.trace = trace

    def __getattr__(self, name):
        value = getattr(self.generator, name)
        if not callable(value):
            return value

        def call(*args, **kwargs):
            before = copy.deepcopy(self.generator.bit_generator.state)
            result = value(*args, **kwargs)

            def plain(v):
                if isinstance(v, np.ndarray):
                    return v.tolist()
                if isinstance(v, np.generic):
                    return v.item()
                return v

            self.trace.append(
                dict(
                    method=name,
                    args=[plain(a) for a in args],
                    kwargs=kwargs,
                    result=plain(result),
                    before=before,
                    after=copy.deepcopy(self.generator.bit_generator.state),
                )
            )
            return result

        return call


def run(root):
    source = importlib.import_module("tools.validate.validate_source_isolated_split")
    pack = importlib.import_module("tools.annotation.build_source_isolated_trial")
    boxes = importlib.import_module("tools.experiments.prepare_box_robustness")
    bedlam = importlib.import_module("tools.annotation.generate_bedlam_densekp")
    write(root / "source_boundaries.json", source_boundaries(root, source.validate_split))
    write(root / "contract.json", contract_cases(root))
    write(root / "boxes.json", pure_boxes(root, boxes, bedlam.official_bbox))
    stage = root / "parent_pack"
    stage.mkdir()
    for name in ("split_v2.json", "train_index.json", "val_index.json"):
        shutil.copyfile(DATA / name, stage / name)
    random.seed(2402)
    np.random.seed(2402)
    torch.manual_seed(2402)
    with patch.object(sys, "argv", ["build_source_isolated_trial.py", str(stage)]):
        pack.main()
    for side in ("train", "val"):
        with (
            np.load(stage / "loader" / f"{side}.npz") as actual,
            np.load(DATA / "loader" / f"{side}.npz") as old,
        ):
            assert actual.files == old.files
            for key in actual.files:
                assert actual[key].dtype == old[key].dtype
                np.testing.assert_array_equal(actual[key], old[key])
    trace = []
    factory = np.random.default_rng

    def traced(seed=None):
        return TracedGenerator(factory(seed), trace)

    with (
        patch.object(
            sys,
            "argv",
            [
                "prepare_box_robustness.py",
                "--data-root",
                str(DATA),
                "--parent-trial",
                str(TRIAL),
                "--output",
                str(root / "derivatives"),
            ],
        ),
        patch.object(np.random, "default_rng", traced),
    ):
        boxes.main()
    write(root / "rng_calls.json", trace)
    py = random.getstate()
    ns = np.random.get_state()
    write(root / "rng_state.json", {"python": py, "numpy_head": [ns[0], ns[2], ns[3], ns[4]]})
    np.savez(root / "rng_state.npz", numpy=ns[1], torch=torch.get_rng_state().numpy())
    write(
        root / "result.json",
        {
            "status": "captured",
            "source_cases": len(json.loads((root / "source_boundaries.json").read_text())),
            "rng_calls": len(trace),
            "model_inference": False,
            "gpu": False,
        },
    )


def compare(before, after):
    old_files = {p.relative_to(before) for p in before.rglob("*") if p.is_file()}
    new_files = {p.relative_to(after) for p in after.rglob("*") if p.is_file()}
    assert old_files == new_files, (old_files - new_files, new_files - old_files)
    counts = dict(json=0, arrays=0, images=0, other=0)
    allowed = {
        "derivatives/config.json": ["frozen_utc"],
        "derivatives/train_index.json": ["frozen_utc"],
        "derivatives/val_index.json": ["frozen_utc"],
        "derivatives/acceptance.json": ["seconds"],
    }
    for prefix in ("fallback", "fallback_failure"):
        for filename, fields in {
            "config.json": ["frozen_utc"],
            "train_index.json": ["frozen_utc"],
            "val_index.json": ["frozen_utc"],
            "acceptance.json": ["seconds"],
        }.items():
            if Path(prefix, filename) in old_files:
                allowed[prefix + "/" + filename] = fields
    for name in sorted(old_files):
        old, new = before / name, after / name
        if name.suffix == ".json":
            left = normalize(json.loads(old.read_text()), before)
            right = normalize(json.loads(new.read_text()), after)
            fields = allowed.get(str(name), [])
            if name.parts[0] in ("fallback", "fallback_failure"):
                fields = {
                    "config.json": ["frozen_utc"],
                    "train_index.json": ["frozen_utc"],
                    "val_index.json": ["frozen_utc"],
                    "acceptance.json": ["seconds"],
                }.get(name.name, [])
            for field in fields:
                left.pop(field)
                right.pop(field)
            assert json.dumps(left) == json.dumps(right), str(name)
            counts["json"] += 1
        elif name.suffix == ".npy":
            left, right = np.load(old, allow_pickle=False), np.load(new, allow_pickle=False)
            assert left.dtype == right.dtype and left.shape == right.shape, str(name)
            np.testing.assert_array_equal(left, right, err_msg=str(name))
            counts["arrays"] += 1
        elif name.suffix == ".npz":
            with (
                np.load(old, allow_pickle=False) as left,
                np.load(new, allow_pickle=False) as right,
            ):
                assert left.files == right.files, str(name)
                for key in left.files:
                    assert (
                        left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
                    ), (str(name), key)
                    np.testing.assert_array_equal(
                        left[key], right[key], err_msg=str(name) + "/" + key
                    )
                    counts["arrays"] += 1
        elif name.suffix.lower() in (".png", ".jpg"):
            np.testing.assert_array_equal(
                np.asarray(Image.open(old)), np.asarray(Image.open(new)), err_msg=str(name)
            )
            counts["images"] += 1
        else:
            assert old.read_bytes() == new.read_bytes(), str(name)
            counts["other"] += 1
    return dict(
        status="passed",
        counts=counts,
        allowed_changes=allowed,
        path_change="only this acceptance run root prefix",
        comparison="exact arrays including dtype/shape; JSON order retained in files; decoded image pixels",
        gpu=False,
        model_inference=False,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--before", type=Path)
    a = p.parse_args()
    if not a.before:
        from tools.validate.validate_source_isolated_split import validate_split
        if validate_split.__module__ != "tools.validate.validate_source_isolated_split":
            raise RuntimeError("Capture requires the independent pre-migration implementations")
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    run(out)
    if a.before:
        result = compare(a.before.resolve(), out)
        write(out.parent / (out.name + "_comparison.json"), result)
        print(result)


if __name__ == "__main__":
    main()
