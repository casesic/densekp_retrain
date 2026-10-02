"""Small source/packing contract inputs; expected values live in frozen fixtures."""

import copy
import json

import numpy as np


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
