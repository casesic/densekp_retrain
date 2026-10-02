"""Audit declared sources and every rendered person, including untargeted occluders.

This preserves the frozen-split validator, including its legacy field meanings,
error ordering and paired-camera keys. It does not infer new source identities.
"""

import json
from pathlib import Path


def _declared_sources(split, allow_proposal):
    """Check approval, identity/appearance ancestry and held-out pose sources."""
    allowed = {"frozen_before_new_combinations"}
    if allow_proposal:
        allowed.add("proposal_awaiting_scope_decision")
    if split["status"] not in allowed:
        raise ValueError("Split is not approved/frozen")
    audit_path = split.get("source_audit")
    if not audit_path:
        raise ValueError("Updated facial-source audit is required; old split is rejected")
    audit = json.loads(Path(audit_path).read_text())
    characters = {character["character_id"]: character for character in split["characters"]}
    families = {
        character["identity_family_id"]: character["partition"] for character in characters.values()
    }
    exclusions = {character["identity_family_id"] for character in split["excluded_families"]}
    if exclusions & families.keys():
        raise ValueError("Excluded family is still included")
    for members in audit["conservative_possible_source_groups"].values():
        sides = {
            families["cycling_" + partition]
            for partition in members
            if "cycling_" + partition in families
        }
        if len(sides) > 1:
            raise ValueError("Possible facial source crosses partitions")
    memberships = {}
    for key in ["identity_family_id", "shape_source_group", "face_source_group"]:
        seen = {}
        for character in characters.values():
            previous_partition = seen.setdefault(character[key], character["partition"])
            if previous_partition != character["partition"]:
                raise ValueError(f"{key} crosses partitions")
        memberships[key] = {
            partition: sorted(member for member, side in seen.items() if side == partition)
            for partition in ["train", "val"]
        }
    for key in ["all_texture_members", "ancestry"]:
        seen = {}
        for character in characters.values():
            for value in character.get(key, []):
                identity = value["asset_id"] if isinstance(value, dict) else value
                if seen.setdefault(identity, character["partition"]) != character["partition"]:
                    raise ValueError(f"{key} crosses partitions")
        memberships[key] = {
            partition: sorted(member for member, side in seen.items() if side == partition)
            for partition in ["train", "val"]
        }
    for key in ["cycling", "pedestrian", "pedestrian_person"]:
        if split["source_groups"]["train"][key] == split["source_groups"]["val"][key]:
            raise ValueError(f"{key} not held out")
        memberships[key] = {
            partition: [split["source_groups"][partition][key]] for partition in ["train", "val"]
        }
    character_counts = {
        partition: sum(character["partition"] == partition for character in characters.values())
        for partition in ["train", "val"]
    }
    for side in character_counts:
        if {
            character["gender"]
            for character in characters.values()
            if character["partition"] == side
        } != {
            "male",
            "female",
        }:
            raise ValueError("Each partition must include both genders")
    return characters, memberships, character_counts


def _index_memberships(split, indices, characters, character_counts, memberships):
    """Check images, all scene people and fixed paired-box signatures."""
    images = {}
    scenes = {}
    paired = {}
    rows_count = {partition: 0 for partition in character_counts}
    for side, index in (indices or {}).items():
        if side not in character_counts:
            raise ValueError("Unknown partition")
        if index["split"] != side or not index.get("frozen_utc"):
            raise ValueError("Unfrozen candidate index")
        for row in index["rows"]:
            if row["quality_state"] != "candidate_accepted" or row["split"] != side:
                raise ValueError("Unaccepted row")
            image = str(Path(row["image"]).resolve())
            scene = str(Path(row["scene"]).resolve())
            if images.setdefault(image, side) != side or scenes.setdefault(scene, side) != side:
                raise ValueError("Same image/dual-view layout crosses partitions")
            # Audit everyone rendered, including people with no target box.
            people = json.loads((Path(scene) / "people.json").read_text())["people"]
            for person in people:
                character = characters.get(person["character_id"])
                if character is None or character["partition"] != side:
                    raise ValueError("Target/occluder identity leakage")
                sources = split["source_groups"][side]
                expected = (
                    sources["cycling"] if person["action"] == "cycling" else sources["pedestrian"]
                )
                if person["source_group"] != expected:
                    raise ValueError("Target/occluder pose-source leakage")
            if row.get("occlusion_pair_id"):
                key = (row["occlusion_pair_id"], row["camera"])
                signature = (
                    side,
                    row["character_id"],
                    row["pose_source_id"],
                    row["camera"],
                    tuple(row["given_bbox_xyxy"]),
                    tuple(row["affine_flat"]),
                    tuple(row["target_object_to_world_flat"]),
                    tuple(row["camera_to_world_flat"]),
                )
                if paired.setdefault(key, signature) != signature:
                    raise ValueError("Paired target/camera/box/affine changed")
            rows_count[side] += 1
    memberships["image"] = {
        partition: sorted(member for member, side in images.items() if side == partition)
        for partition in character_counts
    }
    memberships["layout"] = {
        partition: sorted(member for member, side in scenes.items() if side == partition)
        for partition in character_counts
    }
    return images, scenes, rows_count


def validate_split(split, indices=None, allow_proposal=False):
    """Preserve frozen source fields and row order; reject known leakage."""
    characters, memberships, character_counts = _declared_sources(split, allow_proposal)
    images, scenes, rows_count = _index_memberships(
        split, indices, characters, character_counts, memberships
    )
    intersections = {
        key: sorted(set(values["train"]) & set(values["val"]))
        for key, values in memberships.items()
    }
    if any(intersections.values()):
        raise ValueError("Unexpected train/val intersection")
    return dict(
        status="passed_proposal_only" if allow_proposal else "passed",
        characters=character_counts,
        candidate_rows=rows_count,
        images=len(images),
        layouts=len(scenes),
        memberships=memberships,
        train_val_intersections=intersections,
        meaning="Declared source-risk groups and every rendered target/occluder checked; not proof of unknown photographed face identities",
    )
