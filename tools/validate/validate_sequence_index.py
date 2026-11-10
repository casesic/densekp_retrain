#!/usr/bin/env python3
"""Validate sequence sidecars, or run the six-sample static A1 consumer acceptance."""
import argparse
import json
from pathlib import Path
import resource
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from densekp_retrain.data.sequence_index import SCHEMA, SequenceIndexError, row_context, validate_sequence_index
from densekp_retrain.io.json import write_json

COMPACT = ROOT / "outputs/acceptance/ds_output_modes_20261008/attempt_4/compact"
A1 = ROOT / "outputs/acceptance/high_res_boxes_a1_20261008/compact_run_1/consumer"


def inventory(root):
    return {str(p.relative_to(root)): [p.stat().st_size, p.stat().st_mtime_ns]
            for p in sorted(root.rglob("*")) if p.is_file()}


def static_a1_sidecar():
    """Reference existing boxes only. One static scene, no new crops or poses."""
    people = json.loads((COMPACT / "people.json").read_text())["people"]
    report = json.loads((COMPACT / "report.json").read_text())
    ep = dict(schema=SCHEMA, episode_id="a1_compact_static_reuse", sequence_id="a1_compact_single_snapshot",
        data_kind="synthetic", temporal_kind="static", time_base=[1, 1], time_semantics="static_anchor",
        time_origin="local zero for a single existing snapshot; not a capture timestamp or elapsed motion",
        timeline=dict(first_frame_index=0, frame_step=1, frame_count=1, expected_tick_step=None),
        synchronization=dict(status="synthetic_same_state", basis="shared rendered scene state",
                             evidence_ref=str(COMPACT / "people.json")),
        source_refs=[str(COMPACT / "people.json"), str(COMPACT / "source_recipe.json")],
        recipe_ref=str(COMPACT / "source_recipe.json"), seed=0,
        generator_version="legacy code_version retained in source_recipe.json; not rerun",
        cameras=[dict(camera_id=c, rig_ref=str(COMPACT / "rig.json"), calibration_ref=str(COMPACT / "rig.json"),
                      image_size_wh=report["final_erp_resolution"]) for c in ("cam2", "cam4")],
        assets=[dict(asset_id="legacy_compact_geometry_reference", asset_ref=str(COMPACT / "people.json"),
                     motion_source_ref=None, motion_source_unknown_reason=
                     "Retained people.json identifies geometry frame selectors, not an accepted continuous cycling asset")],
        people=[dict(person_id=p["id"], asset_id="legacy_compact_geometry_reference",
                     source_ref=str(COMPACT / "people.json"), character_family_id=None,
                     family_unknown_reason="compact people manifest does not certify character/source family",
                     identity_status="synthetic_instance") for p in people])
    table = dict(schema=SCHEMA, episode_id=ep["episode_id"], sequence_id=ep["sequence_id"],
                 packed=dict(path=str(A1 / "packed.npz"), row_count=24),
                 frames=[dict(frame_index=0, time_tick=0, status="available")], observations=[], boxes=[])
    boxes_by_camera = {}
    for camera in ("cam2", "cam4"):
        boxes_by_camera[camera] = json.loads((COMPACT / camera / "boxes.json").read_text())
        counts = next(r["in_view"] for r in report["cameras"] if r["camera"] == camera)
        for j, p in enumerate(people):
            given = next(b for b in boxes_by_camera[camera] if b["person_id"] == p["id"])
            state = "observed" if given["bbox_xyxy_exclusive"] is not None else (
                "empty" if counts[j] else "person_absent")
            table["observations"].append(dict(
                observation_id=f'f0_{camera}_p{p["id"]}', frame_index=0, time_tick=0,
                person_id=p["id"], camera_id=camera,
                source_ref=str(COMPACT / "people.json"), character_family_id=None,
                asset_id="legacy_compact_geometry_reference", asset_frame_index=p["frame"], asset_frame_status="known",
                status=state, reason="" if state == "observed" else (
                    "in_view GT exists but no visible instance box" if state == "empty" else "zero in_view GT and no instance box"),
                image=str(COMPACT / camera / "normal_erp.png"), annotations=str(COMPACT / camera / "annotations.npz"),
                annotation_person_index=j if state != "person_absent" else None,
                valid_mask=str(COMPACT / camera / "valid_mask.png"),
                person_mask=str(COMPACT / camera / f'person_{p["id"]}_mask.png') if state != "person_absent" else None,
                raster_manifest=str(COMPACT / "report.json"), image_size_wh=report["final_erp_resolution"]))
    original = json.loads((A1 / "index.json").read_text())["rows"]
    assert len(original) == 24
    parent_ids = set()
    for i, row in enumerate(original):
        parent_id = f'parent_{row["camera"]}_p{row["person_id"]}'
        oid = f'f0_{row["camera"]}_p{row["person_id"]}'
        if parent_id not in parent_ids:
            parent_ids.add(parent_id)
            camera_boxes = boxes_by_camera[row["camera"]]
            origin_row = next(j for j, b in enumerate(camera_boxes) if b["person_id"] == row["person_id"])
            table["boxes"].append(dict(box_id=parent_id, observation_id=oid, packed_row_id=None,
                bbox_xyxy=row["parent_bbox_xyxy"], bbox_kind=row["parent_bbox_kind"],
                bbox_source=camera_boxes[origin_row]["source"], crop_policy="historical_trial", parent_box_id=None,
                origin_ref=str(COMPACT / row["camera"] / "boxes.json"), origin_row=origin_row))
        table["boxes"].append(dict(box_id=f"a1_row_{i:03d}", observation_id=oid, packed_row_id=i,
            bbox_xyxy=row["given_bbox_xyxy"], bbox_kind=row["bbox_kind"], bbox_source=row["bbox_source"],
            crop_policy=row["crop_policy"], parent_box_id=parent_id,
            origin_ref=str(A1 / "index.json"), origin_row=i))
    return ep, table


def run_a1(output):
    # The six image decodes below are the entire actual Dataset item budget.
    import cv2
    from densekp_retrain.data.loading import load_box_dataset
    from densekp.core.keypoint_supervision import supervision_for_crop
    cv2.setNumThreads(1)
    started, cpu_started = time.monotonic(), time.process_time()
    before = {str(root): inventory(root) for root in (COMPACT, A1)}
    write_json(output / "source_inventory_before.json", before)
    episode, table = static_a1_sidecar()
    write_json(output / "episode.json", episode)
    write_json(output / "sequence_index.json", table)
    # Tables and path metadata first; only bounded image headers, no RGB decode.
    tables = validate_sequence_index(episode, table, base_dir=output)
    checked = validate_sequence_index(episode, table, base_dir=output,
                                      check_image_headers=True, check_packed=True)
    dataset = load_box_dataset(A1 / "packed.npz")
    legacy = load_box_dataset(COMPACT / "diagnostic_loader.npz")
    assert len(legacy) == 6  # No sequence sidecar argument or Dataset item read here.
    selected = [0, 1, 4, 8, 16, 20]
    checks = []
    for i in selected:
        context = row_context(episode, table, i)
        obs, box = context["observation"], context["box"]
        item = dataset[i]
        with np.load(obs["annotations"], allow_pickle=False) as ann:
            j = obs["annotation_person_index"]
            assert int(ann["person_ids"][j]) == obs["person_id"]
            np.testing.assert_array_equal(item["proj_verts"][:, :2], ann["xy"][j].astype(np.float32))
            np.testing.assert_array_equal(dataset.data["in_view"][i], ann["in_view"][j])
            np.testing.assert_array_equal(dataset.data["visible"][i], ann["visible"][j])
        assert item["proj_verts"].shape == (138, 3)
        np.testing.assert_array_equal(item["img_size"], [2880, 5760])
        assert Path(item["imgname"]).resolve() == Path(obs["image"]).resolve()
        source = json.loads((A1 / "index.json").read_text())["rows"][box["origin_row"]]
        with np.load(source["geometry"]) as geom:
            np.testing.assert_array_equal(item["_trans"], geom["affine"])
            np.testing.assert_array_equal(item["proj_verts_cropped"], geom["crop_xy"])
        supervised = supervision_for_crop(dataset.data["in_view"][i], item["proj_verts_cropped"][:, :2])
        np.testing.assert_array_equal(supervised, item["proj_verts_cropped"][:, 2] > 0)
        checks.append(dict(packed_row_id=i, episode_id=context["episode_id"], sequence_id=context["sequence_id"],
            frame_index=obs["frame_index"], time_tick=obs["time_tick"], time_base=context["time_base"],
            person_id=obs["person_id"], camera_id=obs["camera_id"], asset_id=obs["asset_id"],
            asset_frame_index=obs["asset_frame_index"], box_id=box["box_id"], parent_box_id=box["parent_box_id"],
            crop_policy=box["crop_policy"], image=obs["image"], points=138,
            supervised_points=int(supervised.sum()),
            supervised_occluded_points=int((supervised & ~dataset.data["visible"][i]).sum()),
            original_annotation_xy_exact=True, prior_a1_affine_and_crop_exact=True))
    after = {str(root): inventory(root) for root in (COMPACT, A1)}
    assert before == after
    write_json(output / "source_inventory_after.json", after)
    result = dict(status="passed", scope="sequence_data_interface_only", table_validation=tables,
        file_and_packed_validation=checked, actual_dataset_samples=len(checks), selected_rows=selected,
        checks=checks, newly_generated_boxes=0, newly_written_npz=0,
        actual_scene_times=1, continuous_motion_validated=False, exposure_sync_validated=False,
        dynamic_render_validated=False, training_effect_validated=False,
        legacy_dataset_constructed_without_sidecar=True, old_file_sizes_and_mtimes_unchanged=True,
        elapsed_seconds=time.monotonic()-started, cpu_seconds=time.process_time()-cpu_started,
        peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        output_bytes_before_report=sum(p.stat().st_size for p in output.rglob("*") if p.is_file()))
    write_json(output / "acceptance.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode", type=Path)
    group.add_argument("--a1-acceptance", action="store_true")
    parser.add_argument("--index", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory; never overwrite previous output")
    parser.add_argument("--check-image-headers", action="store_true")
    parser.add_argument("--check-packed", action="store_true")
    args = parser.parse_args()
    if args.episode and not args.index:
        parser.error("--episode requires --index")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.a1_acceptance:
        run_a1(output)
    else:
        try:
            result = validate_sequence_index(json.loads(args.episode.read_text()), json.loads(args.index.read_text()),
                base_dir=args.index.resolve().parent, check_image_headers=args.check_image_headers,
                check_packed=args.check_packed)
        except (SequenceIndexError, ValueError, OSError) as error:
            write_json(output / "validation.json", dict(status="rejected",
                code=getattr(error, "code", "input_read_error"), detail=str(error)))
            raise SystemExit(1) from error
        write_json(output / "validation.json", result)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
