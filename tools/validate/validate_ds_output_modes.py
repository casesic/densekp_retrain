#!/usr/bin/env python3
"""Bounded full/compact acceptance from one existing C-resolution DS source."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import numpy as np
from PIL import Image

from densekp_retrain.data.boxes import trial_crop_from_box
from densekp_retrain.data.packing import pack_index_rows
from densekp_retrain.imaging.ds import process_scene
from densekp_retrain.imaging.output_modes import (
    complete_output,
    prepare_source_workspace,
    preserve_source_recipe,
    record_failure,
)
from densekp_retrain.io.json import write_json
from tools.validate.validate_ds_scene_pair_pilot import validate


CAMERAS = ("cam2", "cam4")


def directory_bytes(path):
    return sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())


def copy_source(source, target):
    for path in source.iterdir():
        destination = target / path.name
        shutil.copytree(path, destination) if path.is_dir() else shutil.copy2(path, destination)
    manifest_path = target / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for row in manifest["sources"]:
        row["source"] = str((target / f"{row['id']}.exr").resolve())
    write_json(manifest_path, manifest)
    return manifest


def consumer_checks(directory):
    archive_path = directory / "diagnostic_loader.npz"
    with np.load(archive_path) as archive:
        rows = []
        for i in range(len(archive["person_id"])):
            camera = str(archive["camera_id"][i])
            person_id = int(archive["person_id"][i])
            rows.append(dict(
                loader_input=str(archive_path), loader_row=i,
                image=str((directory / camera / "normal_erp.png").resolve()),
            ))
        original = {key: archive[key].copy() for key in archive.files}
    packed = pack_index_rows(dict(rows=rows))
    for key in ("center", "scale", "proj_verts", "in_view", "visible", "person_id", "camera_id"):
        np.testing.assert_array_equal(packed[key], original[key])
    mask_boxes = 0
    for camera in CAMERAS:
        valid = np.asarray(Image.open(directory / camera / "valid_mask.png")) > 0
        for row in json.loads((directory / camera / "boxes.json").read_text()):
            if row["bbox_xyxy_exclusive"] is None:
                continue
            person = np.asarray(Image.open(
                directory / camera / f"person_{row['person_id']}_mask.png"
            )) > 0
            person &= valid
            yy, xx = np.where(person)
            actual = [int(xx.min()), int(yy.min()), int(xx.max()) + 1, int(yy.max()) + 1]
            if actual != row["bbox_xyxy_exclusive"]:
                raise ValueError("Mask-derived box differs from boxes.json")
            center, scale = trial_crop_from_box(np.asarray(actual, np.float32))
            if not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
                raise ValueError("Box augmentation crop primitive rejected retained mask")
            mask_boxes += 1
    return dict(dataset_readback_rows=len(rows), packed_rows=len(packed["imgname"]),
                mask_derived_boxes=mask_boxes)


def compare_outputs(full, compact):
    compared = []
    for camera in CAMERAS:
        people = [p.name for p in (full / camera).glob("person_*_mask.png")]
        for filename in ["normal_erp.png", "valid_mask.png", *people]:
            np.testing.assert_array_equal(
                np.asarray(Image.open(full / camera / filename)),
                np.asarray(Image.open(compact / camera / filename)),
            )
            compared.append(f"{camera}/{filename}")
        if json.loads((full/camera/"boxes.json").read_text()) != json.loads((compact/camera/"boxes.json").read_text()):
            raise ValueError(f"{camera}: boxes differ")
        for filename in ("annotations.npz",):
            with np.load(full/camera/filename) as a, np.load(compact/camera/filename) as b:
                if a.files != b.files: raise ValueError(f"{camera}: fields differ")
                for key in a.files: np.testing.assert_array_equal(a[key], b[key])
            compared.append(f"{camera}/{filename}")
    for filename in ("pair_annotations.npz", "shared_world_densekp.npy", "diagnostic_loader.npz"):
        if filename.endswith(".npz"):
            with np.load(full/filename) as a, np.load(compact/filename) as b:
                if a.files != b.files: raise ValueError(f"{filename}: fields differ")
                for key in a.files: np.testing.assert_array_equal(a[key], b[key])
        else:
            np.testing.assert_array_equal(np.load(full/filename), np.load(compact/filename))
        compared.append(filename)
    return compared


def run_mode(root, reference, mode, people, poses, masks, lenses, lens_metadata,
             vertices, dense):
    output = root / mode
    output.mkdir()
    shutil.copy2(reference / "people.json", output / "people.json")
    shutil.copy2(reference / "rig.json", output / "rig.json")
    source = prepare_source_workspace(output, mode)
    stage = "source_copy"
    timings = {}
    try:
        started = time.monotonic(); manifest = copy_source(reference / "source", source)
        timings["source_copy_seconds"] = time.monotonic() - started
        temporary_peak = directory_bytes(source.parent if mode == "compact" else source)
        stage = "postprocess"; started = time.monotonic()
        result = process_scene(
            output, people, poses, masks, vertices, dense,
            lens_masks=lenses, lens_metadata=lens_metadata,
            output_size=(5760, 2880), output_mode=mode, source_dir=source,
        )
        timings["postprocess_seconds"] = time.monotonic() - started
        result.update(scene="urban_street_assets", people=people,
                      baseline_m=.18, split="acceptance_only", output_mode=mode,
                      source_resolution=[5760,2880], final_erp_resolution=[5760,2880],
                      fisheye_resolution=[2880,2880])
        write_json(output / "report.json", result)
        stage = "readback_validation"; started = time.monotonic()
        validation = validate(output, source_dir=source, output_mode=mode)
        consumers = consumer_checks(output)
        timings["validation_seconds"] = time.monotonic() - started
        preserve_source_recipe(
            output, source, render_command=["acceptance-reuse", str(reference / "source"), str(source)],
            source_buffers_retained=mode == "full",
            extra=dict(output_mode=mode, reused_existing_c_source=str((reference/"source").resolve()),
                       resolution=dict(source_erp=[5760,2880],ds=[2880,2880],final_erp=[5760,2880]),
                       original_render_settings={k:manifest[k] for k in ("samples","seed","animated_seed","device")}),
        )
        stage = "cleanup"; started = time.monotonic()
        complete_output(output, mode, source, checks={**validation, **consumers})
        timings["cleanup_seconds"] = time.monotonic() - started
        return dict(output=str(output), temporary_source_bytes=temporary_peak,
                    timings=timings, validation=validation, consumers=consumers)
    except Exception as error:
        record_failure(output, mode, source, stage, error)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=ROOT/"outputs/experiments/ds_source_resolution_20261008/C")
    args = parser.parse_args()
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    reference=args.reference.resolve()
    people=json.loads((reference/"people.json").read_text())["people"]
    rig=json.loads((reference/"rig.json").read_text())
    poses={row["id"]:np.asarray(row["camera_to_world"]) for row in rig["cameras"]}
    with np.load(ROOT/"assets_external/real_erp_valid_masks.npz") as archive:
        masks={name:archive[f"valid_masks_{name}"].copy() for name in CAMERAS}
    lenses={name:np.asarray(Image.open(reference/name/"reference_lens_mask.png"))>0 for name in CAMERAS}
    old_report=json.loads((reference/"report.json").read_text())
    lens_metadata={row["camera"]:row["lens_mask"] for row in old_report["cameras"]}
    geometry=ROOT/"outputs/baselines/temporal_case_aaron_1095/geometry"
    vertices=np.load(geometry/"vertices.npy",mmap_mode="r")
    dense=np.load(geometry/"densekp_3d.npy",mmap_mode="r")
    started=time.monotonic()
    runs={mode:run_mode(output,reference,mode,people,poses,masks,lenses,lens_metadata,
                        vertices,dense)
          for mode in ("full","compact")}
    compared=compare_outputs(output/"full",output/"compact")
    compact_forbidden=[]
    for directory in ("source", "_temporary_ds_source"):
        if (output/"compact"/directory).exists(): compact_forbidden.append(directory)
    for filename in ("fisheye.png", "reference_lens_mask.png", "overlay.png",
                     "pair_review.jpg", "crop_review.jpg"):
        compact_forbidden += [
            str(path.relative_to(output/"compact"))
            for path in (output/"compact").rglob(filename)
        ]
    if compact_forbidden: raise ValueError(f"Compact retained debug/source files: {compact_forbidden}")
    for required in ("cam2.exr","cam2_preview.png","cam4.exr","cam4_preview.png"):
        if not (output/"full"/"source"/required).is_file(): raise ValueError(f"Full missing {required}")
    recipe_text=(output/"compact"/"source_recipe.json").read_text()
    cleaned=str((output/"compact"/"_temporary_ds_source").resolve())
    if cleaned in recipe_text: raise ValueError("Compact final recipe requires cleaned temporary path")
    runs["full"]["retained_bytes"]=directory_bytes(output/"full")
    runs["compact"]["retained_bytes"]=directory_bytes(output/"compact")
    report=dict(status="passed",reference_source=str(reference),render_count=0,
        note="Read-only reuse of the existing C source; active entry handoff/lifecycle exercised without a new render.",
        resolutions=dict(source_erp=[5760,2880],ds=[2880,2880],final_erp=[5760,2880]),
        exact_comparisons=compared,runs=runs,total_seconds=time.monotonic()-started,
        compact_forbidden_files=[],full_intermediates_readable=True,
        compact_manifest_has_cleaned_required_path=False)
    write_json(output/"acceptance_report.json",report)
    print(json.dumps(report,indent=2))


if __name__ == "__main__": main()
