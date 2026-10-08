#!/usr/bin/env python3
"""Freeze and evaluate fixed DenseKP crops for the source-resolution experiment."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

import cv2
from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.configuration import audit_cfg
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.evaluation import point_errors, summarize
from densekp.core.inference import load_lightning_checkpoint, predict_batch
from densekp.core.keypoint_supervision import (
    SUPERVISION_POLICY,
    make_targets,
    points_in_mask,
)


GROUPS = ("A", "B", "C")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read_annotations(root):
    values = {}
    for group in GROUPS:
        for camera in ("cam2", "cam4"):
            with np.load(root / group / camera / "annotations.npz") as archive:
                values[group, camera] = {key: archive[key] for key in archive.files}
    return values


def person_row(data, person_id):
    return data["person_ids"].tolist().index(person_id)


def build_dataset(root, group, rows, annotations, output):
    targets, in_view, visible = [], [], []
    for row in rows:
        data = annotations[group, row["camera"]]
        index = person_row(data, row["person_id"])
        targets.append(
            make_targets(data["xy"][index], data["in_view"][index], row["center"], row["scale"])
        )
        in_view.append(data["in_view"][index])
        visible.append(data["visible"][index])
    archive = output / group / "fixed_box_loader.npz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        archive,
        imgname=np.asarray([f'{row["camera"]}/normal_erp.png' for row in rows]),
        center=np.asarray([row["center"] for row in rows], dtype=np.float32),
        scale=np.asarray([row["scale"] for row in rows], dtype=np.float32),
        proj_verts=np.asarray(targets, dtype=np.float32),
        supervision_policy=np.asarray(SUPERVISION_POLICY),
        in_view=np.stack(in_view),
        visible=np.stack(visible),
        camera_id=np.asarray([row["camera"] for row in rows]),
        person_id=np.asarray([row["person_id"] for row in rows]),
    )
    return DatasetKeypoints(
        audit_cfg("supervision_weight"),
        "source-resolution",
        is_train=False,
        use_augmentation=False,
        annotation_path=str(archive),
        image_root=str(root / group),
    )


def sharpness(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    laplacian = cv2.Laplacian(gray, cv2.CV_32F)
    return dict(
        laplacian_variance=float(laplacian.var()),
        mean_absolute_laplacian=float(np.abs(laplacian).mean()),
    )


def draw_overlay(item, prediction, gt_mask, visible, title):
    canvas = Image.fromarray(cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB)).resize(
        (512, 512), Image.Resampling.NEAREST
    )
    draw = ImageDraw.Draw(canvas)
    gt = (item["proj_verts_cropped"][:, :2] + .5) * 512
    pred = (prediction[:, :2] + .5) * 512
    for index in np.flatnonzero(gt_mask):
        gx, gy = gt[index]
        px, py = pred[index]
        draw.line((gx, gy, px, py), fill=(240, 190, 60))
        draw.ellipse((gx - 3, gy - 3, gx + 3, gy + 3),
                     fill=(30, 255, 80) if visible[index] else (255, 170, 40))
        draw.ellipse((px - 2, py - 2, px + 2, py + 2), fill=(255, 50, 70))
    draw.rectangle((0, 0, 512, 22), fill="black")
    draw.text((5, 5), title, fill="white")
    return canvas


def horizontal(images):
    sheet = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
    x = 0
    for image in images:
        sheet.paste(image, (x, 0))
        x += image.width
    return sheet


def prepare(root, output):
    plan = json.loads((root / "experiment_plan.json").read_text())
    rows = plan["fixed_boxes"]
    annotations = read_annotations(root)
    checks = dict(final_image_sizes={}, exact_world=True, exact_xy=True, exact_in_view=True,
                  exact_valid_mask=True, exact_affine=True)

    for group in GROUPS:
        for camera in ("cam2", "cam4"):
            size = Image.open(root / group / camera / "normal_erp.png").size
            checks["final_image_sizes"][f"{group}/{camera}"] = list(size)
            if size != tuple(plan["final_erp_size"]):
                raise ValueError(f"Unexpected final ERP size: {group}/{camera} {size}")
            reference = annotations["A", camera]
            current = annotations[group, camera]
            for key, check_name in (
                ("world_densekp", "exact_world"),
                ("xy", "exact_xy"),
                ("in_view", "exact_in_view"),
            ):
                equal = np.array_equal(current[key], reference[key])
                checks[check_name] &= bool(equal)
                if not equal:
                    raise ValueError(f"{key} changed across resolution groups: {group}/{camera}")
            mask_equal = np.array_equal(
                np.asarray(Image.open(root / group / camera / "valid_mask.png")),
                np.asarray(Image.open(root / "A" / camera / "valid_mask.png")),
            )
            checks["exact_valid_mask"] &= bool(mask_equal)
            if not mask_equal:
                raise ValueError(f"valid mask changed: {group}/{camera}")

    datasets = {group: build_dataset(root, group, rows, annotations, output) for group in GROUPS}
    samples = []
    for row_index, row in enumerate(rows):
        items = {group: datasets[group][row_index] for group in GROUPS}
        reference = items["A"]
        affines_equal = all(np.array_equal(items[group]["_trans"], reference["_trans"])
                            for group in GROUPS)
        targets_equal = all(np.array_equal(items[group]["proj_verts_cropped"],
                                           reference["proj_verts_cropped"])
                            for group in GROUPS)
        checks["exact_affine"] &= affines_equal
        if not affines_equal or not targets_equal:
            raise ValueError(f"Crop geometry changed for row {row_index}")
        camera, person_id = row["camera"], row["person_id"]
        visible = []
        in_view = []
        for group in GROUPS:
            data = annotations[group, camera]
            index = person_row(data, person_id)
            visible.append(data["visible"][index])
            in_view.append(data["in_view"][index])
        supervised = reference["proj_verts_cropped"][:, 2] > 0
        common_visible = supervised & np.logical_and.reduce(visible)
        common_occluded = supervised & ~np.logical_or.reduce(visible)
        changed_visibility = supervised & ~(common_visible | common_occluded)
        samples.append(
            dict(
                row=row_index,
                camera=camera,
                person_id=person_id,
                center=row["center"],
                scale=row["scale"],
                affine=reference["_trans"].tolist(),
                supervised=int(supervised.sum()),
                in_view={group: int(in_view[i].sum()) for i, group in enumerate(GROUPS)},
                visible={group: int(visible[i].sum()) for i, group in enumerate(GROUPS)},
                common_visible=int(common_visible.sum()),
                common_occluded=int(common_occluded.sum()),
                changed_visibility=int(changed_visibility.sum()),
            )
        )
        np.savez(
            output / f"row_{row_index}_frozen_masks.npz",
            supervised=supervised,
            common_visible=common_visible,
            common_occluded=common_occluded,
            changed_visibility=changed_visibility,
            **{f"visible_{group}": visible[i] for i, group in enumerate(GROUPS)},
        )
    manifest = dict(
        status="frozen_before_inference",
        groups=list(GROUPS),
        rows=samples,
        checks=checks,
        primary_support=(
            "same GT row and exact common supervised mask across A/B/C; common_visible and "
            "common_occluded exclude points whose raster visibility classification changes"
        ),
        self_support="also report each group's own visible/occluded partition",
        selection="all five pre-existing clear-box rows; no prediction-based selection",
    )
    write_json(output / "evaluation_manifest.json", manifest)
    return datasets, annotations, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "assets_external/densekp.ckpt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output.exists() and not (output / "evaluation_manifest.json").is_file():
        raise FileExistsError(f"Incomplete evaluation directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if (output / "report.json").exists():
        raise FileExistsError(f"Completed evaluation exists: {output / 'report.json'}")

    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    datasets, annotations, manifest = prepare(root, output)
    if args.prepare_only:
        print(f"Frozen: {output / 'evaluation_manifest.json'}", flush=True)
        return

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    with initialize_config_dir(version_base="1.2", config_dir=str(ROOT / "configs/hydra")):
        cfg = compose(config_name="train", overrides=["experiment=densekp"])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS = "supervision_weight"
    OmegaConf.save(cfg, output / "model_config.yaml")
    (output / "loader_config.yaml").write_text(audit_cfg("supervision_weight").dump())

    load_started = time.monotonic()
    model = load_lightning_checkpoint(args.checkpoint, cfg).to(device).eval()
    load_seconds = time.monotonic() - load_started
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    rows = json.loads((root / "experiment_plan.json").read_text())["fixed_boxes"]
    results = []
    buckets = {
        group: {name: {"crop": [], "erp": []} for name in
                ("common_supervised", "common_visible", "common_occluded",
                 "self_visible", "self_occluded")}
        for group in GROUPS
    }
    detail_panels = {index: [] for index in range(len(rows))}
    input_panels = {index: [] for index in range(len(rows))}
    overlay_panels = {index: [] for index in range(len(rows))}
    inference_seconds = 0.0
    nonfinite = 0

    for row_index, row in enumerate(rows):
        with np.load(output / f"row_{row_index}_frozen_masks.npz") as frozen:
            frozen_masks = {key: frozen[key].copy() for key in frozen.files}
        for group in GROUPS:
            item = datasets[group][row_index]
            started = time.monotonic()
            prediction = predict_batch(model, item["img"][None], device)[0].float().cpu().numpy()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds += time.monotonic() - started
            finite = np.isfinite(prediction).all(axis=1)
            nonfinite += int((~finite).sum())
            if not finite.all():
                raise ValueError(f"Nonfinite prediction: {group} row {row_index}")
            restored = restore_crop_xy(prediction[:, :2], item["_trans"], (256, 256))
            target_crop = item["proj_verts_cropped"][:, :2]
            camera, person_id = row["camera"], row["person_id"]
            data = annotations[group, camera]
            person_index = person_row(data, person_id)
            target_erp = data["xy"][person_index]
            crop_error, erp_error = point_errors(
                prediction[:, :2], target_crop, restored, target_erp, 5760
            )
            visible = data["visible"][person_index]
            supervised = frozen_masks["supervised"]
            masks = dict(
                common_supervised=supervised,
                common_visible=frozen_masks["common_visible"],
                common_occluded=frozen_masks["common_occluded"],
                self_visible=supervised & visible,
                self_occluded=supervised & ~visible,
            )
            lens = np.asarray(Image.open(root / group / camera / "valid_mask.png")) > 0
            prediction_valid = points_in_mask(restored, lens)
            metrics = {}
            for name, mask in masks.items():
                metrics[name] = dict(
                    crop_px=summarize(crop_error[mask]),
                    erp_periodic_px=summarize(erp_error[mask]),
                )
                buckets[group][name]["crop"].extend(crop_error[mask].tolist())
                buckets[group][name]["erp"].extend(erp_error[mask].tolist())
            sample_dir = output / group
            np.savez(
                sample_dir / f'{camera}_person_{person_id}.npz',
                prediction_raw=prediction,
                prediction_erp_xy=restored,
                target_crop_xy=target_crop,
                target_erp_xy=target_erp,
                affine=item["_trans"],
                crop_error_px=crop_error,
                erp_periodic_error_px=erp_error,
                prediction_valid=prediction_valid,
                **masks,
            )
            Image.fromarray(cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB)).save(
                sample_dir / f'{camera}_person_{person_id}_model_input.png'
            )
            source_size = json.loads((root / group / "source/source_manifest.json").read_text())["resolution"]
            results.append(
                dict(
                    group=group,
                    source_size=source_size,
                    row=row_index,
                    camera=camera,
                    person_id=person_id,
                    metrics=metrics,
                    prediction_valid=dict(
                        supervised=int((prediction_valid & supervised).sum()),
                        supervised_total=int(supervised.sum()),
                    ),
                    detail=sharpness(item["img_disp"]),
                )
            )

            rgb = Image.open(root / group / camera / "normal_erp.png").convert("RGB")
            center = row["center"]
            side = row["scale"] * 200
            raw = rgb.crop((center[0] - side / 2, center[1] - side / 2,
                            center[0] + side / 2, center[1] + side / 2))
            raw = raw.resize((512, 512), Image.Resampling.LANCZOS)
            label = ImageDraw.Draw(raw)
            label.rectangle((0, 0, 512, 24), fill="black")
            label.text((5, 5), f"{group} source {source_size[0]}x{source_size[1]} / display resize", fill="white")
            detail_panels[row_index].append(raw)
            input_panel = Image.fromarray(
                cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB)
            ).resize((512, 512), Image.Resampling.NEAREST)
            label = ImageDraw.Draw(input_panel)
            label.rectangle((0, 0, 512, 24), fill="black")
            label.text((5, 5), f"{group} actual 256-square input shown 2x nearest", fill="white")
            input_panels[row_index].append(input_panel)
            overlay_panels[row_index].append(
                draw_overlay(item, prediction, supervised, visible,
                             f"{group} {camera}/person{person_id}; crop px; n={int(supervised.sum())}")
            )

    for row_index, row in enumerate(rows):
        stem = f'row_{row_index}_{row["camera"]}_person_{row["person_id"]}'
        horizontal(detail_panels[row_index]).save(output / f"{stem}_detail_compare.jpg", quality=94)
        horizontal(input_panels[row_index]).save(output / f"{stem}_model_input_compare.png")
        horizontal(overlay_panels[row_index]).save(output / f"{stem}_overlay_compare.jpg", quality=94)

    aggregate = {
        group: {
            name: dict(crop_px=summarize(values["crop"]),
                       erp_periodic_px=summarize(values["erp"]))
            for name, values in buckets[group].items()
        }
        for group in GROUPS
    }
    report = dict(
        status="completed",
        root=str(root),
        checkpoint=str(args.checkpoint.resolve()),
        device=str(device),
        precision="float32, no autocast, TF32 disabled",
        model_load_seconds=load_seconds,
        inference_seconds=inference_seconds,
        peak_cuda_memory_bytes=(torch.cuda.max_memory_allocated(device)
                                if device.type == "cuda" else None),
        evaluated_rows=len(rows),
        predictions=len(results),
        finite_output_values=len(results) * 138 * 3,
        nonfinite_points=nonfinite,
        third_value="saved unchanged; not confidence, filtering, or metric weight",
        support_manifest=str((output / "evaluation_manifest.json").resolve()),
        aggregate=aggregate,
        cases=results,
    )
    write_json(output / "report.json", report)
    print(f"Completed: {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
