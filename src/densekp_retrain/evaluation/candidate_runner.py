"""Read, check, predict, score and save one candidate observation at a time."""

import json
from pathlib import Path
import shutil
import time
import cv2
import numpy as np
from PIL import Image
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.configuration import audit_cfg
from densekp.core.evaluation import evaluation_masks, point_errors
from densekp.core.inference import load_checkpoint_state_dict, predict_batch
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, points_in_mask
from densekp_retrain.io.json import write_json
from densekp_retrain.visualization.keypoints import render_candidate_panel
from .candidates import summary
from .records import LoadedEvaluation, SampleCache, CandidateSample, ScoredCandidate, Observations


def load_evaluation_model(checkpoint_path, model_config, device_name, config_directory, output):
    "Preserve seed, precision, config and strict-load order; report elapsed loading time."
    torch.manual_seed(0)
    np.random.seed(0)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    with initialize_config_dir(version_base="1.2", config_dir=str(config_directory)):
        cfg = compose(config_name="train", overrides=["experiment=densekp"])
    if model_config:
        cfg = OmegaConf.load(model_config)
    cfg.DATASETS.TARGET_VALUE_SEMANTICS = "supervision_weight"
    OmegaConf.save(cfg, output / "model_config.yaml")
    (output / "loader_config.yaml").write_text(audit_cfg("supervision_weight").dump())
    checkpoint = checkpoint_path.resolve()
    stat = checkpoint.stat()
    load_start = time.monotonic()
    model, load_details = load_checkpoint_state_dict(checkpoint, cfg)
    load_audit = dict(
        checkpoint=str(checkpoint.resolve()),
        bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        **load_details,
        source="explicit strict checkpoint evaluation",
        with_var=bool(cfg.MODEL.with_var),
    )
    device = torch.device(device_name)
    model.to(device).eval()
    load_audit["seconds"] = time.monotonic() - load_start
    write_json(output / "checkpoint_load.json", load_audit)
    return LoadedEvaluation(model, device, checkpoint, stat, load_audit)


def read_candidate(row, cache):
    "Read the existing Dataset interface and verify row identity, geometry and GT supervision before prediction."
    datasets = cache.datasets
    scenes = cache.scenes
    loader_ids = cache.loader_ids
    directory = Path(row["scene"])
    dataset_key = str(Path(row["loader_input"]).resolve())
    if dataset_key not in datasets:
        if json.loads((directory / "validation.json").read_text())["status"] != "passed":
            raise ValueError("Scene numeric validation failed")
        datasets[dataset_key] = DatasetKeypoints(
            audit_cfg("supervision_weight"),
            "frozen-baseline",
            is_train=False,
            use_augmentation=False,
            annotation_path=row["loader_input"],
            image_root=str(directory),
        )
        with np.load(row["loader_input"]) as loader:
            loader_ids[dataset_key] = {name: loader[name] for name in ("camera_id", "person_id")}
    scene_key = str(directory.resolve())
    if scene_key not in scenes:
        if json.loads((directory / "validation.json").read_text())["status"] != "passed":
            raise ValueError("Scene numeric validation failed")
        scenes[scene_key] = json.loads((directory / "report.json").read_text())
    dataset = datasets[dataset_key]
    loader_row = row["loader_row"]
    item = dataset[loader_row]
    if dataset.data.get("supervision_policy") != SUPERVISION_POLICY:
        raise ValueError("Wrong supervision policy")
    if (
        str(loader_ids[dataset_key]["camera_id"][loader_row]) != row["camera"]
        or int(loader_ids[dataset_key]["person_id"][loader_row]) != row["person_id"]
    ):
        raise ValueError("Frozen candidate index differs from loader row")
    person = next((p for p in scenes[scene_key]["people"] if p["id"] == row["person_id"]))
    if (
        person["character_id"] != row["character_id"]
        or person["pose_source_id"] != row["pose_source_id"]
    ):
        raise ValueError("Frozen character/pose differs from rendered person")
    annotations = np.load(directory / row["camera"] / "annotations.npz")
    person_index = annotations["person_ids"].tolist().index(row["person_id"])
    target_crop = item["proj_verts_cropped"][:, :2]
    visible = annotations["visible"][person_index]
    np.testing.assert_allclose(
        restore_crop_xy(target_crop, item["_trans"], (256, 256)),
        annotations["xy"][person_index],
        atol=0.0002,
        rtol=0,
    )
    masks = evaluation_masks(annotations["in_view"][person_index], visible, target_crop)
    masks["visible_supervised"] = masks["supervised"] & visible
    np.testing.assert_array_equal(masks["supervised"], item["proj_verts_cropped"][:, 2] > 0)
    if int(masks["supervised"].sum()) != row["supervised_points"]:
        raise ValueError("Frozen supervision count changed")
    return CandidateSample(
        directory, scene_key, item, person, annotations, person_index, target_crop, visible, masks
    )


def compare_reference(sample, row, case_id, previous):
    "Compare frozen inputs/GT/affine/masks and square RGB exactly; never compare predictions to select samples."
    item = sample.item
    annotations = sample.annotations
    person_index = sample.person_index
    target_crop = sample.target_crop
    visible = sample.visible
    masks = sample.masks
    for field in ["scene", "camera", "person_id", "character_id", "pose_source_id", "loader_row"]:
        if row[field] != previous[field]:
            raise ValueError(f"Frozen case changed: {case_id}/{field}")
    with np.load(previous["prediction"]) as reference:
        np.testing.assert_array_equal(item["img"][:, :, 32:-32], reference["network_input"])
        np.testing.assert_array_equal(target_crop, reference["target_crop_xy"])
        np.testing.assert_array_equal(annotations["xy"][person_index], reference["target_erp_xy"])
        np.testing.assert_array_equal(
            annotations["world_densekp"][person_index], reference["target_world_xyz"]
        )
        np.testing.assert_array_equal(item["_trans"], reference["affine"])
        np.testing.assert_array_equal(visible, reference["visible"])
        np.testing.assert_array_equal(annotations["in_view"][person_index], reference["in_view"])
        np.testing.assert_array_equal(masks["supervised"], reference["supervised"])
    np.testing.assert_array_equal(
        cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB),
        np.array(Image.open(Path(previous["prediction"]).parent / "square.png")),
    )


def predict_candidate(model, item, device, folder):
    "Predict one full normalized square and preserve invalid-shape artifact/error behavior. Nonfinite output remains available for failure reporting."
    with torch.inference_mode():
        prediction = predict_batch(model, item["img"][None], device)[0].float().cpu().numpy()
    if prediction.shape != (138, 3):
        np.save(folder / "invalid_shape.npy", prediction)
        raise ValueError("Model output shape is not 138 x 3")
    return prediction


def score_candidate(row, sample, prediction, case_id, batch, folder):
    "Compute full-256-square and periodic ERP pixel errors, preserving every eligible GT point and raw third value."
    item = sample.item
    annotations = sample.annotations
    person_index = sample.person_index
    target_crop = sample.target_crop
    visible = sample.visible
    masks = sample.masks
    directory = sample.directory
    person = sample.person
    restored = restore_crop_xy(prediction[:, :2], item["_trans"], (256, 256))
    width, height = Image.open(row["image"]).size
    crop_error, erp_error = point_errors(
        prediction[:, :2], target_crop, restored, annotations["xy"][person_index], width
    )
    invalid = ~np.isfinite(prediction[:, :2]).all(1)
    crop_error[invalid] = np.inf
    erp_error[invalid] = np.inf
    valid_mask = np.array(Image.open(directory / row["camera"] / "valid_mask.png")) > 0
    predicted_in_lens = points_in_mask(restored, valid_mask)
    box = next(
        (
            b
            for b in json.loads((directory / row["camera"] / "boxes.json").read_text())
            if b["person_id"] == row["person_id"]
        )
    )["bbox_xyxy_exclusive"]
    box = row.get("given_bbox_xyxy", box)
    if "affine_flat" in row:
        np.testing.assert_allclose(item["_trans"].ravel(), row["affine_flat"], atol=1e-07, rtol=0)
    action = person.get("action", "cycling")
    category = person["category"]
    bbox_height = box[3] - box[1]
    scale = "small" if bbox_height < 96 else "medium" if bbox_height < 192 else "large"
    metrics = {
        name: dict(crop_256_px=summary(crop_error[mask]), erp_periodic_px=summary(erp_error[mask]))
        for name, mask in masks.items()
    }
    case = dict(row)
    case.update(
        id=case_id,
        batch=batch,
        action=action,
        category=category,
        scale_group=scale,
        bbox_xyxy=box,
        prediction_status="finite" if np.isfinite(prediction).all() else "nonfinite_failure",
        bbox_height_erp_px=bbox_height,
        box_size=float(item["box_size"]),
        metrics=metrics,
        nonfinite_xy_points=int(invalid.sum()),
        nonfinite_keypoint_value=int((~np.isfinite(prediction[:, 2])).sum()),
        supervised_prediction_outside_lens=int((masks["supervised"] & ~predicted_in_lens).sum()),
        prediction=str(folder / "prediction.npz"),
    )
    arrays = dict(
        prediction_raw=prediction,
        prediction_erp_xy=restored,
        target_crop_xy=target_crop,
        target_erp_xy=annotations["xy"][person_index],
        target_world_xyz=annotations["world_densekp"][person_index],
        affine=item["_trans"],
        bbox_xyxy=box,
        box_center=item["box_center"],
        box_size=item["box_size"],
        visible=visible,
        in_view=annotations["in_view"][person_index],
        supervised=masks["supervised"],
        crop_error_px=crop_error,
        erp_periodic_error_px=erp_error,
        prediction_in_lens=predicted_in_lens,
        character_id=row["character_id"],
        pose_source_id=row["pose_source_id"],
        camera_id=row["camera"],
        network_input=item["img"][:, :, 32:-32],
    )
    state = dict(crop=crop_error, erp=erp_error, masks=masks, visible=visible)
    return ScoredCandidate(case, state, arrays)


def save_candidate(sample, scored, output, image_cache):
    "Write arrays, copied source image, square/center images, point panel and case JSON in historical order."
    directory = sample.directory
    scene_key = sample.scene_key
    item = sample.item
    masks = sample.masks
    visible = sample.visible
    case = scored.case
    row = case
    case_id = case["id"]
    batch = case["batch"]
    action = case["action"]
    prediction = scored.arrays["prediction_raw"]
    folder = output / case_id
    np.savez(folder / "prediction.npz", **scored.arrays)
    image_key = (scene_key, row["camera"])
    if image_key not in image_cache:
        destination = output / "images" / f"{batch}_{directory.name}_{row['camera']}.png"
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(row["image"], destination)
        image_cache[image_key] = str(destination)
    case["saved_image"] = image_cache[image_key]
    rgb = cv2.cvtColor(item["img_disp"], cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(folder / "square.png")
    Image.fromarray(rgb[:, 32:-32]).save(folder / "network_center.png")
    panel = render_candidate_panel(
        item,
        prediction,
        masks["supervised"],
        visible,
        f"{case_id} {action} {row['character_id']} {row['camera']}",
    )
    panel.save(folder / "prediction.jpg")
    write_json(folder / "case.json", case)


def evaluate_rows(inputs, loaded, output, *, compare_frozen=False):
    """Process supplied rows in order; experiments remain combinations of these stages."""
    cache = SampleCache()
    observations = Observations()
    image_cache = {}
    for batch, rows in inputs.sources:
        for position, row in enumerate(rows):
            sample = read_candidate(row, cache)
            case_id = f"{batch}_{position:03d}"
            folder = output / case_id
            folder.mkdir()
            if compare_frozen:
                compare_reference(sample, row, case_id, inputs.references[case_id])
            prediction = predict_candidate(loaded.model, sample.item, loaded.device, folder)
            scored = score_candidate(row, sample, prediction, case_id, batch, folder)
            save_candidate(sample, scored, output, image_cache)
            observations.states[case_id] = scored.state
            observations.cases.append(scored.case)
            pair_key = (
                batch,
                str(sample.directory),
                row["person_id"],
                row.get("box_variant", "reference"),
            )
            observations.pairs.setdefault(pair_key, {})[row["camera"]] = case_id
            observations.fixed.setdefault(
                (batch, scored.case["action"], scored.case["category"]), case_id
            )
            print(case_id, scored.case["metrics"]["supervised"]["crop_256_px"], flush=True)
    return observations
