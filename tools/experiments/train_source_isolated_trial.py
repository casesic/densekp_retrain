"""Next-turn bounded trial. Defaults to data/config checks; never implicitly trains."""

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
from datetime import datetime, timezone
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.densekp_trainer import DenseKP
from densekp.core.datasets.configuration import audit_cfg
from densekp_retrain.data.source_split import validate_split
from densekp_retrain.training.trial_callbacks import FixedValidation  # Historical re-export.
from densekp_retrain.io.json import write_json_ascii
from densekp_retrain.io.json import write_json_ascii as write  # Historical re-export.
from densekp.core.training_audit import CheckUpdates, probes
from densekp.core.inference import predict_item as predict

RUN_OUTPUT = None


class Tee:
    def __init__(self, stream, log):
        self.stream, self.log = stream, log

    def write(self, value):
        self.stream.write(value)
        self.log.write(value)
        self.log.flush()

    def flush(self):
        self.stream.flush()
        self.log.flush()

    def isatty(self):
        return False


class IndexedRows(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return dict(self.data[index], _frozen_row=index)


class RunAudit(CheckUpdates):
    def __init__(self, out):
        super().__init__()
        self.out = out
        self.records = []
        self.started = time.monotonic()

    def on_train_start(self, trainer, model):
        optimizer = trainer.optimizers[0]
        expected = {id(p) for p in model.get_parameters()}
        actual = {id(p) for g in optimizer.param_groups for p in g["params"]}
        if not isinstance(optimizer, torch.optim.AdamW) or expected != actual:
            raise ValueError("Actual optimizer/type/parameter coverage mismatch")
        if (
            model.cfg.MODEL.with_var
            or model.automatic_optimization
            or trainer.accumulate_grad_batches != 1
        ):
            raise ValueError("Actual loss/manual optimization/accumulation mismatch")
        if any(not p.requires_grad for p in model.get_parameters()):
            raise ValueError("Frozen model parameter")
        groups = []
        for g in optimizer.param_groups:
            if g["lr"] != 1e-6 or g["weight_decay"] != 1e-4:
                raise ValueError("Actual LR/weight decay mismatch")
            groups.append(
                dict(
                    lr=g["lr"],
                    weight_decay=g["weight_decay"],
                    parameters=sum(p.numel() for p in g["params"]),
                )
            )
        if trainer.lr_scheduler_configs:
            raise ValueError("Unexpected LR scheduler")
        write_json_ascii(
            self.out / "optimizer_audit.json",
            dict(
                status="passed",
                type=type(optimizer).__name__,
                groups=groups,
                with_var=model.cfg.MODEL.with_var,
                manual_optimization=True,
                accumulation=1,
                precision=str(trainer.precision),
                all_backbone_head_trainable=True,
            ),
        )

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        if not model.training:
            raise ValueError("Validation did not restore training mode")
        self.row = int(batch["img"]["_frozen_row"].item())
        if batch["img"]["img"].shape[0] != 1:
            raise ValueError("Actual batch size mismatch")

    def on_before_backward(self, trainer, model, loss):
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite loss before backward")
        self.loss = float(loss.detach())

    def on_before_optimizer_step(self, trainer, model, optimizer):
        super().on_before_optimizer_step(trainer, model, optimizer)
        row = dict(
            update_attempt=trainer.global_step + 1,
            row_index=self.row,
            loss=self.loss,
            seconds_since_start=time.monotonic() - self.started,
            **self.attempts[-1],
        )
        self.records.append(row)
        with (self.out / "updates.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        record = dict(
            completed_update=trainer.global_step,
            row_index=self.row,
            loss=self.loss,
            seconds_since_start=time.monotonic() - self.started,
            peak_cuda_bytes=torch.cuda.max_memory_allocated(model.device),
            peak_reserved_cuda_bytes=torch.cuda.max_memory_reserved(model.device),
        )
        with (self.out / "completed_updates.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")


def dataset(root, side):
    train = side == "train"
    return DatasetKeypoints(
        audit_cfg("supervision_weight"),
        "source-trial",
        is_train=train,
        use_augmentation=False,
        annotation_path=str(root / "loader" / f"{side}.npz"),
        image_root=str(root),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute-training", action="store_true")
    parser.add_argument(
        "--box-data",
        type=Path,
        help="Pre-frozen box-only 400-exposure schedule; original data stays the reference",
    )
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    recommendation = json.loads((data_root / "training_recommendation.json").read_text())
    split = json.loads((data_root / "split_v2.json").read_text())
    indices = {s: json.loads((data_root / f"{s}_index.json").read_text()) for s in ["train", "val"]}
    isolation = validate_split(split, indices)
    if json.loads((data_root / "loader/acceptance.json").read_text())["status"] != "passed":
        raise ValueError("Loader not accepted")
    baseline = json.loads((data_root / "original_baseline/report.json").read_text())
    if baseline["status"] != "completed":
        raise ValueError("Original baseline missing")
    if baseline.get("prediction_failure_cases"):
        raise ValueError("Original baseline has unresolved nonfinite predictions")
    for label, side in [("train_diagnostic", "train"), ("val_baseline", "val")]:
        if (data_root / f"{side}_index.json").read_bytes() != (
            data_root / "original_baseline" / f"{label}_frozen_index.json"
        ).read_bytes():
            raise ValueError("Input index changed after baseline")
    data = {s: dataset(data_root, s) for s in ["train", "val"]}
    if any(len(data[s]) != len(indices[s]["rows"]) for s in data):
        raise ValueError("Loader rows differ")
    checkpoint = ROOT / "assets_external/densekp.ckpt"
    checkpoint_stat = checkpoint.stat()
    old = baseline["checkpoint"]
    if (checkpoint_stat.st_size, checkpoint_stat.st_mtime_ns) != (old["bytes"], old["mtime_ns"]):
        raise ValueError("Original checkpoint changed")
    settings = dict(
        data_root=str(data_root),
        init_checkpoint=str(checkpoint),
        checkpoint_bytes=checkpoint_stat.st_size,
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        logical_gpu=args.gpu,
        checkpoint_mtime_ns=checkpoint_stat.st_mtime_ns,
        strict_load=True,
        trainable="all backbone/head",
        learning_rate=1e-6,
        weight_decay=1e-4,
        batch_size=1,
        accumulation=1,
        steps=400,
        precision="bf16-mixed",
        validation_precision="float32",
        seed=930,
        augmentation=False,
        with_var=False,
        loss="60 * mean supervised squared XY distance; third prediction channel does not weight residuals",
        validation_every=100,
        sample_exposures=400,
        equivalent_train_passes=400 / len(data["train"]),
        source_isolation=isolation,
    )
    box_data = None
    if args.box_data:
        from densekp_retrain.data.loading import load_box_dataset

        box_root = args.box_data.resolve()
        box_indices = {
            s: json.loads((box_root / f"{s}_index.json").read_text()) for s in ("train", "val")
        }
        box_acceptance = json.loads((box_root / "acceptance.json").read_text())
        if box_acceptance["status"] != "passed" or len(box_indices["train"]["rows"]) != 400:
            raise ValueError("Box schedule not accepted or not 400 exposures")
        validate_split(split, box_indices)
        box_data = load_box_dataset(box_root / "loader/train.npz", train=True)
        for row_index, row in enumerate(box_indices["train"]["rows"]):
            item = box_data[row_index]
            with np.load(row["geometry"]) as frozen:
                np.testing.assert_array_equal(item["img"][:, :, 32:-32], frozen["network_input"])
                np.testing.assert_array_equal(item["_trans"], frozen["affine"])
                np.testing.assert_array_equal(
                    item["proj_verts_cropped"][:, 2] > 0, frozen["supervised"]
                )
            parent = int(row["parent_row"])
            np.testing.assert_array_equal(
                box_data.proj_verts[row_index, :, :2], data["train"].proj_verts[parent, :, :2]
            )
            np.testing.assert_array_equal(
                box_data.data["in_view"][row_index], data["train"].data["in_view"][parent]
            )
        settings["box_sampling"] = dict(
            config=json.loads((box_root / "config.json").read_text()),
            actual_variants=box_acceptance["actual_variants"]["train"],
            order="Already shuffled seed-930 target order replay; no second shuffle",
            changed_factor="training boxes and sampling only",
            input_checks="all 400 actual crops exact to frozen geometry/network input",
        )
    if not args.execute_training:
        print(json.dumps(dict(status="ready_data_config_checks_only", **settings), indent=2))
        return
    if not args.output:
        parser.error("--output is required for a new training run")
    torch.cuda.set_device(args.gpu)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    global RUN_OUTPUT
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    RUN_OUTPUT = output
    log = (output / "run.log").open("a", buffering=1)
    sys.stdout = Tee(sys.stdout, log)
    sys.stderr = Tee(sys.stderr, log)
    started = time.monotonic()
    settings["started_utc"] = datetime.now(timezone.utc).isoformat()
    write_json_ascii(output / "settings.json", settings)
    (output / "training_entry.py").write_bytes(Path(__file__).read_bytes())
    (output / "frozen_split_v2.json").write_bytes((data_root / "split_v2.json").read_bytes())
    for side in ["train", "val"]:
        (output / f"frozen_{side}_index.json").write_bytes(
            (data_root / f"{side}_index.json").read_bytes()
        )
    if box_data is not None:
        for name in ["train_index.json", "val_index.json", "config.json", "acceptance.json"]:
            (output / f"box_{name}").write_bytes((box_root / name).read_bytes())
    pl.seed_everything(930, workers=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    with initialize_config_dir(version_base="1.2", config_dir=str(ROOT / "configs/hydra")):
        cfg = compose(config_name="train", overrides=["experiment=densekp"])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS = "supervision_weight"
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.LR = settings["learning_rate"]
    cfg.TRAIN.WEIGHT_DECAY = settings["weight_decay"]
    cfg.MODEL.with_var = False
    cfg.TRAIN.LOSS_TYPE = "l2"  # Metadata; keypoint_objective implements squared XY directly.
    OmegaConf.save(cfg, output / "model_config.yaml")
    device = torch.device(f"cuda:{args.gpu}")
    model = DenseKP.load_from_checkpoint(
        str(checkpoint), cfg=cfg, map_location="cpu", strict=True
    ).to(device)
    # Reuse frozen predictions to check current images/labels/actual merged loaders.
    input_checks = []
    for side, label in [("train", "train_diagnostic"), ("val", "val_baseline")]:
        cases = [c for c in baseline["cases"] if c["batch"] == label]
        for row_index, case in enumerate(cases):
            item = data[side][row_index]
            with np.load(case["prediction"]) as z:
                np.testing.assert_array_equal(item["img"][:, :, 32:-32], z["network_input"])
                np.testing.assert_array_equal(
                    item["proj_verts_cropped"][:, :2], z["target_crop_xy"]
                )
                np.testing.assert_array_equal(item["_trans"], z["affine"])
                np.testing.assert_array_equal(item["proj_verts_cropped"][:, 2] > 0, z["supervised"])
        input_checks.append(dict(split=side, rows=len(cases), actual_input_gt_affine_mask="exact"))
    fixed_rows = [0, 1, 2, 26, 78]
    precheck = []
    for row_index in fixed_rows:
        actual_predictions = predict(model, data["val"][row_index], device)[0]
        with np.load(
            data_root / "original_baseline" / f"val_baseline_{row_index:03d}" / "prediction.npz"
        ) as z:
            expected_predictions = z["prediction_raw"]
        error = float(np.abs(actual_predictions - expected_predictions).max())
        np.testing.assert_allclose(actual_predictions, expected_predictions, atol=1e-5, rtol=0)
        precheck.append(dict(val_row=row_index, max_abs_error=error))
    write_json_ascii(
        output / "pretraining_baseline_check.json",
        dict(
            status="passed",
            input_checks=input_checks,
            fixed_val_rows=precheck,
            tolerance=1e-5,
            precision="FP32",
            with_var=False,
        ),
    )
    torch.cuda.reset_peak_memory_stats(device)
    initial_probes = probes(model)
    checker = RunAudit(output)
    validation = FixedValidation(data["val"], indices["val"]["rows"], output, 100)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[args.gpu],
        precision="bf16-mixed",
        max_steps=400,
        max_epochs=-1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[checker, validation],
        default_root_dir=str(output),
    )
    fit_start = time.monotonic()
    train_data = box_data if box_data is not None else data["train"]
    trainer.fit(
        model,
        train_dataloaders={
            "img": DataLoader(
                IndexedRows(train_data), batch_size=1, shuffle=box_data is None, num_workers=0
            )
        },
    )
    fit_seconds = time.monotonic() - fit_start
    steps = {float(s["step"]) for s in trainer.optimizers[0].state.values() if "step" in s}
    if len(checker.attempts) != 400 or steps != {400.0}:
        raise ValueError("Optimizer update count mismatch")
    changes = {k: float((v - initial_probes[k]).abs().max()) for k, v in probes(model).items()}
    if not all(v > 0 for v in changes.values()):
        raise ValueError("Parameter group did not update")
    if [r["step"] for r in validation.reports] != [100, 200, 300, 400]:
        raise ValueError("Scheduled validations missing")
    model.to(device)
    expected_predictions = np.concatenate(
        [predict(model, data["val"][row_index], device) for row_index in fixed_rows]
    )
    peak = torch.cuda.max_memory_allocated(device)
    reserved = torch.cuda.max_memory_reserved(device)
    del trainer, model
    gc.collect()
    torch.cuda.empty_cache()
    restored = DenseKP.load_from_checkpoint(
        str(output / "step_0400/checkpoint.ckpt"), cfg=cfg, map_location="cpu", strict=True
    ).to(device)
    actual_predictions = np.concatenate(
        [predict(restored, data["val"][row_index], device) for row_index in fixed_rows]
    )
    np.testing.assert_allclose(actual_predictions, expected_predictions, atol=1e-5, rtol=0)
    np.savez(
        output / "reload_predictions.npz",
        before_save=expected_predictions,
        after_load=actual_predictions,
        val_rows=fixed_rows,
    )
    if (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns) != (
        checkpoint_stat.st_size,
        checkpoint_stat.st_mtime_ns,
    ):
        raise ValueError("Original checkpoint changed")
    write_json_ascii(
        output / "report.json",
        dict(
            status="workflow_passed_not_automatic_effect_pass",
            settings=settings,
            actual_updates=400,
            parameter_changes=changes,
            reload_max_abs_error=float(np.abs(actual_predictions - expected_predictions).max()),
            peak_cuda_bytes=peak,
            peak_reserved_cuda_bytes=reserved,
            fit_seconds=fit_seconds,
            total_seconds=time.monotonic() - started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            sample_exposures=len(checker.records),
            unique_exposed_rows=len({r["row_index"] for r in checker.records}),
            unique_parent_rows=(
                len(
                    {
                        box_indices["train"]["rows"][r["row_index"]]["parent_row"]
                        for r in checker.records
                    }
                )
                if box_data is not None
                else len({r["row_index"] for r in checker.records})
            ),
            loss_range=[
                min(r["loss"] for r in checker.records),
                max(r["loss"] for r in checker.records),
            ],
            validation_steps=[r["step"] for r in validation.reports],
            comparison="Use frozen original baseline; primary predeclared final step 400, report all scheduled points and complete-control regressions",
        ),
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if RUN_OUTPUT is not None:
            write_json_ascii(
                RUN_OUTPUT / "failure.json",
                dict(
                    error=repr(error),
                    traceback=traceback.format_exc(),
                    utc=datetime.now(timezone.utc).isoformat(),
                    policy="stopped; no sample skipping or automatic retry",
                ),
            )
        raise
