"""Fixed-input validation for the source-isolated and box-robustness trials."""

import time
from typing import NamedTuple
import numpy as np
import pytorch_lightning as pl
import torch
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.evaluation import evaluation_masks, point_errors
from densekp_retrain.evaluation.candidates import aggregate
from densekp_retrain.io.json import write_json_ascii


class ValidationRows(NamedTuple):
    states: dict
    cases: list
    predictions: list  # Each array is float32 [138,3], including the raw third value.


class FixedValidation(pl.Callback):
    """Scheduled FP32 validation and checkpointing for bounded experiments.

    Success order is eval, rows, invariant checks, report, checkpoint, train.
    Exceptions stop at the original point without restoring mode or retrying.
    The class name preserves Lightning state_key; reports are not persistent
    callback state, matching the original default empty state_dict.
    """

    def __init__(self, data, rows, out, every):
        self.data = data
        self.rows = rows
        self.out = out
        self.every = every
        self.reports = []

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        step = trainer.global_step
        if not step or step % self.every:
            return
        folder = self.out / f"step_{step:04d}"
        folder.mkdir(exist_ok=False)
        model.eval()
        started = time.monotonic()
        versions = [p._version for p in model.parameters()]
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(model.device).clone()
        with torch.inference_mode(), torch.autocast(device_type="cuda", enabled=False):
            states, cases, predictions = self._evaluate_rows(model, folder)
        np.savez(folder / "val_predictions.npz", prediction_raw=predictions)
        if versions != [p._version for p in model.parameters()] or trainer.global_step != step:
            raise ValueError("Validation changed parameters or optimizer update count")
        if not torch.equal(cpu_rng, torch.get_rng_state()) or not torch.equal(
            cuda_rng, torch.cuda.get_rng_state(model.device)
        ):
            raise ValueError("Validation changed RNG state")
        report = dict(
            step=step,
            metrics=aggregate(cases, states),
            rows=len(self.rows),
            seconds=time.monotonic() - started,
            parameters_steps_rng_unchanged=True,
            precision="float32; no autocast",
            index="unchanged frozen val_index.json",
            keypoint_value="raw third column preserved; never used to select or weight evaluation points",
        )
        write_json_ascii(folder / "validation.json", report)
        self.reports.append(report)
        trainer.save_checkpoint(str(folder / "checkpoint.ckpt"), weights_only=False)
        model.train()
        print(f"Validation/checkpoint completed: step {step}, rows {len(self.rows)}", flush=True)

    def _evaluate_rows(self, model, folder):
        """Score rows in frozen order and retain float32 [138,3] predictions.

        XY uses full 256-square coordinates and 1536-wide synthetic ERP pixels.
        This callback belongs to these bounded synthetic trials; the fixed ERP
        width must not be reused as real-dataset calibration.
        """
        states = {}
        cases = []
        predictions = []
        for row_index, row in enumerate(self.rows):
            item = self.data[row_index]
            prediction = (
                model({"img": torch.from_numpy(item["img"][None]).to(model.device)})[
                    "pred_keypoints"
                ][0]
                .float()
                .cpu()
                .numpy()
            )
            if prediction.shape != (138, 3) or not np.isfinite(prediction).all():
                np.save(folder / f"failed_row_{row_index}.npy", prediction)
                raise ValueError("Nonfinite/invalid scheduled validation prediction")
            predictions.append(prediction)
            target_crop = item["proj_verts_cropped"][:, :2]
            prediction_erp = restore_crop_xy(prediction[:, :2], item["_trans"], (256, 256))
            target_erp = self.data.proj_verts[row_index, :, :2]
            crop_error, erp_error = point_errors(
                prediction[:, :2], target_crop, prediction_erp, target_erp, 1536
            )
            invalid = ~np.isfinite(prediction[:, :2]).all(1)
            crop_error[invalid] = np.inf
            erp_error[invalid] = np.inf
            visible = self.data.data["visible"][row_index]
            in_view = self.data.data["in_view"][row_index]
            masks = evaluation_masks(in_view, visible, target_crop)
            masks["visible_supervised"] = masks["supervised"] & visible
            case_id = str(row_index)
            box_height = row["given_bbox_xyxy"][3] - row["given_bbox_xyxy"][1]
            cases.append(
                dict(
                    id=case_id,
                    action=row["action"],
                    category=row["category"],
                    scale_group=(
                        "small" if box_height < 96 else "medium" if box_height < 192 else "large"
                    ),
                )
            )
            states[case_id] = dict(crop=crop_error, erp=erp_error, masks=masks)
        return ValidationRows(states, cases, predictions)
