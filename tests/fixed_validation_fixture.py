"""Real CUDA/Lightning fixture for scheduled validation, failures and resume."""

import json
from pathlib import Path
import random

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from densekp.core.keypoint_objective import keypoint_loss


class SavedRows(Dataset):
    def __init__(self, source):
        self.items = []
        self.read_order = []
        cases = {case["id"]: case for case in json.loads((source / "cases.json").read_text())}
        self.rows = []
        for case_id in ("val_baseline_002", "val_baseline_001", "val_baseline_004"):
            with np.load(source / (case_id + ".npz"), allow_pickle=False) as archive:
                self.items.append({key: archive[key] for key in archive.files})
            case = cases[case_id]
            self.rows.append(
                dict(
                    action=case["action"],
                    category=case["category"],
                    given_bbox_xyxy=case["bbox_xyxy"],
                )
            )
        self.proj_verts = np.stack([item["proj_verts"] for item in self.items])
        self.data = {
            name: np.stack([item[name] for item in self.items]) for name in ("visible", "in_view")
        }

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        self.read_order.append(index)
        return self.items[index]


class TrainRows(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return 2

    def __getitem__(self, index):
        item = self.data.items[index]
        return dict(img=item["img"], proj_verts_cropped=item["proj_verts_cropped"], row=index)


class TinyModel(pl.LightningModule):
    def __init__(self, trace, failure=None):
        super().__init__()
        self.backbone = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(3, 4),
            torch.nn.Dropout(0.25),
        )
        self.head = torch.nn.Linear(4, 138 * 3)
        self.trace = trace
        self.failure = failure
        self.automatic_optimization = False

    def forward(self, batch):
        raw = self.head(self.backbone(batch["img"])).reshape(-1, 138, 3)
        self.trace.append(
            dict(
                event="forward",
                training=self.training,
                autocast=torch.is_autocast_enabled(),
                dtype=str(raw.dtype),
                inference=torch.is_inference_mode_enabled(),
            )
        )
        if not self.training:
            if self.failure == "nan":
                raw[0, 0, 0] = float("nan")
            elif self.failure == "invalid_shape":
                raw = raw[:, :137]
            elif self.failure == "cpu_rng":
                torch.rand(1)
            elif self.failure == "cuda_rng":
                torch.rand(1, device=self.device)
            elif self.failure == "parameter_version":
                self.head.weight.add_(0)
            elif self.failure == "step_mutation":
                self.trainer.fit_loop.epoch_loop.manual_optimization.optim_step_progress.total.completed += (
                    1
                )
        return {"pred_keypoints": raw}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            list(self.head.parameters()) + list(self.backbone.parameters()),
            lr=1e-6,
            weight_decay=1e-4,
        )

    def training_step(self, joint_batch, batch_idx):
        batch = joint_batch["img"]
        optimizer = self.optimizers(use_pl_optimizer=True)
        prediction = self(batch)["pred_keypoints"]
        loss, _ = keypoint_loss(prediction, batch["proj_verts_cropped"], False)
        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
        return prediction


class TraceBatch(pl.Callback):
    def __init__(self, stage, events, arrays):
        self.stage, self.events, self.arrays = stage, events, arrays

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        capture_state(
            self.stage, trainer, model, self.events, self.arrays, int(batch["img"]["row"].item())
        )


def capture_state(stage, trainer, model, events, arrays, row=None):
    index = len(events)
    names = {id(value): name for name, value in model.named_parameters()}
    optimizer = trainer.optimizers[0]
    events.append(
        dict(
            event=stage,
            step=trainer.global_step,
            training=model.training,
            row=row,
            versions=[value._version for value in model.parameters()],
            optimizer_order=[
                [names[id(value)] for value in group["params"]] for group in optimizer.param_groups
            ],
        )
    )
    prefix = f"{index:03d}_{stage}"
    arrays[prefix + "_cpu_rng"] = torch.get_rng_state().numpy().copy()
    arrays[prefix + "_cuda_rng"] = torch.cuda.get_rng_state(model.device).cpu().numpy().copy()
    for name, value in model.named_parameters():
        arrays[prefix + "_parameter_" + name] = value.detach().float().cpu().numpy().copy()
        for key, item in optimizer.state[value].items():
            if torch.is_tensor(item):
                arrays[prefix + "_optimizer_" + name + "_" + key] = (
                    item.detach().cpu().numpy().copy()
                )


def normalize_report(report):
    return {key: value for key, value in report.items() if key != "seconds"}


def run_case(callback_class, source, directory, failure=None, resume=None):
    directory.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(161)
    random.seed(161)
    np.random.seed(161)
    events, arrays = [], {}
    data = SavedRows(source)
    model = TinyModel(events, failure)
    callback = callback_class(data, data.rows, directory, 2)
    if failure == "duplicate_folder":
        (directory / "step_0002").mkdir()
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        precision="bf16-mixed",
        max_steps=4 if resume else (2 if failure else 4),
        max_epochs=-1,
        num_sanity_val_steps=0,
        limit_val_batches=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[
            TraceBatch("before_callback", events, arrays),
            callback,
            TraceBatch("after_callback", events, arrays),
        ],
        default_root_dir=str(directory),
    )
    save = trainer.save_checkpoint

    def traced_save(path, weights_only=False, **kwargs):
        capture_state("save_checkpoint", trainer, model, events, arrays)
        if failure == "save_failure":
            raise OSError("fixture checkpoint save failure")
        return save(path, weights_only=weights_only, **kwargs)

    trainer.save_checkpoint = traced_save
    error = None
    try:
        trainer.fit(
            model,
            train_dataloaders={
                "img": DataLoader(TrainRows(data), batch_size=1, shuffle=True, num_workers=0)
            },
            ckpt_path=resume,
        )
    except Exception as exception:
        error = dict(
            type=type(exception).__name__,
            message=str(exception).replace(str(directory), "<OUTPUT>"),
        )
    artifacts = []
    checkpoints = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        name = str(path.relative_to(directory))
        artifacts.append(name)
        if path.suffix == ".ckpt":
            payload = torch.load(path, map_location="cpu")
            checkpoints.append(
                dict(
                    file=name,
                    step=payload["global_step"],
                    callback_states=payload["callbacks"],
                    optimizer_groups=payload["optimizer_states"][0]["param_groups"],
                    schema=[
                        dict(key=key, shape=list(value.shape), dtype=str(value.dtype))
                        for key, value in payload["state_dict"].items()
                    ],
                )
            )
            for key, value in payload["state_dict"].items():
                arrays[name + "_" + key] = value.numpy()
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                for key in archive.files:
                    arrays[name + "_" + key] = archive[key]
        elif path.suffix == ".npy":
            arrays[name] = np.load(path, allow_pickle=False)
    result = dict(
        events=events,
        data_read_order=data.read_order,
        state_key=callback.state_key,
        callback_state=callback.state_dict(),
        reports=[normalize_report(report) for report in callback.reports],
        global_step=trainer.global_step,
        final_training=model.training,
        error=error,
        artifacts=artifacts,
        checkpoints=checkpoints,
        resumed=resume is not None,
    )
    if not failure:
        assert error is None, error
        assert [report["step"] for report in callback.reports] == ([4] if resume else [2, 4])
        assert all(
            event["dtype"] == "torch.float32" and not event["autocast"]
            for event in events
            if event["event"] == "forward" and not event["training"]
        )
    else:
        assert error is not None, failure
    return result, arrays
