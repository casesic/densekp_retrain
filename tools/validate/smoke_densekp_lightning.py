#!/usr/bin/env python3
"""Exercise the upstream DenseKP Lightning training loop on the tiny dataset."""

import argparse
import json
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from densekp.core.configs import dataset_config  # noqa: E402
from densekp.core.datasets.datamodule_keypoints import DataModule  # noqa: E402
from densekp.core.densekp_trainer import DenseKP  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--dataset", default="p0_dressed_tiny")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/densekp_tiny_smoke/lightning_report.json")
    args = parser.parse_args()
    if args.steps < 1 or args.gpu < 0:
        parser.error("--steps must be positive and --gpu must be nonnegative")
    if not torch.cuda.is_available() or args.gpu >= torch.cuda.device_count():
        parser.error("requested NVIDIA GPU is unavailable in this process")

    pl.seed_everything(0, workers=True)
    torch.set_float32_matmul_precision("medium")
    with initialize_config_dir(version_base="1.2", config_dir=str(ROOT / "configs/hydra")):
        cfg = compose(config_name="train", overrides=[f"data={args.dataset}", "experiment=densekp"])
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.GENERAL.NUM_WORKERS = 1
    cfg.GENERAL.PREFETCH_FACTOR = 2
    augmentation = cfg.DATASETS.CONFIG
    augmentation.SCALE_FACTOR = 0
    augmentation.COLOR_SCALE = 0
    augmentation.ROT_AUG_RATE = 0
    augmentation.TRANS_AUG_RATE = 0
    augmentation.EXTREME_CROP_AUG_RATE = 0
    augmentation.DO_FLIP = False
    augmentation.USE_ALB = False

    datamodule = DataModule(cfg, dataset_config())
    model = (DenseKP.load_from_checkpoint(str(args.checkpoint), cfg=cfg, map_location="cpu")
             if args.checkpoint else DenseKP(cfg))
    trainer = pl.Trainer(
        accelerator="gpu", devices=[args.gpu], precision="16-mixed",
        max_steps=args.steps, val_check_interval=1, limit_val_batches=1,
        num_sanity_val_steps=0, log_every_n_steps=1,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, logger=False,
    )
    trainer.fit(model, datamodule=datamodule)
    report = {
        "trainer": "upstream DenseKP.training_step and DataModule",
        "gpu": args.gpu, "gpu_name": torch.cuda.get_device_name(args.gpu),
        "precision": "16-mixed", "batch_size": 1,
        "completed_steps": trainer.global_step,
        "dataset": args.dataset,
        "initial_checkpoint": str(args.checkpoint.resolve()) if args.checkpoint else None,
        "initialization": "weights only; optimizer and step counter are fresh" if args.checkpoint else "ViTPose backbone",
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(args.gpu),
        "metrics": {name: float(value) for name, value in trainer.callback_metrics.items()},
        "checkpoint_saved": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
