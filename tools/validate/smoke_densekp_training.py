#!/usr/bin/env python3
"""Run a small training check against the real DenseKP loader and loss."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from densekp.core.datasets.dataset_keypoints import DatasetKeypoints  # noqa: E402
from densekp.core.densekp_trainer import DenseKP  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--full-model", action="store_true", help="update backbone and head")
    parser.add_argument("--device", default="cpu", help="PyTorch device, for example cpu or cuda:0")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/experiments/densekp_tiny_smoke/report.json")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this process")

    torch.manual_seed(0)
    torch.set_num_threads(4)
    with initialize_config_dir(version_base="1.2", config_dir=str(ROOT / "configs/hydra")):
        cfg = compose(config_name="train", overrides=["data=p0_dressed_tiny", "experiment=densekp"])
    augmentation = cfg.DATASETS.CONFIG
    augmentation.SCALE_FACTOR = 0
    augmentation.COLOR_SCALE = 0
    augmentation.ROT_AUG_RATE = 0
    augmentation.TRANS_AUG_RATE = 0
    augmentation.EXTREME_CROP_AUG_RATE = 0
    augmentation.DO_FLIP = False
    augmentation.USE_ALB = False

    train = DatasetKeypoints(cfg, "p0-dressed-tiny", is_train=True)
    val = DatasetKeypoints(cfg, "p0-dressed-tiny", is_train=False)
    if len(train) != 12 or len(val) != 4:
        raise RuntimeError(f"unexpected split sizes: {len(train)} train, {len(val)} val")
    sample = train[0]
    batch = {
        "img": torch.from_numpy(np.asarray(sample["img"])[None]).float().to(device),
        "proj_verts_cropped": torch.from_numpy(
            np.asarray(sample["proj_verts_cropped"])[None]
        ).float().to(device),
    }
    model = DenseKP(cfg).to(device)
    if args.full_model:
        model.train()
        optimizer = model.configure_optimizers()
        parameter = next(model.backbone.parameters())
    else:
        model.backbone.requires_grad_(False)
        model.backbone.eval()
        model.head.train()
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=cfg.TRAIN.LR)
        parameter = next(model.head.parameters())
    initial = parameter.detach().clone()
    losses = []
    for _ in range(args.steps):
        optimizer.zero_grad()
        output = model.forward_step(batch, train=True)
        loss, _ = model.compute_loss(batch, output, train=True)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    change = float((parameter.detach() - initial).abs().max())
    if change <= 0:
        raise RuntimeError("checked model parameters did not update")
    report = {
        "mode": "full-model optimization" if args.full_model else "head-only optimization; pretrained backbone frozen",
        "device": str(device),
        "train_samples": len(train), "val_samples": len(val),
        "steps": args.steps, "losses": losses, "max_checked_parameter_change": change,
        "weight_path": str(ROOT / "assets_external/vitpose_backbone.pth"),
    }
    if device.type == "cuda":
        report["max_cuda_memory_bytes"] = torch.cuda.max_memory_allocated(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
