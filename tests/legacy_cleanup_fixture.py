"""Small independent legacy behavior capture; no model or optimizer updates."""

import base64
import copy
import json
import pickle
import random
from collections import OrderedDict
from pathlib import Path
import zlib

import cv2
import numpy as np
import torch


def encode(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return dict(
            dtype=str(value.dtype),
            shape=list(value.shape),
            bytes=base64.b64encode(zlib.compress(value.tobytes())).decode(),
        )
    if isinstance(value, (tuple, list)):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, np.generic):
        return value.item()
    return value


def capture(folder, crop, training, losses):
    from densekp.core.datasets.configuration import audit_cfg

    folder.mkdir(parents=True, exist_ok=False)
    rows = {}
    y, x = np.indices((80, 96))
    img = np.stack(((x + y) % 256, x * 2 % 256, y * 3 % 256), axis=-1).astype(np.uint8)
    points = np.column_stack(
        (np.linspace(5, 85, 44), np.linspace(7, 65, 44), np.ones(44))
    ).astype(np.float32)
    for augment in [False, True]:
        for return_trans in [False, True]:
            random.seed(17)
            np.random.seed(17)
            torch.manual_seed(17)
            cfg = audit_cfg("supervision_weight").DATASETS.CONFIG
            cfg.SCALE_FACTOR = 0.2
            cfg.ROT_FACTOR = 20
            cfg.ROT_AUG_RATE = 1.0
            cfg.TRANS_FACTOR = 0.1
            cfg.TRANS_AUG_RATE = 1.0
            cfg.COLOR_SCALE = 0.1
            value = crop.get_example(
                img.copy(),
                48.0,
                40.0,
                72.0,
                72.0,
                points.copy(),
                list(range(44)),
                32,
                32,
                np.array([123.0, 116.0, 103.0]),
                np.array([58.0, 57.0, 57.0]),
                augment,
                cfg,
                return_trans=return_trans,
            )
            rows[f"get_example_{augment}_{return_trans}"] = encode(value)
            rows[f"rng_{augment}_{return_trans}"] = encode(
                [random.getstate(), np.random.get_state(), torch.get_rng_state()]
            )
    rows["resize"] = encode(crop.resize_image(img, 32))
    for rot in [0, 20]:
        rows[f"skimage_{rot}"] = encode(
            crop.generate_image_patch_skimage(
                img.copy(), 48, 40, 64, 64, 32, 32, False, 1, rot
            )
        )
    try:
        crop.generate_image_patch_skimage(img, 48, 40, 60, 64, 32, 32, False, 1, 0)
    except Exception as e:
        rows["skimage_rejection"] = [type(e).__name__, str(e)]
    rows["flipped_points"] = encode(
        crop.fliplr_keypoints(points.copy(), 96, list(reversed(range(44))))
    )
    rows["rotation_aa"] = encode(crop.rot_aa(np.array([0.1, 0.2, 0.3], np.float32), 20))
    keypoints3 = np.ones((44, 4), np.float32)
    keypoints3[:, :3] = np.arange(132).reshape(44, 3) / 100
    rows["keypoints3"] = encode(
        crop.keypoint_3d_processing(keypoints3, list(range(44)), 20, False)
    )
    rows["denormalize"] = encode(
        training.denormalize_images(
            torch.arange(24, dtype=torch.float32).reshape(1, 3, 2, 4) / 10
        )
    )
    rows["tensor_affine"] = encode(
        training.trans_points2d_parallel(
            torch.ones(2, 3, 2),
            torch.tensor([[[1.0, 0, 2], [0, 1, 3]]] * 2, dtype=torch.float64),
        )
    )
    states = OrderedDict(
        [("model.x", torch.ones(1)), ("other.model.y", torch.zeros(2))]
    )
    rows["prefix"] = encode(training.strip_prefix_if_present(states, "model"))
    xyz = torch.tensor([[[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]]])
    rotation = torch.eye(3)[None]
    translation = torch.ones(1, 3)
    intrinsics = torch.eye(3)[None]
    rows["projection"] = encode(
        training.perspective_projection(xyz, rotation, translation, intrinsics)
    )
    rows["full_camera"] = encode(
        training.convert_to_full_img_cam(
            torch.tensor([[1.0, 0.1, 0.2]]),
            torch.tensor([200.0]),
            torch.tensor([[120.0, 90.0]]),
            torch.tensor([320.0]),
            torch.tensor([240.0]),
            torch.tensor([500.0]),
        )
    )
    model = torch.nn.Linear(2, 1)
    model.weight.data.fill_(0.3)
    model.bias.data.fill_(0.2)
    path = folder / "tiny.pt"
    torch.save(
        {
            "state_dict": OrderedDict(
                [
                    ("model.weight", torch.tensor([[0.5, 0.6]])),
                    ("model.bias", torch.tensor([0.4])),
                ]
            )
        },
        path,
    )
    training.load_valid(model, path, skip_list=["weight"])
    rows["load_valid_skip_ignored"] = encode(model.state_dict())
    torch.save({"state_dict": {"model.extra": torch.ones(1)}}, path)
    try:
        training.load_valid(model, path)
    except Exception as e:
        rows["load_valid_rejection"] = [type(e).__name__, str(e)]
    for name in [
        "Keypoint2DLoss",
        "Keypoint2DLossScaled",
        "VerticesLoss",
        "Keypoint3DLoss",
        "ParameterLoss",
        "TranslationLoss",
    ]:
        for kind in (["default"] if name == "ParameterLoss" else ["l1", "l2"]):
            loss = (
                getattr(losses, name)()
                if kind == "default"
                else getattr(losses, name)(kind)
            )
            pred = torch.arange(264, dtype=torch.float32).reshape(2, 44, 3) / 100
            pred.requires_grad_(True)
            gt = (
                torch.ones(2, 44, 4)
                if name.startswith("Keypoint")
                else torch.ones_like(pred)
            )
            args = [pred, gt]
            if name == "Keypoint2DLossScaled":
                args.extend(
                    [
                        torch.tensor([80.0, 100.0]),
                        torch.tensor([[[96.0, 80.0]], [[120.0, 90.0]]]),
                    ]
                )
            value = loss(*args)
            value.backward()
            rows[name + "_" + kind] = encode([value, pred.grad])
            # Keep old class module names for complete-object pickle recovery.
            (folder / (name + "_" + kind + ".pickle")).write_bytes(pickle.dumps(loss))
    rows["class_modules"] = {
        name: getattr(losses, name).__module__
        for name in [
            "Keypoint2DLoss",
            "Keypoint2DLossScaled",
            "VerticesLoss",
            "Keypoint3DLoss",
            "ParameterLoss",
            "TranslationLoss",
        ]
    }
    return rows
