"""Deterministic small images/labels for independent Dataset and crop baselines."""

from contextlib import ExitStack
import copy
from pathlib import Path
import random
from unittest.mock import patch

import cv2
import numpy as np
import torch

from densekp.core.configs import DATASET_FILES, DATASET_FOLDERS
from densekp.core.datasets.configuration import audit_cfg
from densekp.core.keypoint_supervision import SUPERVISION_POLICY


def make_inputs(directory):
    directory.mkdir(parents=True, exist_ok=False)
    y, x = np.indices((96, 128))
    image = np.stack(((x * 3 + y) % 256, (y * 5) % 256, (x + y * 7) % 256), axis=-1).astype(
        np.uint8
    )
    cv2.imwrite(str(directory / "image.png"), image)
    xy = np.column_stack((np.linspace(-8, 140, 138), np.linspace(4, 90, 138))).astype(np.float32)
    points = np.column_stack((xy, np.ones(138, np.float32)))
    auxiliary = np.column_stack(
        (np.linspace(25, 100, 44), np.linspace(15, 80, 44), np.ones(44))
    ).astype(np.float32)
    in_view = np.arange(138) % 5 != 0
    visible = in_view & (np.arange(138) % 3 == 0)
    common = dict(
        center=np.array([[62.25, 45.5]], np.float32),
        scale=np.array([0.48], np.float32),
        proj_verts=points[None],
        part=auxiliary[None],
    )
    policy = dict(
        supervision_policy=SUPERVISION_POLICY, in_view=in_view[None], visible=visible[None]
    )
    for name, image_name in [
        ("relative", "image.png"),
        ("absolute", str((directory / "image.png").resolve())),
    ]:
        np.savez(directory / (name + ".npz"), **common, **policy, imgname=[image_name])
    np.savez(directory / "legacy.npz", **common, imgname=["image.png"])
    common["scale"] = np.array([[0.38, 0.48]], np.float32)
    np.savez(directory / "aspect.npz", **common, **policy, imgname=["image.png"])


def specifications(directory):
    common = dict(
        annotation_path=str(directory / "relative.npz"),
        image_root=str(directory),
        name="crop-fixture",
        row=0,
        image_size=32,
        seed=7,
        augment=False,
        settings={},
    )
    cases = []

    def add(label, **kwargs):
        cases.append(dict(common, id=label, **kwargs))

    add("relative")
    add("absolute", annotation_path=str(directory / "absolute.npz"), image_root="/unused-root")
    add("portrait", name="fixture-portrait")
    add("closeup", name="fixture-closeup")
    add("legacy_weights", annotation_path=str(directory / "legacy.npz"))
    add("aspect", annotation_path=str(directory / "aspect.npz"), aspect=[192, 256])
    add("translation", augment=True, settings=dict(TRANS_FACTOR=0.2, TRANS_AUG_RATE=1.0))
    add("scale", augment=True, settings=dict(SCALE_FACTOR=0.3))
    add("rotation", augment=True, settings=dict(ROT_FACTOR=30.0, ROT_AUG_RATE=1.0))
    add("color", augment=True, settings=dict(COLOR_SCALE=0.3))
    for seed in (1, 22):
        add(
            "mixed_" + str(seed),
            augment=True,
            seed=seed,
            settings=dict(
                TRANS_FACTOR=0.2,
                TRANS_AUG_RATE=0.5,
                SCALE_FACTOR=0.3,
                ROT_FACTOR=30.0,
                ROT_AUG_RATE=0.6,
                COLOR_SCALE=0.3,
            ),
        )
    for seed in (0, 1, 19):
        add(
            "extreme_" + str(seed),
            augment=True,
            seed=seed,
            settings=dict(EXTREME_CROP_AUG_RATE=1.0),
        )
    # Fixed seeds cover seven aggressive branches; head/shoulder helpers are covered above.
    for seed in (0, 1, 2, 3, 4, 5, 9, 19, 26, 28):
        add(
            "aggressive_" + str(seed),
            augment=True,
            seed=seed,
            settings=dict(EXTREME_CROP_AUG_RATE=1.0, EXTREME_CROP_AUG_LEVEL=1),
        )
    add("antialias", antialias=True, augment=True, settings=dict(SCALE_FACTOR=0.2, COLOR_SCALE=0.1))
    add(
        "replicate",
        border="replicate",
        augment=True,
        settings=dict(TRANS_FACTOR=1.0, TRANS_AUG_RATE=1.0),
    )
    for seed in (1, 7, 22):
        add(
            "alb_" + str(seed),
            augment=True,
            seed=seed,
            settings=dict(
                USE_ALB=True,
                ALB_PROB=1.0,
                SCALE_FACTOR=0.1,
                ROT_FACTOR=15.0,
                ROT_AUG_RATE=1.0,
                COLOR_SCALE=0.2,
            ),
        )
    add("flip_rejected", augment=True, settings=dict(DO_FLIP=True, FLIP_AUG_RATE=1.0))
    return cases


def configuration(spec):
    cfg = audit_cfg("supervision_weight")
    cfg.MODEL.IMAGE_SIZE = spec.get("image_size", 256)
    if spec.get("aspect"):
        cfg.MODEL.BBOX_SHAPE = spec["aspect"]
    cfg.DATASETS.USE_SKIMAGE_ANTIALIAS = spec.get("antialias", False)
    cfg.DATASETS.BORDER_MODE = spec.get("border", "constant")
    for key, value in spec.get("settings", {}).items():
        setattr(cfg.DATASETS.CONFIG, key, value)
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def rng_state(prefix, arrays):
    python = random.getstate()
    numpy = np.random.get_state()
    arrays[prefix + "_python_rng"] = np.asarray(python[1], np.uint64)
    arrays[prefix + "_numpy_rng"] = numpy[1].copy()
    arrays[prefix + "_torch_rng"] = torch.get_rng_state().numpy().copy()
    return dict(
        python_version=python[0],
        python_gauss=python[2],
        numpy_kind=numpy[0],
        numpy_position=numpy[2],
        numpy_has_gauss=numpy[3],
        numpy_cached_gauss=numpy[4],
    )


def run_dataset(spec, *, explicit=False, original=False):
    if original:
        raise RuntimeError("original Dataset capture retired in B2b; use the saved independent baseline")

    from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
    from densekp.core.datasets import crop_pipeline as sampling_module
    from densekp.core.datasets import image_processing as image_module

    sampling_name = "sample_augmentation"
    arrays, trace = {}, []
    sampler = getattr(sampling_module, sampling_name)
    warp = image_module.generate_image_patch_cv2

    def sample(config):
        values = sampler(config)
        for index, value in enumerate(values):
            arrays["augmentation_" + str(index)] = np.asarray(value)
        trace.append("sample_augmentation")
        return values

    def crop(*args, **kwargs):
        for name, value in zip(
            (
                "center_x",
                "center_y",
                "width",
                "height",
                "patch_width",
                "patch_height",
                "flip",
                "scale",
                "rotation",
            ),
            args[1:],
        ):
            arrays["crop_" + name] = np.asarray(value)
        trace.append("warp_image")
        return warp(*args, **kwargs)

    cfg = configuration(spec)
    seed_all(spec["seed"])
    before = rng_state("before", arrays)
    error, item = None, {}
    with ExitStack() as stack:
        stack.enter_context(patch.object(sampling_module, sampling_name, sample))
        stack.enter_context(patch.object(image_module, "generate_image_patch_cv2", crop))
        kwargs = dict(is_train=True, use_augmentation=spec["augment"])
        if explicit:
            kwargs.update(
                annotation_path=Path(spec["annotation_path"]), image_root=Path(spec["image_root"])
            )
        else:
            stack.enter_context(patch.dict(DATASET_FOLDERS, {spec["name"]: spec["image_root"]}))
            stack.enter_context(
                patch.dict(DATASET_FILES[True], {spec["name"]: spec["annotation_path"]})
            )
        try:
            dataset = DatasetKeypoints(cfg, spec["name"], **kwargs)
            item = dataset[spec["row"]]
        except Exception as failure:
            error = dict(type=type(failure).__name__, message=str(failure))
    metadata = dict(
        error=error,
        trace=trace,
        rng_before=before,
        rng_after=rng_state("after", arrays),
        item_keys=list(item),
        values={},
        types={},
    )
    for name, value in item.items():
        metadata["types"][name] = type(value).__name__
        if isinstance(value, (np.ndarray, np.generic)):
            arrays["item_" + name] = np.asarray(value)
        else:
            metadata["values"][name] = value
    return metadata, arrays


def lowlevel_specs():
    return [
        dict(id=name, seed=41)
        for name in (
            "array_rgb",
            "array_rgba_replicate",
            "float_image",
            "no_normalization",
            "no_dense_targets",
            "return_trans_flag",
            "augmented",
            "invalid_width",
            "invalid_type",
            "missing_image",
            "flip_before_read",
        )
    ]


def run_lowlevel(spec, directory):
    from densekp.core.datasets.crop_pipeline import CropBox, crop_example

    cfg = audit_cfg("supervision_weight").DATASETS.CONFIG
    image = cv2.imread(str(directory / "image.png"))
    keypoints = np.column_stack(
        (np.linspace(10, 100, 44), np.linspace(20, 80, 44), np.ones(44))
    ).astype(np.float32)
    points = np.tile(np.array([62.0, 45.0, 1.0], np.float32), (138, 1))
    options = dict(
        patch_size=(17, 13),
        mean=np.array([100.0, 110.0, 120.0]),
        std=np.array([50.0, 60.0, 70.0]),
        is_bgr=False,
        border_mode=cv2.BORDER_CONSTANT,
    )
    width, do_augment = 120.0, False
    name = spec["id"]
    if name == "array_rgba_replicate":
        image = np.concatenate((image, image[:, :, :1]), axis=2)
        options.update(border_mode=cv2.BORDER_REPLICATE, is_bgr=True)
    elif name == "float_image":
        image = image.astype(np.float32) / 2
        options["use_skimage_antialias"] = True
    elif name == "no_normalization":
        options.update(mean=None, std=None)
    elif name == "no_dense_targets":
        points = None
    elif name == "return_trans_flag":
        # The retired wrapper ignored this flag. Keep the original fixture case;
        # the named result always includes affine without a flag.
        pass
    elif name in ("augmented", "invalid_width"):
        do_augment = True
        cfg.SCALE_FACTOR = 0.3
        cfg.ROT_FACTOR = 35.0
        cfg.ROT_AUG_RATE = 1.0
        cfg.TRANS_FACTOR = 0.2
        cfg.TRANS_AUG_RATE = 1.0
        cfg.COLOR_SCALE = 0.1
        if name == "invalid_width":
            width = 0.0
    elif name == "invalid_type":
        image = Path("image.png")  # Historical low-level interface rejects Path.
    elif name == "missing_image":
        image = str(directory / "missing.png")
    elif name == "flip_before_read":
        image = str(directory / "missing.png")
        do_augment = True
        cfg.DO_FLIP = True
        cfg.FLIP_AUG_RATE = 1.0
    arrays = {}
    seed_all(spec["seed"])
    before = rng_state("before", arrays)
    error, result = None, ()
    try:
        crop = crop_example(
            image,
            CropBox(62.25, 45.5, width, 100.0),
            keypoints,
            points,
            do_augment=do_augment,
            augm_config=cfg,
            **options,
        )
        # Test-only encoding into the original frozen result_N layout. Do not
        # regenerate the fixture or add this compatibility tuple to runtime code.
        box = crop.plan.box
        result = (
            crop.image, crop.display, crop.image_size,
            box.center_x, box.center_y, box.width, box.height,
            crop.keypoints, crop.dense_points, crop.affine,
        )
    except Exception as failure:
        error = dict(type=type(failure).__name__, message=str(failure))
    arrays["mutated_keypoints"] = keypoints
    if points is not None:
        arrays["mutated_dense_points"] = points
    metadata = dict(
        error=error,
        types=[type(value).__name__ for value in result],
        none_indices=[],
        rng_before=before,
        rng_after=rng_state("after", arrays),
    )
    for index, value in enumerate(result):
        if value is None:
            metadata["none_indices"].append(index)
        else:
            arrays["result_" + str(index)] = np.asarray(value)
    return metadata, arrays
