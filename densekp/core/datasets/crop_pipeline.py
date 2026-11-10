"""Named stages connecting source images, boxes and DenseKP crop coordinates."""

from dataclasses import dataclass
import cv2
import numpy as np

from .augmentation import (
    Augmentation,
    sample_augmentation,
    extreme_cropping,
    extreme_cropping_aggressive,
)
from . import image_processing as image_ops
from .image_layout import warp_roi_image


@dataclass(frozen=True)
class CropBox:
    """Center and width/height in source-image pixels, before augmentation scale."""

    center_x: float
    center_y: float
    width: float
    height: float


@dataclass(frozen=True)
class CropPlan:
    box: CropBox  # extreme crop and translation applied; scale remains separate
    augmentation: Augmentation


@dataclass
class CropResult:
    """Image and GT share affine [2,3]; XY is pixel / full-patch size - .5.

    Labels are not clipped: points outside the crop can exceed [-.5,.5).

    image is float32 CHW; display is the original warped HWC BGR pixels.
    Auxiliary/dense arrays are transformed in place, retaining their third value.
    """

    image: np.ndarray
    display: np.ndarray
    image_size: np.ndarray
    plan: CropPlan
    keypoints: np.ndarray
    dense_points: np.ndarray | None
    affine: np.ndarray

def check_flip_augmentation(do_augment, augm_config):
    """Reject unverified semantics before image IO or random sampling."""
    if do_augment and augm_config.DO_FLIP and augm_config.FLIP_AUG_RATE > 0:
        raise ValueError("DenseKP flipping requires a verified 138-point semantic permutation")


def determine_crop(box, keypoints, do_augment, augm_config):
    """Sample, validate, optionally restrict the body, then translate the center."""
    if do_augment:
        augmentation = sample_augmentation(augm_config)
    else:
        augmentation = Augmentation(1.0, 0, False, False, 0, [1.0, 1.0, 1.0], 0.0, 0.0)
    center_x, center_y, width, height = box.center_x, box.center_y, box.width, box.height
    scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, _, tx, ty = augmentation
    if (
        not np.isfinite([center_x, center_y, width, height, scale, rot]).all()
        or min(width, height, scale) <= 0
    ):
        raise ValueError(
            "Crop center, size and augmentation must be finite with positive size/scale"
        )
    if do_flip:
        raise ValueError("DenseKP flipping requires a verified 138-point semantic permutation")
    if do_extreme_crop:
        if extreme_crop_lvl == 0:
            center_x1, center_y1, width1, height1 = extreme_cropping(
                center_x, center_y, width, height, keypoints
            )
        elif extreme_crop_lvl == 1:
            center_x1, center_y1, width1, height1 = extreme_cropping_aggressive(
                center_x, center_y, width, height, keypoints
            )
        if width1 >= 4 and height1 >= 4:
            center_x, center_y, width, height = center_x1, center_y1, width1, height1
    center_x += width * tx
    center_y += height * ty
    return CropPlan(CropBox(center_x, center_y, width, height), augmentation)


def transform_crop(
    image,
    plan,
    keypoints,
    dense_points,
    *,
    patch_size,
    mean=None,
    std=None,
    do_augment=False,
    augm_config=None,
    is_bgr=True,
    use_skimage_antialias=False,
    border_mode=cv2.BORDER_CONSTANT,
    image_layout=None,
):
    """Preprocess → warp → normalize image → transform labels, in that order."""
    box, augmentation = plan.box, plan.augmentation
    patch_width, patch_height = patch_size
    if image_layout is not None and tuple(image.size_hw[::-1]) != image_layout.stored_size_wh:
        raise ValueError("stored_image_size_conflict")
    if image_layout is not None and image_layout.kind == "roi_image":
        display, affine = warp_roi_image(image.pixels, plan, patch_size, image_layout,
            use_skimage_antialias=use_skimage_antialias, do_augment=do_augment,
            augm_config=augm_config, border_mode=border_mode)
    else:
        pixels = image_ops.preprocess_image(
            image.pixels, box, augmentation, patch_size, use_skimage_antialias, do_augment, augm_config
        )
        display, affine = image_ops.generate_image_patch_cv2(
            pixels,
            box.center_x,
            box.center_y,
            box.width,
            box.height,
            patch_width,
            patch_height,
            augmentation.flip,
            augmentation.scale,
            augmentation.rotation,
            border_mode=border_mode,
        )
    normalized = image_ops.normalize_image(
        display, image.channels, augmentation.color_scale, mean, std, is_bgr
    )
    image_ops.transform_labels(keypoints, affine, patch_size)
    if dense_points is not None:
        image_ops.transform_labels(dense_points, affine, patch_size)
    size_hw = np.array(image_layout.full_size_wh[::-1]) if image_layout is not None else image.size_hw
    return CropResult(normalized, display, size_hw, plan, keypoints, dense_points, affine)


def crop_example(
    img_path,
    box,
    keypoints,
    dense_points,
    *,
    patch_size,
    mean=None,
    std=None,
    do_augment=False,
    augm_config=None,
    is_bgr=True,
    use_skimage_antialias=False,
    border_mode=cv2.BORDER_CONSTANT,
    dataset=None,
):
    """Named replacement for the legacy ten-value crop interface."""
    check_flip_augmentation(do_augment, augm_config)
    image = image_ops.read_image(img_path, dataset)
    plan = determine_crop(box, keypoints, do_augment, augm_config)
    return transform_crop(
        image,
        plan,
        keypoints,
        dense_points,
        patch_size=patch_size,
        mean=mean,
        std=std,
        do_augment=do_augment,
        augm_config=augm_config,
        is_bgr=is_bgr,
        use_skimage_antialias=use_skimage_antialias,
        border_mode=border_mode,
    )
