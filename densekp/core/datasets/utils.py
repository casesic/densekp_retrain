"""Compatibility imports and tuple adapters; Dataset uses crop_pipeline directly.

Historical SMPL/HMR variants live in densekp.legacy.crop. No active Dataset
implementation imports this compatibility module.
"""

import torch
import numpy as np
from skimage.transform import rotate, resize
from skimage.filters import gaussian
import random
import cv2
from typing import List, Dict, Tuple
from yacs.config import CfgNode
from ..crop_geometry import (
    rotate_2d,
    gen_trans_from_patch_cv,
    trans_point2d,
    trans_points2d_parallel,
    normalize_crop_xy,
)


from densekp.legacy.crop import (
    resize_image,
    get_transform,
    transform,
    crop_img,
    generate_image_patch_skimage,
    fliplr_params,
    fliplr_keypoints,
    keypoint_3d_processing,
    rot_aa,
    get_example,
    do_augmentation,
)

# Historical public imports; new Dataset code imports the defining modules.
from .image_processing import (
    expand_to_aspect_ratio,
    generate_image_patch_cv2,
    convert_cvimg_to_tensor,
)
from .augmentation import (
    crop_to_hips,
    crop_to_shoulders,
    crop_to_head,
    crop_torso_only,
    crop_rightarm_only,
    crop_leftarm_only,
    crop_legs_only,
    crop_rightleg_only,
    crop_leftleg_only,
    full_body,
    upper_body,
    get_bbox,
    extreme_cropping,
    extreme_cropping_aggressive,
)
from .augmentation import sample_augmentation
from .crop_pipeline import crop_example, CropBox


def get_example_projverts(
    img_path: str | np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    keypoints_2d: np.array,
    flip_kp_permutation: List[int],
    proj_verts: np.array = None,
    patch_width: int = 0,
    patch_height: int = 0,
    mean: np.array = None,
    std: np.array = None,
    do_augment: bool = False,
    augm_config: CfgNode = None,
    is_bgr: bool = True,
    use_skimage_antialias: bool = False,
    border_mode: int = cv2.BORDER_CONSTANT,
    return_trans: bool = False,
    dataset: str = None,
):
    """Historical ten-value tuple; flip permutation/return_trans remain unused."""
    result = crop_example(
        img_path,
        CropBox(center_x, center_y, width, height),
        keypoints_2d,
        proj_verts,
        patch_size=(patch_width, patch_height),
        mean=mean,
        std=std,
        do_augment=do_augment,
        augm_config=augm_config,
        is_bgr=is_bgr,
        use_skimage_antialias=use_skimage_antialias,
        border_mode=border_mode,
        dataset=dataset,
    )
    return result.legacy_tuple()
