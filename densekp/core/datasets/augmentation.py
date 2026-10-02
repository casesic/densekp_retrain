"""Crop augmentation sampling and historical auxiliary-joint extreme crops.

Python, NumPy and Torch RNG calls keep their original order. These rules use
44 auxiliary joints, not the 138 DenseKP semantic rows.
"""

from typing import NamedTuple, List, Tuple
import random
import numpy as np
import torch
from yacs.config import CfgNode


class Augmentation(NamedTuple):
    scale: float
    rotation: float  # degrees
    flip: bool
    extreme_crop: bool
    extreme_crop_level: int
    color_scale: list
    translation_x: float  # fraction of the selected crop width
    translation_y: float  # fraction of the selected crop height


def sample_augmentation(aug_config: CfgNode) -> Augmentation:
    """
    Compute random augmentation parameters.
    Args:
        aug_config (CfgNode): Config containing augmentation parameters.
    Returns:
        scale (float): Box rescaling factor.
        rot (float): Random image rotation.
        do_flip (bool): Whether to flip image or not.
        do_extreme_crop (bool): Whether to apply extreme cropping (as proposed in EFT).
        color_scale (List): Color rescaling factor
        tx (float): Random translation along the x axis.
        ty (float): Random translation along the y axis.
    """

    tx = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    ty = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.TRANS_FACTOR
    if random.random() >= aug_config.TRANS_AUG_RATE:
        tx = ty = 0.0
    scale = np.clip(np.random.randn(), -1.0, 1.0) * aug_config.SCALE_FACTOR + 1.0

    rot = (
        np.clip(np.random.randn(), -2.0, 2.0) * aug_config.ROT_FACTOR
        if random.random() <= aug_config.ROT_AUG_RATE
        else 0
    )
    do_flip = aug_config.DO_FLIP and random.random() <= aug_config.FLIP_AUG_RATE
    do_extreme_crop = random.random() <= aug_config.EXTREME_CROP_AUG_RATE
    extreme_crop_lvl = aug_config.get("EXTREME_CROP_AUG_LEVEL", 0)
    # extreme_crop_lvl = 0
    c_up = 1.0 + aug_config.COLOR_SCALE
    c_low = 1.0 - aug_config.COLOR_SCALE
    color_scale = [
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up),
        random.uniform(c_low, c_up),
    ]
    return Augmentation(scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty)


def crop_to_hips(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
) -> Tuple:
    """
    Extreme cropping: Crop the box up to the hip locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [10, 11, 13, 14, 19, 20, 21, 22, 23, 24, 25 + 0, 25 + 1, 25 + 4, 25 + 5]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_to_shoulders(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box up to the shoulder locations.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [
        25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 10, 11, 14, 15, 16]
    ]
    keypoints_2d[lower_body_keypoints, :] = 0
    center, scale = get_bbox(keypoints_2d)
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.2 * scale[0]
        height = 1.2 * scale[1]
    return center_x, center_y, width, height


def crop_to_head(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box and keep on only the head.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    lower_body_keypoints = [3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24] + [
        25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 14, 15, 16]
    ]
    keypoints_2d[lower_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.3 * scale[0]
        height = 1.3 * scale[1]
    return center_x, center_y, width, height


def crop_torso_only(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box and keep on only the torso.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nontorso_body_keypoints = [
        0,
        3,
        4,
        6,
        7,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ] + [25 + i for i in [0, 1, 4, 5, 6, 7, 10, 11, 13, 17, 18]]
    keypoints_2d[nontorso_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_rightarm_only(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box and keep on only the right arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightarm_body_keypoints = [
        0,
        1,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ] + [25 + i for i in [0, 1, 2, 3, 4, 5, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_leftarm_only(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box and keep on only the left arm.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftarm_body_keypoints = [
        0,
        1,
        2,
        3,
        4,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ] + [25 + i for i in [0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftarm_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_legs_only(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box and keep on only the legs.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonlegs_body_keypoints = [0, 1, 2, 3, 4, 5, 6, 7, 15, 16, 17, 18] + [
        25 + i for i in [6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 18]
    ]
    keypoints_2d[nonlegs_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_rightleg_only(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box and keep on only the right leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonrightleg_body_keypoints = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
    ] + [25 + i for i in [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonrightleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def crop_leftleg_only(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
):
    """
    Extreme cropping: Crop the box and keep on only the left leg.
    Args:
        center_x (float): x coordinate of the bounding box center.
        center_y (float): y coordinate of the bounding box center.
        width (float): Bounding box width.
        height (float): Bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        center_x (float): x coordinate of the new bounding box center.
        center_y (float): y coordinate of the new bounding box center.
        width (float): New bounding box width.
        height (float): New bounding box height.
    """
    keypoints_2d = keypoints_2d.copy()
    nonleftleg_body_keypoints = [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        15,
        16,
        17,
        18,
        22,
        23,
        24,
    ] + [25 + i for i in [0, 1, 2, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]]
    keypoints_2d[nonleftleg_body_keypoints, :] = 0
    if keypoints_2d[:, -1].sum() > 1:
        center, scale = get_bbox(keypoints_2d)
        center_x = center[0]
        center_y = center[1]
        width = 1.1 * scale[0]
        height = 1.1 * scale[1]
    return center_x, center_y, width, height


def full_body(keypoints_2d: np.array) -> bool:
    """
    Check if all main body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """

    body_keypoints_openpose = [2, 3, 4, 5, 6, 7, 10, 11, 13, 14]
    body_keypoints = [25 + i for i in [8, 7, 6, 9, 10, 11, 1, 0, 4, 5]]
    return (
        np.maximum(keypoints_2d[body_keypoints, -1], keypoints_2d[body_keypoints_openpose, -1]) > 0
    ).sum() == len(body_keypoints)


def upper_body(keypoints_2d: np.array):
    """
    Check if all upper body joints are visible.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
    Returns:
        bool: True if all main body joints are visible.
    """
    lower_body_keypoints_openpose = [10, 11, 13, 14]
    lower_body_keypoints = [25 + i for i in [1, 0, 4, 5]]
    upper_body_keypoints_openpose = [0, 1, 15, 16, 17, 18]
    upper_body_keypoints = [25 + 8, 25 + 9, 25 + 12, 25 + 13, 25 + 17, 25 + 18]
    return (
        (keypoints_2d[lower_body_keypoints + lower_body_keypoints_openpose, -1] > 0).sum() == 0
    ) and ((keypoints_2d[upper_body_keypoints + upper_body_keypoints_openpose, -1] > 0).sum() >= 2)


def get_bbox(keypoints_2d: np.array, rescale: float = 1.2) -> Tuple:
    """
    Get center and scale for bounding box from openpose detections.
    Args:
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center (np.array): Array of shape (2,) containing the new bounding box center.
        scale (float): New bounding box scale.
    """
    valid = keypoints_2d[:, -1] > 0
    valid_keypoints = keypoints_2d[valid][:, :-1]
    center = 0.5 * (valid_keypoints.max(axis=0) + valid_keypoints.min(axis=0))
    bbox_size = valid_keypoints.max(axis=0) - valid_keypoints.min(axis=0)
    # adjust bounding box tightness
    scale = bbox_size
    scale *= rescale
    return center, scale


def extreme_cropping(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
) -> Tuple:
    """
    Perform extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.7:
            center_x, center_y, width, height = crop_to_hips(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(
                center_x, center_y, width, height, keypoints_2d
            )
        else:
            center_x, center_y, width, height = crop_to_head(
                center_x, center_y, width, height, keypoints_2d
            )
    elif upper_body(keypoints_2d):
        if p < 0.9:
            center_x, center_y, width, height = crop_to_shoulders(
                center_x, center_y, width, height, keypoints_2d
            )
        else:
            center_x, center_y, width, height = crop_to_head(
                center_x, center_y, width, height, keypoints_2d
            )

    return center_x, center_y, max(width, height), max(width, height)


def extreme_cropping_aggressive(
    center_x: float, center_y: float, width: float, height: float, keypoints_2d: np.array
) -> Tuple:
    """
    Perform aggressive extreme cropping
    Args:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
        keypoints_2d (np.array): Array of shape (N, 3) containing 2D keypoint locations.
        rescale (float): Scale factor to rescale bounding boxes computed from the keypoints.
    Returns:
        center_x (float): x coordinate of bounding box center.
        center_y (float): y coordinate of bounding box center.
        width (float): bounding box width.
        height (float): bounding box height.
    """
    p = torch.rand(1).item()
    if full_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_hips(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.3:
            center_x, center_y, width, height = crop_to_shoulders(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.5:
            center_x, center_y, width, height = crop_torso_only(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.6:
            center_x, center_y, width, height = crop_rightarm_only(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.7:
            center_x, center_y, width, height = crop_leftarm_only(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.8:
            center_x, center_y, width, height = crop_legs_only(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.9:
            center_x, center_y, width, height = crop_rightleg_only(
                center_x, center_y, width, height, keypoints_2d
            )
        else:
            center_x, center_y, width, height = crop_leftleg_only(
                center_x, center_y, width, height, keypoints_2d
            )
    elif upper_body(keypoints_2d):
        if p < 0.2:
            center_x, center_y, width, height = crop_to_shoulders(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.4:
            center_x, center_y, width, height = crop_to_head(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.6:
            center_x, center_y, width, height = crop_torso_only(
                center_x, center_y, width, height, keypoints_2d
            )
        elif p < 0.8:
            center_x, center_y, width, height = crop_rightarm_only(
                center_x, center_y, width, height, keypoints_2d
            )
        else:
            center_x, center_y, width, height = crop_leftarm_only(
                center_x, center_y, width, height, keypoints_2d
            )
    return center_x, center_y, max(width, height), max(width, height)
