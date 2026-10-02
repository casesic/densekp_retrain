"""Pure box calculations with explicit historical policies.

Coordinates are source-raster pixels. change_box retains the synthetic trial's
1536 x 768 clipping; official_bbox retains BEDLAM's joint-based inclusive gate.
Neither function samples random numbers or chooses experiment quotas.
"""

import numpy as np


def change_box(box, dx=0.0, dy=0.0, factor=1.0, edge=None, fraction=0.0):
    box = np.asarray(box, float).copy()
    center = (box[:2] + box[2:]) / 2
    size = box[2:] - box[:2]
    center += np.array([dx, dy]) * size.max() * 1.2
    box = np.r_[center - size * factor / 2, center + size * factor / 2]
    if edge is not None:
        axis = edge % 2
        box[edge] += (1 if edge < 2 else -1) * size[axis] * fraction
    # Match detector clipping to the available raster; synthetic images do not wrap at their boxes.
    box[[0, 2]] = np.clip(box[[0, 2]], 0, 1536)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, 768)
    return box


def official_bbox(joints2d: np.ndarray, image_height: int, image_width: int):
    """复现 BEDLAM 主体 bbox：前 22 joints、图内过滤、1.2 扩张。"""
    body = np.asarray(joints2d, dtype=np.float64)[:22]
    finite = np.isfinite(body[:, :2]).all(axis=1)
    valid = finite & (body[:, 0] >= 0) & (body[:, 0] <= image_width)
    valid &= (body[:, 1] >= 0) & (body[:, 1] <= image_height)
    selected = body[valid]
    if len(selected) < 8:
        return None
    xmin, ymin = selected[:, :2].min(axis=0)
    xmax, ymax = selected[:, :2].max(axis=0)
    tight_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)
    scale = np.float32(max(xmax - xmin, ymax - ymin) / 200.0 * 1.2)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return center, scale, np.int32(len(selected)), tight_bbox


def box_rejection(box):
    """Trial detector gate, after the caller's float32 conversion."""
    if not np.isfinite(box).all() or min(box[2:] - box[:2]) < 2:
        return "invalid_box"
    return None


def trial_crop_from_box(box):
    """Float32 trial xyxy -> center and scale (pixel side / 200).

    Caller converts xyxy to float32 before the rejection gate, as historically.
    Keep multiplication before division to retain the original rounding.
    """
    center = (box[:2] + box[2:]) / 2
    scale = np.float32(max(box[2:] - box[:2]) * 1.2 / 200)
    return center, scale


def support_rejection(supervision, target_pixels, min_points, min_pixels):
    """The experiment supplies its thresholds; visibility never gates GT."""
    if supervision.sum() < min_points or target_pixels.sum() < min_pixels:
        return "empty_or_insufficient_GT_or_visible_target_support"
    return None
