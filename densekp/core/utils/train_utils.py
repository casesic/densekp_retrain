"""Compatibility exports for historical training/camera helpers."""

from densekp.core.image_normalization import denormalize_images
from densekp.legacy.train_utils import (
    trans_points2d_parallel,
    strip_prefix_if_present,
    load_valid,
    perspective_projection,
    convert_to_full_img_cam,
)
