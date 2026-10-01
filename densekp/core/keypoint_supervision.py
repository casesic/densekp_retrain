"""Eligibility for synthetic 138-point labels; visibility is separate metadata."""
import numpy as np

from .crop_geometry import gen_trans_from_patch_cv, trans_points2d_parallel


SUPERVISION_POLICY = 'in_view_full_crop'


def inside_full_crop(normalized_xy):
    """The full crop is [-0.5, 0.5), including the strips outside the backbone."""
    xy = np.asarray(normalized_xy)
    return np.isfinite(xy).all(axis=-1) & ((xy >= -.5) & (xy < .5)).all(axis=-1)


def points_in_mask(xy, mask):
    """Finite, in-bounds pixels in a binary lens mask; match delivery rounding."""
    xy = np.asarray(xy)
    mask = np.asarray(mask, dtype=bool)
    if xy.shape[-1] != 2 or mask.ndim != 2:
        raise ValueError('Expected xy [...,2] and mask [H,W]')
    h, w = mask.shape
    valid = np.isfinite(xy).all(axis=-1)
    valid &= (xy[..., 0] >= 0) & (xy[..., 0] < w) & (xy[..., 1] >= 0) & (xy[..., 1] < h)
    pixels = np.rint(np.where(valid[..., None], xy, 0)).astype(np.int64)
    valid &= (pixels[..., 0] < w) & (pixels[..., 1] < h)
    return valid & mask[np.clip(pixels[..., 1], 0, h-1), np.clip(pixels[..., 0], 0, w-1)]


def make_targets(xy, in_view, center, scale):
    """Store original xy plus supervision for the unaugmented square crop.

    The loader recomputes eligibility after augmentation from saved in_view.
    Occlusion never removes reliable synthetic coordinates from training.
    """
    xy = np.asarray(xy, dtype=np.float32)
    center = np.asarray(center, dtype=np.float32)
    size = float(np.float32(scale) * 200)
    affine = gen_trans_from_patch_cv(*center, size, size, 256, 256, 1., 0.)
    crop_xy = (trans_points2d_parallel(xy, affine) / 256 - .5).astype(np.float32)
    supervision = np.asarray(in_view, dtype=bool) & inside_full_crop(crop_xy)
    return np.column_stack((xy, supervision)).astype(np.float32)
