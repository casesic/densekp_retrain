"""Versioned raster storage; all GT and returned affine coordinates stay logical.

This core module has no dependency on the application package. ROI support is
checked after the actual augmentation plan, before any pixel preprocessing.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from ..crop_geometry import gen_trans_from_patch_cv

LAYOUT_FIELDS=('image_layout','image_layout_version','full_image_size_wh','roi_xyxy','offset_xy','stored_size_wh')

@dataclass(frozen=True)
class ImageLayout:
    kind: str
    full_size_wh: tuple
    roi_xyxy: tuple
    offset_xy: tuple
    stored_size_wh: tuple


def _integers(value,shape,name):
    a=np.asarray(value)
    if a.shape!=shape or a.dtype.kind not in 'iu' or (a<0).any():
        raise ValueError('invalid_image_layout_'+name)
    return a.astype(np.int64)


def parse_image_layout(row):
    if not any(k in row for k in LAYOUT_FIELDS):return None
    if not all(k in row for k in LAYOUT_FIELDS):raise ValueError('missing_image_layout_metadata')
    if np.asarray(row['image_layout_version']).shape!=() or np.asarray(row['image_layout_version']).dtype.kind not in 'iu' or row['image_layout_version']!=1:
        raise ValueError('unsupported_image_layout_version')
    kind=str(row['image_layout'])
    if kind not in ('full_image','roi_image'):raise ValueError('unknown_image_layout')
    full=_integers(row['full_image_size_wh'],(2,),'full_size');stored=_integers(row['stored_size_wh'],(2,),'stored_size')
    roi=_integers(row['roi_xyxy'],(4,),'roi');offset=_integers(row['offset_xy'],(2,),'offset')
    if (full<=0).any() or (stored<=0).any() or (roi[2:]<=roi[:2]).any() or (roi[2:]>full).any():
        raise ValueError('image_layout_out_of_bounds')
    if not np.array_equal(offset,roi[:2]) or not np.array_equal(stored,roi[2:]-roi[:2]):
        raise ValueError('image_layout_offset_or_size_conflict')
    if kind=='full_image' and (offset.any() or not np.array_equal(stored,full)):
        raise ValueError('full_image_layout_conflict')
    return ImageLayout(kind,tuple(map(int,full)),tuple(map(int,roi)),tuple(map(int,offset)),tuple(map(int,stored)))


def read_layouts(data,count):
    present=[k for k in LAYOUT_FIELDS if k in data]
    if not present:return None
    if len(present)!=len(LAYOUT_FIELDS):raise ValueError('missing_image_layout_metadata')
    if any(np.asarray(data[k]).ndim==0 or len(data[k])!=count for k in LAYOUT_FIELDS):
        raise ValueError('image_layout_row_count_conflict')
    return [parse_image_layout({k:data[k][i] for k in LAYOUT_FIELDS}) for i in range(count)]


def crop_affine_and_support(plan,patch_size,layout):
    """Same full-image affine as the legacy path; conservative interpolation guard.

    Require the entire continuous square (including its unsampled far edge)
    plus one source pixel within the stored region. This covers bilinear
    neighbor support and OpenCV's fixed-point rounding, for arbitrary rotations.
    Border filling is never used as observation. No prefilter is supported yet.
    """
    b=plan.box;a=plan.augmentation;w,h=patch_size
    affine=gen_trans_from_patch_cv(b.center_x,b.center_y,b.width,b.height,w,h,a.scale,a.rotation)
    if a.flip:raise ValueError('DenseKP flipping requires a verified 138-point semantic permutation')
    if not np.isfinite(affine).all() or abs(np.linalg.det(affine[:,:2]))<1e-12:
        raise ValueError('invalid_crop_affine')
    inverse=cv2.invertAffineTransform(affine)
    corners=np.array([[0,0,1],[w,0,1],[0,h,1],[w,h,1]])@inverse.T
    r=np.array(layout.roi_xyxy)
    # 1e-9px only accommodates double affine inversion at an exact guard edge.
    if (corners.min(0)-1<r[:2]-1e-9).any() or (corners.max(0)+1>r[2:]+1e-9).any():
        raise ValueError('augmented_crop_outside_roi_support')
    return affine,inverse


def warp_roi_image(pixels,plan,patch_size,layout,*,use_skimage_antialias,do_augment,augm_config,border_mode):
    if use_skimage_antialias:raise ValueError('roi_image_prefilter_not_supported')
    if do_augment and augm_config.USE_ALB:raise ValueError('roi_image_albumentations_not_supported')
    affine,inverse=crop_affine_and_support(plan,patch_size,layout)
    # Invert the SAME full transform first. Moving the crop center before
    # gen_trans would change its float32 rounding. The integer translation only
    # changes image addresses, never labels, img_size or the returned affine.
    local_inverse=inverse.copy();local_inverse[:,2]-=layout.offset_xy
    display=cv2.warpAffine(pixels,local_inverse,tuple(map(int,patch_size)),
        flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP,borderMode=border_mode)
    return display,affine
