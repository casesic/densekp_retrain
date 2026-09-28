"""DenseKP crop geometry. Sizes are (width, height); xy are raster pixels.

The affine maps the label raster to the full crop, before the backbone's
central-width slice. Normalize against the full crop; never add a slice offset.
Only xy is transformed. A third model/annotation field has no spatial meaning.
"""
import cv2
import numpy as np

def rotate_2d(pt_2d: np.array, rot_rad: float) -> np.array:
    """
    Rotate a 2D point on the x-y plane.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        rot_rad (float): Rotation angle
    Returns:
        np.array: Rotated 2D point.
    """
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x: float, c_y: float,
                            src_width: float, src_height: float,
                            dst_width: float, dst_height: float,
                            scale: float, rot: float) -> np.array:
    """
    Create transformation matrix for the bounding box crop.
    Args:
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        src_width (float): Bounding box width.
        src_height (float): Bounding box height.
        dst_width (float): Output box width.
        dst_height (float): Output box height.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        trans (np.array): Target geometric transformation.
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.zeros(2)
    src_center[0] = c_x
    src_center[1] = c_y
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def trans_point2d(pt_2d: np.array, trans: np.array):
    """
    Transform a 2D point using translation matrix trans.
    Args:
        pt_2d (np.array): Input 2D point with shape (2,).
        trans (np.array): Transformation matrix.
    Returns:
        np.array: Transformed 2D point.
    """
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]

def trans_points2d_parallel(keypoints_2d: np.array, trans:np.array):

    ones = np.ones((keypoints_2d.shape[0], 1))
    keypoints_augmented = np.hstack([keypoints_2d, ones])
    
    # Perform the transformation: (3, 3) dot (N, 3)^T => (3, N)
    transformed_keypoints = np.dot(trans, keypoints_augmented.T).T
    
    # Extract the transformed 2D points by discarding the last row
    transformed_keypoints_2d = transformed_keypoints[:, :2]
    
    return transformed_keypoints_2d


def normalize_crop_xy(xy, size_wh):
    """Full-crop pixels -> dimensionless xy, centered at zero."""
    size = np.asarray(size_wh, dtype=np.float64)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Crop size must be positive finite (width, height)")
    return np.asarray(xy) / size - 0.5


def restore_crop_xy(normalized_xy, trans, size_wh):
    """Normalized crop xy -> label-raster pixels using the exact saved affine.

    Portrait rotation / ERP roll / ROI offset, if present, are separate outer
    transforms. This function does not reinterpret any third output value.
    """
    size = np.asarray(size_wh, dtype=np.float64)
    normalize_crop_xy(np.zeros((1, 2)), size)  # validate dimensions
    matrix = np.asarray(trans, dtype=np.float64)
    if matrix.shape != (2, 3) or not np.isfinite(matrix).all():
        raise ValueError("Expected a finite 2x3 affine")
    if abs(np.linalg.det(matrix[:, :2])) < 1e-12:
        raise ValueError("Cannot invert a singular crop affine")
    return trans_points2d_parallel((np.asarray(normalized_xy) + 0.5) * size,
                                   cv2.invertAffineTransform(matrix))
