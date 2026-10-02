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


def resize_image(img, target_size):
    height, width = img.shape[:2]
    aspect_ratio = width / height

    # Calculate the new size while maintaining the aspect ratio
    if aspect_ratio > 1:
        new_width = target_size
        new_height = int(target_size / aspect_ratio)
    else:
        new_width = int(target_size * aspect_ratio)
        new_height = target_size

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # Create a new blank image with the target size
    final_img = np.ones((target_size, target_size, 3), dtype=np.uint8) * 255

    # Paste the resized image onto the blank image, centering it
    start_x = (target_size - new_width) // 2
    start_y = (target_size - new_height) // 2
    final_img[start_y : start_y + new_height, start_x : start_x + new_width] = resized_img

    return aspect_ratio, final_img


def get_transform(center, scale, res, rot=0):
    """Generate transformation matrix."""
    """Taken from PARE: https://github.com/mkocabas/PARE/blob/6e0caca86c6ab49ff80014b661350958e5b72fd8/pare/utils/image_utils.py"""
    h = 200 * scale
    t = np.zeros((3, 3))
    t[0, 0] = float(res[1]) / h
    t[1, 1] = float(res[0]) / h
    t[0, 2] = res[1] * (-float(center[0]) / h + 0.5)
    t[1, 2] = res[0] * (-float(center[1]) / h + 0.5)
    t[2, 2] = 1
    if not rot == 0:
        rot = -rot  # To match direction of rotation from cropping
        rot_mat = np.zeros((3, 3))
        rot_rad = rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
        rot_mat[2, 2] = 1
        # Need to rotate around center
        t_mat = np.eye(3)
        t_mat[0, 2] = -res[1] / 2
        t_mat[1, 2] = -res[0] / 2
        t_inv = t_mat.copy()
        t_inv[:2, 2] *= -1
        t = np.dot(t_inv, np.dot(rot_mat, np.dot(t_mat, t)))
    return t


def transform(pt, center, scale, res, invert=0, rot=0, as_int=True):
    """Transform pixel location to different reference."""
    """Taken from PARE: https://github.com/mkocabas/PARE/blob/6e0caca86c6ab49ff80014b661350958e5b72fd8/pare/utils/image_utils.py"""
    t = get_transform(center, scale, res, rot=rot)
    if invert:
        t = np.linalg.inv(t)
    new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.0]).T
    new_pt = np.dot(t, new_pt)
    if as_int:
        new_pt = new_pt.astype(int)
    return new_pt[:2] + 1


def crop_img(img, ul, br, border_mode=cv2.BORDER_CONSTANT, border_value=0):
    c_x = (ul[0] + br[0]) / 2
    c_y = (ul[1] + br[1]) / 2
    bb_width = patch_width = br[0] - ul[0]
    bb_height = patch_height = br[1] - ul[1]
    trans = gen_trans_from_patch_cv(
        c_x, c_y, bb_width, bb_height, patch_width, patch_height, 1.0, 0
    )
    img_patch = cv2.warpAffine(
        img,
        trans,
        (int(patch_width), int(patch_height)),
        flags=cv2.INTER_LINEAR,
        borderMode=border_mode,
        borderValue=border_value,
    )

    # Force borderValue=cv2.BORDER_CONSTANT for alpha channel
    if (img.shape[2] == 4) and (border_mode != cv2.BORDER_CONSTANT):
        img_patch[:, :, 3] = cv2.warpAffine(
            img[:, :, 3],
            trans,
            (int(patch_width), int(patch_height)),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    return img_patch


def generate_image_patch_skimage(
    img: np.array,
    c_x: float,
    c_y: float,
    bb_width: float,
    bb_height: float,
    patch_width: float,
    patch_height: float,
    do_flip: bool,
    scale: float,
    rot: float,
    border_mode=cv2.BORDER_CONSTANT,
    border_value=0,
) -> Tuple[np.array, np.array]:
    """
    Crop image according to the supplied bounding box.
    Args:
        img (np.array): Input image of shape (H, W, 3)
        c_x (float): Bounding box center x coordinate in the original image.
        c_y (float): Bounding box center y coordinate in the original image.
        bb_width (float): Bounding box width.
        bb_height (float): Bounding box height.
        patch_width (float): Output box width.
        patch_height (float): Output box height.
        do_flip (bool): Whether to flip image or not.
        scale (float): Rescaling factor for the bounding box (augmentation).
        rot (float): Random rotation applied to the box.
    Returns:
        img_patch (np.array): Cropped image patch of shape (patch_height, patch_height, 3)
        trans (np.array): Transformation matrix.
    """

    img_height, img_width, img_channels = img.shape
    if do_flip:
        img = img[:, ::-1, :]
        c_x = img_width - c_x - 1

    trans = gen_trans_from_patch_cv(
        c_x, c_y, bb_width, bb_height, patch_width, patch_height, scale, rot
    )

    # img_patch = cv2.warpAffine(img, trans, (int(patch_width), int(patch_height)), flags=cv2.INTER_LINEAR)

    # skimage
    center = np.zeros(2)
    center[0] = c_x
    center[1] = c_y
    res = np.zeros(2)
    res[0] = patch_width
    res[1] = patch_height
    # assumes bb_width = bb_height
    # assumes patch_width = patch_height
    assert bb_width == bb_height, f"{bb_width=} != {bb_height=}"
    assert patch_width == patch_height, f"{patch_width=} != {patch_height=}"
    scale1 = scale * bb_width / 200.0

    # Upper left point
    ul = np.array(transform([1, 1], center, scale1, res, invert=1, as_int=False)) - 1
    # Bottom right point
    br = (
        np.array(transform([res[0] + 1, res[1] + 1], center, scale1, res, invert=1, as_int=False))
        - 1
    )

    # Padding so that when rotated proper amount of context is included
    try:
        pad = int(np.linalg.norm(br - ul) / 2 - float(br[1] - ul[1]) / 2) + 1
    except:
        breakpoint()
    if not rot == 0:
        ul -= pad
        br += pad

    if False:
        # Old way of cropping image
        ul_int = ul.astype(int)
        br_int = br.astype(int)
        new_shape = [br_int[1] - ul_int[1], br_int[0] - ul_int[0]]
        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape)

        # Range to fill new array
        new_x = max(0, -ul_int[0]), min(br_int[0], len(img[0])) - ul_int[0]
        new_y = max(0, -ul_int[1]), min(br_int[1], len(img)) - ul_int[1]
        # Range to sample from original image
        old_x = max(0, ul_int[0]), min(len(img[0]), br_int[0])
        old_y = max(0, ul_int[1]), min(len(img), br_int[1])
        new_img[new_y[0] : new_y[1], new_x[0] : new_x[1]] = img[
            old_y[0] : old_y[1], old_x[0] : old_x[1]
        ]

    # New way of cropping image
    new_img = crop_img(img, ul, br, border_mode=border_mode, border_value=border_value).astype(
        np.float32
    )

    # print(f'{new_img.shape=}')
    # print(f'{new_img1.shape=}')
    # print(f'{np.allclose(new_img, new_img1)=}')
    # print(f'{img.dtype=}')

    if not rot == 0:
        # Remove padding

        new_img = rotate(new_img, rot)  # scipy.misc.imrotate(new_img, rot)
        new_img = new_img[pad:-pad, pad:-pad]

    if new_img.shape[0] < 1 or new_img.shape[1] < 1:
        print(f"{img.shape=}")
        print(f"{new_img.shape=}")
        print(f"{ul=}")
        print(f"{br=}")
        print(f"{pad=}")
        print(f"{rot=}")

        breakpoint()

    # resize image
    new_img = resize(new_img, res)  # scipy.misc.imresize(new_img, res)

    new_img = np.clip(new_img, 0, 255).astype(np.uint8)

    return new_img, trans


def fliplr_params(smpl_params: Dict, has_smpl_params: Dict) -> Tuple[Dict, Dict]:
    """
    Flip SMPL parameters when flipping the image.
    Args:
        smpl_params (Dict): SMPL parameter annotations.
        has_smpl_params (Dict): Whether SMPL annotations are valid.
    Returns:
        Dict, Dict: Flipped SMPL parameters and valid flags.
    """
    global_orient = smpl_params["global_orient"].copy()
    body_pose = smpl_params["body_pose"].copy()
    betas = smpl_params["betas"].copy()
    has_global_orient = has_smpl_params["global_orient"]
    has_body_pose = has_smpl_params["body_pose"]
    has_betas = has_smpl_params["betas"]

    body_pose_permutation = [
        6,
        7,
        8,
        3,
        4,
        5,
        9,
        10,
        11,
        15,
        16,
        17,
        12,
        13,
        14,
        18,
        19,
        20,
        24,
        25,
        26,
        21,
        22,
        23,
        27,
        28,
        29,
        33,
        34,
        35,
        30,
        31,
        32,
        36,
        37,
        38,
        42,
        43,
        44,
        39,
        40,
        41,
        45,
        46,
        47,
        51,
        52,
        53,
        48,
        49,
        50,
        57,
        58,
        59,
        54,
        55,
        56,
        63,
        64,
        65,
        60,
        61,
        62,
        69,
        70,
        71,
        66,
        67,
        68,
    ]
    body_pose_permutation = body_pose_permutation[: len(body_pose)]
    body_pose_permutation = [i - 3 for i in body_pose_permutation]

    body_pose = body_pose[body_pose_permutation]

    global_orient[1::3] *= -1
    global_orient[2::3] *= -1
    body_pose[1::3] *= -1
    body_pose[2::3] *= -1

    smpl_params = {
        "global_orient": global_orient.astype(np.float32),
        "body_pose": body_pose.astype(np.float32),
        "betas": betas.astype(np.float32),
    }

    has_smpl_params = {
        "global_orient": has_global_orient,
        "body_pose": has_body_pose,
        "betas": has_betas,
    }

    return smpl_params, has_smpl_params


def fliplr_keypoints(joints: np.array, width: float, flip_permutation: List[int]) -> np.array:
    """
    Flip 2D or 3D keypoints.
    Args:
        joints (np.array): Array of shape (N, 3) or (N, 4) containing 2D or 3D keypoint locations and confidence.
        flip_permutation (List): Permutation to apply after flipping.
    Returns:
        np.array: Flipped 2D or 3D keypoints with shape (N, 3) or (N, 4) respectively.
    """
    joints = joints.copy()
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1
    joints = joints[flip_permutation, :]

    return joints


def keypoint_3d_processing(
    keypoints_3d: np.array, flip_permutation: List[int], rot: float, do_flip: float
) -> np.array:
    """
    Process 3D keypoints (rotation/flipping).
    Args:
        keypoints_3d (np.array): Input array of shape (N, 4) containing the 3D keypoints and confidence.
        flip_permutation (List): Permutation to apply after flipping.
        rot (float): Random rotation applied to the keypoints.
        do_flip (bool): Whether to flip keypoints or not.
    Returns:
        np.array: Transformed 3D keypoints with shape (N, 4).
    """
    if do_flip:
        keypoints_3d = fliplr_keypoints(keypoints_3d, 1, flip_permutation)
    # in-plane rotation
    rot_mat = np.eye(3)
    if not rot == 0:
        rot_rad = -rot * np.pi / 180
        sn, cs = np.sin(rot_rad), np.cos(rot_rad)
        rot_mat[0, :2] = [cs, -sn]
        rot_mat[1, :2] = [sn, cs]
    keypoints_3d[:, :-1] = np.einsum("ij,kj->ki", rot_mat, keypoints_3d[:, :-1])
    # flip the x coordinates
    keypoints_3d = keypoints_3d.astype("float32")
    return keypoints_3d


def rot_aa(aa: np.array, rot: float) -> np.array:
    """
    Rotate axis angle parameters.
    Args:
        aa (np.array): Axis-angle vector of shape (3,).
        rot (np.array): Rotation angle in degrees.
    Returns:
        np.array: Rotated axis-angle vector.
    """
    # pose parameters
    R = np.array(
        [
            [np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
            [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
            [0, 0, 1],
        ]
    )
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa.astype(np.float32)


def get_example(
    img_path: str | np.ndarray,
    center_x: float,
    center_y: float,
    width: float,
    height: float,
    keypoints_2d: np.array,
    flip_kp_permutation: List[int],
    patch_width: int,
    patch_height: int,
    mean: np.array,
    std: np.array,
    do_augment: bool,
    augm_config: CfgNode,
    is_bgr: bool = True,
    use_skimage_antialias: bool = False,
    border_mode: int = cv2.BORDER_CONSTANT,
    return_trans: bool = False,
    dataset: str = None,
):

    if isinstance(img_path, str):
        # 1. load image
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if dataset is not None and ("closeup" in dataset or "portrait" in dataset):
            cvimg = cv2.rotate(cvimg, cv2.ROTATE_90_CLOCKWISE)
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % img_path)
    elif isinstance(img_path, np.ndarray):
        cvimg = img_path
    else:
        raise TypeError("img_path must be either a string or a numpy array")

    img_height, img_width, img_channels = cvimg.shape
    img_size = np.array([img_height, img_width])

    if do_augment:
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = (
            do_augmentation(augm_config)
        )
    else:
        scale, rot, do_flip, do_extreme_crop, extreme_crop_lvl, color_scale, tx, ty = (
            1.0,
            0,
            False,
            False,
            0,
            [1.0, 1.0, 1.0],
            0.0,
            0.0,
        )

    if width < 1 or height < 1:
        # Todo proper exit
        breakpoint()

    if do_extreme_crop:
        if extreme_crop_lvl == 0:
            center_x1, center_y1, width1, height1 = extreme_cropping(
                center_x, center_y, width, height, keypoints_2d
            )
        elif extreme_crop_lvl == 1:
            center_x1, center_y1, width1, height1 = extreme_cropping_aggressive(
                center_x, center_y, width, height, keypoints_2d
            )

        THRESH = 4
        if width1 < THRESH or height1 < THRESH:
            pass
        else:
            center_x, center_y, width, height = center_x1, center_y1, width1, height1

    center_x += width * tx
    center_y += height * ty

    # 3. generate image patch
    if use_skimage_antialias:
        # Blur image to avoid aliasing artifacts
        downsampling_factor = patch_width / (width * scale)
        if downsampling_factor > 1.1:
            cvimg = gaussian(
                cvimg,
                sigma=(downsampling_factor - 1) / 2,
                channel_axis=2,
                preserve_range=True,
                truncate=3.0,
            )

    if do_augment and augm_config.USE_ALB:
        import albumentations as A

        aug_comp = [
            A.Downscale(0.5, 0.9, interpolation=0, p=0.1),
            A.ImageCompression(20, 100, p=0.1),
            A.RandomRain(blur_value=4, p=0.1),
            A.MotionBlur(blur_limit=(3, 15), p=0.2),
            A.Blur(blur_limit=(3, 10), p=0.1),
            A.RandomSnow(brightness_coeff=1.5, snow_point_lower=0.2, snow_point_upper=0.4),
        ]
        aug_mod = [
            A.CLAHE((1, 11), (10, 10), p=0.2),
            A.ToGray(p=0.2),
            A.RandomBrightnessContrast(p=0.2),
            A.MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                always_apply=False,
                p=0.2,
            ),
            A.Posterize(p=0.1),
            A.RandomGamma(gamma_limit=(80, 200), p=0.1),
            A.Equalize(mode="cv", p=0.1),
        ]
        albumentation_aug = A.Compose(
            [A.OneOf(aug_comp, p=augm_config.ALB_PROB), A.OneOf(aug_mod, p=augm_config.ALB_PROB)]
        )
        cvimg = albumentation_aug(image=cvimg)["image"]

    img_patch_cv, trans = generate_image_patch_cv2(
        cvimg,
        center_x,
        center_y,
        width,
        height,
        patch_width,
        patch_height,
        do_flip,
        scale,
        rot,
        border_mode=border_mode,
    )

    image = img_patch_cv.copy()

    if is_bgr:
        image = image[:, :, ::-1]
    img_patch_cv = image.copy()
    img_patch = convert_cvimg_to_tensor(image)

    for n_c in range(min(img_channels, 3)):
        img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        if mean is not None and std is not None:
            img_patch[n_c, :, :] = (img_patch[n_c, :, :] - mean[n_c]) / std[n_c]

    keypoints_2d[:, :2] = trans_points2d_parallel(keypoints_2d[:, 0:2], trans)
    keypoints_2d[:, :-1] = keypoints_2d[:, :-1] / patch_width - 0.5

    if not return_trans:
        return (
            img_patch,
            img_patch_cv,
            keypoints_2d,
            img_size,
            center_x,
            center_y,
            width,
            height,
            scale,
        )
    else:
        return (
            img_patch,
            img_patch_cv,
            keypoints_2d,
            img_size,
            center_x,
            center_y,
            width,
            height,
            trans,
            scale,
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


def do_augmentation(aug_config: CfgNode) -> Tuple:
    """Historical positional tuple; active code uses named Augmentation."""
    return tuple(sample_augmentation(aug_config))


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
