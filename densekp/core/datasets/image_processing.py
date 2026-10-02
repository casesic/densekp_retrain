"""Image IO, OpenCV warping and pixel/label transformations for crop processing."""

from typing import NamedTuple, Tuple
import cv2
import numpy as np
from skimage.filters import gaussian
from ..crop_geometry import gen_trans_from_patch_cv, trans_points2d_parallel, normalize_crop_xy


class LoadedImage(NamedTuple):
    pixels: np.ndarray
    size_hw: np.ndarray
    channels: int


def read_image(img_path, dataset=None):
    """Keep original BGR/orientation flags and dataset-name rotation branches.

    Array inputs are used as provided; portrait/closeup rotation applies only
    to file inputs. Label rotation is a separate unresolved historical issue.
    """
    if isinstance(img_path, str):
        cvimg = cv2.imread(img_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if dataset and ("closeup" in dataset or "portrait" in dataset):
            cvimg = cv2.rotate(cvimg, cv2.ROTATE_90_CLOCKWISE)
        if not isinstance(cvimg, np.ndarray):
            raise IOError(f"Fail to read {img_path}")
    elif isinstance(img_path, np.ndarray):
        cvimg = img_path
    else:
        raise TypeError("img_path must be either a string or a numpy array")
    img_height, img_width, img_channels = cvimg.shape
    return LoadedImage(cvimg, np.array([img_height, img_width]), img_channels)


def expand_to_aspect_ratio(input_shape, target_aspect_ratio=None):
    """Increase the size of the bounding box to match the target shape."""
    if target_aspect_ratio is None:
        return input_shape

    try:
        w, h = input_shape
    except (ValueError, TypeError):
        return input_shape

    w_t, h_t = target_aspect_ratio
    if h / w < h_t / w_t:
        h_new = max(w * h_t / w_t, h)
        w_new = w
    else:
        h_new = h
        w_new = max(h * w_t / h_t, w)
    if h_new < h or w_new < w:
        breakpoint()
    return np.array([w_new, h_new])


def generate_image_patch_cv2(
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
    Crop the input image and return the crop and the corresponding transformation matrix.
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

    return img_patch, trans


def convert_cvimg_to_tensor(cvimg: np.array):
    """
    Convert image from HWC to CHW format.
    Args:
        cvimg (np.array): Image of shape (H, W, 3) as loaded by OpenCV.
    Returns:
        np.array: Output image of shape (3, H, W).
    """
    # from h,w,c(OpenCV) to c,h,w
    img = cvimg.copy()
    img = np.transpose(img, (2, 0, 1))
    # from int to float
    img = img.astype(np.float32)
    return img


def preprocess_image(
    cvimg, box, augmentation, patch_size, use_skimage_antialias, do_augment, augm_config
):
    """Keep Gaussian prefilter and Albumentations before affine warping."""
    width, height, scale = box.width, box.height, augmentation.scale
    patch_width, patch_height = patch_size
    if use_skimage_antialias:
        downsampling_factor = max(width * scale / patch_width, height * scale / patch_height)
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
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.2),
            A.Posterize(p=0.1),
            A.RandomGamma(gamma_limit=(80, 200), p=0.1),
            A.Equalize(mode="cv", p=0.1),
        ]
        albumentation_aug = A.Compose(
            [A.OneOf(aug_comp, p=augm_config.ALB_PROB), A.OneOf(aug_mod, p=augm_config.ALB_PROB)]
        )
        cvimg = albumentation_aug(image=cvimg)["image"]
    return cvimg


def normalize_image(display, channels, color_scale, mean, std, is_bgr):
    """Convert HWC to float32 CHW, then color/clip/normalize each channel."""
    image = display[:, :, ::-1] if is_bgr else display.copy()
    normalized = convert_cvimg_to_tensor(image)
    for channel in range(min(channels, 3)):
        normalized[channel, :, :] = np.clip(
            normalized[channel, :, :] * color_scale[channel], 0, 255
        )
        if mean is not None and std is not None:
            normalized[channel, :, :] = (normalized[channel, :, :] - mean[channel]) / std[channel]
    return normalized


def transform_labels(points, affine, patch_size):
    """Mutate XY using the image affine; preserve the intermediate dtype cast.

    Assigning pixel XY before normalization intentionally rounds to the label
    array's dtype twice. Collapsing these assignments changes float32 results.
    Third values are untouched here; Dataset recomputes supervision afterwards.
    """
    points[:, :2] = trans_points2d_parallel(points[:, 0:2], affine)
    points[:, :2] = normalize_crop_xy(points[:, :2], patch_size)
