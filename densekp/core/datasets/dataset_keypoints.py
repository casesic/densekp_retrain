import cv2
import copy
import numpy as np
from torch.utils.data import Dataset

from .image_processing import expand_to_aspect_ratio, read_image
from .crop_pipeline import CropBox, check_flip_augmentation, determine_crop, transform_crop
from .loading import resolve_source, resolve_image_path, read_annotations
from ..keypoint_contract import validate_annotations
from ..keypoint_supervision import SUPERVISION_POLICY, supervision_for_crop
from ..constants import NUM_JOINTS
from ..utils.pylogger import get_pylogger

log = get_pylogger(__name__)


class DatasetKeypoints(Dataset):
    """138-point samples from explicit annotation/image paths or a legacy ID.

    ``dataset`` keeps its original label and portrait/closeup branch meaning;
    it need not be registered when both paths are supplied. Relative image names
    are joined to image_root exactly as before, and absolute names override it.
    """

    def __init__(
        self,
        cfg,
        dataset="explicit",
        use_augmentation=True,
        is_train=True,
        *,
        annotation_path=None,
        image_root=None,
    ):
        super(DatasetKeypoints, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.cfg = cfg
        self.use_augmentation = bool(use_augmentation)
        if (
            is_train
            and self.use_augmentation
            and cfg.DATASETS.CONFIG.DO_FLIP
            and cfg.DATASETS.CONFIG.FLIP_AUG_RATE > 0
        ):
            raise ValueError("DenseKP flipping requires a verified 138-point semantic permutation")

        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.BBOX_SHAPE = cfg.MODEL.get("BBOX_SHAPE", None)
        self.MEAN = 255.0 * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255.0 * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get("USE_SKIMAGE_ANTIALIAS", False)
        self.border_mode = {
            "constant": cv2.BORDER_CONSTANT,
            "replicate": cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get("BORDER_MODE", "constant")]

        self.source = resolve_source(
            dataset, is_train, annotation_path=annotation_path, image_root=image_root
        )
        self.img_dir = self.source.image_root
        self.annotation_path = self.source.annotation_path
        self.data = read_annotations(self.annotation_path)
        validate_annotations(self.data, cfg.DATASETS.get("TARGET_VALUE_SEMANTICS"))
        self.imgname = self.data["imgname"]
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data["scale"]
        self.center = self.data["center"]

        if "part" in self.data:
            self.keypoints = self.data["part"]
        elif "gtkps" in self.data:
            self.keypoints = self.data["gtkps"][:, :NUM_JOINTS]
        else:
            self.keypoints = np.zeros((len(self.imgname), NUM_JOINTS, 3))
        if self.keypoints.shape != (len(self.imgname), NUM_JOINTS, 3):
            raise ValueError(f"Auxiliary keypoints must have shape [N,{NUM_JOINTS},3]")
        self.keypoints = self.keypoints.astype(np.float32)
        self.proj_verts = self.data["proj_verts"].astype(np.float32)
        self.length = self.scale.shape[0]

        log.info(f"Loaded {self.dataset} dataset, num samples {self.length}")

    def __getitem__(self, index):
        # Select the stored box/labels; scale is in original-image pixels / 200.
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints = self.keypoints[index].copy()
        bbox_size = expand_to_aspect_ratio(scale * 200, target_aspect_ratio=self.BBOX_SHAPE).max()
        if bbox_size < 1:
            raise ValueError(f"{self.dataset}[{index}]: bbox size must be at least one pixel")
        imgname = resolve_image_path(self.img_dir, self.imgname[index])
        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)
        dense_points = self.proj_verts[index].copy()
        do_augment = self.is_train and self.use_augmentation

        # Read → determine crop → transform pixels and labels → recompute supervision.
        check_flip_augmentation(do_augment, augm_config)
        image = read_image(imgname, self.dataset)
        plan = determine_crop(
            CropBox(center[0], center[1], bbox_size, bbox_size), keypoints, do_augment, augm_config
        )
        example = transform_crop(
            image,
            plan,
            keypoints,
            dense_points,
            patch_size=(self.IMG_SIZE, self.IMG_SIZE),
            mean=self.MEAN,
            std=self.STD,
            do_augment=do_augment,
            augm_config=augm_config,
            is_bgr=True,
            use_skimage_antialias=self.use_skimage_antialias,
            border_mode=self.border_mode,
        )
        new_center = np.array([plan.box.center_x, plan.box.center_y])
        img_patch = example.image[:3, :, :]
        item = {"proj_verts": self.proj_verts[index].copy()}
        visible, visibility_known = self._supervision(index, example.dense_points)
        item["visible"] = visible
        item["visibility_known"] = visibility_known
        item["proj_verts_cropped"] = example.dense_points
        item["img"] = img_patch
        item["img_disp"] = example.display
        item["box_center"] = new_center
        # Historical metadata: width after extreme crop, before sampled scale.
        item["box_size"] = plan.box.width
        item["img_size"] = 1.0 * example.image_size.copy()
        item["_scale"] = scale
        item["_trans"] = example.affine
        item["imgname"] = imgname
        item["dataset"] = self.dataset
        return item

    def _supervision(self, index, cropped_points):
        """Recompute eligible GT after the actual affine, retaining legacy weights.

        Visibility is metadata, never a condition for supervising reliable GT.
        Both old and synthetic annotations expose the same returned fields.
        """
        visible = np.zeros(len(cropped_points), dtype=bool)
        visibility_known = np.zeros(len(cropped_points), dtype=bool)
        if self.data.get("supervision_policy") == SUPERVISION_POLICY:
            cropped_points[:, 2] = supervision_for_crop(
                self.data["in_view"][index], cropped_points[:, :2]
            )
            visible = self.data["visible"][index].copy()
            visibility_known[:] = True
        return visible, visibility_known

    def __len__(self):
        return int(len(self.imgname))
