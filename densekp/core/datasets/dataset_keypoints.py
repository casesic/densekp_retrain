import cv2
import copy
import numpy as np
from torch.utils.data import Dataset

from .image_processing import expand_to_aspect_ratio, read_image
from .crop_pipeline import CropBox, check_flip_augmentation, determine_crop, transform_crop
from .loading import resolve_source, resolve_image_path, read_annotations
from .image_layout import read_layouts
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
        self.image_layouts = read_layouts(self.data, len(self.imgname))
        # Optional bounded box perturbation uses a separate deterministic RNG.
        # Old configs and validation never enter this branch.
        self.box_augmentation = cfg.DATASETS.get("BOX_AUGMENTATION", {"enabled": False})
        self.box_augmentation_epoch = 0
        self.box_augmentation_visit_index = 0
        if self.box_augmentation.get("enabled", False) and is_train:
            from densekp_retrain.data.box_augmentation import (
                validate_box_policy, require_full_box_family_support, policy_within_reservation,
            )
            import json
            policy = validate_box_policy(dict(self.box_augmentation["policy"]))
            geometric = ("EXTREME_CROP_AUG_RATE", "ROT_AUG_RATE", "SCALE_FACTOR",
                         "TRANS_AUG_RATE", "FLIP_AUG_RATE")
            if self.use_augmentation and (any(cfg.DATASETS.CONFIG.get(k, 0) for k in geometric)
                    or cfg.DATASETS.CONFIG.get("USE_ALB", False)
                    or self.use_skimage_antialias):
                raise ValueError("box_augmentation_combined_geometric_or_prefilter_not_supported")
            if "parent_bbox_xyxy" not in self.data or "box_augmentation_reserved_policy" not in self.data:
                raise ValueError("box_augmentation_requires_parent_and_reserved_policy")
            for i in range(len(self.imgname)):
                declared = json.loads(str(self.data["box_augmentation_reserved_policy"][i]))
                if not policy_within_reservation(policy,declared):
                    raise ValueError("box_augmentation_policy_exceeds_reserved_support")
                layout = self.image_layouts[i] if self.image_layouts is not None else None
                if layout is None:
                    raise ValueError("box_augmentation_requires_explicit_image_layout")
                require_full_box_family_support(self.data["parent_bbox_xyxy"][i],
                    layout.roi_xyxy, layout.full_size_wh, policy)
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
        if self.is_train and self.box_augmentation.get("enabled", False):
            from densekp_retrain.data.box_augmentation import sample_box_variant
            layout = self.image_layouts[index]
            variant = sample_box_variant(self.data["parent_bbox_xyxy"][index],
                policy=dict(self.box_augmentation["policy"]),
                seed=self.box_augmentation["seed"],
                observation_key=(self.box_augmentation_epoch, index, self.box_augmentation_visit_index),
                roi_xyxy=layout.roi_xyxy, image_size_wh=layout.full_size_wh)
            center = np.asarray(variant["center"], dtype=np.float32)
            scale = np.float32(variant["scale"])
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
        layout = self.image_layouts[index] if self.image_layouts is not None else None
        if layout is not None and layout.kind == "roi_image" and ("portrait" in self.dataset or "closeup" in self.dataset):
            raise ValueError("roi_image_dataset_rotation_not_supported")
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
            image_layout=layout,
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

    def set_box_augmentation_epoch(self, epoch):
        """Set before worker creation; repeated visits in one epoch use draw zero."""
        if type(epoch) is not int or epoch < 0:
            raise ValueError("invalid_box_augmentation_epoch")
        self.box_augmentation_epoch = epoch

    def set_box_augmentation_visit_index(self, visit_index):
        """Optional explicit repeat index; never inferred from worker scheduling."""
        if type(visit_index) is not int or visit_index < 0:
            raise ValueError("invalid_box_augmentation_visit_index")
        self.box_augmentation_visit_index = visit_index
