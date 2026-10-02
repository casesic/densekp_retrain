"""Actual Dataset configuration used for frozen box derivatives (no augmentation)."""

from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.configuration import audit_cfg


def load_box_dataset(path, train=False):
    return DatasetKeypoints(
        audit_cfg("supervision_weight"),
        "box-trial",
        is_train=train,
        use_augmentation=False,
        annotation_path=str(path),
        image_root="/",
    )
