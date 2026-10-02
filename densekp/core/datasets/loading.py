"""Resolve explicit data sources, retaining historical ID and image-path rules."""

from dataclasses import dataclass
import os
import numpy as np
from ..configs import DATASET_FILES, DATASET_FOLDERS


@dataclass(frozen=True)
class DatasetSource:
    annotation_path: str
    image_root: str


def resolve_image_path(image_root, image_name):
    """Join like the old Dataset; absolute names override the image root.

    Do not resolve against the NPZ directory or canonicalize relative strings.
    """
    return os.path.join(image_root, image_name)


def resolve_source(dataset, is_train, *, annotation_path=None, image_root=None):
    """Explicit arguments override each registry field independently.

    The dataset name remains a label and a selector for historical image-reading
    branches. Supplying both paths requires no registered ID and never mutates
    the registry. Missing fields retain the original KeyError behavior.
    """
    root = DATASET_FOLDERS[dataset] if image_root is None else os.fspath(image_root)
    annotation = (
        DATASET_FILES[is_train][dataset] if annotation_path is None else os.fspath(annotation_path)
    )
    return DatasetSource(annotation, root)


def read_annotations(annotation_path):
    """Load only fields used by DatasetKeypoints, with the historical pickle flag."""
    with np.load(annotation_path, allow_pickle=True) as archive:
        fields = (
            "imgname",
            "scale",
            "center",
            "proj_verts",
            "part",
            "gtkps",
            "supervision_policy",
            "in_view",
            "visible",
        )
        return {key: archive[key] for key in fields if key in archive}
