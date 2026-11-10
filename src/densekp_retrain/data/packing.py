"""Pack index rows in their saved order without changing label or source semantics.

Source validation belongs before packing. DatasetKeypoints applies the existing
annotation contract on readback; this layer retains original KeyError/empty-row
behavior rather than silently filtering or repairing rows.
"""

from pathlib import Path
import numpy as np
from densekp.core.keypoint_supervision import SUPERVISION_POLICY
from densekp.core.datasets.image_layout import LAYOUT_FIELDS,read_layouts


def pack_index_rows(index):
    fields = {
        k: []
        for k in [
            "imgname",
            "center",
            "scale",
            "proj_verts",
            "in_view",
            "visible",
            "person_id",
            "camera_id",
        ]
    }
    cache = {}
    layout_flags = {}
    expected_layout=None
    expected_box_reservation=None
    for row in index["rows"]:
        path = row["loader_input"]
        if path not in cache:
            with np.load(path) as archive:
                cache[path] = {key: archive[key] for key in archive.files}
            layout_flags[path]=read_layouts(cache[path],len(cache[path]['imgname'])) is not None
        source_arrays = cache[path]
        source_row = row["loader_row"]
        has_layout=layout_flags[path]
        box_fields=('parent_bbox_xyxy','box_augmentation_reserved_policy')
        box_audit_fields=('box_augmentation_family_supported','box_augmentation_family_reason')
        has_box=all(k in source_arrays for k in box_fields)
        has_audit=all(k in source_arrays for k in box_audit_fields)
        if (any(k in source_arrays for k in box_fields) and not has_box) or (any(k in source_arrays for k in box_audit_fields) and not has_audit):
            raise ValueError('incomplete_box_augmentation_reservation')
        if expected_box_reservation is None:
            expected_box_reservation=(has_box,has_audit)
            if has_box:fields.update({key:[] for key in box_fields})
            if has_audit:fields.update({key:[] for key in box_audit_fields})
        elif expected_box_reservation!=(has_box,has_audit):
            raise ValueError('mixed_box_augmentation_reservation')
        if expected_layout is None:
            expected_layout=has_layout
            if has_layout:fields.update({key:[] for key in LAYOUT_FIELDS})
        elif expected_layout!=has_layout:
            raise ValueError('mixed_explicit_and_legacy_image_layouts_require_explicit_full_metadata')
        if str(source_arrays["supervision_policy"]) != SUPERVISION_POLICY:
            raise ValueError("Supervision mismatch")
        for key in fields:
            fields[key].append(
                str(Path(row["image"]).resolve())
                if key == "imgname"
                else source_arrays[key][source_row]
            )
    return {key: np.asarray(values) for key, values in fields.items()}


def save_index_loader(index, target, side):
    fields = pack_index_rows(index)
    np.savez(target, **fields, supervision_policy=SUPERVISION_POLICY, split=side)


def pack_derivative_rows(rows):
    """Keep all per-row fields and their order; empty input retains IndexError."""
    keys=set(rows[0])
    if any(set(row)!=keys for row in rows):
        raise ValueError("inconsistent_row_fields; image layout metadata must not be dropped")
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
