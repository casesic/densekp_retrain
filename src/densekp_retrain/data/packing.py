"""Pack index rows in their saved order without changing label or source semantics.

Source validation belongs before packing. DatasetKeypoints applies the existing
annotation contract on readback; this layer retains original KeyError/empty-row
behavior rather than silently filtering or repairing rows.
"""

from pathlib import Path
import numpy as np
from densekp.core.keypoint_supervision import SUPERVISION_POLICY


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
    for row in index["rows"]:
        path = row["loader_input"]
        if path not in cache:
            with np.load(path) as archive:
                cache[path] = {key: archive[key] for key in archive.files}
        source_arrays = cache[path]
        source_row = row["loader_row"]
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
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}
