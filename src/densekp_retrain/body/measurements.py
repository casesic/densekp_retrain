"""Frozen anatomical-band measurements, in SMPL-X axes and meters.

These are engineering fit constraints, not clinical anthropometry. Experimental
shape targets and optimization recipes remain in their callers.
"""

import numpy as np


def measurement_rows(vertices, joints, labels, head_weights):
    """Fixed anatomical bands, used as measurement constraints, never mesh edits."""
    body = labels == 0
    head = body & (head_weights > 0.5)
    eyes_y = vertices[labels != 0, 1].mean()
    masks = {
        "height": (body, 1),
        "shoulder_span": (
            body
            & (np.abs(vertices[:, 1] - joints[16:18, 1].mean()) < 0.025)
            & (np.abs(vertices[:, 0]) < 0.25),
            0,
        ),
        "waist_span": (body & (np.abs(vertices[:, 1] - 0.13) < 0.025), 0),
        "head_width": (head & (np.abs(vertices[:, 1] - (eyes_y - 0.025)) < 0.018), 0),
        "jaw_width": (head & (np.abs(vertices[:, 1] - (eyes_y - 0.075)) < 0.015), 0),
        "head_height": (head, 1),
    }
    rows = []
    for name, (mask, axis) in masks.items():
        ids = np.flatnonzero(mask)
        if len(ids) < 8:
            raise ValueError(f"Insufficient measurement band: {name}")
        ordered = ids[np.argsort(vertices[ids, axis])]
        count = 1 if name == "height" else max(2, len(ids) // 10)
        rows.append((name, axis, ordered[-count:], ordered[:count]))
    return rows


def measure(v, rows):
    """Measure band spans in meters from (name, axis, high_ids, low_ids) rows."""
    return np.array([v[high, axis].mean() - v[low, axis].mean() for _, axis, high, low in rows])
