"""Historical visibility tests with distinct neighborhood and boundary rules.

ERP depth is Euclidean ray range in meters, with instance IDs and 138-row person
blocks; a periodic 5x5 neighborhood and strict 4cm range difference are used.
The legacy DS instance-only rule rounds pixels, samples 7x7 without seam wrap,
and does not compare depth. These policies intentionally remain separate.
"""

import numpy as np


def erp_visibility(
    xy: np.ndarray,
    distance: np.ndarray,
    depth: np.ndarray,
    index: np.ndarray,
    person_ids: list[int],
):
    """Return finite/vertical-valid and surface-visible masks for 138-row person blocks.

    depth[H,W] is ray range in meters; index[H,W] holds rounded instance IDs.
    Horizontal samples wrap; vertical neighbors clip. Eligibility is not supervision."""
    height, width = depth.shape
    visible = np.zeros(xy.shape[0], dtype=bool)
    valid = np.isfinite(xy).all(axis=1) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    for j, (x, y) in enumerate(xy):
        if not valid[j]:
            continue
        person_id = person_ids[j // 138]
        ix, iy = int(np.floor(x)) % width, int(np.floor(y))
        rows = np.arange(max(0, iy - 2), min(height, iy + 3))
        cols = np.arange(ix - 2, ix + 3) % width
        sample_depth = depth[np.ix_(rows, cols)]
        sample_index = index[np.ix_(rows, cols)]
        matched = (sample_depth > 0) & (np.rint(sample_index) == person_id)
        visible[j] = bool(
            matched.any() and np.min(np.abs(sample_depth[matched] - distance[j])) < 0.04
        )
    return valid, visible


def visible_at_instance_points(
    xy: np.ndarray, instance_mask: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    """Legacy instance-only visibility: rounded center, nonperiodic 7x7 neighborhood.

    valid_mask gates the rounded center; no ray-distance or occlusion-depth test."""
    height, width = instance_mask.shape
    visible = np.zeros(len(xy), dtype=bool)
    for index, (x, y) in enumerate(xy):
        if not np.isfinite([x, y]).all() or not (0 <= x < width and 0 <= y < height):
            continue
        ix, iy = int(round(x)), int(round(y))
        if not (0 <= ix < width and 0 <= iy < height) or not valid_mask[iy, ix]:
            continue
        region = instance_mask[max(0, iy - 3) : iy + 4, max(0, ix - 3) : ix + 4]
        visible[index] = bool(region.any())
    return visible
