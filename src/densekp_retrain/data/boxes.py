"""Pure box calculations with explicit historical policies.

Coordinates are source-raster pixels. change_box retains the synthetic trial's
1536 x 768 clipping; official_bbox retains BEDLAM's joint-based inclusive gate.
Neither function samples random numbers or chooses experiment quotas.
"""

import numpy as np


HISTORICAL_MIN_SIDE_PX = 2
HISTORICAL_SHIFT_REFERENCE_EXPANSION = 1.2
CROP_EXPANSIONS = {"deployment": 1.0, "historical_trial": 1.2}
BBOX_KINDS = frozenset({"visible_instance_tight", "fixed_reference", "detection_tracking",
                        "manual", "synthetic_perturbation"})


def _perturb_box(box, dx, dy, factor, edge, fraction, shift_reference_expansion):
    box = np.asarray(box, float).copy()
    center = (box[:2] + box[2:]) / 2
    size = box[2:] - box[:2]
    center += np.array([dx, dy]) * size.max() * shift_reference_expansion
    box = np.r_[center - size * factor / 2, center + size * factor / 2]
    if edge is not None:
        axis = edge % 2
        box[edge] += (1 if edge < 2 else -1) * size[axis] * fraction
    return box


def change_box(box, dx=0.0, dy=0.0, factor=1.0, edge=None, fraction=0.0):
    """Historical 1536x768 raster policy; preserve operations and permissive inputs."""
    box = _perturb_box(box, dx, dy, factor, edge, fraction,
                       HISTORICAL_SHIFT_REFERENCE_EXPANSION)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, 1536)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, 768)
    return box


def official_bbox(joints2d: np.ndarray, image_height: int, image_width: int):
    """复现 BEDLAM 主体 bbox：前 22 joints、图内过滤、1.2 扩张。"""
    body = np.asarray(joints2d, dtype=np.float64)[:22]
    finite = np.isfinite(body[:, :2]).all(axis=1)
    valid = finite & (body[:, 0] >= 0) & (body[:, 0] <= image_width)
    valid &= (body[:, 1] >= 0) & (body[:, 1] <= image_height)
    selected = body[valid]
    if len(selected) < 8:
        return None
    xmin, ymin = selected[:, :2].min(axis=0)
    xmax, ymax = selected[:, :2].max(axis=0)
    tight_bbox = np.array([xmin, ymin, xmax, ymax], dtype=np.float32)
    center = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2], dtype=np.float32)
    scale = np.float32(max(xmax - xmin, ymax - ymin) / 200.0 * 1.2)
    if not np.isfinite(scale) or scale <= 0:
        return None
    return center, scale, np.int32(len(selected)), tight_bbox


def box_rejection(box):
    """Trial detector gate, after the caller's float32 conversion."""
    if not np.isfinite(box).all() or min(box[2:] - box[:2]) < HISTORICAL_MIN_SIDE_PX:
        return "invalid_box"
    return None


def trial_crop_from_box(box):
    """Float32 trial xyxy -> center and scale (pixel side / 200).

    Caller converts xyxy to float32 before the rejection gate, as historically.
    Keep multiplication before division to retain the original rounding.
    """
    return _crop_from_box(box, CROP_EXPANSIONS["historical_trial"])


def _crop_from_box(box, expansion):
    center = (box[:2] + box[2:]) / 2
    scale = np.float32(max(box[2:] - box[:2]) * expansion / 200)
    return center, scale


def support_rejection(supervision, target_pixels, min_points, min_pixels):
    """The experiment supplies its thresholds; visibility never gates GT."""
    if supervision.sum() < min_points or target_pixels.sum() < min_pixels:
        return "empty_or_insufficient_GT_or_visible_target_support"
    return None


class BoxRejected(ValueError):
    """Stable reason code for new consumers; historical errors stay unchanged."""


def rejection_record(reason, **details):
    """Strict JSON-safe diagnostics, including invalid numeric proposals."""
    def safe(value):
        if isinstance(value, np.ndarray):
            return safe(value.tolist())
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not np.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value
    return safe(dict(reason=str(reason), **details))


def validate_image_size(image_size_wh):
    size = np.asarray(image_size_wh, float)
    if (size.shape != (2,) or not np.isfinite(size).all()
            or (size <= 0).any() or (size != np.floor(size)).any()):
        raise BoxRejected("invalid_image_size_wh")
    return tuple(int(v) for v in size)


def _check_box(box, image_size_wh, boundary_policy, min_side_px):
    width, _ = validate_image_size(image_size_wh)
    if boundary_policy not in ("raster_clip", "erp_reject_seam"):
        raise BoxRejected("unknown_boundary_policy")
    if not np.isfinite(min_side_px) or min_side_px <= 0:
        raise BoxRejected("invalid_min_side_px")
    box = np.asarray(box, float)
    if box.shape != (4,) or not np.isfinite(box).all():
        raise BoxRejected("nonfinite_or_malformed_box")
    if (box[2:] <= box[:2]).any():
        raise BoxRejected("reversed_or_empty_box")
    # ERP uses ordered, unwrapped xyxy. No endpoint modulo or min/max repair.
    if boundary_policy == "erp_reject_seam" and (box[0] < 0 or box[2] > width):
        raise BoxRejected("erp_seam_crossing")
    if min(box[2:] - box[:2]) < min_side_px:
        raise BoxRejected("below_min_side_px")
    return box


def change_box_for_image(box, *, image_size_wh, boundary_policy, min_side_px,
                         shift_reference_expansion, dx=0.0, dy=0.0, factor=1.0,
                         edge=None, fraction=0.0):
    """Explicit new path. dx/dy units are original maxside * shift_reference_expansion.

    That normalization is independent of the subsequent crop expansion policy.
    Ordinary rasters clip all four sides; ERP rejects horizontal crossing before
    clipping, while vertical image limits still clip. Thresholds are raster px.
    """
    width, height = validate_image_size(image_size_wh)
    box = _check_box(box, image_size_wh, boundary_policy, min_side_px)
    params = [dx, dy, factor, fraction, shift_reference_expansion]
    if (not np.isfinite(params).all() or factor <= 0 or shift_reference_expansion <= 0
            or not 0 <= fraction <= 1 or (edge is not None and (
                isinstance(edge, (bool, np.bool_)) or not isinstance(edge, (int, np.integer))
                or edge not in (0, 1, 2, 3)))):
        raise BoxRejected("invalid_perturbation_parameters")
    with np.errstate(over="ignore", invalid="ignore"):
        changed = _perturb_box(box, dx, dy, factor, edge, fraction, shift_reference_expansion)
    changed = _check_box(changed, image_size_wh, boundary_policy, min_side_px)
    changed[[0, 2]] = np.clip(changed[[0, 2]], 0, width)
    changed[[1, 3]] = np.clip(changed[[1, 3]], 0, height)
    with np.errstate(over="ignore"):
        changed = changed.astype(np.float32)
    _check_box(changed, image_size_wh, boundary_policy, min_side_px)
    return changed


def crop_from_box(box, *, image_size_wh, boundary_policy, crop_policy, min_side_px):
    """Explicit named expansion; reject an ERP square crop crossing the seam too."""
    _check_box(box, image_size_wh, boundary_policy, min_side_px)
    if crop_policy not in CROP_EXPANSIONS:
        raise BoxRejected("unknown_crop_policy")
    with np.errstate(over="ignore", invalid="ignore"):
        center, scale = _crop_from_box(np.asarray(box, np.float32), CROP_EXPANSIONS[crop_policy])
    if not np.isfinite(center).all() or not np.isfinite(scale) or scale <= 0:
        raise BoxRejected("invalid_crop")
    # Match Dataset's scale * 200 without an extra float32 cast of the side.
    side = scale * 200
    if boundary_policy == "erp_reject_seam" and (
            center[0] - side / 2 < 0 or center[0] + side / 2 > image_size_wh[0]):
        raise BoxRejected("erp_crop_seam_crossing")
    return center, scale


def validate_box_context(box, *, bbox_kind, bbox_source, person_id, mask_person_id,
                         annotation_person_id, image_size_wh, manifest_size_wh,
                         valid_mask, person_mask, boundary_policy, min_side_px):
    """Validate identity, raster agreement and target support separately from clipping.

    All kinds require declared provenance and intersecting valid target pixels.
    Only visible_instance_tight must equal the visible instance mask's tight box.
    Fixed/detector/manual boxes do not claim amodal GT or verified track identity.
    """
    size = validate_image_size(image_size_wh)
    if validate_image_size(manifest_size_wh) != size:
        raise BoxRejected("manifest_image_size_mismatch")
    shape = (size[1], size[0])
    if valid_mask.shape != shape or person_mask.shape != shape:
        raise BoxRejected("mask_image_size_mismatch")
    if bbox_kind not in BBOX_KINDS:
        raise BoxRejected("unknown_bbox_kind")
    if not isinstance(bbox_source, str) or not bbox_source.strip():
        raise BoxRejected("missing_bbox_source")
    if person_id is None or person_id != mask_person_id or person_id != annotation_person_id:
        raise BoxRejected("target_identity_mismatch")
    box = _check_box(box, size, boundary_policy, min_side_px)
    if (box[:2] < 0).any() or box[2] > size[0] or box[3] > size[1]:
        raise BoxRejected("box_outside_image")
    target = (person_mask > 0) & (valid_mask > 0)
    yy, xx = np.where(target)
    if len(xx) == 0:
        raise BoxRejected("empty_visible_target_mask")
    # A single tight rectangle cannot represent support on both seam edges.
    if boundary_policy == "erp_reject_seam" and xx.min() == 0 and xx.max() == size[0] - 1:
        raise BoxRejected("erp_target_support_at_both_seam_edges")
    if bbox_kind == "visible_instance_tight":
        tight = [xx.min(), yy.min(), xx.max() + 1, yy.max() + 1]
        if not np.array_equal(box, tight):
            raise BoxRejected("visible_instance_tight_mismatch")
    if not ((xx >= box[0]) & (xx < box[2]) & (yy >= box[1]) & (yy < box[3])).any():
        raise BoxRejected("box_has_no_valid_target_support")
    return target
