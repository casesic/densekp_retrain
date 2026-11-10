"""Versioned observation availability, separate from raster/GT correctness."""
import numpy as np
from .boxes import BoxRejected, validate_box_context
from .raster_storage import validate_roi_box
from densekp.core.datasets.image_layout import parse_image_layout

LEGACY = 'strict_dual_crop_v1'
RECORD = 'record_unavailable_v1'
EXPECTED_UNAVAILABLE = {'no_visible_instance_box', 'below_min_side_px',
                        'erp_crop_seam_crossing', 'no_GT_inside_full_crop'}


def policy_name(recipe):
    name = recipe.get('observation_policy', LEGACY)
    if name not in (LEGACY, RECORD):
        raise ValueError('unknown_observation_policy')
    return name


def no_box_state(in_view, visible, evidence=None):
    if np.any(visible & ~in_view):
        raise ValueError('visible_outside_physical_domain')
    if not np.any(in_view):
        return 'person_absent', 'outside_physical_domain_no_instance_box'
    if evidence is not None and np.all(evidence[in_view] == 2) and not np.any(visible):
        return 'empty', 'fully_neighbor_occluded_no_native_box'
    return 'empty', 'no_visible_instance_box'


def validate_native_roi(box, entry, pid, annotation_pid, meta, valid, pm):
    """Check provenance/support BEFORE permitting a tiny or absent native box.

    ROI coverage failures, malformed metadata and mismatched tight masks remain
    hard errors. Only the small explicit EXPECTED_UNAVAILABLE set is recoverable.
    """
    layout = parse_image_layout(meta)
    if layout is None or layout.kind != 'roi_image':
        raise ValueError('required_roi_image_layout')
    if valid.shape != pm.shape or tuple(pm.shape[::-1]) != layout.stored_size_wh:
        raise ValueError('mask_image_size_mismatch')
    if pid != entry['person_id'] or pid != annotation_pid or not entry.get('source'):
        raise ValueError('native_box_identity_or_source_conflict')
    if np.any(pm & ~valid):
        raise ValueError('person_mask_outside_physical_domain')
    if box is None:
        if pm.any():
            raise ValueError('null_box_with_nonempty_instance_mask')
        raise BoxRejected('no_visible_instance_box')
    local = np.asarray(box) - np.tile(layout.offset_xy, 2)
    validate_box_context(local, bbox_kind='visible_instance_tight', bbox_source=entry['source'],
        person_id=pid, mask_person_id=entry['person_id'], annotation_person_id=annotation_pid,
        image_size_wh=layout.stored_size_wh, manifest_size_wh=layout.stored_size_wh,
        valid_mask=valid, person_mask=pm, boundary_policy='raster_clip', min_side_px=0.001)
    return validate_roi_box(box, entry, pid, annotation_pid, meta, valid, pm)


def pair_availability(support, cameras=('cam2', 'cam4'), minimum=8):
    """None means no legal crop; a legal crop with zero GT has an empty set.

    has_dual_input describes legal image crops, not NPZ row eligibility.
    Zero-supervision crops are still excluded from training rows.
    """
    missing = [c for c in cameras if support.get(c) is None]
    common = None if missing else int(np.logical_and(*(support[c] for c in cameras)).sum())
    return dict(common_points=common, common_set_defined=not missing,
        evaluable_point_count=0 if common is None else common,
        has_dual_input=not missing, sufficient_common_support=common is not None and common >= minimum,
        single_side_only=len(missing) == 1, unavailable_cameras=missing,
        reason=('no_legal_crop:' + ','.join(missing)) if missing else
               ('empty_common_crop_support' if common == 0 else
                ('common_crop_below_minimum' if common < minimum else '')))


def enforce_crop_policy(recipe, simultaneous, availability):
    if policy_name(recipe) == LEGACY and simultaneous and not availability['sufficient_common_support']:
        raise ValueError('actual_dual_edge_crop_common_below8')
