"""GT-selected groups: rejected predictions remain in the error denominator."""
import numpy as np

from densekp.core.keypoint_supervision import inside_full_crop


def summarize(values):
    """Summarize errors strictly: empty statistics are None; nonfinite raises.

    Keep this policy separate from the failure-status candidate summary in
    ``densekp_retrain.evaluation.candidates``. Neither drops rejected points.
    """
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite errors must not disappear from the denominator')
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
                median=float(np.median(values)) if values.size else None,
                p90=float(np.percentile(values, 90)) if values.size else None)


def point_errors(predicted_crop, target_crop, predicted_erp, target_erp, width):
    """Return (crop_px, erp_px), each shape (N,), from four (N, 2) arrays.

    Crop XY uses the full 256-square normalized frame (center = 0), including
    the columns outside the network's middle 192-column input. ERP XY and
    ``width`` must use the same image's pixel units; X wraps periodically.
    No prediction third value is consumed or interpreted as a weight.
    """
    crop = np.linalg.norm(predicted_crop - target_crop, axis=1) * 256
    delta = predicted_erp - target_erp
    delta[:, 0] = (delta[:, 0] + width / 2) % width - width / 2
    return crop, np.linalg.norm(delta, axis=1)


def evaluation_masks(in_view, visible, target_crop):
    """Return named (N,) boolean groups from GT, independent of prediction.

    ``in_view`` and ``visible`` are (N,); ``target_crop`` is (N, 2) normalized
    full-square XY. Reliable occluded GT inside the full crop is supervised.
    Middle-192-column groups are diagnostics, not the supervision boundary.
    """
    in_view = np.asarray(in_view, dtype=bool)
    visible = np.asarray(visible, dtype=bool) & in_view
    in_crop = inside_full_crop(target_crop)
    retained = in_crop & (target_crop[:, 0] >= -.375) & (target_crop[:, 0] < .375)
    occluded = in_view & ~visible
    return dict(all_in_view=in_view, visible=visible, occluded=occluded,
                supervised=in_view & in_crop, occluded_in_crop=occluded & in_crop,
                outside_crop=in_view & ~in_crop,
                visible_retained_input=visible & retained,
                occluded_retained_input=occluded & retained)


def prediction_retention(eligible, prediction_valid):
    """Count rejection for two (N,) boolean masks; empty rates remain None."""
    count = int(eligible.sum())
    retained = int((eligible & prediction_valid).sum())
    return dict(eligible=count, retained=retained, rejected=count-retained,
                rejection_rate=(count-retained)/count if count else None)


def pair_masks(in_view_a, visible_a, in_view_b, visible_b):
    """Partition shared eligibility using aligned (N,) boolean masks of two views."""
    shared = in_view_a & in_view_b
    return dict(both_in_view=shared,
                both_visible=shared & visible_a & visible_b,
                one_visible=shared & (visible_a ^ visible_b),
                both_occluded=shared & ~visible_a & ~visible_b)
