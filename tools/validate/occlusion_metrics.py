"""GT-selected groups: rejected predictions remain in the error denominator."""
import numpy as np

from densekp.core.keypoint_supervision import inside_full_crop


def summarize(values):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite errors must not disappear from the denominator')
    return dict(count=int(values.size), mean=float(values.mean()) if values.size else None,
                median=float(np.median(values)) if values.size else None,
                p90=float(np.percentile(values, 90)) if values.size else None)


def point_errors(predicted_crop, target_crop, predicted_erp, target_erp, width):
    crop = np.linalg.norm(predicted_crop - target_crop, axis=1) * 256
    delta = predicted_erp - target_erp
    delta[:, 0] = (delta[:, 0] + width / 2) % width - width / 2
    return crop, np.linalg.norm(delta, axis=1)


def evaluation_masks(in_view, visible, target_crop):
    """Use GT only; full-image eligibility and crop eligibility stay separate."""
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
    """Count runtime rejection, including empty groups explicitly as null rates."""
    count = int(eligible.sum())
    retained = int((eligible & prediction_valid).sum())
    return dict(eligible=count, retained=retained, rejected=count-retained,
                rejection_rate=(count-retained)/count if count else None)


def pair_masks(in_view_a, visible_a, in_view_b, visible_b):
    shared = in_view_a & in_view_b
    return dict(both_in_view=shared,
                both_visible=shared & visible_a & visible_b,
                one_visible=shared & (visible_a ^ visible_b),
                both_occluded=shared & ~visible_a & ~visible_b)
