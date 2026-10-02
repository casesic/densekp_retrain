"""Pure candidate-report metrics, independent of model loading and CLI code."""
import numpy as np


def summary(values):
    """Summarize 1-D errors without removing nonfinite values from the count.

    Empty input produces ``empty``/None statistics. Any nonfinite error makes
    all statistics None with ``nonfinite_failure`` and its explicit count.
    Unlike ``densekp.core.evaluation.summarize``, this policy does not raise.
    """
    values = np.asarray(values, float)
    nonfinite_count = int((~np.isfinite(values)).sum())
    valid = len(values) > 0 and nonfinite_count == 0
    return dict(
        count=int(values.size),
        nonfinite_count=nonfinite_count,
        status='nonfinite_failure' if nonfinite_count else ('scored' if len(values) else 'empty'),
        mean=float(values.mean()) if valid else None,
        median=float(np.median(values)) if valid else None,
        p90=float(np.percentile(values, 90)) if valid else None,
    )


def aggregate(cases, states):
    """Group candidate errors by metadata and GT mask, preserving report order.

    Cases require id/action/category/scale_group; the three actual_* groups
    appear only when their source metadata exists. ``states[id]`` supplies
    (N,) crop/ERP pixel errors and named (N,) boolean masks. Crop errors are in
    full-256-square pixels; ERP errors use the source image resolution.

    ``rows`` includes empty-mask rows. Point weighting pools selected points;
    person weighting averages each nonempty row first, in its original dtype.
    Failed rows retain their denominator and propagate failure via summary.
    Iteration/reduction order is part of historical numerical compatibility.
    """
    groups = {}
    for case in cases:
        state = states[case['id']]
        prefixes = ['all', f"action/{case['action']}", f"layout/{case['category']}",
                    f"scale/{case['scale_group']}"]
        if 'lens_truncated_gt' in case:
            prefixes.append('actual_lens_gt/' + ('truncated' if case['lens_truncated_gt'] else 'complete'))
        if 'neighbor_crop_pixels' in case:
            prefixes.append('actual_crop_neighbor/' + ('present' if case['neighbor_crop_pixels'] > 0 else 'absent'))
        if 'in_view_outside_full_crop' in case:
            prefixes.append('actual_crop_gt/' + ('truncated' if case['in_view_outside_full_crop'] else 'complete'))
        for name, mask in state['masks'].items():
            for prefix in prefixes:
                group = groups.setdefault(prefix + '/' + name,
                                          dict(crop=[], erp=[], person_crop=[], person_erp=[], rows=0))
                group['crop'].extend(state['crop'][mask])
                group['erp'].extend(state['erp'][mask])
                group['rows'] += 1
                if mask.any():
                    group['person_crop'].append(float(state['crop'][mask].mean()))
                    group['person_erp'].append(float(state['erp'][mask].mean()))
    return {
        key: dict(
            rows=group['rows'],
            point_weighted=dict(crop_256_px=summary(group['crop']),
                                erp_periodic_px=summary(group['erp'])),
            person_weighted=dict(crop_256_px=summary(group['person_crop']),
                                 erp_periodic_px=summary(group['person_erp'])),
        )
        for key, group in groups.items()
    }
