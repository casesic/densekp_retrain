"""Paired-view endpoint errors; preserve GT denominators and report order."""

import numpy as np
from .candidates import summary


def collect_pair_metrics(pairs, states, *, box_variants=True):
    """Return pair reports and ordered raw error buckets.

    Candidate keys include a box variant; historical recovery keys do not.
    Each state has per-point crop/ERP pixel errors and bool supervision/visibility
    masks. Missing views and empty intersections retain the original policy.
    """
    pair_results = []
    pair_values = {}
    for group_key, views in pairs.items():
        if box_variants:
            batch, scene, person_id, box_variant = group_key
        else:
            batch, scene, person_id = group_key
        if box_variants:
            result = dict(
                batch=batch, scene=scene, person_id=person_id, box_variant=box_variant, views=views
            )
        else:
            result = dict(batch=batch, scene=scene, person_id=person_id, views=views)
        if len(views) != 2:
            result.update(status="missing accepted given-box view; not detector failure")
            pair_results.append(result)
            continue
        left, right = [states[views[name]] for name in ("cam2", "cam4")]
        shared = left["masks"]["supervised"] & right["masks"]["supervised"]
        groups = dict(
            both_supervised=shared,
            both_visible=shared & left["visible"] & right["visible"],
            one_visible=shared & (left["visible"] ^ right["visible"]),
            both_occluded=shared & ~left["visible"] & ~right["visible"],
            both_in_view_diagnostic=left["masks"]["all_in_view"] & right["masks"]["all_in_view"],
        )
        result.update(status="scored", metrics={})
        for name, mask in groups.items():
            result["metrics"][name] = {}
            for unit in ("crop", "erp"):
                mean = (left[unit][mask] + right[unit][mask]) / 2
                worst = np.maximum(left[unit][mask], right[unit][mask])
                result["metrics"][name][unit] = dict(
                    mean_endpoint=summary(mean), worse_endpoint=summary(worst)
                )
                for label in (batch, "combined"):
                    bucket = pair_values.setdefault(
                        f"{label}/{name}/{unit}", dict(mean=[], worst=[])
                    )
                    bucket["mean"].extend(mean)
                    bucket["worst"].extend(worst)
        pair_results.append(result)
    return pair_results, pair_values
