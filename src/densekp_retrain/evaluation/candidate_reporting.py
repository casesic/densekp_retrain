"""Application-level paired/group summaries and candidate report output."""

import time
from .candidates import aggregate, summary
from .pairs import collect_pair_metrics
from densekp_retrain.io.json import write_json
from densekp_retrain.visualization.keypoints import save_contact_sheets


def summarize_observations(inputs, observations):
    "Return paired reports/raw buckets, per-batch metrics and stable worst-case order. Explicit empty actions remain null, never zero-pixel results."
    cases = observations.cases
    states = observations.states
    sources = inputs.sources
    pair_results, pair_values = collect_pair_metrics(observations.pairs, states)
    by_batch = {
        batch: aggregate([c for c in cases if c["batch"] == batch], states) for batch, _ in sources
    }
    for metrics in by_batch.values():
        for action in ("cycling", "standing", "walking"):
            metrics.setdefault(
                "action/" + action + "/supervised",
                dict(
                    rows=0,
                    point_weighted=dict(crop_256_px=summary([]), erp_periodic_px=summary([])),
                    person_weighted=dict(crop_256_px=summary([]), erp_periodic_px=summary([])),
                ),
            )
    worst = sorted(
        cases,
        key=lambda c: (
            float("inf")
            if c["metrics"]["supervised"]["crop_256_px"]["nonfinite_count"]
            else c["metrics"]["supervised"]["crop_256_px"]["mean"] or -1
        ),
        reverse=True,
    )[:8]
    return pair_results, pair_values, by_batch, worst


def write_evaluation_report(inputs, loaded, observations, output, reference_report, started):
    "Summarize and write the report after panels, retaining checkpoint and reference-denominator checks."
    cases = observations.cases
    states = observations.states
    fixed = observations.fixed
    references = inputs.references
    partitions = inputs.partitions
    split_scope = inputs.split_scope
    checkpoint = loaded.checkpoint
    stat = loaded.checkpoint_stat
    load_audit = loaded.audit
    pair_results, pair_values, by_batch, worst = summarize_observations(inputs, observations)
    save_contact_sheets(output, list(fixed.values()), [case["id"] for case in worst])
    if (
        checkpoint.stat().st_mtime_ns != stat.st_mtime_ns
        or checkpoint.stat().st_size != stat.st_size
    ):
        raise ValueError("Checkpoint changed during inference")
    if reference_report and set(references) != {c["id"] for c in cases}:
        raise ValueError("Reference denominator changed")
    report = dict(
        status="completed",
        split=split_scope,
        partitions=partitions,
        checkpoint=load_audit,
        cases=cases,
        by_batch=by_batch,
        combined=aggregate(cases, states),
        pairs=pair_results,
        paired_aggregate={
            key: dict(mean_endpoint=summary(v["mean"]), worse_endpoint=summary(v["worst"]))
            for key, v in pair_values.items()
        },
        fixed_controls=list(fixed.values()),
        worst_samples=[c["id"] for c in worst],
        seconds=time.monotonic() - started,
        prediction_failure_cases=[
            c["id"] for c in cases if c["prediction_status"] == "nonfinite_failure"
        ],
        metric_definition="Unweighted valid-GT point means and equal-person means, full 256-square px and periodic ERP px; no 32px shift",
        visibility="visible_supervised and occluded_in_crop partition supervised; all_in_view additionally diagnoses GT outside crop",
        nonfinite_policy="nonfinite XY is infinite error; count retained, aggregate mean/quantiles null and failure reported; raw third value never screens points",
        keypoint_value="all raw third values saved unchanged; not confidence, weight, or selection criterion",
        pair_definition="same world person/semantic row, mean and max of two per-view GT endpoint errors; both supervised required except named diagnostic group",
        reference_report=str(reference_report) if reference_report else None,
        frozen_input_comparison=(
            "all actual inputs, GT, affine, visibility and supervision equal"
            if references
            else None
        ),
        limitation="few source groups; batches retain distinct purposes, combined is not a generalization score; no independent test or detector",
    )
    write_json(output / "report.json", report)
