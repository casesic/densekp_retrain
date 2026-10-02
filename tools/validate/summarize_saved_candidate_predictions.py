"""Finish aggregation of a fully saved batch after a report-only failure; no inference."""

import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.evaluation.candidates import aggregate, summary
from densekp_retrain.io.json import write_json
from densekp_retrain.evaluation.pairs import collect_pair_metrics
from densekp.core.evaluation import evaluation_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    started = time.monotonic()
    output = args.directory.resolve()
    if (output / "report.json").exists():
        raise FileExistsError("Do not overwrite a completed report")
    baseline = json.loads(args.reference.read_text())
    refs = {case["id"]: case for case in baseline["cases"] if case["batch"] == args.label}
    index = json.loads((output / f"{args.label}_frozen_index.json").read_text())
    cases = [
        json.loads((output / f"{args.label}_{position:03d}/case.json").read_text())
        for position in range(len(index["rows"]))
    ]
    assert set(refs) == {case["id"] for case in cases}
    states = {}
    pairs = {}
    fixed = {}
    for case in cases:
        saved = np.load(case["prediction"])
        reference = np.load(refs[case["id"]]["prediction"])
        for key in [
            "target_crop_xy",
            "target_erp_xy",
            "target_world_xyz",
            "affine",
            "visible",
            "in_view",
            "supervised",
            "network_input",
        ]:
            np.testing.assert_array_equal(saved[key], reference[key])
        assert (
            saved["prediction_raw"].shape == (138, 3) and np.isfinite(saved["prediction_raw"]).all()
        )
        masks = evaluation_masks(saved["in_view"], saved["visible"], saved["target_crop_xy"])
        masks["visible_supervised"] = masks["supervised"] & saved["visible"]
        states[case["id"]] = dict(
            crop=saved["crop_error_px"],
            erp=saved["erp_periodic_error_px"],
            masks=masks,
            visible=saved["visible"],
        )
        pairs.setdefault((case["batch"], case["scene"], case["person_id"]), {})[case["camera"]] = (
            case["id"]
        )
        fixed.setdefault((case["action"], case["category"]), case["id"])
    pair_results, buckets = collect_pair_metrics(pairs, states, box_variants=False)
    write_json(
        output / "report.json",
        dict(
            status="completed",
            cases=cases,
            by_batch={args.label: aggregate(cases, states)},
            combined=aggregate(cases, states),
            checkpoint=json.loads((output / "checkpoint_load.json").read_text()),
            pairs=pair_results,
            paired_aggregate={
                k: dict(mean_endpoint=summary(v["mean"]), worse_endpoint=summary(v["worst"]))
                for k, v in buckets.items()
            },
            prediction_failure_cases=[],
            fixed_controls=list(fixed.values()),
            recovery="79 predictions completed before overbroad train+val reference-set check failed; exact selected-batch denominator and arrays verified; CPU aggregation only",
            reference_report=str(args.reference),
            seconds=time.monotonic() - started,
        ),
    )


if __name__ == "__main__":
    main()
