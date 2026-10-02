"""Replay saved predictions through old/new candidate CLI; never load a model."""

import argparse
import importlib
import json
from pathlib import Path
import shutil
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
from PIL import Image
import torch


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def prepare_inputs(acceptance):
    inputs = acceptance / "inputs"
    inputs.mkdir(exist_ok=False)
    chosen = json.loads((acceptance / "selected_case_ids.json").read_text())
    sources, references, replay = [], [], []
    for report_path, ids in chosen.items():
        report = json.loads(Path(report_path).read_text())
        cases = {case["id"]: case for case in report["cases"]}
        label = cases[ids[0]]["batch"]
        original_index = json.loads(
            (Path(report_path).parent / (label + "_frozen_index.json")).read_text()
        )
        rows = []
        for position, case_id in enumerate(ids):
            case = dict(cases[case_id])
            original_position = int(case_id.rsplit("_", 1)[1])
            rows.append(original_index["rows"][original_position])
            case["id"] = f"{label}_{position:03d}"
            references.append(case)
            replay.append(dict(id=case["id"], prediction=case["prediction"]))
        index = dict(original_index, split="integration_only", rows=rows)
        path = inputs / (label + ".json")
        write(path, index)
        sources.append(dict(label=label, index=str(path)))
    write(inputs / "reference.json", dict(cases=references))
    write(
        inputs / "replay.json",
        dict(
            sources=sources,
            predictions=replay,
            scope="11 saved observations; integration-only replay fixture, not a new source split",
        ),
    )


def load_original(path):
    module = types.ModuleType("original_candidate_reporting")
    module.__file__ = str(ROOT / "tools/validate/evaluate_candidate_baseline.py")
    exec(compile(path.read_text(), module.__file__, "exec"), module.__dict__)
    return module


def normalized_json(path, output):
    value = json.loads(path.read_text())
    # Exhaustive allowlist: no metric, count, failure status or source path removed.
    if path.name == "selection_frozen.json":
        value["utc"] = "<UTC>"
    if path.name == "checkpoint_load.json":
        value["seconds"] = "<ELAPSED>"
    if path.name == "report.json":
        value["seconds"] = "<ELAPSED>"
        value["checkpoint"]["seconds"] = "<ELAPSED>"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).replace(
        str(output.resolve()), "<OUTPUT>"
    )


def compare_outputs(expected, actual):
    old_files = sorted(
        str(path.relative_to(expected)) for path in expected.rglob("*") if path.is_file()
    )
    new_files = sorted(
        str(path.relative_to(actual)) for path in actual.rglob("*") if path.is_file()
    )
    assert old_files == new_files, (old_files, new_files)
    counts = dict(json=0, arrays=0, images=0, other=0)
    for name in old_files:
        before, after = expected / name, actual / name
        if after.suffix == ".json":
            assert normalized_json(before, expected) == normalized_json(after, actual), name
            counts["json"] += 1
        elif after.suffix in (".npz", ".npy"):
            if after.suffix == ".npz":
                with (
                    np.load(before, allow_pickle=False) as left,
                    np.load(after, allow_pickle=False) as right,
                ):
                    assert left.files == right.files, name
                    for key in left.files:
                        assert (
                            left[key].shape == right[key].shape
                            and left[key].dtype == right[key].dtype
                        )
                        np.testing.assert_array_equal(
                            left[key], right[key], err_msg=name + "/" + key
                        )
                        counts["arrays"] += 1
            else:
                np.testing.assert_array_equal(np.load(before), np.load(after))
                counts["arrays"] += 1
        elif after.suffix in (".png", ".jpg"):
            assert before.read_bytes() == after.read_bytes(), name
            np.testing.assert_array_equal(
                np.asarray(Image.open(before)), np.asarray(Image.open(after))
            )
            counts["images"] += 1
        else:
            assert before.read_bytes() == after.read_bytes(), name
            counts["other"] += 1
    return counts


def replay_recovery(acceptance, output, capture):
    """Exercise the historical report-only CLI on already saved finite cases."""
    if capture:
        source = (
            acceptance
            / "before/source_snapshot/tools/validate/summarize_saved_candidate_predictions.py"
        )
        module = types.ModuleType("original_saved_report_recovery")
        module.__file__ = str(ROOT / "tools/validate/summarize_saved_candidate_predictions.py")
        exec(compile(source.read_text(), module.__file__, "exec"), module.__dict__)
        # Use the independent old writer too, not its current compatibility export.
        module.write = load_original(
            acceptance / "before/source_snapshot/tools/validate/evaluate_candidate_baseline.py"
        ).write
    else:
        module = importlib.import_module("tools.validate.summarize_saved_candidate_predictions")
    source = acceptance / "candidate_before/finite"
    records = []
    for label in ("val_baseline", "challenge"):
        destination = output / label
        destination.mkdir()
        index_name = label + "_frozen_index.json"
        for name in (index_name, "checkpoint_load.json"):
            shutil.copyfile(source / name, destination / name)
        rows = json.loads((source / index_name).read_text())["rows"]
        for position in range(len(rows)):
            case_id = f"{label}_{position:03d}"
            (destination / case_id).mkdir()
            shutil.copyfile(source / case_id / "case.json", destination / case_id / "case.json")
        argv = [
            "summarize_saved_candidate_predictions.py",
            "--directory",
            str(destination),
            "--label",
            label,
            "--reference",
            str(acceptance / "inputs/reference.json"),
        ]
        with patch.object(sys, "argv", argv):
            module.main()
        record = dict(label=label, rows=len(rows))
        if not capture:
            record["comparison"] = compare_outputs(
                acceptance / "recovery_before" / label, destination
            )
        records.append(record)
    return dict(
        status="passed",
        phase="capture" if capture else "verify",
        reports=records,
        model_inference=False,
        allowed_changes=["report.seconds", "this run output-root prefix only"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["capture", "verify"], required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recovery-only", action="store_true")
    args = parser.parse_args()
    acceptance = args.acceptance.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    capture = args.phase == "capture"
    if args.recovery_only:
        result = replay_recovery(acceptance, args.output, capture)
        write(args.output / "result.json", result)
        print(json.dumps(result, indent=2))
        return
    if capture:
        prepare_inputs(acceptance)
        module = load_original(
            acceptance / "before/source_snapshot/tools/validate/evaluate_candidate_baseline.py"
        )
        targets = [module]
    else:
        module = importlib.import_module("tools.validate.evaluate_candidate_baseline")
        targets = [importlib.import_module("densekp_retrain.evaluation.candidate_runner")]
    specification = json.loads((acceptance / "inputs/replay.json").read_text())
    records = []
    for scenario in ("finite", "nonfinite", "invalid_shape"):
        predictions = []
        for entry in specification["predictions"]:
            with np.load(entry["prediction"], allow_pickle=False) as archive:
                predictions.append(
                    {key: archive[key] for key in ("prediction_raw", "network_input")}
                )
        if scenario == "nonfinite":
            predictions[0]["prediction_raw"][0, 0] = np.nan
            predictions[1]["prediction_raw"][1, 2] = np.inf
        if scenario == "invalid_shape":
            predictions[1]["prediction_raw"] = predictions[1]["prediction_raw"][:137]
        consumed = []

        def replay_batch(model, images, device):
            saved = predictions[len(consumed)]
            np.testing.assert_array_equal(images[0, :, :, 32:-32], saved["network_input"])
            consumed.append(len(consumed))
            return torch.from_numpy(saved["prediction_raw"][None])

        def load_stub(checkpoint, cfg):
            return torch.nn.Identity(), dict(
                epoch=2,
                global_step=55000,
                state_dict_keys=496,
                missing_keys=[],
                unexpected_keys=[],
                strict=True,
            )

        output = args.output / scenario
        argv = [
            "evaluate_candidate_baseline.py",
            "--indices",
            *[entry["index"] for entry in specification["sources"]],
            "--labels",
            *[entry["label"] for entry in specification["sources"]],
            "--output",
            str(output),
            "--reference-report",
            str(acceptance / "inputs/reference.json"),
            "--device",
            "cpu",
        ]
        error = None
        with (
            patch.object(sys, "argv", argv),
            patch.object(targets[0], "load_checkpoint_state_dict", load_stub),
            patch.object(targets[0], "predict_batch", replay_batch),
        ):
            try:
                module.main()
            except ValueError as failure:
                error = dict(type=type(failure).__name__, message=str(failure))
        if scenario == "invalid_shape":
            assert error == dict(type="ValueError", message="Model output shape is not 138 x 3")
        else:
            assert error is None, error
        record = dict(scenario=scenario, consumed=len(consumed), error=error)
        if not capture:
            record["comparison"] = compare_outputs(
                acceptance / "candidate_before" / scenario, output
            )
        records.append(record)
    result = dict(
        status="passed",
        phase=args.phase,
        scenarios=records,
        model_inference=False,
        allowed_changes=[
            "selection_frozen.utc",
            "checkpoint_load.seconds",
            "report.seconds",
            "report.checkpoint.seconds",
            "this run output-root prefix only",
        ],
    )
    write(args.output / "result.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
