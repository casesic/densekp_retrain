"""Strict checkpoint inference on frozen rows, with optional frozen-input comparison."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from densekp_retrain.evaluation.candidate_inputs import prepare_inputs, write_selection
from densekp_retrain.evaluation.candidate_runner import load_evaluation_model, evaluate_rows
from densekp_retrain.evaluation.candidate_reporting import write_evaluation_report
from densekp_retrain.io.json import write_json

# Historical imports remain available; current callers use their defining modules.
from densekp_retrain.io.json import write_json as write
from densekp_retrain.visualization.keypoints import render_candidate_panel
from densekp_retrain.evaluation.candidates import aggregate, summary


def render_panel(item, pred, supervised, visible, label):
    """Historical import and keyword signature; active code uses visualization."""
    return render_candidate_panel(item, pred, supervised, visible, label)


def parse_args():
    "Parse the unchanged candidate evaluation command-line interface."
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indices", type=Path, nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--split-manifest", type=Path, help="Required for source-isolated train/val indices"
    )
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "assets_external/densekp.ckpt")
    parser.add_argument("--model-config", type=Path)
    parser.add_argument(
        "--reference-report",
        type=Path,
        help="Require identical case IDs, images, GT, affine and masks to this baseline",
    )
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main():
    args = parse_args()
    if len(args.labels) != len(args.indices):
        raise ValueError("Each index needs a batch label")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    inputs = prepare_inputs(args.indices, args.labels, args.reference_report, output)
    if inputs.partition_indices:
        if not args.split_manifest:
            raise ValueError("Source-isolated indices need their split manifest")
        from densekp_retrain.data.source_split import validate_split

        isolation = validate_split(
            json.loads(args.split_manifest.read_text()), inputs.partition_indices
        )
        write_json(output / "source_isolation.json", isolation)
    write_selection(inputs, output)
    loaded = load_evaluation_model(
        args.checkpoint, args.model_config, args.device, ROOT / "configs/hydra", output
    )
    observations = evaluate_rows(inputs, loaded, output, compare_frozen=bool(args.reference_report))
    write_evaluation_report(inputs, loaded, observations, output, args.reference_report, started)


if __name__ == "__main__":
    main()
