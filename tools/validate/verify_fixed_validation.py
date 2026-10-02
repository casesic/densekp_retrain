"""Real small-GPU callback baseline and migration acceptance; no DenseKP updates."""

import argparse
import gc
import importlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import numpy as np
import torch
from tests.fixed_validation_fixture import run_case


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["capture", "verify"], required=True)
    parser.add_argument("--acceptance", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    acceptance = args.acceptance.resolve()
    capture = args.phase == "capture"
    name = (
        "tools.experiments.train_source_isolated_trial"
        if capture
        else "densekp_retrain.training.trial_callbacks"
    )
    callback = importlib.import_module(name).FixedValidation
    if capture and callback.__module__ != name:
        raise RuntimeError(
            "Capture requires the independent old callback, not a compatibility export"
        )
    gpu = subprocess.run(["nvidia-smi", "-i", "1"], capture_output=True, text=True)
    (args.output / "gpu_preflight.txt").write_text(gpu.stdout + gpu.stderr)
    if gpu.returncode or not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; callback acceptance needs real GPU RNG/autocast")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    source = ROOT / "outputs/acceptance/refactor_metrics_config_20261001/before"
    before = acceptance / "callback_before"
    records = []
    scenarios = [
        "success",
        "resumed",
        "nan",
        "invalid_shape",
        "cpu_rng",
        "cuda_rng",
        "parameter_version",
        "step_mutation",
        "save_failure",
        "duplicate_folder",
    ]
    for scenario in scenarios:
        print("Callback scenario: " + scenario, flush=True)
        resume = (
            str(before / "success/step_0002/checkpoint.ckpt") if scenario == "resumed" else None
        )
        metadata, arrays = run_case(
            callback,
            source,
            args.output / scenario,
            None if scenario in ("success", "resumed") else scenario,
            resume,
        )
        (args.output / (scenario + ".json")).write_text(json.dumps(metadata, indent=2) + "\n")
        np.savez(args.output / (scenario + ".npz"), **arrays)
        if not capture:
            expected = json.loads((before / (scenario + ".json")).read_text())
            # Optimizer param_groups includes tuple-valued betas. Compare the
            # serialized representation on both sides, without dropping fields.
            assert json.loads(json.dumps(metadata)) == expected, scenario + " metadata changed"
            with np.load(before / (scenario + ".npz"), allow_pickle=False) as previous:
                assert set(previous.files) == set(arrays), scenario
                for key, value in arrays.items():
                    assert (
                        value.dtype == previous[key].dtype and value.shape == previous[key].shape
                    ), key
                    np.testing.assert_array_equal(
                        value, previous[key], err_msg=scenario + "/" + key
                    )
        records.append(
            dict(
                scenario=scenario,
                arrays=len(arrays),
                step=metadata["global_step"],
                reports=[report["step"] for report in metadata["reports"]],
                error=metadata["error"],
                state_key=metadata["state_key"],
                callback_state=metadata["callback_state"],
            )
        )
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(
        status="passed",
        phase=args.phase,
        callback_module=callback.__module__,
        scenarios=records,
        real_gpu=True,
        training="tiny network only; 4 success updates, 2 resumed, 2 per failure scenario",
        checkpoint_resume="independent old step_0002 checkpoint",
        comparison="exact except report.seconds and output path prefix",
        densekp_parameter_updates=False,
    )
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
