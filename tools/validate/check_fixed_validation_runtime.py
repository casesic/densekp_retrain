"""Explicit runtime acceptance. Default: print a plan without computing or writing."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def checked_file(value):
    path = Path(value).resolve()
    if not path.is_file():
        raise ValueError(f"Missing input: {path}")
    return str(path)


def output_path(value):
    path = Path(value).resolve()
    if path.exists():
        raise ValueError(f"Output must be new: {path}")
    return str(path)


def device_name(value):
    if not (value.isdecimal() or (value.startswith("GPU-") and len(value) > 4)):
        raise ValueError("Device must be an explicit physical index or GPU UUID")
    return value


def gpu_preflight(selector, minimum_free):
    # Called only by --execute, before importing Torch. Never select another GPU.
    import subprocess
    query = ["nvidia-smi", "-i", selector,
             "--query-gpu=uuid,name,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(query, capture_output=True, text=True, check=True)
    lines = result.stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("Device query must identify exactly one GPU")
    uuid, name, free = [part.strip() for part in lines[0].split(",")]
    if not uuid.startswith("GPU-") or (selector.startswith("GPU-") and uuid != selector):
        raise RuntimeError("GPU UUID mismatch")
    if int(free) < minimum_free:
        raise RuntimeError("Insufficient free GPU memory; no automatic fallback")
    processes = subprocess.run(
        ["nvidia-smi", "-i", selector,
         "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    import os
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = uuid
    return dict(uuid=uuid, name=name, free_mib=int(free), logical_device="cuda:0",
                processes=processes.stdout.strip(), query=query)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


SCENARIOS = ("success", "resumed", "nan", "invalid_shape", "cpu_rng", "cuda_rng",
             "parameter_version", "step_mutation", "save_failure", "duplicate_folder")


def parser():
    p = argparse.ArgumentParser(description=__doc__ + " Requires source checkout including tests.")
    for name in ("source", "before", "output", "device"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--scenario", nargs="+", choices=SCENARIOS, required=True)
    p.add_argument("--resume-checkpoint")
    p.add_argument("--max-updates", type=int, required=True, help="Hard cumulative attempted update budget")
    p.add_argument("--execute", action="store_true")
    return p


def build_plan(args):
    if len(set(args.scenario)) != len(args.scenario):
        raise ValueError("Duplicate scenarios")
    if ("resumed" in args.scenario) != bool(args.resume_checkpoint):
        raise ValueError("Resume checkpoint is required exactly when resumed is selected")
    source, before = Path(args.source).resolve(), Path(args.before).resolve()
    checked_file(source / "cases.json")
    for case in ("val_baseline_002", "val_baseline_001", "val_baseline_004"):
        checked_file(source / (case + ".npz"))
    for scenario in args.scenario:
        checked_file(before / (scenario + ".json"))
        checked_file(before / (scenario + ".npz"))
    checked_file(ROOT / "tests/fixed_validation_fixture.py")
    required = sum(4 if s == "success" else 2 for s in args.scenario)
    if args.max_updates < required:
        raise ValueError(f"Selected scenarios require {required} updates")
    return dict(source=str(source), before=str(before), output=output_path(args.output),
                device=device_name(args.device), scenario=args.scenario,
                resume_checkpoint=checked_file(args.resume_checkpoint) if args.resume_checkpoint else None,
                max_updates=args.max_updates, expected_updates=required,
                mode="execute" if args.execute else "plan",
                comparison="original run_case metadata and arrays exact; existing fixture normalization only",
                dependency="source checkout tests.fixed_validation_fixture; no DenseKP construction")


def case_arguments(plan, scenario):
    return dict(source=Path(plan["source"]), directory=Path(plan["output"]) / scenario,
                failure=None if scenario in ("success", "resumed") else scenario,
                resume=plan["resume_checkpoint"] if scenario == "resumed" else None)


def execute(plan):
    if "torch" in sys.modules:
        raise RuntimeError("Execute in a fresh process so device selection precedes Torch")
    device_record = gpu_preflight(plan["device"], 1024)
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import functools
    import gc
    import numpy as np
    import torch
    from unittest.mock import patch
    from tests.fixed_validation_fixture import run_case
    from densekp_retrain.training.trial_callbacks import FixedValidation
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    if plan["resume_checkpoint"]:
        resume = torch.load(plan["resume_checkpoint"], map_location="cpu")
        if resume["global_step"] != 2:
            raise ValueError("Resume fixture must be the independent step2 checkpoint")
        del resume
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "device.json", device_record)
    counts = dict(attempted_updates=0, completed_updates=0, scenario=None)
    records = []
    original_step = torch.optim.AdamW.step
    @functools.wraps(original_step)
    def counted_step(optimizer, *args, **kwargs):
        shapes = [tuple(p.shape) for g in optimizer.param_groups for p in g["params"]]
        if shapes != [(414, 4), (414,), (4, 3), (4,)]:
            raise RuntimeError("Only the existing TinyModel optimizer is authorized")
        if counts["attempted_updates"] >= plan["max_updates"]:
            raise RuntimeError("Cumulative update budget exhausted; no retry")
        counts["attempted_updates"] += 1
        write_json(output / "counts.json", counts)
        result = original_step(optimizer, *args, **kwargs)
        counts["completed_updates"] += 1
        write_json(output / "counts.json", counts)
        return result
    def block_densekp(event, args):
        if event == "import" and args[0] == "densekp.core.densekp_trainer":
            raise RuntimeError("DenseKP model prohibited in callback check")
    sys.addaudithook(block_densekp)
    with patch.object(torch.optim.AdamW, "step", counted_step):
        for scenario in plan["scenario"]:
            counts["scenario"] = scenario
            start = counts["attempted_updates"]
            metadata, arrays = run_case(FixedValidation, **case_arguments(plan, scenario))
            write_json(output / (scenario + ".json"), metadata)
            np.savez(output / (scenario + ".npz"), **arrays)
            before = Path(plan["before"])
            expected = json.loads((before / (scenario + ".json")).read_text())
            assert json.loads(json.dumps(metadata)) == expected, scenario + " metadata changed"
            with np.load(before / (scenario + ".npz"), allow_pickle=False) as previous:
                assert set(previous.files) == set(arrays), scenario
                for key, value in arrays.items():
                    assert value.dtype == previous[key].dtype and value.shape == previous[key].shape, key
                    np.testing.assert_array_equal(value, previous[key], err_msg=scenario + "/" + key)
            updates = counts["attempted_updates"] - start
            assert updates == (4 if scenario == "success" else 2)
            records.append(dict(scenario=scenario, arrays=len(arrays), updates=updates,
                                state_key=metadata["state_key"], error=metadata["error"]))
            gc.collect()
            torch.cuda.empty_cache()
    report = dict(status="passed", scenarios=records, counts=counts,
                  real_gpu=True, densekp_updates=0)
    write_json(output / "result.json", report)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    print(json.dumps(execute(plan) if args.execute else plan, indent=2))


if __name__ == "__main__":
    main()
