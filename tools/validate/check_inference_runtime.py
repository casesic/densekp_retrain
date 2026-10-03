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


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "model-config", "backbone", "device", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--inputs", nargs="+", required=True, help="NPZ files: full float32 img, 3x256x256")
    p.add_argument("--route", nargs="+", choices=("explicit", "lightning"), required=True)
    p.add_argument("--expected", nargs="+", required=True, help="One independent JSON + sibling NPZ per route")
    p.add_argument("--profile", nargs="+", choices=("batch", "item"), required=True)
    p.add_argument("--execute", action="store_true")
    return p


def build_plan(args):
    if len(args.route) != len(args.expected) or len(set(args.route)) != len(args.route):
        raise ValueError("Supply one expected JSON per distinct route, in route order")
    if len(set(args.profile)) != len(args.profile):
        raise ValueError("Duplicate profiles")
    expected = [checked_file(p) for p in args.expected]
    for path in expected:
        checked_file(Path(path).with_suffix(".npz"))
    inputs = [checked_file(p) for p in args.inputs]
    if len({Path(p).stem for p in inputs}) != len(inputs):
        raise ValueError("Input stems must be distinct")
    return dict(checkpoint=checked_file(args.checkpoint), model_config=checked_file(args.model_config),
                backbone=checked_file(args.backbone), inputs=inputs, route=args.route,
                expected=expected, profile=args.profile, device=device_name(args.device),
                output=output_path(args.output), mode="execute" if args.execute else "plan",
                comparison="prediction atol=1e-5 rtol=0; schema, input and RNG exact",
                forward_limit=len(inputs) * len(args.route) * len(args.profile), parameter_updates=0)


def execute(plan):
    if "torch" in sys.modules:
        raise RuntimeError("Execute in a fresh process so device selection precedes Torch")
    device_record = gpu_preflight(plan["device"], 12000)
    import os
    os.environ["DENSEKP_VITPOSE_BACKBONE"] = plan["backbone"]
    sys.path[:0] = [str(ROOT), str(ROOT / "src")]
    import gc
    import functools
    import numpy as np
    import torch
    from unittest.mock import patch
    from omegaconf import OmegaConf
    from densekp.core.densekp_trainer import DenseKP
    from densekp.core.inference import (load_checkpoint_state_dict, load_lightning_checkpoint,
                                       predict_batch, predict_item)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    output = Path(plan["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "plan.json", plan)
    write_json(output / "device.json", device_record)
    cfg = OmegaConf.load(plan["model_config"])
    images = {}
    for path in plan["inputs"]:
        with np.load(path, allow_pickle=False) as archive:
            image = archive["img"].copy()
        if image.shape != (3, 256, 256) or image.dtype != np.float32:
            raise ValueError("Expected complete normalized float32 3x256x256 input")
        images[Path(path).stem] = image
    def blocked(*args, **kwargs):
        raise RuntimeError("Parameter updates prohibited in inference check")
    def schema(state):
        return [dict(key=k, shape=list(v.shape), dtype=str(v.dtype)) for k, v in state.items()]
    records = []
    forwards = 0
    for route, reference in zip(plan["route"], plan["expected"]):
        expected = json.loads(Path(reference).read_text())
        previous = np.load(Path(reference).with_suffix(".npz"), allow_pickle=False)
        torch.manual_seed(0)
        np.random.seed(0)
        arrays = {}
        def rng(stage):
            arrays["cpu_rng_" + stage] = torch.get_rng_state().numpy().copy()
            arrays["cuda_rng_" + stage] = torch.cuda.get_rng_state(device).cpu().numpy().copy()
        rng("before")
        loads, modes, initialization = [], [], []
        original_load = DenseKP.load_state_dict
        original_init = DenseKP.__init__
        original_read = torch.load
        @functools.wraps(original_init)
        def observe_init(model, *args, **kwargs):
            initialization.append("DenseKP.__init__:begin")
            original_init(model, *args, **kwargs)
            initialization.append("DenseKP.__init__:end")
        def observe_read(path, *args, **kwargs):
            initialization.append("torch.load:" + str(getattr(path, "name", path)))
            return original_read(path, *args, **kwargs)
        def observe_load(model, state, strict=True):
            incoming = schema(state)
            result = original_load(model, state, strict=strict)
            loads.append(dict(strict=strict, incoming=incoming, actual=schema(model.state_dict()),
                              missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys))
            return result
        with patch.object(DenseKP, "load_state_dict", observe_load), \
             patch.object(DenseKP, "__init__", observe_init), patch.object(torch, "load", observe_read), \
             patch.object(torch.optim.AdamW, "step", blocked), patch.object(torch.optim.SGD, "step", blocked), \
             patch("pytorch_lightning.Trainer.fit", blocked):
            if route == "explicit":
                model, details = load_checkpoint_state_dict(Path(plan["checkpoint"]), cfg)
            else:
                model = load_lightning_checkpoint(Path(plan["checkpoint"]), cfg)
                details = None
            raw_mode = model.training
            assert raw_mode is True
            model.to(device).eval()
            assert len(loads) == 1 and loads[0] == expected["load"]
            assert model.training == expected["mode_after_load"]
            rng("after_load")
            for case, image in images.items():
                arrays[case + "_input"] = image.copy()
                for profile in plan["profile"]:
                    before_mode = model.training
                    if profile == "batch":
                        value = predict_batch(model, image[None], device)[0].float().cpu().numpy()
                        assert model.training == before_mode
                        suffix = "candidate"
                    else:
                        value = predict_item(model, {"img": image}, device)
                        assert model.training is False
                        suffix = "smoke"
                    forwards += 1
                    write_json(output / "counts.json", dict(forwards=forwards, parameter_updates=0))
                    arrays[case + "_" + suffix] = value
                    modes.append(dict(case=case, profile=profile, before=before_mode, after=model.training))
            rng("after_predictions")
        np.savez(output / (route + ".npz"), **arrays)
        maximum = 0.0
        for key, actual in arrays.items():
            old = previous[key]
            assert actual.shape == old.shape and actual.dtype == old.dtype, key
            if key.endswith(("_candidate", "_smoke")):
                np.testing.assert_allclose(actual, old, atol=1e-5, rtol=0, err_msg=key)
                maximum = max(maximum, float(np.max(np.abs(actual - old))))
            else:
                np.testing.assert_array_equal(actual, old, err_msg=key)
        record = dict(route=route, max_abs=maximum, load=loads[0], raw_mode=raw_mode,
                      modes=modes, initialization=initialization, load_details=details)
        write_json(output / (route + ".json"), record)
        records.append(record)
        previous.close()
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report = dict(status="passed", routes=records, forwards=forwards, parameter_updates=0)
    write_json(output / "result.json", report)
    return report


def main(argv=None):
    args = parser().parse_args(argv)
    plan = build_plan(args)
    print(json.dumps(execute(plan) if args.execute else plan, indent=2))


if __name__ == "__main__":
    main()
