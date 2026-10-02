"""Independent original/current inference comparison on seven frozen inputs.

Runs only three named checkpoints and two historical loading routes. Recipes
are extracted from entry-point ASTs so each original precision/mode/check policy
is exercised independently before moving any inference implementation.
"""
import argparse
import ast
import gc
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import numpy as np
import torch
from omegaconf import OmegaConf
from densekp.core.densekp_trainer import DenseKP

ENTRIES = {
    'candidate': 'tools/validate/evaluate_candidate_baseline.py',
    'scene': 'tools/validate/evaluate_ds_scene_pair.py',
    'occlusion': 'tools/validate/evaluate_ds_occlusion_control.py',
    'box': 'tools/validate/evaluate_ds_box_control.py',
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_recipe(path, profile):
    """Extract only forward/conversion/check statements, without data IO loops."""
    tree = ast.parse(path.read_text())
    for parent in ast.walk(tree):
        for field in ('body', 'orelse'):
            body = getattr(parent, field, None)
            if not isinstance(body, list):
                continue
            for index, node in enumerate(body):
                if not isinstance(node, ast.With):
                    continue
                text = ast.unparse(node)
                if 'pred_keypoints' not in text and 'predict_batch(' not in text:
                    continue
                count = 2 if profile == 'candidate' else 3
                nodes = body[index:index + count]
                if profile == 'occlusion':
                    finite_check = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                                        and ast.unparse(n.test).startswith('prediction.shape !='))
                    nodes.append(finite_check)
                return '\n'.join(ast.unparse(n) for n in nodes)
    if profile == 'candidate' and path == ROOT / ENTRIES['candidate']:
        return source_recipe(ROOT / 'src/densekp_retrain/evaluation/candidate_runner.py', profile)
    raise ValueError(f'No prediction recipe in {path}')


def loader_recipe(path, route):
    tree = ast.parse(path.read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name in ('main', 'load_evaluation_model'))
    if route == 'explicit':
        start = next((i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                      and ast.unparse(n.targets[0]) in ('payload', '(model, load_details)')), None)
        if start is None and path == ROOT / ENTRIES['candidate']:
            return loader_recipe(ROOT / 'src/densekp_retrain/evaluation/candidate_runner.py', route)
        if start is None:
            raise ValueError(f'No explicit load recipe in {path}')
        end = next(i for i in range(start, len(main.body)) if isinstance(main.body[i], ast.Expr)
                   and ast.unparse(main.body[i]).startswith(("write(out / 'checkpoint_load.json'",
                                                            "write_json(output / 'checkpoint_load.json'")))
        return '\n'.join(ast.unparse(n) for n in main.body[start:end])
    start = next(i for i, n in enumerate(main.body) if isinstance(n, ast.Assign)
                 and ast.unparse(n.targets[0]) == 'model')
    return '\n'.join(ast.unparse(n) for n in main.body[start:start + 2])


def schema(state):
    return [dict(key=key, shape=list(value.shape), dtype=str(value.dtype)) for key, value in state.items()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', choices=['capture', 'verify'], required=True)
    parser.add_argument('--acceptance', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, help='Independent baseline directory; never overwritten')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    capture = args.phase == 'capture'
    args.output.mkdir(parents=True, exist_ok=False)
    acceptance = args.acceptance.resolve()
    baseline = args.baseline.resolve() if args.baseline else acceptance / 'before/inference'
    if capture:
        baseline.mkdir(exist_ok=False)
    # Physical GPU 1 only. The launcher must select its UUID, not a numeric mask.
    preflight = subprocess.run(['nvidia-smi', '-i', '1'], capture_output=True, text=True)
    (args.output / 'gpu_preflight.txt').write_text(preflight.stdout + preflight.stderr)
    if preflight.returncode or not torch.cuda.is_available():
        raise RuntimeError('Selected CUDA device unavailable; do not migrate inference')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    source_root = acceptance / 'before/source_snapshot' if capture else ROOT
    recipes = {name: source_recipe(source_root / rel, name) for name, rel in ENTRIES.items()}
    loaders = {
        'explicit': loader_recipe(source_root / ENTRIES['candidate'], 'explicit'),
        'lightning': loader_recipe(source_root / ENTRIES['scene'], 'lightning'),
    }
    namespace = dict(torch=torch, np=np, time=time, DenseKP=DenseKP)
    if capture:
        # The original snapshot contains a true definition, not a re-export.
        smoke = ast.parse((source_root / 'tools/experiments/train_cycling_smoke.py').read_text())
        old_predict = next(n for n in smoke.body if isinstance(n, ast.FunctionDef) and n.name == 'predict')
        exec(compile(ast.Module(body=[old_predict], type_ignores=[]), '<original predict>', 'exec'), namespace)
        write(baseline / 'recipes.json', dict(predictions=recipes, loaders=loaders, smoke=ast.unparse(old_predict)))
        previous = ROOT / 'outputs/acceptance/refactor_metrics_config_20261001/before'
        selection = json.loads((previous / 'selection.json').read_text())
        write(baseline / 'selection.json', selection)
        (baseline / 'inputs').mkdir()
        for entry in selection['samples']:
            shutil.copyfile(previous / (entry['id'] + '.npz'), baseline / 'inputs' / (entry['id'] + '.npz'))
    else:
        from densekp.core.inference import load_checkpoint_state_dict, load_lightning_checkpoint, predict_batch, predict_item
        namespace.update(load_checkpoint_state_dict=load_checkpoint_state_dict,
                         load_lightning_checkpoint=load_lightning_checkpoint,
                         predict_batch=predict_batch, predict=predict_item)
        selection = json.loads((baseline / 'selection.json').read_text())
    write(args.output / 'recipes.json', dict(predictions=recipes, loaders=loaders))
    inputs = {}
    for entry in selection['samples']:
        with np.load(baseline / 'inputs' / (entry['id'] + '.npz'), allow_pickle=False) as archive:
            image = archive['img'].copy()
        assert image.shape == (3, 256, 256) and image.dtype == np.float32
        inputs[entry['id']] = image
    sources = json.loads((acceptance / 'before/checkpoint_loads/result.json').read_text())['models']
    results = []
    for source in sources:
        for route in ('explicit', 'lightning'):
            label = source['label'] + '_' + route
            print('Loading ' + label, flush=True)
            cfg = OmegaConf.load(acceptance / 'before/checkpoint_loads' / (source['label'] + '_config.yaml'))
            checkpoint = Path(source['checkpoint'])
            stat = checkpoint.stat()
            assert (stat.st_size, stat.st_mtime_ns) == (source['bytes'], source['mtime_ns'])
            torch.manual_seed(0)
            np.random.seed(0)
            arrays = dict(cpu_rng_before=torch.get_rng_state().numpy().copy(),
                          cuda_rng_before=torch.cuda.get_rng_state(device).cpu().numpy().copy())
            loads = []
            original_load = DenseKP.load_state_dict

            def observe_load(model, state, strict=True):
                incoming = schema(state)
                result = original_load(model, state, strict=strict)
                loads.append(dict(strict=strict, incoming=incoming, actual=schema(model.state_dict()),
                                  missing_keys=result.missing_keys, unexpected_keys=result.unexpected_keys))
                return result

            current = dict(namespace, cfg=cfg, checkpoint=checkpoint, stat=stat, device=device,
                           device_name=str(device),
                           args=SimpleNamespace(checkpoint=checkpoint, device=str(device)),
                           load_start=time.monotonic())
            with patch.object(DenseKP, 'load_state_dict', observe_load):
                exec(loaders[route], current)
            model = current['model']
            assert len(loads) == 1 and loads[0]['strict']
            assert loads[0]['incoming'] == loads[0]['actual']
            assert not loads[0]['missing_keys'] and not loads[0]['unexpected_keys']
            arrays['cpu_rng_after_load'] = torch.get_rng_state().numpy().copy()
            arrays['cuda_rng_after_load'] = torch.cuda.get_rng_state(device).cpu().numpy().copy()
            metadata = dict(load=loads[0], mode_after_load=model.training, cases=[],
                            checkpoint=source['checkpoint'], bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
            if route == 'explicit':
                metadata['load_audit'] = {key: value for key, value in current['load_audit'].items() if key != 'seconds'}
            for case_id, image in inputs.items():
                arrays[case_id + '_input'] = image.copy()
                for profile, code in recipes.items():
                    case = dict(namespace, model=model, device=device, item={'img':image},
                                folder=args.output, output=args.output, camera='cam2', person_id=1)
                    exec(code, case)
                    output_name = {'candidate':'pred', 'scene':'prediction', 'occlusion':'prediction', 'box':'raw'}[profile]
                    if profile == 'candidate' and 'prediction' in case:
                        output_name = 'prediction'
                    arrays[case_id + '_' + profile] = case[output_name]
                    metadata['cases'].append(dict(id=case_id, profile=profile, training=model.training))
                arrays[case_id + '_smoke'] = namespace['predict'](model, {'img':image}, device)
            arrays['cpu_rng_after_predictions'] = torch.get_rng_state().numpy().copy()
            arrays['cuda_rng_after_predictions'] = torch.cuda.get_rng_state(device).cpu().numpy().copy()
            write(args.output / (label + '.json'), metadata)
            np.savez(args.output / (label + '.npz'), **arrays)
            maximum = 0.
            if capture:
                write(baseline / (label + '.json'), metadata)
                np.savez(baseline / (label + '.npz'), **arrays)
            else:
                assert metadata == json.loads((baseline / (label + '.json')).read_text()), label
                with np.load(baseline / (label + '.npz'), allow_pickle=False) as previous:
                    assert set(arrays) == set(previous.files)
                    for key, value in arrays.items():
                        assert value.shape == previous[key].shape and value.dtype == previous[key].dtype, key
                        if any(key.endswith('_' + name) for name in [*ENTRIES, 'smoke']):
                            np.testing.assert_allclose(value, previous[key], atol=1e-5, rtol=0, err_msg=label + '/' + key)
                            maximum = max(maximum, float(np.abs(value - previous[key]).max()))
                        else:
                            np.testing.assert_array_equal(value, previous[key], err_msg=label + '/' + key)
            results.append(dict(label=label, state_keys=len(loads[0]['actual']), inputs=len(inputs),
                                prediction_profiles=5, max_abs=maximum if not capture else None))
            # Recipe namespaces also own the model; release them before next load.
            del model, current, case
            gc.collect()
            torch.cuda.empty_cache()
            print('Passed ' + label, flush=True)
    report = dict(status='passed', phase=args.phase, models=results, real_predictions=True,
                  physical_gpu_uuid='GPU-6f046d56-acbe-d49b-774a-9d463e768cac',
                  torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(device),
                  prediction_tolerance=dict(atol=1e-5, rtol=0), precision='float32 highest, no autocast, cudnn TF32 disabled',
                  training=False, rendering=False)
    write(args.output / 'result.json', report)
    if capture:
        write(baseline / 'result.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
