"""Display frozen regression tasks; --execute explicitly enables GPU inference.

Only orchestration is shared. Evaluators retain their distinct metrics and units.
This module uses only the standard library and never imports an evaluator.
"""
import argparse
import csv
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / 'configs/regression/suites.json'


def source_trial_tasks(trial, data, config):
    """Build the original three candidate batches, in their original argv order."""
    out = trial / 'evaluation'
    old = ROOT / config['history']['mixed']
    riding = ROOT / config['history']['riding']
    batches = [
        ('main', [data / 'train_index.json', data / 'val_index.json'],
         ['train_diagnostic', 'val_baseline'], data / 'original_baseline/report.json'),
        ('paired', [data / 'paired_train_index.json', data / 'paired_val_index.json'],
         ['paired_train', 'paired_val'], data / 'paired_baseline/report.json'),
        ('regression', [riding / 'candidate_index.json', old / 'candidate_index.json'],
         ['riding_A', 'mixed_B'], old / 'original_baseline/report.json'),
    ]
    tasks = []
    for name, indices, labels, reference in batches:
        command = [
            sys.executable, str(ROOT / 'tools/validate/evaluate_candidate_baseline.py'),
            '--indices', *map(str, indices), '--labels', *labels, '--output', str(out / name),
            '--checkpoint', str(trial / 'step_0400/checkpoint.ckpt'),
            '--model-config', str(trial / 'model_config.yaml'),
            '--reference-report', str(reference), '--device', 'cuda:0',
        ]
        if name != 'regression':
            command += ['--split-manifest', str(data / 'split_v2.json')]
        tasks.append((name, command))
    return tasks


def box_robustness_tasks(root, output_name, config):
    """Build A/B/C challenges, validation, regression, and two history tasks."""
    settings = config['box_robustness']
    trial = root / 'training'
    out = root / output_name
    data = ROOT / settings['data']
    plain = ROOT / settings['plain']
    old = ROOT / config['history']['mixed']
    riding = ROOT / config['history']['riding']
    checkpoints = [
        ('A', ROOT / settings['checkpoint_a']),
        ('B', plain / 'step_0400/checkpoint.ckpt'),
        ('C', trial / 'step_0400/checkpoint.ckpt'),
    ]
    tasks = []
    for name, checkpoint in checkpoints:
        command = [
            sys.executable, str(ROOT / 'tools/validate/evaluate_candidate_baseline.py'),
            '--indices', str(root / 'data_v2/val_index.json'), '--labels', 'challenge',
            '--split-manifest', str(data / 'split_v2.json'), '--checkpoint', str(checkpoint),
            '--output', str(out / f'challenge_{name}'),
        ]
        if name != 'A':
            command += ['--reference-report', str(out / 'challenge_A/report.json')]
        tasks.append((f'challenge_{name}', command))
    tasks.append(('val_C', [
        sys.executable, str(ROOT / 'tools/validate/evaluate_candidate_baseline.py'),
        '--indices', str(data / 'val_index.json'), '--labels', 'val_baseline',
        '--split-manifest', str(data / 'split_v2.json'),
        '--checkpoint', str(trial / 'step_0400/checkpoint.ckpt'),
        '--reference-report', str(data / 'original_baseline/report.json'),
        '--output', str(out / 'val_C'),
    ]))
    tasks.append(('regression_C', [
        sys.executable, str(ROOT / 'tools/validate/evaluate_candidate_baseline.py'),
        '--indices', str(riding / 'candidate_index.json'), str(old / 'candidate_index.json'),
        '--labels', 'riding_A', 'mixed_B', '--checkpoint', str(trial / 'step_0400/checkpoint.ckpt'),
        '--reference-report', str(old / 'original_baseline/report.json'),
        '--output', str(out / 'regression_C'),
    ]))
    for name, checkpoint in checkpoints[1:]:
        tasks.append((f'history_{name}', [
            sys.executable, str(ROOT / 'tools/validate/evaluate_ds_box_control.py'),
            '--control-root', str(ROOT / settings['control']),
            '--old-root', str(ROOT / settings['old_scene']),
            '--checkpoint', str(checkpoint), '--output', str(out / f'history_{name}'),
        ]))
    return tasks


def source_postprocessing(trial, data):
    return [
        [sys.executable, str(ROOT / 'tools/validate/summarize_source_occlusion_pairs.py'),
         str(trial / 'evaluation/paired')],
        [sys.executable, str(ROOT / 'tools/validate/compare_source_trial.py'),
         '--data-root', str(data), '--trial', str(trial)],
    ]


def environment_delta(suite, config, gpu_uuid):
    return dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=gpu_uuid,
                OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4',
                MPLCONFIGDIR=config[suite]['mplconfigdir'])


def check_gpu(gpu_uuid, physical_gpu):
    """Query the selected device and processes, resolve exactly one full UUID.

    No busy-device policy, automatic fallback, reset or process termination is added.
    A physical index plus UUID must identify the same device. Recheck before tasks.
    """
    selector = physical_gpu if physical_gpu is not None else gpu_uuid
    if selector is None:
        raise ValueError('--execute requires --gpu-uuid or --physical-gpu')
    try:
        status = subprocess.run([
            'nvidia-smi', '-i', selector,
            '--query-gpu=uuid,memory.used,memory.free,utilization.gpu', '--format=csv',
        ], capture_output=True, text=True)
        processes = subprocess.run([
            'nvidia-smi', '-i', selector,
            '--query-compute-apps=pid,process_name,used_gpu_memory', '--format=csv',
        ], capture_output=True, text=True)
    except OSError as error:
        raise RuntimeError('GPU preflight failed') from error
    rows = list(csv.reader(io.StringIO(status.stdout)))
    if (status.returncode or processes.returncode or len(rows) != 2
            or len(rows[1]) != 4 or not rows[1][0].strip().startswith('GPU-')):
        raise RuntimeError('GPU preflight failed')
    resolved = rows[1][0].strip()
    if gpu_uuid is not None and resolved != gpu_uuid:
        raise RuntimeError('GPU preflight failed')
    return resolved, status.stdout, processes.stdout


def training_gate(report_path, message):
    report = json.loads(report_path.read_text())
    if report['actual_updates'] != 400 or report['reload_max_abs_error'] > 1e-5:
        raise ValueError(message)


def run_logged(command, log_path, env):
    started = time.monotonic()
    with log_path.open('w') as log:
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return result.returncode, time.monotonic() - started


def execute_source(args, config):
    trial, data = args.trial.resolve(), args.data_root.resolve()
    training_gate(trial / 'report.json', 'Final workflow gate failed')
    out = trial / 'evaluation'
    out.mkdir(exist_ok=False)
    (out / 'evaluation_entry.py').write_bytes(Path(__file__).read_bytes())
    (out / 'metric_entry.py').write_bytes((ROOT / 'tools/validate/evaluate_candidate_baseline.py').read_bytes())
    env = dict(os.environ, **environment_delta('source_trial', config, args.gpu_uuid))
    records = []
    for name, command in source_trial_tasks(trial, data, config):
        check_gpu(args.gpu_uuid, args.physical_gpu)
        returncode, seconds = run_logged(command, out / f'{name}.log', env)
        records.append(dict(batch=name, command=command, returncode=returncode, seconds=seconds))
        (out / 'execution.json').write_text(json.dumps(dict(gpu_uuid=args.gpu_uuid, batches=records), indent=2) + '\n')
        if returncode:
            raise RuntimeError(f'{name} failed; preserved log, no skipping/retry')
        print(f'{name} evaluation complete', flush=True)
    for command in source_postprocessing(trial, data):
        subprocess.run(command, cwd=ROOT, env=env, check=True)


def execute_box(args, config):
    root = args.experiment.resolve()
    out = root / args.output_name
    out.mkdir(exist_ok=bool(args.only))
    training_gate(root / 'training/report.json', 'Training/reload not accepted')
    env = dict(os.environ, **environment_delta('box_robustness', config, args.gpu_uuid))
    tasks = box_robustness_tasks(root, args.output_name, config)
    records = json.loads((out / 'execution.json').read_text()) if args.only else []
    for name, command in tasks:
        if args.only and name not in args.only:
            continue
        _, status, processes = check_gpu(args.gpu_uuid, args.physical_gpu)
        returncode, seconds = run_logged(command, out / f'{name}.log', env)
        records.append(dict(name=name, command=command, returncode=returncode, seconds=seconds,
                            gpu=status, compute_processes=processes))
        (out / 'execution.json').write_text(json.dumps(records, indent=2) + '\n')
        if returncode:
            raise RuntimeError(f'{name} failed; stop without retry')
        print(name, 'complete', flush=True)


def display_plan(args, config):
    """Read at most the small gate report. No mkdir, subprocess or weight IO."""
    if args.suite == 'source_trial':
        trial, data = args.trial.resolve(), args.data_root.resolve()
        report = trial / 'report.json'
        out = trial / 'evaluation'
        tasks = source_trial_tasks(trial, data, config)
        postprocessing = source_postprocessing(trial, data)
        postprocessing_outputs = [str(out / 'paired/occlusion_control_metrics.json'),
                                  str(trial / 'comparison')]
        independent = []
        gate_order = 'report -> gate -> mkdir(exist_ok=False) -> snapshots -> tasks -> postprocessing'
    else:
        root = args.experiment.resolve()
        report = root / 'training/report.json'
        out = root / args.output_name
        tasks = box_robustness_tasks(root, args.output_name, config)
        tasks = [(name, command) for name, command in tasks if not args.only or name in args.only]
        postprocessing = []
        postprocessing_outputs = []
        independent = [[sys.executable, str(ROOT / 'tools/validate/compare_box_robustness.py'),
                        '--experiment', str(root)]]
        gate_order = 'mkdir(exist_ok=bool(only)) -> report -> gate -> read execution if only -> tasks'
    try:
        training_gate(report, 'Training/reload not accepted')
        gate = dict(passed=True)
    except (OSError, ValueError, KeyError, TypeError) as error:
        gate = dict(passed=False, error_type=type(error).__name__, reason=str(error))
    rows = []
    for name, command in tasks:
        # Decode the already-built argv for display; never maintain a second task table.
        options = {}
        key = None
        for token in command[2:]:
            if token.startswith('--'):
                key = token
                options[key] = []
            elif key:
                options[key].append(token)
        rows.append(dict(name=name, argv=command, resolved_options=options,
                         model_config='explicit' if '--model-config' in options else 'evaluator default (not passed)',
                         device='explicit cuda:0' if '--device' in options else 'evaluator cuda:0 (not passed)'))
    return dict(
        suite=args.suite, mode='display_only', cwd=str(ROOT), output=str(out), tasks=rows,
        gate_report=str(report), gate=gate, execution_order=gate_order,
        gate_condition='actual_updates == 400; reject reload_max_abs_error > 1e-5 (original comparison, no new finite check)',
        execute_precondition='Explicit device; initial device/process query before business gate, then recheck before each inference task',
        gpu_selection=dict(uuid=args.gpu_uuid, physical=args.physical_gpu, child_logical_device='cuda:0'),
        environment_delta=environment_delta(args.suite, config, args.gpu_uuid or '<resolved UUID at execution>'),
        postprocessing=postprocessing, postprocessing_outputs=postprocessing_outputs,
        independent_followup=independent,
        notes=[
            'reference_report checks inputs; it does not reuse predictions or disable inference.',
            'Unknown --only names retain zero-task behavior; --execute still reads existing execution.json.',
            'Box compare is independent and reads evaluation_v2 even with a different suite output-name.',
            'No automatic checkpoint/version selection; box weights are not approved for deployment.',
        ],
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', choices=('source_trial', 'box_robustness'), required=True)
    parser.add_argument('--trial', type=Path)
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--experiment', type=Path)
    parser.add_argument('--output-name', default='evaluation_v2')
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--physical-gpu', help='Explicit physical GPU index; no implicit device 0 or 1')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args(argv)
    if args.suite == 'source_trial':
        if args.trial is None or args.data_root is None:
            parser.error('source_trial requires --trial and --data-root')
        if args.only or args.experiment or args.output_name != 'evaluation_v2':
            parser.error('--only/--experiment/--output-name belong to box_robustness')
    elif args.experiment is None or args.trial or args.data_root:
        parser.error('box_robustness requires --experiment; --trial/--data-root belong to source_trial')
    if args.execute and args.gpu_uuid is None and args.physical_gpu is None:
        parser.error('--execute requires --gpu-uuid or --physical-gpu')
    return args


def main(argv=None):
    args = parse_args(argv)
    config = json.loads(CONFIG_PATH.read_text())
    if not args.execute:
        print(json.dumps(display_plan(args, config), indent=2))
        return
    # Deliberate new precondition, separate from both original business gates.
    args.gpu_uuid, status, processes = check_gpu(args.gpu_uuid, args.physical_gpu)
    print(json.dumps(dict(gpu_preflight=dict(uuid=args.gpu_uuid, status=status, processes=processes))), file=sys.stderr)
    if args.suite == 'source_trial':
        execute_source(args, config)
    else:
        execute_box(args, config)


if __name__ == '__main__':
    main()
