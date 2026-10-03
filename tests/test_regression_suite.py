"""Stdlib-only regression scheduling contracts; never run a business subprocess."""
import builtins
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools.validate import run_regression_suite as suite

ROOT = Path(__file__).resolve().parents[1]


class RegressionSuiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.trial = self.root / 'trial'
        self.experiment = self.root / 'experiment'
        self.data = self.root / 'data'
        self.trial.mkdir()
        self.experiment.mkdir()
        (self.experiment / 'training').mkdir()
        self.data.mkdir()
        self.good_report = json.dumps(dict(actual_updates=400, reload_max_abs_error=1e-5))
        (self.trial / 'report.json').write_text(self.good_report)
        (self.experiment / 'training/report.json').write_text(self.good_report)
        metric = self.root / 'tools/validate/evaluate_candidate_baseline.py'
        metric.parent.mkdir(parents=True)
        metric.write_text('# inert snapshot\n')
        self.config = json.loads((ROOT / 'configs/regression/suites.json').read_text())
        self.patch = patch.object(suite, 'ROOT', self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)
        self.calls = []
        self.status = 'uuid, memory.used, memory.free, utilization.gpu\nGPU-TEST, 0 MiB, 40000 MiB, 0 %\n'
        self.processes = 'pid, process_name, used_gpu_memory\n'

    def argv(self, kind, *extra):
        paths = (['--trial', str(self.trial), '--data-root', str(self.data)]
                 if kind == 'source_trial' else ['--experiment', str(self.experiment)])
        return ['--suite', kind, *paths, *extra]

    def stub(self, command, **kwargs):
        self.calls.append((command, kwargs))
        if command[0] == 'nvidia-smi':
            processes = any('compute-apps' in token for token in command)
            return SimpleNamespace(returncode=0, stdout=self.processes if processes else self.status)
        if 'stdout' in kwargs:
            kwargs['stdout'].write('stub output\n')
        return SimpleNamespace(returncode=0)

    def execute(self, kind, *extra, runner=None):
        with patch.object(subprocess, 'run', runner or self.stub), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('real process')), \
             patch.object(os, 'system', side_effect=AssertionError('real shell')), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            suite.main(self.argv(kind, '--execute', '--gpu-uuid', 'GPU-TEST', *extra))

    def test_source_plan_argv_and_postprocessing(self):
        args = suite.parse_args(self.argv('source_trial'))
        plan = suite.display_plan(args, self.config)
        self.assertEqual([t['name'] for t in plan['tasks']], ['main', 'paired', 'regression'])
        first = plan['tasks'][0]['argv']
        self.assertEqual(first[2:], [
            '--indices', str(self.data / 'train_index.json'), str(self.data / 'val_index.json'),
            '--labels', 'train_diagnostic', 'val_baseline', '--output', str(self.trial / 'evaluation/main'),
            '--checkpoint', str(self.trial / 'step_0400/checkpoint.ckpt'),
            '--model-config', str(self.trial / 'model_config.yaml'),
            '--reference-report', str(self.data / 'original_baseline/report.json'), '--device', 'cuda:0',
            '--split-manifest', str(self.data / 'split_v2.json')])
        self.assertNotIn('--split-manifest', plan['tasks'][2]['argv'])
        self.assertEqual([Path(c[1]).name for c in plan['postprocessing']],
                         ['summarize_source_occlusion_pairs.py', 'compare_source_trial.py'])
        self.assertEqual(plan['independent_followup'], [])
        self.assertTrue(plan['gate']['passed'])

    def test_box_plan_order_defaults_and_only(self):
        args = suite.parse_args(self.argv('box_robustness'))
        plan = suite.display_plan(args, self.config)
        names = ['challenge_A', 'challenge_B', 'challenge_C', 'val_C', 'regression_C', 'history_B', 'history_C']
        self.assertEqual([t['name'] for t in plan['tasks']], names)
        for task in plan['tasks']:
            self.assertNotIn('--model-config', task['argv'])
            self.assertNotIn('--device', task['argv'])
        self.assertEqual(plan['postprocessing'], [])
        self.assertEqual(Path(plan['independent_followup'][0][1]).name, 'compare_box_robustness.py')
        self.assertNotIn('--reference-report', plan['tasks'][0]['argv'])
        for task in plan['tasks'][1:3]:
            self.assertEqual(task['resolved_options']['--reference-report'],
                             [str(self.experiment / 'evaluation_v2/challenge_A/report.json')])
        args.only = ['history_C', 'challenge_B']
        self.assertEqual([t['name'] for t in suite.display_plan(args, self.config)['tasks']], ['challenge_B', 'history_C'])
        args.only = ['unknown']
        self.assertEqual(suite.display_plan(args, self.config)['tasks'], [])

    def test_display_has_no_write_process_heavy_import_or_weight_read(self):
        original_open = io.open
        original_import = builtins.__import__
        def guarded_open(file, mode='r', *args, **kwargs):
            if any(flag in mode for flag in 'wax+'):
                raise AssertionError(('write', file))
            if str(file).endswith(('.ckpt', '.pth', '.pt', '.npz')):
                raise AssertionError(('asset read', file))
            return original_open(file, mode, *args, **kwargs)
        def guarded_import(name, *args, **kwargs):
            if name.split('.')[0] in ('torch', 'numpy', 'bpy', 'smplx', 'densekp', 'densekp_retrain') or 'evaluate_' in name:
                raise AssertionError(('heavy import', name))
            return original_import(name, *args, **kwargs)
        with patch.object(io, 'open', guarded_open), patch.object(builtins, '__import__', guarded_import), \
             patch.object(Path, 'mkdir', side_effect=AssertionError('mkdir')), \
             patch.object(subprocess, 'run', side_effect=AssertionError('subprocess')), \
             patch.object(subprocess, 'Popen', side_effect=AssertionError('Popen')), \
             patch.object(os, 'system', side_effect=AssertionError('shell')):
            for kind in ('source_trial', 'box_robustness'):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    suite.main(self.argv(kind))
                self.assertEqual(json.loads(output.getvalue())['mode'], 'display_only')
        self.assertFalse((self.trial / 'evaluation').exists())
        self.assertFalse((self.experiment / 'evaluation_v2').exists())

    def test_gate_and_existing_directory_order(self):
        for kind, report, out in (
            ('source_trial', self.trial / 'report.json', self.trial / 'evaluation'),
            ('box_robustness', self.experiment / 'training/report.json', self.experiment / 'evaluation_v2'),
        ):
            report.write_text('{"actual_updates": 399, "reload_max_abs_error": 0}')
            with self.assertRaises(ValueError):
                self.execute(kind)
            self.assertEqual(out.exists(), kind == 'box_robustness')
            if not out.exists():
                out.mkdir()
            report.unlink()
            # Source reads missing report first; box fails mkdir first.
            with self.assertRaises(FileNotFoundError if kind == 'source_trial' else FileExistsError):
                self.execute(kind)

    def test_reload_gate_threshold_and_original_nonfinite_rule(self):
        report = self.trial / 'report.json'
        for value, accepted in [(1e-5, True), (1.00001e-5, False), (float('nan'), True)]:
            report.write_text(json.dumps(dict(actual_updates=400, reload_max_abs_error=value)))
            if accepted:
                suite.training_gate(report, 'gate')
            else:
                with self.assertRaisesRegex(ValueError, 'gate'):
                    suite.training_gate(report, 'gate')

    def test_source_success_environment_and_failure_stopping(self):
        self.execute('source_trial')
        business = [(c, k) for c, k in self.calls if c[0] != 'nvidia-smi']
        self.assertEqual(len(business), 5)
        for command, kw in business:
            self.assertEqual(kw['cwd'], self.root)
            self.assertEqual({k:kw['env'][k] for k in suite.environment_delta('source_trial', self.config, 'GPU-TEST')},
                             dict(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='GPU-TEST', OMP_NUM_THREADS='4',
                                  OPENBLAS_NUM_THREADS='4', MPLCONFIGDIR='/tmp/densekp-trial-mpl'))
        records = json.loads((self.trial / 'evaluation/execution.json').read_text())
        self.assertEqual([r['batch'] for r in records['batches']], ['main', 'paired', 'regression'])
        self.assertEqual(list(records['batches'][0]), ['batch', 'command', 'returncode', 'seconds'])

    def test_task_and_source_postprocessing_failure(self):
        for failing in ('paired', 'summarize_source_occlusion_pairs.py', 'compare_source_trial.py'):
            with self.subTest(failing=failing), tempfile.TemporaryDirectory() as directory:
                self.trial = Path(directory)
                (self.trial / 'report.json').write_text(self.good_report)
                self.calls = []
                def runner(command, **kw):
                    result = self.stub(command, **kw)
                    if command[0] != 'nvidia-smi' and (failing == Path(command[1]).name or 'paired_train' in command and failing == 'paired'):
                        if kw.get('check'):
                            raise subprocess.CalledProcessError(9, command)
                        return SimpleNamespace(returncode=9)
                    return result
                with self.assertRaises(RuntimeError if failing == 'paired' else subprocess.CalledProcessError):
                    self.execute('source_trial', runner=runner)
                business = [c for c, _ in self.calls if c[0] != 'nvidia-smi']
                self.assertEqual(len(business), {'paired':2, 'summarize_source_occlusion_pairs.py':4, 'compare_source_trial.py':5}[failing])
                record = json.loads((self.trial / 'evaluation/execution.json').read_text())
                self.assertEqual(len(record['batches']), 2 if failing == 'paired' else 3)
                if failing == 'paired':
                    self.assertEqual(record['batches'][-1]['returncode'], 9)

    def test_box_only_keeps_order_and_requires_execution_even_unknown(self):
        out = self.experiment / 'evaluation_v2'
        with self.assertRaises(FileNotFoundError):
            self.execute('box_robustness', '--only', 'unknown')
        (out / 'execution.json').write_text('[{"prior":true}]')
        self.calls = []
        self.execute('box_robustness', '--only', 'unknown')
        self.assertTrue(all(c[0]=='nvidia-smi' for c, _ in self.calls))
        self.execute('box_robustness', '--only', 'history_C', 'challenge_B')
        records = json.loads((out / 'execution.json').read_text())
        self.assertEqual(records[0], dict(prior=True))
        self.assertEqual([r['name'] for r in records[1:]], ['challenge_B', 'history_C'])
        self.assertEqual(records[1]['gpu'], self.status)
        self.assertEqual(records[1]['compute_processes'], self.processes)

    def test_gpu_resolution_mismatch_failures_and_explicit_selection(self):
        with patch.object(subprocess, 'run', self.stub):
            self.assertEqual(suite.check_gpu(None, '2')[0], 'GPU-TEST')
            self.assertEqual(self.calls[-1][0][2], '2')
            self.assertEqual(suite.check_gpu('GPU-TEST', None)[0], 'GPU-TEST')
            with self.assertRaises(RuntimeError):
                suite.check_gpu('GPU-TE', '2')  # substring is not sufficient
        for failure in ('query', 'process', 'malformed', 'missing'):
            def runner(command, **kw):
                if failure == 'missing':
                    raise FileNotFoundError('nvidia-smi')
                result = self.stub(command, **kw)
                processes = any('compute-apps' in t for t in command)
                if (failure=='query' and not processes) or (failure=='process' and processes):
                    result.returncode = 1
                if failure == 'malformed' and not processes:
                    result.stdout = 'bad response'
                return result
            with self.subTest(failure=failure), self.assertRaises(RuntimeError):
                self.execute('box_robustness', runner=runner)
            self.assertFalse((self.experiment / 'evaluation_v2').exists())
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            suite.parse_args(self.argv('source_trial', '--execute'))

    def test_relative_paths_follow_cwd_but_fixed_roles_follow_root(self):
        previous = Path.cwd()
        try:
            for cwd in (self.root, self.data):
                os.chdir(cwd)
                args = suite.parse_args(['--suite','source_trial','--trial','trial','--data-root','data'])
                plan = suite.display_plan(args, self.config)
                self.assertEqual(plan['output'], str(cwd / 'trial/evaluation'))
                self.assertIn(str(self.root / self.config['history']['mixed'] / 'candidate_index.json'), plan['tasks'][2]['argv'])
        finally:
            os.chdir(previous)

    def test_physical_selection_maps_uuid_and_recheck_stops_before_log(self):
        with patch.object(subprocess, 'run', self.stub), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            suite.main(self.argv('box_robustness', '--execute', '--physical-gpu', '2'))
        business = [kw for command, kw in self.calls if command[0] != 'nvidia-smi']
        self.assertEqual(len(business), 7)
        self.assertTrue(all(kw['env']['CUDA_VISIBLE_DEVICES'] == 'GPU-TEST' for kw in business))
        self.assertTrue(all(command[2] == '2' for command, _ in self.calls if command[0] == 'nvidia-smi'))
        queries = 0
        def change_device(command, **kwargs):
            nonlocal queries
            result = self.stub(command, **kwargs)
            if command[0] == 'nvidia-smi' and any('query-gpu' in token for token in command):
                queries += 1
                if queries == 2:
                    result.stdout = self.status.replace('GPU-TEST', 'GPU-CHANGED')
            return result
        with self.assertRaisesRegex(RuntimeError, 'GPU preflight failed'):
            self.execute('source_trial', runner=change_device)
        out = self.trial / 'evaluation'
        self.assertEqual(sorted(p.name for p in out.iterdir()), ['evaluation_entry.py', 'metric_entry.py'])
