"""Small standard-library plan/dispatch checks; never execute GPU or model code."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def entry(name):
    path = ROOT / 'tools/validate' / ('check_' + name + '_runtime.py')
    spec = importlib.util.spec_from_file_location('runtime_' + name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class RuntimeEntryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inference = entry('inference')
        self.callback = entry('fixed_validation')
        for name in ('checkpoint', 'config.yaml', 'backbone', 'image.npz', 'expected.json',
                     'expected.npz', 'resume.ckpt', 'cases.json', 'val_baseline_002.npz',
                     'val_baseline_001.npz', 'val_baseline_004.npz'):
            (self.root / name).write_text('{}')
        for scenario in self.callback.SCENARIOS:
            for suffix in ('.json', '.npz'):
                (self.root / (scenario + suffix)).write_text('{}')
        self.inf = ['--checkpoint', str(self.root/'checkpoint'), '--model-config', str(self.root/'config.yaml'),
                    '--backbone', str(self.root/'backbone'), '--inputs', str(self.root/'image.npz'),
                    '--expected', str(self.root/'expected.json'), '--route', 'explicit',
                    '--profile', 'batch', 'item', '--device', 'GPU-test', '--output', str(self.root/'inf')]
        self.cb = ['--source', str(self.root), '--before', str(self.root), '--output', str(self.root/'cb'),
                   '--scenario', 'success', '--device', '1', '--max-updates', '4']

    def test_default_cold_plan_has_no_writes_processes_or_heavy_imports(self):
        code = '''import sys, runpy, json
sys.dont_write_bytecode = True
script, argv = sys.argv[1], json.loads(sys.argv[2])
def guard(event, args):
    if event == 'import' and args[0].split('.')[0] in ('torch','numpy','bpy','smplx','pytorch_lightning','densekp','densekp_retrain'):
        raise RuntimeError('heavy import: ' + args[0])
    if event in ('subprocess.Popen','os.system','os.mkdir','os.remove','os.rename'):
        raise RuntimeError('side effect: ' + event)
    if event == 'open':
        path, mode, flags = args
        if isinstance(mode, str) and any(c in mode for c in 'wax+'):
            raise RuntimeError('write')
        if isinstance(flags, int) and flags & (1|2|64|512|1024):
            raise RuntimeError('write flags')
        if str(path).endswith(('checkpoint','backbone','.npz','.ckpt')):
            raise RuntimeError('binary input read')
sys.addaudithook(guard)
sys.argv = [script] + argv
runpy.run_path(script, run_name='__main__')
'''
        for module, args in ((self.inference, self.inf), (self.callback, self.cb)):
            for cwd in (ROOT, self.root):
                result = subprocess.run([sys.executable, '-I', '-B', '-c', code,
                                         module.__file__, json.dumps(args)], cwd=cwd,
                                        text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(json.loads(result.stdout)['mode'], 'plan')
        self.assertFalse((self.root/'inf').exists())
        self.assertFalse((self.root/'cb').exists())

    def test_explicit_execute_dispatch_keeps_plan(self):
        for module, args in ((self.inference, self.inf), (self.callback, self.cb)):
            with patch.object(module, 'execute', return_value={'stub': True}) as execute:
                with contextlib.redirect_stdout(io.StringIO()):
                    module.main(args + ['--execute'])
                plan = execute.call_args.args[0]
                self.assertEqual(plan['mode'], 'execute')
                self.assertEqual(plan['device'], 'GPU-test' if module is self.inference else '1')
                self.assertFalse(Path(plan['output']).exists())

    def test_missing_input_and_existing_output_rejected(self):
        for module, args in ((self.inference, self.inf), (self.callback, self.cb)):
            plan = module.build_plan(module.parser().parse_args(args))
            Path(plan['output']).mkdir()
            with self.assertRaisesRegex(ValueError, 'Output must be new'):
                module.build_plan(module.parser().parse_args(args))
        (self.root/'checkpoint').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing input'):
            self.inference.build_plan(self.inference.parser().parse_args(self.inf))

    def test_required_route_profile_device_and_pairs(self):
        args = self.inference.parser().parse_args(self.inf)
        args.route = ['explicit', 'lightning']
        with self.assertRaisesRegex(ValueError, 'one expected'):
            self.inference.build_plan(args)
        args.route = ['explicit']
        args.profile = ['item', 'item']
        with self.assertRaisesRegex(ValueError, 'Duplicate profiles'):
            self.inference.build_plan(args)
        for value in ('cuda:0', 'auto', '-1'):
            with self.assertRaises(ValueError):
                self.inference.device_name(value)

    def test_all_callback_scenarios_map_without_computation(self):
        for scenario in self.callback.SCENARIOS:
            args = self.callback.parser().parse_args(self.cb)
            args.scenario = [scenario]
            args.resume_checkpoint = str(self.root/'resume.ckpt') if scenario == 'resumed' else None
            plan = self.callback.build_plan(args)
            call = self.callback.case_arguments(plan, scenario)
            self.assertEqual(call['failure'], None if scenario in ('success','resumed') else scenario)
            self.assertEqual(call['resume'], args.resume_checkpoint)
            self.assertEqual(call['directory'], self.root/'cb'/scenario)
            self.assertEqual(plan['expected_updates'], 4 if scenario == 'success' else 2)

    def test_resume_and_update_budget_are_explicit(self):
        args = self.callback.parser().parse_args(self.cb)
        args.scenario = ['resumed']
        with self.assertRaisesRegex(ValueError, 'Resume checkpoint'):
            self.callback.build_plan(args)
        args.resume_checkpoint = str(self.root/'resume.ckpt')
        args.scenario = ['success', 'resumed', 'save_failure']
        with self.assertRaisesRegex(ValueError, 'require 8'):
            self.callback.build_plan(args)
        args.max_updates = 8
        self.assertEqual(self.callback.build_plan(args)['expected_updates'], 8)

    def test_device_query_failures_and_mapping(self):
        for module in (self.inference, self.callback):
            with patch('subprocess.run', side_effect=subprocess.CalledProcessError(1, 'query')):
                with self.assertRaises(subprocess.CalledProcessError):
                    module.gpu_preflight('GPU-test', 100)
            with patch('subprocess.run', return_value=SimpleNamespace(stdout='GPU-other, board, 10000')):
                with self.assertRaisesRegex(RuntimeError, 'mismatch'):
                    module.gpu_preflight('GPU-test', 100)
            with patch('subprocess.run', return_value=SimpleNamespace(stdout='GPU-test, board, 1')):
                with self.assertRaisesRegex(RuntimeError, 'Insufficient'):
                    module.gpu_preflight('GPU-test', 100)
            with patch('subprocess.run', side_effect=[SimpleNamespace(stdout='GPU-test, board, 10000'),
                                                       SimpleNamespace(stdout='GPU-test, 123, other, 20')]), \
                 patch.dict('os.environ', {}, clear=True):
                record = module.gpu_preflight('GPU-test', 100)
                import os
                self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], 'GPU-test')
                self.assertEqual(record['logical_device'], 'cuda:0')
                self.assertIn('123', record['processes'])

    def test_relative_paths_follow_invocation_cwd(self):
        import os
        cwd = os.getcwd()
        try:
            os.chdir(self.root)
            self.assertEqual(self.inference.checked_file('checkpoint'), str(self.root/'checkpoint'))
            self.assertEqual(self.callback.output_path('new'), str(self.root/'new'))
        finally:
            os.chdir(cwd)


if __name__ == '__main__':
    unittest.main()
