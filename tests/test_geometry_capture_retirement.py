"""Capture refusal and verify dispatch only; never instantiate a geometry model."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class GeometryCaptureRetirementTests(unittest.TestCase):
    def test_cli_capture_rejects_before_imports_assets_or_writes(self):
        code = r"""
import os, runpy, sys
sys.dont_write_bytecode = True
script, output = sys.argv[1:]
def audit(event, args):
    if event == 'import' and args[0].split('.')[0] in (
        'numpy', 'torch', 'smplx', 'densekp', 'densekp_retrain', 'tests', 'bpy'
    ):
        raise AssertionError(('computation import before refusal', args[0]))
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError(('write before refusal', path))
        if isinstance(path, str) and (
            '/assets/' in path or ('/outputs/' in path and not path.endswith(('.py', '.pyc')))
            or path.endswith(('.npy', '.npz', '.pkl', '.pt', '.ckpt'))
        ):
            raise AssertionError(('asset read before refusal', path))
    if event in ('os.mkdir', 'os.remove', 'os.rename', 'subprocess.Popen', 'os.system'):
        raise AssertionError(('side effect before refusal', event))
sys.addaudithook(audit)
sys.argv = [script, '--output', output]
runpy.run_path(script, run_name='__main__')
"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code,
                 str(ROOT / "tools/validate/verify_geometry_camera.py"),
                 str(root / "output")], text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("geometry capture retired in B2b", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_original_api_and_run_reject_before_imports_or_computation(self):
        from tests import geometry_camera_fixture as fixture
        from tools.validate import verify_geometry_camera as verifier

        def forbidden(*args, **kwargs):
            raise AssertionError("capture reached imports, computation or IO")

        with (
            patch("builtins.__import__", side_effect=forbidden),
            patch.object(fixture.importlib, "import_module", side_effect=forbidden),
            patch.object(fixture, "synthetic", side_effect=forbidden),
            patch.object(fixture.np, "load", side_effect=forbidden),
            patch.object(Path, "open", side_effect=forbidden),
            patch.object(Path, "mkdir", side_effect=forbidden),
        ):
            with self.assertRaisesRegex(RuntimeError, "original geometry capture retired"):
                fixture.api(original=True)
            with self.assertRaisesRegex(RuntimeError, "original geometry capture retired"):
                verifier.run(None, original=True)

    def test_before_selects_verify_branch_without_running_geometry(self):
        # Dispatch evidence only. Real asset replay is deliberately interrupted.
        from tools.validate import verify_geometry_camera as verifier

        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            with (
                patch.object(sys, "argv", ["verify_geometry_camera.py", "--before",
                                         str(root / "before"), "--output", str(output)]),
                patch.object(verifier, "run", side_effect=RuntimeError("verify dispatch sentinel")) as run,
                patch.object(verifier, "compare") as compare,
            ):
                with self.assertRaisesRegex(RuntimeError, "verify dispatch sentinel"):
                    verifier.main()
                run.assert_called_once_with(output.resolve(), original=False)
                compare.assert_not_called()
            self.assertTrue(output.is_dir())
            self.assertEqual(list(output.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
