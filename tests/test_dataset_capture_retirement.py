"""Retired Dataset capture rejects before imports, sampling or filesystem effects."""

from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


class DatasetCaptureRetirementTests(unittest.TestCase):
    def test_cli_capture_rejects_before_data_imports_or_writes(self):
        code = r"""
import os, runpy, sys
from pathlib import Path
sys.dont_write_bytecode = True
script, acceptance, output = sys.argv[1:]
def audit(event, args):
    if event == 'import' and args[0].split('.')[0] in (
        'numpy', 'torch', 'densekp', 'densekp_retrain', 'tests', 'bpy'
    ):
        raise AssertionError(('data import before rejection', args[0]))
    if event == 'open':
        path, mode, flags = args
        if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
            raise AssertionError(('write before rejection', path))
        if isinstance(path, str) and (
            path == acceptance or path.startswith(acceptance + os.sep)
            or path == output or path.startswith(output + os.sep)
        ):
            raise AssertionError(('data read before rejection', path))
    if event in ('os.mkdir', 'os.remove', 'os.rename', 'subprocess.Popen', 'os.system'):
        raise AssertionError(('filesystem/process side effect', event))
sys.addaudithook(audit)
sys.argv = [script, '--phase', 'capture', '--acceptance', acceptance, '--output', output]
runpy.run_path(script, run_name='__main__')
"""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            result = subprocess.run(
                [sys.executable, "-I", "-B", "-c", code,
                 str(ROOT / "tools/validate/verify_dataset_crop.py"),
                 str(root / "absent-baseline"), str(root / "output")],
                text=True, capture_output=True,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("capture retired in B2b", result.stderr)
            self.assertNotIn("Traceback", result.stderr)
            self.assertEqual(list(root.iterdir()), [])

    def test_original_helper_rejects_before_imports_sampling_or_io(self):
        from tests import dataset_crop_fixture as fixture

        def forbidden(*args, **kwargs):
            raise AssertionError("original capture reached imports, sampling or data IO")

        with (
            patch("builtins.__import__", side_effect=forbidden),
            patch.object(fixture, "configuration", side_effect=forbidden),
            patch.object(fixture, "seed_all", side_effect=forbidden),
            patch.object(fixture.np, "load", side_effect=forbidden),
            patch.object(fixture.cv2, "imread", side_effect=forbidden),
            patch.object(Path, "mkdir", side_effect=forbidden),
            patch.object(Path, "open", side_effect=forbidden),
        ):
            with self.assertRaisesRegex(RuntimeError, "original Dataset capture retired in B2b"):
                fixture.run_dataset(None, original=True)


if __name__ == "__main__":
    unittest.main()
