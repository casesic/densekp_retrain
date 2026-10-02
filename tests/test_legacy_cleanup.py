"""Independent historical behavior and import boundaries, without large assets."""

import ast
import base64
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
import zlib

from tests.legacy_cleanup_fixture import capture

ROOT = Path(__file__).resolve().parents[1]


class LegacyCleanupTests(unittest.TestCase):
    def test_legacy_values_rng_and_failure_behavior(self):
        from densekp.core.datasets import utils as crop
        from densekp.core.utils import train_utils as training
        from densekp.core import losses

        expected = json.loads(
            (ROOT / "tests/fixtures/legacy_cleanup_v1.json").read_text()
        )
        with TemporaryDirectory() as temp:
            actual = capture(Path(temp) / "capture", crop, training, losses)
        self.assertEqual(actual, expected)

    def test_recipe_direct_execution_body_unchanged(self):
        frozen = json.loads(
            (ROOT / "tests/fixtures/legacy_import_bodies_v1.json").read_text()
        )
        for name, encoded in frozen.items():
            with self.subTest(script=name):
                tree = ast.parse((ROOT / name).read_text())
                main = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main"
                )
                self.assertEqual(
                    ast.dump(
                        ast.Module(body=main.body, type_ignores=[]),
                        include_attributes=False,
                    ),
                    zlib.decompress(base64.b64decode(encoded)).decode(),
                )

    def test_imports_are_inert_without_optional_dependencies(self):
        # -I prevents the calling environment's PYTHONPATH from hiding imports.
        code = r"""
import json,os,random,runpy,sys,pkgutil
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(sys.argv[1]);paths=json.loads((root/'tests/fixtures/legacy_import_bodies_v1.json').read_text())
active=False
blocked=[]
def audit(event,args):
    if not active:return
    if event=='open':
        path,mode,flags=args
        if flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
            raise AssertionError(('write on import',path))
        if isinstance(path,str) and Path(path).suffix not in ('.py','.pyc'):
            raise AssertionError(('asset read on import',path))
    if event in ('os.mkdir','os.remove','os.rename','os.listdir','os.scandir','subprocess.Popen','os.system'):
        raise AssertionError(('mutation, scan or process on import',event,args))
sys.addaudithook(audit)
for name in paths:
    state=random.getstate();path_before=list(sys.path);active=True
    try:namespace=runpy.run_path(str(root/name),run_name='legacy_import_check')
    finally:active=False
    assert callable(namespace['main'])
    assert state==random.getstate() and path_before==sys.path
assert not any(n in sys.modules for n in ['bpy','torch','numpy','smplx','matplotlib'])
print(json.dumps(dict(imported=len(paths),asset_reads=False,writes=False,processes=False,optional_dependencies_loaded=False)))
"""
        result = subprocess.run(
            [sys.executable, "-I", "-c", code, str(ROOT)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["imported"], 12)
