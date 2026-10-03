"""Ordinary-Python recipe imports without assets or optional dependencies."""

import json
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ImportBoundaryTests(unittest.TestCase):
    def test_imports_are_inert_without_optional_dependencies(self):
        # -I prevents the calling environment's PYTHONPATH from hiding imports.
        code = r"""
import json,os,random,runpy,sys,pkgutil
from pathlib import Path
sys.dont_write_bytecode=True
root=Path(sys.argv[1])
paths=[
    'tools/experiments/audit_skin_texture.py',
    'tools/experiments/prepare_procedural_eyes.py',
    'tools/validate/blender_headless_smoke_test.py',
]
active=False
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
        self.assertEqual(json.loads(result.stdout)["imported"], 3)
