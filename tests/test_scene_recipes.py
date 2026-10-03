"""Render job contracts and explicit real-Blender recipe import checks."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from densekp_retrain.rendering import jobs
from tests.scene_jobs_fixture import capture

ROOT = Path(__file__).resolve().parents[1]


class SceneRecipeTests(unittest.TestCase):
    def test_jobs_match_independent_before_commands_paths_failures_and_artifacts(self):
        before = json.loads((ROOT / "tests/fixtures/scene_jobs_v1.json").read_text())
        with TemporaryDirectory() as temporary:
            after = capture(Path(temporary) / "new", jobs)
        self.assertEqual(after, before)


    def test_job_builder_reads_manifest_without_creating_output_or_launching(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            baseline = root / "baseline"
            baseline.mkdir()
            (baseline / "manifest.jsonl").write_text(
                json.dumps(
                    dict(
                        body_diffuse_texture_path="body.png",
                        body_uv_path="uv.npz",
                        hdri_path="sky.exr",
                    )
                )
                + "\n"
            )
            args = SimpleNamespace(
                baseline_dir=Path("baseline"),
                clothing_texture_dir=Path("textures"),
                blender_bin=None,
                output_dir=root / "output",
                frames=1,
                resolution=512,
                camera_placement=None,
            )
            with patch.object(
                jobs, "find_blender_executable", return_value=Path("/fixture/blender")
            ), patch.object(
                jobs,
                "run_logged",
                side_effect=AssertionError("builder launched a process"),
            ):
                job = jobs.build_eye_background_job(args, root)
            self.assertFalse(args.output_dir.exists())
            self.assertEqual(job.geometry, baseline / "geometry")
            self.assertEqual(
                job.command[job.command.index("--output-dir") + 1], str(args.output_dir)
            )


def check_blender_recipe_imports(root):
    """Import real Blender recipes with IO/process and scene-access guards.

    This checks definitions only, never a recipe main or scene assembly.
    Call explicitly inside Blender; it is not an ordinary-Python unittest.
    """
    import argparse, csv, dis, json, math, os, pkgutil, random, runpy, sys
    from pathlib import Path
    import bpy
    from mathutils import Vector

    paths = [root / 'tools/scene' / name for name in (
        'build_cycling_forest.py', 'prepare_hidden_alley.py', 'validate_cycling_forest.py'
    )]
    target_names = {str(p) for p in paths}
    active = False
    instructions = {}

    def audit(event, args):
        if not active:
            return
        if event == 'open':
            path, mode, flags = args
            if flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                raise AssertionError(('write on Blender recipe import', path))
            if isinstance(path, str) and Path(path).suffix not in ('.py', '.pyc'):
                raise AssertionError(('asset read on Blender recipe import', path))
        if event in ('os.mkdir', 'os.remove', 'os.rename', 'os.listdir', 'os.scandir',
                     'subprocess.Popen', 'os.system'):
            raise AssertionError(('IO/process on Blender recipe import', event))

    def trace(frame, event, arg):
        if frame.f_code.co_filename not in target_names:
            return None
        if event == 'call':
            if frame.f_code.co_name == 'main':
                raise AssertionError('recipe main called during import')
            frame.f_trace_opcodes = True
        if event == 'opcode':
            ops = instructions.setdefault(frame.f_code, {
                op.offset: op for op in dis.get_instructions(frame.f_code)
            })
            op = ops.get(frame.f_lasti)
            if op and op.opname in ('LOAD_ATTR', 'LOAD_METHOD', 'IMPORT_FROM') and op.argval in ('ops', 'data', 'context'):
                raise AssertionError(('Blender scene access during recipe import', op.argval))
        return trace

    sys.addaudithook(audit)
    original_trace = sys.gettrace()
    original_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    before_objects = [(o.name, o.as_pointer(), tuple(tuple(r) for r in o.matrix_world)) for o in bpy.data.objects]
    before_scenes = [(s.name, s.as_pointer()) for s in bpy.data.scenes]
    try:
        for path in paths:
            rng, search_path = random.getstate(), list(sys.path)
            active = True
            sys.settrace(trace)
            try:
                namespace = runpy.run_path(str(path), run_name='blender_recipe_import_check')
            finally:
                active = False
                sys.settrace(original_trace)
            assert callable(namespace['main'])
            assert random.getstate() == rng and sys.path == search_path
        assert before_objects == [(o.name, o.as_pointer(), tuple(tuple(r) for r in o.matrix_world)) for o in bpy.data.objects]
        assert before_scenes == [(s.name, s.as_pointer()) for s in bpy.data.scenes]
    finally:
        active = False
        sys.settrace(original_trace)
        sys.dont_write_bytecode = original_bytecode
    return dict(imported=[str(p.relative_to(root)) for p in paths], real_blender=bpy.app.version_string,
                asset_reads=False, writes=False, processes=False, scene_access=False, main_called=False)
