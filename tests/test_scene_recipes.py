"""Frozen recipe ordering and command contracts; no Blender simulation claims."""

import ast
import base64
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zlib

from densekp_retrain.rendering import jobs
from tests.scene_jobs_fixture import capture

ROOT = Path(__file__).resolve().parents[1]


class SceneRecipeTests(unittest.TestCase):
    def test_jobs_match_independent_before_commands_paths_failures_and_artifacts(self):
        before = json.loads((ROOT / "tests/fixtures/scene_jobs_v1.json").read_text())
        with TemporaryDirectory() as temporary:
            after = capture(Path(temporary) / "new", jobs)
        self.assertEqual(after, before)

    def test_original_recipe_statement_order_and_private_helpers_are_unchanged(self):
        frozen = json.loads(
            (ROOT / "tests/fixtures/scene_recipe_bodies_v1.json").read_text()
        )["bodies"]
        for name, encoded in frozen.items():
            with self.subTest(recipe=name):
                tree = ast.parse((ROOT / "tools/scene" / name).read_text())
                main = next(
                    n
                    for n in tree.body
                    if isinstance(n, ast.FunctionDef) and n.name == "main"
                )
                actual = ast.dump(
                    ast.Module(body=main.body, type_ignores=[]),
                    include_attributes=False,
                )
                self.assertEqual(
                    actual, zlib.decompress(base64.b64decode(encoded)).decode()
                )
                # Import-time statements are only imports, docstring, definition and guard.
                for node in tree.body:
                    self.assertIsInstance(
                        node,
                        (ast.Import, ast.ImportFrom, ast.Expr, ast.FunctionDef, ast.If),
                    )
                    if isinstance(node, ast.Expr):
                        self.assertIsInstance(node.value, ast.Constant)
                    if isinstance(node, ast.If):
                        self.assertEqual(
                            ast.unparse(node.test), "__name__ == '__main__'"
                        )

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
