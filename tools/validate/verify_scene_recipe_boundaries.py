"""Blender-only import/CLI boundary checks, without building or rendering scenes.

Reads frozen original parser prefixes for comparison. Never executes a recipe
body: new modules are imported, and three direct CLI paths stop at --help.
"""

import argparse
import ast
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import random
import runpy
import sys
import bpy
import numpy as np

RECIPES = (
    "build_cycling_forest.py",
    "build_reference_riverfront.py",
    "prepare_hidden_alley.py",
    "validate_cycling_forest.py",
)


def scene_state():
    return dict(
        file=bpy.data.filepath,
        scene=bpy.context.scene.as_pointer(),
        objects=[
            (o.as_pointer(), o.name, o.type, list(map(list, o.matrix_world)))
            for o in bpy.data.objects
        ],
        scenes=[s.as_pointer() for s in bpy.data.scenes],
        meshes=[m.as_pointer() for m in bpy.data.meshes],
        materials=[m.as_pointer() for m in bpy.data.materials],
        images=[m.as_pointer() for m in bpy.data.images],
        cameras=[m.as_pointer() for m in bpy.data.cameras],
    )


def main():
    # Blender ignores PYTHONDONTWRITEBYTECODE unless --python-use-system-env is
    # selected. Suppress interpreter cache writes before auditing recipe I/O.
    sys.dont_write_bytecode = True
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    evidence = args.evidence.resolve()
    active = False
    events = []

    def audit(event, values):
        if not active:
            return
        if event == "open":
            _, mode, flags = values
            if flags & (
                os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
            ):
                events.append([event, str(values)])
                raise RuntimeError("recipe import attempted file write")
        if event in (
            "os.mkdir",
            "os.remove",
            "os.rename",
            "subprocess.Popen",
            "os.system",
        ):
            events.append([event, str(values)])
            raise RuntimeError("recipe import attempted mutation or launch")

    sys.addaudithook(audit)
    results = []
    for name in RECIPES:
        script = root / "tools/scene" / name
        old = evidence / "source_snapshot/tools/scene" / name
        state = scene_state()
        rng = random.getstate()
        numpy_rng = np.random.get_state()
        active = True
        try:
            namespace = runpy.run_path(
                str(script), run_name="scene_recipe_import_check"
            )
        finally:
            active = False
        assert scene_state() == state, name
        assert random.getstate() == rng, name
        after = np.random.get_state()
        assert (
            numpy_rng[0] == after[0]
            and np.array_equal(numpy_rng[1], after[1])
            and numpy_rng[2:] == after[2:]
        )
        assert callable(namespace["main"])
        row = dict(
            script=name,
            import_scene_unchanged=True,
            import_rng_unchanged=True,
            write_or_launch_attempts=0,
        )
        if name != "validate_cycling_forest.py":
            # Capture only the old prefix ending at parse_args. No mkdir or bpy calls follow.
            tree = ast.parse(old.read_text())
            prefix = []
            for node in tree.body:
                prefix.append(node)
                if (
                    isinstance(node, ast.Assign)
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute)
                    and node.value.func.attr == "parse_args"
                ):
                    break
            else:
                raise AssertionError("No parser boundary")
            outputs = []
            statuses = []
            original_argv = sys.argv
            try:
                for version in ("old", "new"):
                    sys.argv = [str(script), "--", "--help"]
                    buffer = io.StringIO()
                    active = True
                    try:
                        with redirect_stdout(buffer):
                            if version == "old":
                                exec(
                                    compile(
                                        ast.Module(body=prefix, type_ignores=[]),
                                        str(old),
                                        "exec",
                                    ),
                                    dict(__name__="__main__", __file__=str(script)),
                                )
                            else:
                                runpy.run_path(str(script), run_name="__main__")
                    except SystemExit as error:
                        statuses.append(error.code)
                    finally:
                        active = False
                    outputs.append(buffer.getvalue())
                assert statuses == [0, 0] and outputs[0] == outputs[1], name
                assert scene_state() == state and random.getstate() == rng, name
            finally:
                sys.argv = original_argv
            row["cli_help_exact"] = True
            (evidence / (name + ".help.txt")).write_text(outputs[1])
        results.append(row)
    assert not events
    (evidence / "blender_import_cli.json").write_text(
        json.dumps(
            dict(
                blender=bpy.app.version_string,
                cases=results,
                old_parser_source="frozen original prefix only, executed after migration independently",
                scene_build_executed=False,
                render_executed=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
