"""Runtime executable and frozen-baseline discovery for tool wrappers."""

import json
import os
import shutil
from pathlib import Path

from tools.common.paths import DEFAULT_APPEARANCE_MANIFEST


def find_blender_executable(cli_value: str | Path | None = None) -> Path:
    """Resolve Blender from CLI, ``BLENDER_BIN``, then ``PATH``."""
    candidate = cli_value or os.environ.get("BLENDER_BIN") or shutil.which("blender")
    if not candidate:
        raise RuntimeError(
            "Blender executable not found. Pass --blender-bin, set BLENDER_BIN, "
            "or add blender to PATH."
        )
    executable = Path(candidate).expanduser()
    if not executable.is_file():
        raise RuntimeError(f"Blender executable does not exist: {executable}")
    if not os.access(executable, os.X_OK):
        raise RuntimeError(f"Blender path is not executable: {executable}")
    return executable.resolve()


def load_appearance_baseline(
    manifest_path: str | Path | None = None,
) -> tuple[dict, Path]:
    """Load the first record of the frozen appearance manifest unchanged."""
    path = Path(manifest_path) if manifest_path else DEFAULT_APPEARANCE_MANIFEST
    path = path.expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.is_file():
        raise RuntimeError(
            f"Appearance baseline manifest not found: {path}. "
            "Pass --appearance-manifest to override it."
        )
    lines = path.read_text().splitlines()
    if not lines:
        raise RuntimeError(f"Appearance baseline manifest is empty: {path}")
    return json.loads(lines[0]), path.resolve()
