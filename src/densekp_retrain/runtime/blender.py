"""Blender executable discovery."""

import os
import shutil
from pathlib import Path


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

