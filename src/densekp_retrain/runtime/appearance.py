"""Frozen appearance baseline discovery."""

import json
from pathlib import Path

from .paths import DEFAULT_APPEARANCE_MANIFEST


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

