"""Subprocess helpers for render jobs."""

import subprocess
from pathlib import Path
from typing import Sequence


def run_logged(
    command: Sequence[str],
    *,
    cwd: Path,
    log_path: Path,
) -> None:
    """Run a command in a fixed cwd and combine its output in one log."""
    with log_path.open("w") as log:
        subprocess.run(
            list(command),
            cwd=cwd,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )

