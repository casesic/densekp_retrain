"""Repository paths shared by directly executable tools."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_APPEARANCE_MANIFEST = (
    REPO_ROOT / "outputs/baselines/p0_dressed_eye_background_fixed/manifest.jsonl"
)


def repo_root() -> Path:
    """Return the repository root without depending on the process cwd."""
    return REPO_ROOT
