"""Runtime discovery and repository path helpers."""

from .appearance import load_appearance_baseline
from .blender import find_blender_executable
from .paths import DEFAULT_APPEARANCE_MANIFEST, REPO_ROOT, repo_root

__all__ = [
    "DEFAULT_APPEARANCE_MANIFEST",
    "REPO_ROOT",
    "find_blender_executable",
    "load_appearance_baseline",
    "repo_root",
]

