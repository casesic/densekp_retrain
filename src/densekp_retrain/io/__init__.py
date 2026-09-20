"""File-format helpers for synthetic-data tooling."""

from .exr import read_combined_depth_index
from .images import linear_to_srgb

__all__ = ["linear_to_srgb", "read_combined_depth_index"]

