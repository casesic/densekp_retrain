"""Reusable DenseKP annotation infrastructure."""

from .serialization import write_bedlam_tiny_dataset
from .surface import describe_surface_operator, surface_operator_rows

__all__ = [
    "describe_surface_operator",
    "surface_operator_rows",
    "write_bedlam_tiny_dataset",
]

