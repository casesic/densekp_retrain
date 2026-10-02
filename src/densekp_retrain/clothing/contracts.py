"""Checks for loose, surface-bound garments; never substitutes for visual acceptance."""

import json
from pathlib import Path
import numpy as np


def validate_binding(folder, current_reference, vertices, *, require_studio=False):
    folder = Path(folder)
    info = json.loads((folder / "clothing_manifest.json").read_text())
    if info.get("contract") != "commuter_surface_binding_v1":
        raise ValueError("Unknown commuter garment contract")
    reference = np.load(folder / "reference_body_vertices.npy")
    if not np.array_equal(reference, current_reference):
        raise ValueError("Garment attachment no longer matches the posed body")
    offsets = np.load(folder / "rest_offsets.npy")
    rotations = np.load(folder / "binding_rotations.npy")
    correction = np.load(folder / "collision_correction.npy")
    count = current_reference.shape[1]
    if (
        offsets.shape != (count, 3)
        or rotations.shape != (count, 3, 3)
        or correction.shape != (count, 3)
    ):
        raise ValueError("Malformed garment binding arrays")
    expected = reference + np.einsum("nab,nb->na", rotations, offsets)[None] + correction[None]
    if (
        vertices.shape != expected.shape
        or not np.isfinite(vertices).all()
        or not np.isfinite(expected).all()
    ):
        raise ValueError("Nonfinite or malformed garment mesh")
    error = float(np.abs(vertices - expected).max())
    if error > 2e-6:
        raise ValueError("Saved garment does not reconstruct from its binding")
    if require_studio:
        if info.get("status") != "studio_accepted":
            raise ValueError("Commuter garment has not passed studio acceptance")
        collision = info["collision"]
        if (
            collision["negative_vertex_count"]
            or collision["negative_face_center_count"]
            or collision.get("correction_budget_exhausted", False)
            or np.linalg.norm(correction, axis=1).max() >= 0.06
        ):
            raise ValueError("Commuter garment collision gate failed")
    return dict(
        binding_reconstruction_max_m=error,
        collision_correction_max_m=float(np.linalg.norm(correction, axis=1).max()),
    )
