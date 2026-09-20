"""Shared, numerically frozen SMPL-X geometry operations."""

from pathlib import Path
from typing import Any

import numpy as np


def load_smplx_model(
    model_dir: Path,
    *,
    gender: str,
    num_betas: int,
    batch_size: int,
) -> Any:
    import smplx

    kwargs = dict(
        gender=gender,
        ext="npz",
        num_betas=num_betas,
        use_pca=False,
        flat_hand_mean=True,
        batch_size=batch_size,
    )
    if (model_dir / f"SMPLX_{gender.upper()}.npz").is_file():
        return smplx.SMPLX(str(model_dir), **kwargs)
    return smplx.create(str(model_dir), model_type="smplx", **kwargs)


def validate_geometry(
    vertices: np.ndarray,
    regressor: np.ndarray,
    *,
    frames: int,
) -> None:
    if vertices.shape != (frames, 10475, 3):
        raise RuntimeError(f"vertices shape 错误：{vertices.shape}")
    if regressor.shape != (138, 10475):
        raise RuntimeError(f"DenseKP regressor shape 错误：{regressor.shape}")
    if not np.isfinite(vertices).all():
        raise RuntimeError("vertices 存在非有限值")


def compute_densekp_3d(
    regressor: np.ndarray, vertices: np.ndarray
) -> np.ndarray:
    return np.einsum(
        "kv,tvc->tkc", regressor, vertices, optimize=True
    ).astype(np.float32)


def write_geometry_arrays(
    output_dir: Path,
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    densekp_3d: np.ndarray,
) -> None:
    np.save(output_dir / "vertices.npy", vertices)
    np.save(output_dir / "faces.npy", faces)
    np.save(output_dir / "densekp_3d.npy", densekp_3d)
