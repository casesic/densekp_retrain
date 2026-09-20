"""Non-mutating inspection helpers for the DenseKP surface operator."""

import numpy as np


def surface_operator_rows(regressor: np.ndarray) -> list[dict]:
    rows = []
    for point_id, row in enumerate(regressor):
        ids = np.flatnonzero(row)
        rows.append(
            {
                "point_id": point_id,
                "vertex_ids": ids.tolist(),
                "weights": row[ids].tolist(),
            }
        )
    return rows


def describe_surface_operator(regressor: np.ndarray) -> str:
    return (
        f"shape={regressor.shape} dtype={regressor.dtype} "
        f"nnz={np.count_nonzero(regressor)} "
        f"row_sum=[{regressor.sum(1).min():.8f},{regressor.sum(1).max():.8f}]"
    )

