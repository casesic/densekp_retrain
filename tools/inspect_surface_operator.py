#!/usr/bin/env python3
"""Inspect/export the sparse definition of all 138 SMPL-X surface points."""
import argparse
import json
from pathlib import Path
import numpy as np

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor", type=Path, default=Path("assets/densekp_regressor_R.npy"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    r = np.load(args.regressor)
    if r.shape != (138, 10475):
        raise SystemExit(f"unexpected shape: {r.shape}")
    rows = []
    for point_id, row in enumerate(r):
        ids = np.flatnonzero(row)
        rows.append({"point_id": point_id, "vertex_ids": ids.tolist(), "weights": row[ids].tolist()})
    print(f"shape={r.shape} dtype={r.dtype} nnz={np.count_nonzero(r)} row_sum=[{r.sum(1).min():.8f},{r.sum(1).max():.8f}]")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
