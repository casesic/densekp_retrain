#!/usr/bin/env python3
"""Inspect/export the sparse definition of all 138 SMPL-X surface points."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from densekp_retrain.annotation import describe_surface_operator, surface_operator_rows

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--regressor", type=Path, default=Path("assets/densekp_regressor_R.npy"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    r = np.load(args.regressor)
    if r.shape != (138, 10475):
        raise SystemExit(f"unexpected shape: {r.shape}")
    rows = surface_operator_rows(r)
    print(describe_surface_operator(r))
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"wrote {args.json}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
