"""Fail closed on identity, source, image, occluder or paired-box leakage."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]


# Historical import compatibility; active callers import the public module.
from densekp_retrain.data.source_split import validate_split


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--train-index", type=Path)
    p.add_argument("--val-index", type=Path)
    p.add_argument("--allow-proposal", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    indices = {
        s: json.loads(v.read_text())
        for s, v in [("train", a.train_index), ("val", a.val_index)]
        if v
    }
    result = validate_split(json.loads(a.split.read_text()), indices, a.allow_proposal)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
