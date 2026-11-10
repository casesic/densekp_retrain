#!/usr/bin/env python3
"""Describe an existing garment or select a contiguous cache-consumption view."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from densekp_retrain.assets.garment_contract import (
    describe_fixed, describe_simple, read_json, select_asset_window,
    validate_description, write_new,
)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--describe-simple", type=Path)
    group.add_argument("--describe-fixed", type=Path)
    group.add_argument("--select-window", type=Path)
    parser.add_argument("--base-authority", type=Path)
    parser.add_argument("--window", type=int, nargs=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.describe_simple:
        result = validate_description(describe_simple(args.describe_simple))
    elif args.describe_fixed:
        if not args.base_authority:
            parser.error("--describe-fixed requires --base-authority")
        result = validate_description(describe_fixed(args.describe_fixed, args.base_authority))
    else:
        if args.window is None:
            parser.error("--select-window requires --window START END")
        result = select_asset_window(read_json(args.select_window), args.window)
        result["description_ref"] = str(args.select_window.resolve())
    write_new(args.output, result)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
