#!/usr/bin/env python3
"""Plan or explicitly build a configured standard-SMPL-X dressed motion."""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from densekp_retrain.assets.composition import build, plan, prepare_preview, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recipes", type=Path, default=ROOT / "configs/asset_composition_a2_20261010/recipes.json")
    parser.add_argument("--name", required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--plan", action="store_true")
    action.add_argument("--build", action="store_true")
    action.add_argument("--prepare-preview", action="store_true")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--preview-frames", type=int, default=60)
    args = parser.parse_args()
    recipes = read(args.recipes)
    if recipes.get("schema") != "densekp_asset_composition_recipes_v1":
        raise ValueError("unsupported_recipe_schema")
    recipe = recipes["recipes"][args.name]
    output_root = args.output_root or ROOT / recipe["output_dir"]
    if args.prepare_preview:
        if not args.gpu_uuid:
            parser.error("--prepare-preview requires --gpu-uuid")
        print(prepare_preview(output_root, args.name, args.gpu_uuid, frames=args.preview_frames))
        return
    resolved = plan(ROOT, read(ROOT / recipes["registry"]), recipe)
    resolved["output_dir"] = str(output_root.resolve())
    if args.plan:
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    out, validation = build(ROOT, resolved, output_root, args.name)
    print(json.dumps({"output": str(out), "validation": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
