"""Bounded daily checks: source/config syntax and asset-free CPU contracts."""

import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
BASIC_TESTS = [
    "tests.test_body_geometry",
    "tests.test_omni_radtan",
    "tests.test_double_sphere_delivery.DoubleSphereDeliveryTest.test_production_axes",
    "tests.test_double_sphere_delivery.DoubleSphereDeliveryTest.test_center_erp_maps_to_respective_fisheye_centers",
    "tests.test_double_sphere_delivery.DoubleSphereDeliveryTest.test_projection_inverse",
]
MODEL_TESTS = [
    "tests.test_dataset_crop_contract",
    "tests.test_source_packing_contract",
    "tests.test_scene_recipes",
    "tests.test_evaluation_contract",
    "tests.test_legacy_cleanup",
    "tests.test_image_normalization",
    "tests.test_import_boundaries",
    "tests.test_imaging_pipeline",
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Only parse source, JSON and YAML; no numerical checks",
    )
    parser.add_argument(
        "--model-contracts",
        action="store_true",
        help="Also run small model-dependency contracts; no model/optimizer training or inference",
    )
    parser.add_argument(
        "--surface-operator",
        type=Path,
        help="Opt in to inspecting only this explicit local regressor file",
    )
    args = parser.parse_args()
    sources = [ROOT / "setup.py"]
    for name in ("densekp", "src/densekp_retrain", "tools", "tests"):
        sources.extend(sorted((ROOT / name).rglob("*.py")))
    for path in sources:
        ast.parse(path.read_text(), filename=str(path.relative_to(ROOT)))
    import yaml

    configs = []
    for path in sorted((ROOT / "configs").rglob("*")):
        if path.suffix == ".json":
            json.loads(path.read_text())
            configs.append(path)
        elif path.suffix in (".yaml", ".yml"):
            yaml.safe_load(path.read_text())
            configs.append(path)
    print(
        f"Source syntax: {len(sources)} files; config syntax: {len(configs)} files",
        flush=True,
    )
    if not args.no_tests:
        names = BASIC_TESTS + (MODEL_TESTS if args.model_contracts else [])
        subprocess.run(
            [sys.executable, "-B", "-m", "unittest", *names], cwd=ROOT, check=True
        )
    if args.surface_operator:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/annotation/inspect_surface_operator.py"),
                "--regressor",
                str(args.surface_operator.resolve()),
            ],
            cwd=ROOT,
            check=True,
        )
    print("Workspace checks passed; no outputs, checkpoint or texture tree scan.")


if __name__ == "__main__":
    main()
