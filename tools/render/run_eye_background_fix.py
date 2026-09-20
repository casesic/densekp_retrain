"""Run the fixed appearance with the previous baseline geometry/camera settings."""

import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from densekp_retrain.rendering.jobs import run_eye_background_fix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir", "--output", dest="output_dir", required=True, type=Path
    )
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--camera-placement", type=Path)
    parser.add_argument("--blender-bin", type=Path)
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/baselines/p0_dressed_hdri_eye_fixed"),
    )
    parser.add_argument(
        "--clothing-texture-dir",
        type=Path,
        default=os.environ.get("BEDLAM_CLOTHING_TEXTURE_DIR"),
    )
    args = parser.parse_args()
    if not args.clothing_texture_dir:
        parser.error(
            "--clothing-texture-dir is required (or set BEDLAM_CLOTHING_TEXTURE_DIR)"
        )
    return args


def main() -> None:
    run_eye_background_fix(parse_args(), ROOT)


if __name__ == "__main__":
    main()
