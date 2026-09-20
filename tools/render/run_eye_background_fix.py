"""Run the fixed appearance with the previous baseline geometry/camera settings."""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from densekp_retrain.rendering.jobs import run_eye_background_fix

p=argparse.ArgumentParser()
p.add_argument('--output-dir','--output',dest='output_dir',required=True,type=Path)
p.add_argument('--frames',type=int,default=1)
p.add_argument('--resolution',type=int,default=512)
p.add_argument('--camera-placement',type=Path)
p.add_argument('--blender-bin',type=Path)
p.add_argument('--baseline-dir',type=Path,default=Path('outputs/p0_dressed_hdri_eye_fixed'))
p.add_argument('--clothing-texture-dir',type=Path,default=os.environ.get('BEDLAM_CLOTHING_TEXTURE_DIR'))
a=p.parse_args()
if not a.clothing_texture_dir:
    p.error('--clothing-texture-dir is required (or set BEDLAM_CLOTHING_TEXTURE_DIR)')
run_eye_background_fix(a, ROOT)
