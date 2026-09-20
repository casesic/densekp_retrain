#!/usr/bin/env python3
"""Render one temporal validation case with the frozen appearance settings."""
import argparse, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))

from densekp_retrain.rendering.jobs import run_temporal_case

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--geometry',type=Path,required=True); p.add_argument('--output-dir','--output',dest='output_dir',type=Path,required=True)
    p.add_argument('--frames',type=int,required=True); p.add_argument('--hdri',type=Path,required=True)
    p.add_argument('--clothing-geometry',type=Path); p.add_argument('--clothing-texture-dir',type=Path)
    p.add_argument('--resolution',type=int,default=512); p.add_argument('--hdri-rotation',type=float,default=0)
    p.add_argument('--blender-bin',type=Path); p.add_argument('--appearance-manifest',type=Path)
    a=p.parse_args()
    if a.clothing_geometry:
        if not a.clothing_texture_dir: p.error('--clothing-texture-dir is required with clothing')
    run_temporal_case(a, ROOT)

if __name__=='__main__': main()
