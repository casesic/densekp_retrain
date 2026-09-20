#!/usr/bin/env python3
"""Render one temporal validation case with the frozen appearance settings."""
import argparse, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.common.runtime import find_blender_executable, load_appearance_baseline

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--geometry',type=Path,required=True); p.add_argument('--output-dir','--output',dest='output_dir',type=Path,required=True)
    p.add_argument('--frames',type=int,required=True); p.add_argument('--hdri',type=Path,required=True)
    p.add_argument('--clothing-geometry',type=Path); p.add_argument('--clothing-texture-dir',type=Path)
    p.add_argument('--resolution',type=int,default=512); p.add_argument('--hdri-rotation',type=float,default=0)
    p.add_argument('--blender-bin',type=Path); p.add_argument('--appearance-manifest',type=Path)
    a=p.parse_args(); root=ROOT
    baseline,_=load_appearance_baseline(a.appearance_manifest)
    cmd=[str(find_blender_executable(a.blender_bin)),'-b','-t','4','--python',str(root/'tools/render/render_p0_blender.py'),'--',
         '--input-geometry',str(a.geometry.resolve()),'--output-dir',str(a.output_dir.resolve()),'--render-frame-count',str(a.frames),'--resolution',str(a.resolution),
         '--appearance-baseline','--body-texture',baseline['body_diffuse_texture_path'],'--body-uv',baseline['body_uv_path'],
         '--environment-mode','hdri','--hdri-path',str(a.hdri.resolve()),'--hdri-strength','1.0','--hdri-rotation-deg',str(a.hdri_rotation),
         '--manual-lights','fill','--eye-render-mode','procedural','--eye-mapping',str((root/'outputs/eye_fix_v2/procedural_eye_coordinates.npz').resolve()),
         '--ground-mode','shadow_catcher']
    if a.clothing_geometry:
        if not a.clothing_texture_dir: p.error('--clothing-texture-dir is required with clothing')
        cmd += ['--clothing-geometry',str(a.clothing_geometry.resolve()),'--clothing-texture-id','texture_01','--clothing-texture-dir',str(a.clothing_texture_dir.resolve())]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    with (a.output_dir/'render.log').open('w') as log: subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (a.output_dir/'p0_render_report.json').is_file(): raise RuntimeError(f'Blender did not finish; inspect {a.output_dir/"render.log"}')
    link=a.output_dir/'geometry'
    if not link.exists(): link.symlink_to(a.geometry.resolve(),target_is_directory=True)

if __name__=='__main__': main()
