#!/usr/bin/env python3
"""Render one temporal validation case with the frozen appearance settings."""
import argparse, json, subprocess
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--geometry',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,required=True); p.add_argument('--hdri',type=Path,required=True)
    p.add_argument('--clothing-geometry',type=Path); p.add_argument('--clothing-texture-dir',type=Path)
    p.add_argument('--resolution',type=int,default=512); p.add_argument('--hdri-rotation',type=float,default=0)
    a=p.parse_args(); root=Path(__file__).resolve().parents[1]
    baseline=json.loads((root/'outputs/p0_dressed_eye_background_fixed/manifest.jsonl').read_text().splitlines()[0])
    cmd=['/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender','-b','-t','4','--python',str(root/'tools/render_p0_blender.py'),'--',
         '--input-geometry',str(a.geometry.resolve()),'--output-dir',str(a.output.resolve()),'--render-frame-count',str(a.frames),'--resolution',str(a.resolution),
         '--appearance-baseline','--body-texture',baseline['body_diffuse_texture_path'],'--body-uv',baseline['body_uv_path'],
         '--environment-mode','hdri','--hdri-path',str(a.hdri.resolve()),'--hdri-strength','1.0','--hdri-rotation-deg',str(a.hdri_rotation),
         '--manual-lights','fill','--eye-render-mode','procedural','--eye-mapping',str((root/'outputs/eye_fix_v2/procedural_eye_coordinates.npz').resolve()),
         '--ground-mode','shadow_catcher']
    if a.clothing_geometry:
        if not a.clothing_texture_dir: p.error('--clothing-texture-dir is required with clothing')
        cmd += ['--clothing-geometry',str(a.clothing_geometry.resolve()),'--clothing-texture-id','texture_01','--clothing-texture-dir',str(a.clothing_texture_dir.resolve())]
    a.output.mkdir(parents=True,exist_ok=True)
    with (a.output/'render.log').open('w') as log: subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    if not (a.output/'p0_render_report.json').is_file(): raise RuntimeError('Blender did not finish; inspect render.log')
    link=a.output/'geometry'
    if not link.exists(): link.symlink_to(a.geometry.resolve(),target_is_directory=True)

if __name__=='__main__': main()
