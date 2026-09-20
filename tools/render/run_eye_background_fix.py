"""Run the fixed appearance with the previous baseline geometry/camera settings."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tools.common.runtime import find_blender_executable

p=argparse.ArgumentParser()
p.add_argument('--output-dir','--output',dest='output_dir',required=True,type=Path)
p.add_argument('--frames',type=int,default=1)
p.add_argument('--resolution',type=int,default=512)
p.add_argument('--camera-placement',type=Path)
p.add_argument('--blender-bin',type=Path)
p.add_argument('--baseline-dir',type=Path,default=Path('outputs/p0_dressed_hdri_eye_fixed'))
p.add_argument('--clothing-texture-dir',type=Path,default=os.environ.get('BEDLAM_CLOTHING_TEXTURE_DIR'))
a=p.parse_args()
root=ROOT
baseline=a.baseline_dir if a.baseline_dir.is_absolute() else root/a.baseline_dir
r=json.loads((baseline/'manifest.jsonl').read_text().splitlines()[0])
if not a.clothing_texture_dir:
    p.error('--clothing-texture-dir is required (or set BEDLAM_CLOTHING_TEXTURE_DIR)')
clothing_texture_dir=a.clothing_texture_dir.expanduser().resolve()
cmd=[str(find_blender_executable(a.blender_bin)),'-b','-t','4','--python',str(root/'tools/render/render_p0_blender.py'),'--',
     '--input-geometry',str(baseline/'geometry'),'--clothing-geometry',str(baseline/'geometry'),
     '--output-dir',str(a.output_dir),'--clothing-texture-id','texture_01',
     '--clothing-texture-dir',str(clothing_texture_dir),
     '--body-texture',r['body_diffuse_texture_path'],'--body-uv',r['body_uv_path'],
     '--environment-mode','hdri','--hdri-path',r['hdri_path'],'--hdri-strength','1.0','--hdri-rotation-deg','0',
     '--manual-lights','fill','--eye-render-mode','procedural','--eye-mapping',str(root/'outputs/eye_fix_v2/procedural_eye_coordinates.npz'),
     '--ground-mode','shadow_catcher','--render-frame-count',str(a.frames)]
cmd += ['--resolution',str(a.resolution)]
if a.camera_placement: cmd += ['--camera-placement',str(a.camera_placement.resolve())]
a.output_dir.mkdir(parents=True,exist_ok=True)
with (a.output_dir/'render.log').open('w') as log:
    subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
if not (a.output_dir/'p0_render_report.json').is_file(): raise RuntimeError(f'Blender did not finish rendering; inspect {a.output_dir/"render.log"}')
if not (a.output_dir/'geometry').exists(): (a.output_dir/'geometry').symlink_to((baseline/'geometry').resolve(),target_is_directory=True)
