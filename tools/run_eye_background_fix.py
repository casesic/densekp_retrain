"""Run the fixed appearance with the previous baseline geometry/camera settings."""
import argparse
import json
import subprocess
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--output',required=True,type=Path)
p.add_argument('--frames',type=int,default=1)
p.add_argument('--resolution',type=int,default=512)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
baseline=root/'outputs/p0_dressed_hdri_eye_fixed'
r=json.loads((baseline/'manifest.jsonl').read_text().splitlines()[0])
cmd=['/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender','-b','-t','4','--python',str(root/'tools/render_p0_blender.py'),'--',
     '--input-geometry',str(baseline/'geometry'),'--clothing-geometry',str(baseline/'geometry'),
     '--output-dir',str(a.output),'--clothing-texture-id','texture_01',
     '--clothing-texture-dir','/home/yy/datasets/BEDLAM/clothing_data/tiny/rp_aaron_posed_009/1095/textures/texture_01',
     '--body-texture',r['body_diffuse_texture_path'],'--body-uv',r['body_uv_path'],
     '--environment-mode','hdri','--hdri-path',r['hdri_path'],'--hdri-strength','1.0','--hdri-rotation-deg','0',
     '--manual-lights','fill','--eye-render-mode','procedural','--eye-mapping',str(root/'outputs/eye_fix_v2/procedural_eye_coordinates.npz'),
     '--ground-mode','shadow_catcher','--render-frame-count',str(a.frames)]
cmd += ['--resolution',str(a.resolution)]
a.output.mkdir(parents=True,exist_ok=True)
with (a.output/'render.log').open('w') as log:
    subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
if not (a.output/'p0_render_report.json').is_file(): raise RuntimeError('Blender did not finish rendering; inspect render.log')
if not (a.output/'geometry').exists(): (a.output/'geometry').symlink_to((baseline/'geometry').resolve(),target_is_directory=True)
