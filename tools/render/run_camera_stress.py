"""Render all eight deterministic configurations through the existing renderer."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser();p.add_argument('--config',help='Optional single config smoke run');p.add_argument('--blender-bin',type=Path);p.add_argument('--clothing-texture-dir',type=Path);a=p.parse_args()
cfg=json.loads((root/'configs/perspective_camera_stress.json').read_text())
out=root/'outputs/perspective_camera_stress'
for c in cfg['configs']:
    if a.config and c['name']!=a.config:continue
    target=out/'_renders'/c['name']
    if (target/'p0_render_report.json').is_file() and not a.config:
        print('Keeping complete render',c['name'],flush=True)
        continue
    print('Rendering',c['name'],flush=True)
    cmd=[sys.executable,str(root/'tools/render/run_eye_background_fix.py'),
         '--output-dir',str(target),'--frames','16','--camera-placement',str(out/'configs'/f"{c['name']}.json")]
    if a.blender_bin: cmd += ['--blender-bin',str(a.blender_bin)]
    if a.clothing_texture_dir: cmd += ['--clothing-texture-dir',str(a.clothing_texture_dir)]
    subprocess.run(cmd,check=True,cwd=root)
    report=json.loads((target/'p0_render_report.json').read_text())
    if report['projection_error_px']['max']>=.1:
        raise RuntimeError('Unexpected projection error; stop before further configs')
    print('PASS projection',report['projection_error_px']['max'],flush=True)
