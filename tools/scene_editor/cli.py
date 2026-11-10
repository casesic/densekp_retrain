"""Save, inspect and replay the same recipe accepted by the browser."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from tools.scene_editor.core import default_recipe, read, write, validate, freeze


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['example','check','freeze','run'])
    parser.add_argument('--scene',choices=['city','riverfront'],default='city')
    parser.add_argument('--recipe',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frame',type=int)
    parser.add_argument('--rig-id')
    parser.add_argument('--gpu-uuid')
    parser.add_argument('--budget-seconds',type=int,default=2400)
    args=parser.parse_args()
    if args.action=='example':
        if args.output.exists():raise FileExistsError(args.output)
        write(args.output,default_recipe(args.scene));print(args.output);return
    if args.recipe is None:parser.error('--recipe required')
    recipe=read(args.recipe)
    frames=None if args.frame is None else [args.frame]
    if args.rig_id is not None and (args.frame is None or args.rig_id not in {r['rig_id'] for r in recipe['rigs']}):
        parser.error('--rig-id requires --frame and a rig from the recipe')
    check=validate(recipe,frames,[args.rig_id] if args.rig_id else None)
    if args.action=='check':
        print(json.dumps(check,indent=2,ensure_ascii=False));return
    if check['status']!='passed':raise ValueError(check['errors'])
    prepared=args.output/'prepared'
    manifest=freeze(recipe,prepared,frames=frames,rig_ids=[args.rig_id] if args.rig_id else None)
    if args.action=='freeze':print(prepared/'rig_list.json');return
    if args.gpu_uuid is None:parser.error('--gpu-uuid required for run')
    command=[sys.executable,str(ROOT/'tools/experiments/run_multirig_roi_a1.py'),
             '--output',str(args.output/'run'),'--gpu-uuid',args.gpu_uuid,
             '--budget-seconds',str(args.budget_seconds),'--rig-list-manifest',str(prepared/'rig_list.json')]
    if args.frame is not None:
        if args.rig_id not in {r['rig_id'] for r in recipe['rigs']}:parser.error('--rig-id must be a recipe rig')
        command+=['--preview','--frame-index',str(args.frame),'--rig-id',args.rig_id]
    elif args.rig_id is not None:parser.error('--rig-id requires --frame')
    subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
