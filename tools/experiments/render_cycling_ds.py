#!/usr/bin/env python3
"""Render two independent cycling bodies in masked, paired DS scene pilots."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
import cv2
import numpy as np
from tools.fisheye.run_ds_scene_pair_pilot import SCENES, load_lens_masks, add_lens_mask_arguments
from densekp_retrain.imaging.ds import process_scene
from densekp_retrain.imaging.output_modes import (prepare_source_workspace,
    preserve_source_recipe, complete_output, record_failure)
from densekp_retrain.camera.double_sphere import DS_20260320
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.body.transforms import world_points
from tools.validate.validate_ds_scene_pair_pilot import validate
from densekp_retrain.camera.double_sphere import build_erp_to_fisheye_lut
from densekp_retrain.runtime import load_appearance_baseline


def layout(assets, origin, yaw_deg, yaw, poses, masks):
    people = []
    for i, (asset, sign) in enumerate(zip(assets, (1, -1))):
        vertices = np.load(asset/'vertices.npy')[0]
        dense = np.load(asset/'densekp_3d.npy')[0]
        choices = []
        for distance in (1.2, 1.5, 1.8, 2.2, 2.6):
            for angle in np.linspace(77,103,27)*sign:
                theta = math.radians(angle)
                local = np.array([distance*math.sin(theta),distance*math.cos(theta),.25])
                person = dict(id=i+1, geometry=str(asset), frame=0,
                              sample_id=asset.parent.name, category='edge_partial',
                              location=(origin+yaw@local).tolist(), yaw_deg=yaw_deg+90)
                points = world_points(dense, vertices, person)
                views = [point_in_view(points,poses[n],n,masks[n])[-1] for n in poses]
                counts = [int(v.sum()) for v in views]
                common = int((views[0]&views[1]).sum())
                if min(counts)>=20 and max(counts)<=115 and common>=8:
                    choices.append((common-1.5*sum(abs(c-90) for c in counts),person))
        if not choices: raise ValueError('No valid paired partial-body layout')
        people.append(max(choices,key=lambda row:row[0])[1])
    rear = dict(people[0], id=3, geometry=str(assets[1]), sample_id=assets[1].parent.name,
                category='rear_overlap')
    relative = np.asarray(people[0]['location'])-origin
    relative[2] = 0
    rear['location'] = (origin+2*relative+yaw@np.array([0,.20,.25])).tolist()
    people.append(rear)
    people.append(dict(id=4, geometry=str(assets[0]), sample_id=assets[0].parent.name,
                       frame=0, category='complete_control', yaw_deg=yaw_deg,
                       location=(origin+yaw@np.array([0,3.8,.25])).tolist()))
    bounds=[]
    for person in people:
        v=np.load(Path(person['geometry'])/'vertices.npy')[0]
        body=world_points(v,v,person); bounds.append((body.min(0),body.max(0)))
    for i,(low,high) in enumerate(bounds):
        for other_low,other_high in bounds[i+1:]:
            if not ((high+.05<other_low)|(other_high+.05<low)).any():
                raise ValueError('Bodies lack 5cm AABB clearance')
    return people


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--converted',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--clothing-root',type=Path,help='Per-sample fitted jersey/shorts assets')
    p.add_argument('--blender-bin',type=Path,default=Path('/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender'))
    p.add_argument('--width',type=int,default=5760)
    p.add_argument('--final-width',type=int,default=5760)
    p.add_argument('--final-height',type=int,default=2880)
    p.add_argument('--output-mode',choices=['compact','full'],default='compact')
    p.add_argument('--temporary-root',type=Path,help='Fresh parent for per-scene compact temporary directories')
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--device',choices=['CPU','CUDA'],default='CUDA')
    add_lens_mask_arguments(p)
    args=p.parse_args()
    if args.width<64 or args.width%2: p.error('ERP width must be positive, even, and at least 64')
    conversion=json.loads((args.converted/'report.json').read_text())
    if len(conversion['cases'])!=2 or not all(c['accepted_for_synthetic_pilot'] for c in conversion['cases']):
        raise ValueError('Expected two accepted independent conversions')
    assets=[(args.converted/c['sample_id']/'geometry').resolve() for c in conversion['cases']]
    lenses,lens_metadata=load_lens_masks(args.fisheye_reference_root,args.reference_frame)
    with np.load(ROOT/'assets_external/real_erp_valid_masks.npz') as z:
        masks={n:z[f'valid_masks_{n}'] for n in lenses}
    layout_masks={}
    for n in lenses:
        x,y=build_erp_to_fisheye_lut(DS_20260320[n],n,args.width,args.width//2)
        layout_masks[n]=cv2.resize(masks[n].astype(np.uint8),(args.width,args.width//2),
                                   interpolation=cv2.INTER_NEAREST).astype(bool)
        layout_masks[n]&=cv2.remap(lenses[n].astype(np.uint8),x,y,cv2.INTER_NEAREST).astype(bool)
    appearance,_=load_appearance_baseline()
    args.output=args.output.resolve(); args.output.mkdir(parents=True,exist_ok=False)
    original_hook=sys.excepthook
    for name,(config_path,origin,yaw_deg) in SCENES.items():
        out=args.output/name; out.mkdir()
        temporary=(args.temporary_root/name if args.temporary_root else None)
        source_dir=prepare_source_workspace(out,args.output_mode,temporary)
        failure_stage={'value':'preflight'}
        def preserve_failure(error_type,error,traceback):
            record_failure(out,args.output_mode,source_dir,failure_stage['value'],error)
            original_hook(error_type,error,traceback)
        sys.excepthook=preserve_failure
        poses,physical,yaw=rig_poses(origin,yaw_deg,.18)
        people=layout(assets,np.asarray(origin),yaw_deg,yaw,poses,layout_masks)
        if args.clothing_root:
            for person in people:
                clothing=(args.clothing_root/person['sample_id']).resolve()
                metadata=json.loads((clothing/'clothing_manifest.json').read_text())
                if Path(metadata['body_geometry']).resolve()!=Path(person['geometry']).resolve():
                    raise ValueError('Clothing and person body differ')
                person['clothing_geometry']=str(clothing)
        rig=dict(camera_model='Double Sphere',calibration_group='20260320',baseline_m=.18,
                 baseline_status='synthetic trial, not measured',
                 intrinsics={n:asdict(DS_20260320[n]) for n in poses},
                 cameras=[dict(id=n,camera_to_world=poses[n].tolist(),
                               physical_ds_camera_to_world=physical[n].tolist()) for n in poses])
        (out/'rig.json').write_text(json.dumps(rig,indent=2)+'\n')
        (out/'people.json').write_text(json.dumps(dict(people=people,sequence=False),indent=2)+'\n')
        scene=json.loads((ROOT/config_path).read_text())
        command=[str(args.blender_bin),'-b','--disable-autoexec','--python-exit-code','1',
                 '--python',str(ROOT/'tools/fisheye/render_equirectangular_source.py'),'--',
                 '--geometry',str(assets[0]),'--people-config',str(out/'people.json'),
                 '--cameras',str(out/'rig.json'),'--scene-blend',str(ROOT/scene['blend_path']),
                 '--output-dir',str(source_dir),'--width',str(args.width),'--height',str(args.width//2),
                 '--samples',str(args.samples),'--seed',str(args.seed),'--device',args.device,
                 '--body-texture',appearance['body_diffuse_texture_path'],'--body-uv',appearance['body_uv_path']]
        (out/'render_command.json').write_text(json.dumps(command,indent=2)+'\n')
        print('Rendering '+name,flush=True)
        failure_stage['value']='render'
        with (out/'render.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        source=json.loads((source_dir/'source_manifest.json').read_text())
        if source['people']!=people: raise ValueError('Rendered people mismatch')
        for camera in source['sources']:
            np.testing.assert_allclose(camera['camera_to_world'],poses[camera['id']],atol=1e-12,rtol=0)
        failure_stage['value']='postprocess'
        report=process_scene(out,people,poses,masks,None,None,lens_masks=lenses,
            lens_metadata=lens_metadata,output_size=(args.final_width,args.final_height),
            output_mode=args.output_mode,source_dir=source_dir)
        report.update(scene=name,people=people,conversion_report=str((args.converted/'report.json').resolve()),
                      sequence=False,calibration_group='20260320',baseline_m=.18,
                      valid_mask_source=str((ROOT/'assets_external/real_erp_valid_masks.npz').resolve()),
                      regressor_source=str((ROOT/'assets/densekp_regressor_R.npy').resolve()),
                      ground_clearance_m=.25,
                      appearance=('per-body fitted cycling jersey and shorts' if args.clothing_root
                                  else 'shared body texture; no motion-specific clothing'),
                      limitation='two estimated poses repeated in two scenes; no recovered fingers or real 3D GT',
                      visibility_rule='source ERP instance plus 4cm ray-range tolerance in 5x5 neighborhood',
                      source_resolution=[args.width,args.width//2],
                      final_erp_resolution=[args.final_width,args.final_height],output_mode=args.output_mode)
        (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        failure_stage['value']='validation'
        checked=validate(out,source_dir=source_dir,output_mode=args.output_mode)
        preserve_source_recipe(out,source_dir,render_command=command,
            source_buffers_retained=args.output_mode=='full',extra=dict(
                output_mode=args.output_mode,final_erp_resolution=[args.final_width,args.final_height],
                ds_resolution=[2880,2880],seed=args.seed,samples=args.samples))
        failure_stage['value']='cleanup'
        complete_output(out,args.output_mode,source_dir,checks=checked)
        print(json.dumps(checked),flush=True)
    sys.excepthook=original_hook


if __name__=='__main__': main()
