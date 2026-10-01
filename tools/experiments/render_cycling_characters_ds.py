#!/usr/bin/env python3
"""Two bounded layouts covering eight independent character/pose derivatives."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from tools.experiments.render_cycling_ds import layout
from tools.experiments.prepare_cycling_characters import POSES, write_json
from tools.fisheye.run_ds_scene_pair_pilot import (
    SCENES, DS_20260320, rig_poses, load_lens_masks, process_scene, add_lens_mask_arguments)
from tools.fisheye.run_fisheye_multiperson_pilot import world_points
from tools.validate.validate_ds_scene_pair_pilot import validate
from densekp_retrain.camera.double_sphere import build_erp_to_fisheye_lut


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--assets',type=Path,required=True)
    p.add_argument('--review-acceptance',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--blender-bin',type=Path,default=Path('/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender'))
    p.add_argument('--width',type=int,default=1536)
    p.add_argument('--samples',type=int,default=32)
    p.add_argument('--seed',type=int,default=29)
    p.add_argument('--device',choices=['CPU','CUDA'],default='CUDA')
    add_lens_mask_arguments(p); a=p.parse_args(); start=time.monotonic()
    if a.width<64 or a.width%2: raise ValueError('ERP width must be even and at least 64')
    acceptance=json.loads(a.review_acceptance.read_text())
    if acceptance['status']!='passed' or Path(acceptance['asset_root']).resolve()!=a.assets.resolve():
        raise ValueError('Preview acceptance must pass for these exact assets before DS rendering')
    assets=json.loads((a.assets/'report.json').read_text())
    characters=[x['character_id'] for x in assets['characters']]
    if len(characters)!=4 or len(assets['derivatives'])!=8: raise ValueError('Expected four identities and eight derivatives')
    for row in assets['characters']:
        manifest=json.loads(Path(row['manifest']).read_text())
        if manifest['status']!='frozen_after_review': raise ValueError('Unfrozen character')
        if manifest['appearance']!=acceptance['appearances'][row['character_id']]:
            raise ValueError('Appearance changed since preview acceptance')
    lenses,lens_metadata=load_lens_masks(a.fisheye_reference_root,a.reference_frame)
    with np.load(ROOT/'assets_external/real_erp_valid_masks.npz') as z:
        masks={n:z[f'valid_masks_{n}'] for n in lenses}
    layout_masks={}
    for n in lenses:
        x,y=build_erp_to_fisheye_lut(DS_20260320[n],n,a.width,a.width//2)
        layout_masks[n]=cv2.resize(masks[n].astype(np.uint8),(a.width,a.width//2),interpolation=cv2.INTER_NEAREST).astype(bool)
        layout_masks[n]&=cv2.remap(lenses[n].astype(np.uint8),x,y,cv2.INTER_NEAREST).astype(bool)
    a.output=a.output.resolve(); a.output.mkdir(parents=True,exist_ok=False)
    validations=[]; coverage=[]
    for scene_index,(name,(config_path,origin,yaw_deg)) in enumerate(SCENES.items()):
        out=a.output/name; out.mkdir()
        pose_ids=[POSES[scene_index],POSES[scene_index],POSES[1-scene_index],POSES[1-scene_index]]
        selected=[next(d for d in assets['derivatives'] if d['character_id']==cid and d['pose_source_id']==pid)
                  for cid,pid in zip(characters,pose_ids)]
        geometries=[Path(d['geometry']) for d in selected]
        poses,physical,yaw=rig_poses(origin,yaw_deg,.18)
        people=layout(geometries[:2],np.asarray(origin),yaw_deg,yaw,poses,layout_masks)
        bounds=[]
        for person,record in zip(people,selected):
            person.update(geometry=record['geometry'],clothing_geometry=record['clothing_geometry'],
                          character_id=record['character_id'],pose_source_id=record['pose_source_id'],
                          sample_id=f"{record['character_id']}__{record['pose_source_id']}",
                          appearance=record['appearance'],split='integration_only')
            v=np.load(Path(person['geometry'])/'vertices.npy')[0]
            c=np.load(Path(person['clothing_geometry'])/'clothing_vertices.npy')[0]
            world=world_points(np.concatenate((v,c)),v,person)
            bounds.append((world.min(0),world.max(0)))
        for i,(low,high) in enumerate(bounds):
            for other_low,other_high in bounds[i+1:]:
                if not ((high+.05<other_low)|(other_high+.05<low)).any():
                    raise ValueError('New character clothing/body lacks existing 5cm AABB clearance')
        rig=dict(camera_model='Double Sphere',calibration_group='20260320',baseline_m=.18,
                 baseline_status='synthetic trial, not measured',split='integration_only',
                 intrinsics={n:asdict(DS_20260320[n]) for n in poses},
                 cameras=[dict(id=n,camera_to_world=poses[n].tolist(),physical_ds_camera_to_world=physical[n].tolist()) for n in poses])
        write_json(out/'rig.json',rig); write_json(out/'people.json',dict(people=people,sequence=False,split='integration_only'))
        scene=json.loads((ROOT/config_path).read_text())
        command=[str(a.blender_bin),'-b','-t','8','--disable-autoexec','--python-exit-code','1',
                 '--python',str(ROOT/'tools/fisheye/render_equirectangular_source.py'),'--',
                 '--geometry',str(geometries[0]),'--people-config',str(out/'people.json'),
                 '--cameras',str(out/'rig.json'),'--scene-blend',str(ROOT/scene['blend_path']),
                 '--output-dir',str(out/'source'),'--width',str(a.width),'--height',str(a.width//2),
                 '--samples',str(a.samples),'--seed',str(a.seed),'--device',a.device]
        write_json(out/'render_command.json',command)
        render_start=time.monotonic()
        with (out/'render.log').open('w') as log:
            subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        source=json.loads((out/'source/source_manifest.json').read_text())
        if source['people']!=people: raise ValueError('Rendered identity differs from labels')
        for camera in source['sources']:
            np.testing.assert_allclose(camera['camera_to_world'],poses[camera['id']],atol=1e-12,rtol=0)
        result=process_scene(out,people,poses,masks,None,None,lens_masks=lenses,lens_metadata=lens_metadata)
        result.update(scene=name,people=people,split='integration_only',sequence=False,
                      render_seconds=time.monotonic()-render_start,baseline_m=.18,calibration_group='20260320',
                      appearance='per-character skin and procedural eyes; fitted common-palette jersey/shorts; albedo hair only',
                      regressor_source=str(ROOT/'assets/densekp_regressor_R.npy'),
                      visibility_rule='fresh source depth/instance, 4cm ray-range tolerance in 5x5 neighborhood',
                      limitation='two estimated photo poses, four synthetic identities; no independent split')
        write_json(out/'report.json',result)
        checked=validate(out); validations.append(checked)
        seen=set()
        for camera in poses:
            boxes=json.loads((out/camera/'boxes.json').read_text())
            seen.update(b['person_id'] for b in boxes if b['bbox_xyxy_exclusive'] is not None)
        if seen!={1,2,3,4}: raise ValueError('A derivative has no actual oracle crop in either camera')
        coverage.extend(dict(character_id=p['character_id'],pose_source_id=p['pose_source_id'],scene=name,person_id=p['id']) for p in people)
        print(json.dumps(checked),flush=True)
    write_json(a.output/'report.json',dict(status='passed',split='integration_only',coverage=coverage,
               scene_validations=validations,seconds=time.monotonic()-start,
               loader_rows=sum(v['loader_rows'] for v in validations),
               supervised_points=sum(v['supervised_points'] for v in validations),
               supervised_occluded_points=sum(v['supervised_occluded_points'] for v in validations)))


if __name__=='__main__': main()
