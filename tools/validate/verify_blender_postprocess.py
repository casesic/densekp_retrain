#!/usr/bin/env python3
"""Connect finite Blender acceptance renders to existing DS and Omni label paths."""
import argparse,json,shutil,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
import numpy as np
from PIL import Image
from densekp_retrain.imaging.ds import process_scene
from densekp_retrain.imaging.omni import warp_case
from densekp_retrain.camera import OmniRadtanIntrinsics
from densekp_retrain.body.transforms import world_points
from densekp.core.keypoint_supervision import make_targets,SUPERVISION_POLICY


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--run',choices=['before','after'],required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    inputs=a.evidence/'inputs'
    for route in ('native','linked'):
        dest=a.output/route;source=dest/'source';source.mkdir(parents=True)
        for file in (a.evidence/f'{a.run}_{route}_render/source').iterdir():
            if file.suffix in ('.exr','.png'):shutil.copyfile(file,source/file.name)
    native=a.output/'native';people=json.loads((inputs/'people.json').read_text())['people']
    poses={row['id']:np.asarray(row['camera_to_world']) for row in json.loads((inputs/'rig.json').read_text())['cameras']}
    original=ROOT/'outputs/experiments/source_isolated_pilot_20260930/layouts/val_000'
    with np.load(ROOT/'assets_external/real_erp_valid_masks.npz') as z:masks={n:z[f'valid_masks_{n}'] for n in poses}
    lenses={n:np.asarray(Image.open(original/n/'reference_lens_mask.png'))>0 for n in poses}
    report=process_scene(native,people,poses,masks,None,None,lens_masks=lenses,lens_metadata={n:dict(source=str(original/n/'reference_lens_mask.png')) for n in poses})
    (native/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    targets={};groups=[]
    for name in poses:
        with np.load(native/name/'annotations.npz') as z:
            for row in json.loads((native/name/'boxes.json').read_text()):
                if row['bbox_xyxy_exclusive'] is None:continue
                i=z['person_ids'].tolist().index(row['person_id']);x1,y1,x2,y2=row['bbox_xyxy_exclusive'];center=[(x1+x2)/2,(y1+y2)/2];scale=max(x2-x1,y2-y1)*1.2/200
                target=make_targets(z['xy'][i],z['in_view'][i],center,scale)
                targets[f"{name}_{row['person_id']}"]=target
                groups.append(dict(camera=name,person_id=row['person_id'],supervised=int(target[:,2].sum()),supervised_occluded=int(((target[:,2]>0)&~z['visible'][i]).sum())))
    np.savez(native/'supervision.npz',**targets)
    (native/'supervision.json').write_text(json.dumps(dict(policy=SUPERVISION_POLICY,groups=groups),indent=2)+'\n')
    linked=a.output/'linked';geometry=ROOT/'outputs/baselines/temporal_case_aaron_1095/geometry'
    body=np.load(geometry/'vertices.npy',mmap_mode='r')[0];dense=np.load(geometry/'densekp_3d.npy',mmap_mode='r')[0]
    points=world_points(dense,body,dict(location=[0,0,0],yaw_deg=0))
    item=json.loads((inputs/'linked_cameras.json').read_text())['cameras'][0]
    warp_case(linked/'source'/f"{item['id']}.exr",linked/'fisheye',points,np.asarray(item['camera_to_world']),OmniRadtanIntrinsics())
    # Historical linked single-person object mask includes body ID 1 and clothes ID 2.
    mask=np.asarray(Image.open(linked/'fisheye/mask.png'))>0
    yy,xx=np.where(mask);bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None
    (linked/'mask_box_check.json').write_text(json.dumps(dict(bbox_xyxy_exclusive=bbox,pixels=int(mask.sum()),scope='acceptance-derived box, not a newly added production output'),indent=2)+'\n')

if __name__=='__main__':main()
