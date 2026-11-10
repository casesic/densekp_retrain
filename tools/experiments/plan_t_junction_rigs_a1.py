"""Freeze, screen, and precisely preflight three paired city-junction rigs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.camera.domain import point_in_view
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.data.viewpoint import classify_view, view_angle
from tools.experiments.prepare_urban_three_multirig_a1 import preflight as exact_preflight

OUT=ROOT/'outputs/experiments/t_junction_multirig_a1_20261011'
CONFIG=ROOT/'configs/t_junction_multirig_a1_20261011.json'
SOURCE_OVERRIDE=None


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as file:
        json.dump(value,file,indent=2,allow_nan=False,ensure_ascii=False);file.write('\n')


def longest(flags):
    n=maximum=0
    for flag in flags:
        n=n+1 if flag else 0;maximum=max(n,maximum)
    return maximum


def candidates(config,world,view):
    spec=config[view]
    cells=[(x,y,o) for x in spec['x_m'] for y in spec['y_m'] for o in spec['yaw_offsets_deg']]
    if len(cells)!=config['candidate_limit_per_rig'] or len(set(cells))!=len(cells):
        raise ValueError('invalid_frozen_camera_cells')
    rng=random.Random(f't_junction_camera_v1|{config["camera_master_seed"]}|{view}')
    rng.shuffle(cells)
    target=np.asarray(world[15,0].mean(axis=0))
    rows=[]
    for number,(x,y,offset) in enumerate(cells):
        seed=rng.getrandbits(63)
        local=random.Random(seed)
        xx=x+local.uniform(-config['position_jitter_m'],config['position_jitter_m'])
        yy=y+local.uniform(-config['position_jitter_m'],config['position_jitter_m'])
        bearing=math.degrees(math.atan2(target[1]-yy,target[0]-xx))
        rows.append(dict(attempt=number,camera_child_seed=seed,origin_m=[xx,yy,0.],
                         yaw_deg=bearing+offset+local.uniform(-config['yaw_jitter_deg'],config['yaw_jitter_deg']),
                         nominal_cell=[x,y,offset],construction='roadside_cell_to_target_midpoint_bearing'))
    return rows


def freeze():
    config=json.loads(CONFIG.read_text())
    source=SOURCE_OVERRIDE or ROOT/config['source_prepared']
    world=np.load(source/'world_densekp.npy')
    if world.shape!=(31,3,138,3):raise ValueError('source_world_axis')
    for view in ('back','front','side'):
        write(OUT/f'frozen_{view}_candidates.json',dict(schema=config['schema'],
              scene_episode_id=config['scene_episode_id'],source_prepared=str(source.resolve()),
              camera_master_seed=config['camera_master_seed'],view_goal=view,
              candidates=candidates(config,world,view),
              selection='light_view_common_then_worst_camera_R138_proxy_height_then_original_attempt; exact_top_3',
              accepted_front_cam4_native_median_reference_px=438,
              source_grouping='all_rigs_same_three_people_same_original_motions'))


def headings(source):
    recipe=json.loads((source/'recipe.json').read_text())
    M=np.load(source/'scene_matrices.npy')
    asset=json.loads(Path(recipe['people'][0]['asset_manifest']).read_text())
    with np.load(asset['parameters']) as params:
        orient=params['global_orient'][recipe['people'][0]['asset_frames']]
    return np.stack([M[k,0,:3,:3]@cv2.Rodrigues(orient[k])[0]@[0.,0.,1.] for k in range(31)])


def light_screen():
    started=time.monotonic()
    config=json.loads(CONFIG.read_text())
    source=SOURCE_OVERRIDE or ROOT/config['source_prepared']
    recipe=json.loads((source/'recipe.json').read_text())
    world=np.load(source/'world_densekp.npy')
    M=np.load(source/'scene_matrices.npy')
    forward=headings(source)
    masks={c:np.asarray(Image.open(source/'calibration'/f'{c}_valid_mask.png'))>0 for c in ('cam2','cam4')}
    for view in ('back','front','side'):
        frozen=json.loads((OUT/f'frozen_{view}_candidates.json').read_text())
        if Path(frozen['source_prepared']).resolve()!=source.resolve():
            raise ValueError('frozen_camera_source_conflict')
        rows=[]
        for candidate in frozen['candidates']:
            poses,_,_=rig_poses(candidate['origin_m'],candidate['yaw_deg'],recipe['rig']['baseline_m'])
            frames=[]
            for k in range(31):
                per={}
                inside={}
                for camera,pose in poses.items():
                    xy,_,_,_,valid=point_in_view(world[k,0],pose,camera,masks[camera])
                    inside[camera]=valid
                    angle=view_angle(M[k,0,:3,3],forward[k],pose[:3,3])
                    height=float(np.ptp(xy[valid,1])) if valid.sum()>=8 else 0.
                    per[camera]=dict(angle_deg=float(angle),view_class=classify_view(angle),
                       in_domain=int(valid.sum()),out_domain=int((~valid).sum()),
                       projected_R138_height_px=height)
                common=int((inside['cam2']&inside['cam4']).sum())
                both=common>=config['minimum_common_points'] and all(per[c]['view_class']==view for c in per)
                edge=common>=8 and all(per[c]['in_domain']>=8 and per[c]['out_domain']>=5 for c in per)
                frames.append(dict(frame_index=k,cameras=per,common_physical_points=common,
                                   matching_view_and_common=both,dual_edge=edge))
            view_flags=[r['matching_view_and_common'] for r in frames]
            edge_flags=[r['dual_edge'] for r in frames]
            matching=[r for r in frames if r['matching_view_and_common']]
            height=min(np.median([r['cameras'][c]['projected_R138_height_px'] for r in matching])
                       for c in ('cam2','cam4')) if matching else 0.
            passed=(longest(view_flags)>=config['minimum_window_frames'] and
                    height>=config['minimum_target_proxy_height_px'][view] and
                    (view!='back' or longest(edge_flags)>=config['minimum_window_frames']))
            rows.append(dict(**candidate,passed=bool(passed),view_window=longest(view_flags),
               edge_window=longest(edge_flags),edge_count=sum(edge_flags),
               common_count=sum(r['common_physical_points']>=8 for r in frames),
               smaller_camera_R138_height_median_px=float(height),frames=frames))
        ranked=sorted((row for row in rows if row['passed']),
                      key=lambda r:(-r['view_window'],-r['smaller_camera_R138_height_median_px'],r['attempt']))
        write(OUT/f'light_{view}_checks.json',dict(view=view,attempt_count=len(rows),passed_count=len(ranked),
              ranked_original_attempts=[r['attempt'] for r in ranked],
              elapsed_seconds=time.monotonic()-started,attempts=rows))
        print(view,'light_pass',len(ranked),'best',ranked[0]['attempt'] if ranked else None,flush=True)


def exact():
    config=json.loads(CONFIG.read_text())
    source=SOURCE_OVERRIDE or ROOT/config['source_prepared']
    for view in ('back','front','side'):
        frozen=json.loads((OUT/f'frozen_{view}_candidates.json').read_text())
        if Path(frozen['source_prepared']).resolve()!=source.resolve():
            raise ValueError('frozen_camera_source_conflict')
        light=json.loads((OUT/f'light_{view}_checks.json').read_text())
        best={r['attempt']:r for r in light['attempts']}
        ranked=[best[n] for n in light['ranked_original_attempts']]
        selected=ranked[:config['exact_support_limit_per_rig']]
        if not selected:continue
        exact_config=dict(schema=config['schema'],source_prepared=str(source),
            scene_episode_id=config['scene_episode_id'],view_goal=view,rig_id=f'rig_{view}',
            coverage_goal=f'{view}_junction_view_control',minimum_common_points=8,
            minimum_common_physical_frames=6,minimum_view_common_run=6,
            minimum_view_common_frames=6,minimum_dual_edge_frames=6 if view=='back' else 0,
            minimum_dual_edge_run=6 if view=='back' else 0,
            require_road_cross_section=False,minimum_camera_to_person_root_m=.5,
            minimum_camera_road_x_abs_m=None,maximum_union_roi_pixels=7_000_000,
            maximum_exact_support_candidates=3)
        write(OUT/f'exact_{view}_config.json',exact_config)
        write(OUT/f'exact_{view}_frozen.json',dict(schema=config['schema'],
              source_prepared=str(source),scene_episode_id=config['scene_episode_id'],
              candidates=[{k:r[k] for k in ('attempt','camera_child_seed','origin_m','yaw_deg')}
                          for r in selected]))
        value=exact_preflight(OUT/f'exact_{view}_config.json',OUT/f'exact_{view}_frozen.json',
                              OUT/f'prepared_{view}',time_limit_seconds=900)
        print(view,'exact',value is not None,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['freeze','light','exact'])
    parser.add_argument('--output-dir',type=Path,default=OUT)
    parser.add_argument('--source-prepared',type=Path)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    SOURCE_OVERRIDE=args.source_prepared.resolve() if args.source_prepared else None
    {'freeze':freeze,'light':light_screen,'exact':exact}[args.command]()
