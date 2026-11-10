"""Freeze a bounded three-rider T-junction episode from accepted city assets."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import random
import shutil
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.data.dynamic_scene import frame_record, geometry_paths
from densekp_retrain.data.multiperson_road_pass import trajectory_separation
from densekp_retrain.data.t_junction import (scene_trajectory, check_trajectory, road_point,
                                              road_supported)

PARENT = ROOT / 'outputs/experiments/cycling_team_relative_layout_a1_20261010/city_attempt5_cpu_only_correction1/prepared'
OUT = ROOT / 'outputs/experiments/t_junction_multirig_a1_20261011'
SCENE_ID = 'city_t_junction_3p_s20261040'


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def candidates(seed, count=12):
    rng = random.Random(f't_junction_trajectory_v1|{seed}')
    values = []
    for k in range(count):
        child = rng.getrandbits(63)
        r = random.Random(child)
        values.append(dict(attempt=k, child_seed=child, turn_radius_m=4.0,
            turn_start_y_m=77.85+r.uniform(-.1,.1),
            p1_start_progress_m=r.uniform(-.08,.12), p1_speed_m_s=r.uniform(2.03,2.12),
            p2_start_x_m=r.uniform(-7.3,-6.8), p2_speed_m_s=r.uniform(1.4,1.65),
            p2_road_y_m=r.uniform(82.65,83.15),
            p3_start_x_m=r.uniform(9.1,9.8), p3_speed_m_s=r.uniform(1.25,1.48),
            p3_road_y_m=r.uniform(85.25,85.75)))
    return values


def world_geometry(recipe, old_matrices, old_world, new_matrices):
    points = np.empty_like(old_world)
    per_person = []
    for j, person in enumerate(recipe['people']):
        asset = json.loads(Path(person['asset_manifest']).read_text())
        paths = geometry_paths(asset)
        indices = person['asset_frames']
        body = np.load(paths['body_vertices'], mmap_mode='r')[indices]
        cloth = np.load(paths['cloth_vertices'], mmap_mode='r')[indices]
        old_R = old_matrices[:, j, :3, :3]
        old_t = old_matrices[:, j, :3, 3]
        local = (old_world[:, j] - old_t[:, None]) @ old_R
        new_R = new_matrices[:, j, :3, :3]
        new_t = new_matrices[:, j, :3, 3]
        points[:, j] = local @ new_R.transpose(0, 2, 1) + new_t[:, None]
        body_world = body @ new_R.transpose(0, 2, 1) + new_t[:, None]
        cloth_world = cloth @ new_R.transpose(0, 2, 1) + new_t[:, None]
        per_person.append((asset, body_world, cloth_world))
    return points, per_person


def check_candidate(recipe, old_matrices, old_world, item):
    records = recipe['episode_timeline']
    lanes = [dict(lane='turn_main_to_positive_x',start=item['p1_start_progress_m'],
                  speed=item['p1_speed_m_s'],lateral=0.,turn_y=item['turn_start_y_m'],
                  radius=item['turn_radius_m']),
             dict(lane='cross_positive_x',start=item['p2_start_x_m'],
                  speed=item['p2_speed_m_s'],lateral=item['p2_road_y_m']-84.),
             dict(lane='cross_negative_x',start=-item['p3_start_x_m'],
                  speed=item['p3_speed_m_s'],lateral=item['p3_road_y_m']-84.)]
    # The negative-X lane uses negative progress so its initial world X is positive.
    matrices = np.stack([scene_trajectory(records, **lane,
                  height=recipe['people'][j]['placement_height_m'],
                  base_rotation=old_matrices[0,j,:3,:3]) for j,lane in enumerate(lanes)],axis=1)
    world, geometry = world_geometry(recipe, old_matrices, old_world, matrices)
    trajectory = [check_trajectory(matrices[:,j]) for j in range(3)]
    bounds = []
    for j,(asset, body, cloth) in enumerate(geometry):
        xy = np.concatenate((body[:,:,:2], cloth[:,:,:2]),axis=1)
        support = np.fromiter((road_supported(float(x),float(y),margin=.04)
                               for x,y in xy.reshape(-1,2)),dtype=bool,count=xy.shape[0]*xy.shape[1])
        if not support.all():
            first = np.flatnonzero(~support)[0]
            frame, vertex = divmod(int(first),xy.shape[1])
            raise ValueError(f'road_union_outside:person={j+1}:frame={frame}:xy={xy[frame,vertex].tolist()}')
        roots = matrices[:,j,:3,3]
        with np.load(asset['parameters']) as params:
            orient = params['global_orient'][recipe['people'][j]['asset_frames']]
        body_forward = np.stack([matrices[k,j,:3,:3] @ cv2.Rodrigues(o)[0] @ [0.,0.,1.]
                                 for k,o in enumerate(orient)])[:,:2]
        steps = np.gradient(roots[:,:2],axis=0)
        angle = np.rad2deg(np.arccos(np.clip(np.sum(body_forward*steps,axis=1)/
            (np.linalg.norm(body_forward,axis=1)*np.linalg.norm(steps,axis=1)), -1,1)))
        if angle.max()>25:
            raise ValueError(f'forward_tangent_misaligned:person={j+1}:max_deg={angle.max():.2f}')
        if np.min(body[:,:,2]) < .05 or np.max(body.min(axis=1)[:,2]) > .7:
            raise ValueError(f'fixed_height_conflict:person={j+1}')
        bounds.append(np.concatenate((xy.min(axis=1),xy.max(axis=1)),axis=1))
        trajectory[j]['maximum_body_forward_tangent_angle_deg']=float(angle.max())
    bounds = np.stack(bounds,axis=1)
    separation = trajectory_separation(bounds,matrices[:,:,:3,3],clearance=.1,min_root_distance=.9)
    return dict(matrices=matrices,world=world,trajectory=trajectory,
                pairwise_separation=separation,lanes=lanes)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['freeze','prepare'])
    parser.add_argument('--seed',type=int,default=20261040)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    out=args.output_dir.resolve()
    if args.command=='freeze':
        out.mkdir(parents=True,exist_ok=True)
        write(out/'frozen_trajectory_candidates.json',dict(schema='t_junction_trajectory_v1',
            master_seed=args.seed,source_prepared=str(PARENT),scene_episode_id=SCENE_ID,
            junction_endpoint='positive_y',road_union=dict(main_x=[-3.5,3.5],main_y=[-80,80],
                cross_x=[-80,80],cross_y=[80,88],height_m=0),
            selection='first_all_hard_geometry_passed_no_render_search',candidates=candidates(args.seed)))
        return
    started=time.monotonic()
    frozen=json.loads((out/'frozen_trajectory_candidates.json').read_text())
    parent=json.loads((PARENT/'recipe.json').read_text())
    old_matrices=np.load(PARENT/'scene_matrices.npy')
    old_world=np.load(PARENT/'world_densekp.npy')
    checks=[];selected=None
    for item in frozen['candidates']:
        try:
            value=check_candidate(parent,old_matrices,old_world,item)
            checks.append(dict(attempt=item['attempt'],status='passed',
                               trajectory=value['trajectory'],pairwise_separation=value['pairwise_separation']))
            selected=(item,value)
            break
        except ValueError as e:
            checks.append(dict(attempt=item['attempt'],status='rejected',reason=str(e)))
    write(out/'trajectory_checks.json',dict(attempts=checks,elapsed_seconds=time.monotonic()-started))
    if selected is None:
        print('NO_TRAJECTORY');return
    item,value=selected
    destination=out/'prepared_scene'
    destination.mkdir(exist_ok=False)
    for name in ('rig.json','episode_timeline.json','person1_timeline.json','person2_timeline.json','person3_timeline.json',
                 'person1_source_timeline.json','person2_source_timeline.json','person3_source_timeline.json'):
        shutil.copy2(PARENT/name,destination/name)
    shutil.copytree(PARENT/'calibration',destination/'calibration')
    np.save(destination/'scene_matrices.npy',value['matrices'])
    np.save(destination/'world_densekp.npy',value['world'])
    recipe=copy.deepcopy(parent)
    recipe.pop('road_bounds_xyxy',None)
    recipe.update(episode_id=SCENE_ID,sequence_id=SCENE_ID,scene_episode_id=SCENE_ID,
                  scene_mode='t_junction_three_cyclists',trajectory_schema='t_junction_trajectory_v1',
                  selected_trajectory_candidate=item,source_prepared_ref=str(PARENT),
                  scene_matrices_ref=str((destination/'scene_matrices.npy').resolve()),
                  road_regions_xyxy=[[-3.5,-80.,3.5,80.],[-80.,80.,80.,88.]],
                  road_support_policy='actual_asphalt_union_all_body_cloth_vertices_31_samples_margin_0p04m',
                  integration_only=True)
    for j,person in enumerate(recipe['people']):
        person['scene_from_model']=value['matrices'][:,j].tolist()
        person.pop('scene_yaw_deg',None)
        person['road_start_m']=value['matrices'][0,j,:3,3].tolist()
        person['speed_m_s']=value['lanes'][j]['speed']
        if j==0:
            person.pop('path_direction_xy',None)
            lane=value['lanes'][0]
            person['path_tangent_xy_by_frame']=[road_point(
                lane['start']+lane['speed']*k/10.,turn_y=lane['turn_y'],radius=lane['radius'])[1].tolist()
                for k in range(31)]
        else:
            person['path_direction_xy']=[1,0] if j==1 else [-1,0]
        person['support_checks']['road_object_names']=sorted(set(
            person['support_checks']['road_object_names']+['Cross_street.001']))
    for k in range(31):frame_record(recipe,k)
    write(destination/'recipe.json',recipe)
    write(destination/'geometry_preflight.json',dict(status='passed',stage='road_trajectory_only',
          rig_preflight_required=True,selected_attempt=item['attempt'],
          trajectories=value['trajectory'],pairwise_separation=value['pairwise_separation'],
          limitation='31 sampled full body+cloth vertices and XY AABB person separation, not continuous triangle collision'))
    print('SELECTED',item['attempt'])


if __name__=='__main__':main()
