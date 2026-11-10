"""Versioned editor recipe and adapter for the existing multirig ROI producer."""
from __future__ import annotations

import copy
import json
import math
import random
import re
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.camera.rig import rig_poses
from densekp_retrain.data.dynamic_scene import frame_record, checked_matrices

VERSION = 'scene_editor_recipe_v1'
OUTPUT = ROOT / 'outputs/experiments/scene_editor_a3_20261013'
SOURCES = {
    'city': {
        'base': ROOT / 'outputs/experiments/t_junction_multirig_a1_20261011/prepared_scene',
        'rigs': {
            'rig_back': ROOT / 'outputs/experiments/t_junction_local_rig_delivery_a1_20261011/prepared_prior_back_88',
            'rig_front': ROOT / 'outputs/experiments/t_junction_local_rig_delivery_a1_20261011/prepared_prior_front_5',
            'rig_side': ROOT / 'outputs/experiments/t_junction_local_rig_delivery_a1_20261011/prepared_prior_side_27',
        },
    },
    'riverfront': {
        'base': ROOT / 'outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront',
        'rigs': {'rig_edge': ROOT / 'outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront'},
    },
}


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def map_data(scene):
    if scene == 'city':
        return dict(schema='scene_editor_map_v1', version='urban_street_assets_road_mesh_20261013',
                    scene='city', bounds=[-90,-90,90,95], z_m=0, units='metres_world_XY_Z_up',
                    roads=[[-3.5,-80,3.5,80],[-80,80,80,88]],
                    obstacles=[], water=[],
                    provenance='Conservative main and positive-Y cross asphalt rectangles from t_junction.road_supported; buildings and other obstacles remain unknown')
    if scene == 'riverfront':
        return dict(schema='scene_editor_map_v1', version='riverfront_promendade_spec_20261013',
                    scene='riverfront', bounds=[-22,-22,22,22], z_m=0, units='metres_world_XY_Z_up',
                    roads=[[-2.5,-20,2.5,20]], water=[[-22,-22,-3,22]], obstacles=[],
                    provenance='riverfront README: Promenade_ground width 5 m, water negative X; exact occupied boundaries outside verified corridor unknown')
    raise ValueError('unregistered_scene')


def _sample_curve(points, distance):
    """Uniform Catmull-Rom curve sampled densely, then indexed by arc length."""
    p=np.asarray(points,dtype=float)
    if p.ndim!=2 or p.shape[1]!=2 or len(p)<2 or not np.isfinite(p).all():
        raise ValueError('route_requires_two_finite_XY_points')
    if np.any(np.linalg.norm(np.diff(p,axis=0),axis=1)<.01):
        raise ValueError('route_control_points_too_close')
    samples=[]
    for i in range(len(p)-1):
        a=p[max(i-1,0)]; b=p[i]; c=p[i+1]; d=p[min(i+2,len(p)-1)]
        for u in np.linspace(0,1,41,endpoint=False):
            samples.append(.5*((2*b)+(-a+c)*u+(2*a-5*b+4*c-d)*u*u+(-a+3*b-3*c+d)*u*u*u))
    samples.append(p[-1]);s=np.asarray(samples)
    lengths=np.r_[0,np.cumsum(np.linalg.norm(np.diff(s,axis=0),axis=1))]
    if not 0<=distance<=lengths[-1]:raise ValueError(f'route_too_short:{distance:.3f}>{lengths[-1]:.3f}')
    xy=np.array([np.interp(distance,lengths,s[:,axis]) for axis in range(2)])
    lo=max(0,distance-.025);hi=min(lengths[-1],distance+.025)
    tangent=np.array([np.interp(hi,lengths,s[:,axis])-np.interp(lo,lengths,s[:,axis]) for axis in range(2)])
    tangent/=np.linalg.norm(tangent)
    return xy,tangent,float(lengths[-1])


def sample_person(person, frame):
    seconds=frame/10
    distance=person['start_progress_m']+person['speed_m_s']*seconds
    return _sample_curve(person['control_points_xy_m'],distance)


def _initial_route(matrices, person_index):
    """Sketch a new route from the registered source's first and last positions."""
    start=matrices[0,person_index,:2,3].tolist()
    end=matrices[-1,person_index,:2,3].tolist()
    # Keep a small endpoint margin so floating-point sampling at frame 30
    # cannot request a distance just beyond the route's sampled arc length.
    return [start,end],float(np.linalg.norm(np.subtract(end,start))*.995/3.)


def default_recipe(scene='city'):
    if scene not in SOURCES:raise ValueError('unregistered_scene')
    base=SOURCES[scene]['base']; raw=read(base/'recipe.json'); matrices=np.load(base/'scene_matrices.npy')
    count=2 if scene=='city' else 1
    people=[]
    for j in range(count):
        p=raw['people'][j]
        anchors,speed=_initial_route(matrices,j)
        people.append(dict(person_id=j+1,asset_id=p['asset_id'],asset_manifest=p['asset_manifest'],
                           source_frames=p['source_frames'],asset_frames=p['asset_frames'],
                           control_points_xy_m=anchors,speed_m_s=speed,start_progress_m=0.,
                           locked=True,role=p.get('role','rider'),root_mode=p['root_mode']))
    rigs=[]
    for rig_id,src in list(SOURCES[scene]['rigs'].items())[:2]:
        meta=read(src/'recipe.json')['rig']
        rigs.append(dict(rig_id=rig_id,origin_m=meta['origin_m'],yaw_deg=meta['yaw_deg'],
                         baseline_m=.18,calibration_ref='20260320-riding-dataset3'))
    return dict(schema=VERSION,recipe_id=f'{scene}_example',revision=1,parent_revision=None,
                scene=scene,map_version=map_data(scene)['version'],world='XY_metres_Z_up_road_Z0',
                seed=20261013,source_pool='integration_diagnostic',people=people,rigs=rigs,
                frame_count=31,time_base=[1,10],frame_indices=list(range(31)),
                render=dict(samples=64,erp_wh=[5760,2880],roi_only=True,box_reservation='source_policy'),
                source_group='shared_source_diagnostic',purpose='ordinary_multiview',
                base_prepared=str(base),algorithm='catmull_dense_arclength_v1')


def additional_person(scene, person_id):
    """Offer a distinct, registered source identity with a two-point sketch."""
    if scene not in SOURCES or type(person_id) is not int or person_id<1:
        raise ValueError('invalid_person_request')
    base=SOURCES[scene]['base'];raw=read(base/'recipe.json')
    if person_id>len(raw['people']):raise ValueError('registered_asset_pool_insufficient')
    p=raw['people'][person_id-1]
    anchors,speed=_initial_route(np.load(base/'scene_matrices.npy'),person_id-1)
    return dict(person_id=person_id,asset_id=p['asset_id'],asset_manifest=p['asset_manifest'],
                source_frames=p['source_frames'],asset_frames=p['asset_frames'],
                control_points_xy_m=anchors,speed_m_s=speed,start_progress_m=0.,
                locked=True,role=p.get('role','rider'),root_mode=p['root_mode'])


def validate(recipe, frames=None, rig_ids=None):
    if recipe.get('schema')!=VERSION or recipe.get('scene') not in SOURCES:raise ValueError('invalid_editor_recipe')
    if not isinstance(recipe.get('recipe_id'),str) or re.fullmatch(r'[a-z0-9_]{1,64}',recipe['recipe_id']) is None:
        raise ValueError('invalid_recipe_id')
    if type(recipe.get('revision')) is not int or not 1<=recipe['revision']<=9999:
        raise ValueError('invalid_revision')
    if recipe.get('purpose')!='ordinary_multiview' or recipe.get('world')!='XY_metres_Z_up_road_Z0':
        raise ValueError('unsupported_purpose_or_world')
    if recipe.get('map_version')!=map_data(recipe['scene'])['version']:raise ValueError('map_version_changed')
    if recipe.get('frame_count')!=31 or recipe.get('time_base')!=[1,10]:raise ValueError('unsupported_timebase')
    if recipe.get('frame_indices')!=list(range(31)):raise ValueError('unsupported_frame_indices')
    if recipe.get('render')!=dict(samples=64,erp_wh=[5760,2880],roi_only=True,box_reservation='source_policy'):
        raise ValueError('unsupported_render_contract')
    if recipe.get('source_pool')!='integration_diagnostic' or recipe.get('source_group')!='shared_source_diagnostic':
        raise ValueError('unsupported_source_pool')
    if recipe.get('algorithm')!='catmull_dense_arclength_v1':raise ValueError('unsupported_trajectory_algorithm')
    if not 1<=len(recipe['people'])<=5 or not 1<=len(recipe['rigs'])<=3:raise ValueError('person_or_rig_count')
    if [p['person_id'] for p in recipe['people']]!=list(range(1,len(recipe['people'])+1)):raise ValueError('unstable_person_id')
    if len({p['asset_id'] for p in recipe['people']})!=len(recipe['people']):raise ValueError('reused_identity')
    if len({r['rig_id'] for r in recipe['rigs']})!=len(recipe['rigs']):raise ValueError('duplicate_rig')
    base=Path(recipe['base_prepared']).resolve()
    if base!=SOURCES[recipe['scene']]['base'].resolve():raise ValueError('unregistered_source')
    raw=read(base/'recipe.json')
    if len(recipe['people'])>len(raw['people']):raise ValueError('asset_pool_insufficient_for_count')
    for j,p in enumerate(recipe['people']):
        if p['asset_id']!=raw['people'][j]['asset_id'] or p['asset_manifest']!=raw['people'][j]['asset_manifest']:
            raise ValueError(f'asset_not_frozen_or_unavailable:p{j+1}')
        if p['asset_frames']!=raw['people'][j]['asset_frames']:raise ValueError('source_time_window_changed')
        if p['source_frames']!=raw['people'][j]['source_frames'] or p['root_mode']!=raw['people'][j]['root_mode']:
            raise ValueError('source_clock_or_root_mode_changed')
    if any(not isinstance(r['rig_id'],str) or
           re.fullmatch(r'rig_[a-z0-9_]+',r['rig_id']) is None or
           r.get('source_rig_id',r['rig_id']) not in SOURCES[recipe['scene']]['rigs'] or
           r['baseline_m']!=.18 or r['calibration_ref']!='20260320-riding-dataset3'
           for r in recipe['rigs']):
        raise ValueError('unregistered_rig_or_calibration')
    frames=list(range(31)) if frames is None else frames
    if not isinstance(frames,list) or not frames or len(frames)>31 or len(set(map(str,frames)))!=len(frames) or any(type(k) is not int or not 0<=k<31 for k in frames):
        raise ValueError('invalid_frame_subset')
    if rig_ids is not None:
        if not isinstance(rig_ids,list) or not rig_ids or any(r not in {v['rig_id'] for v in recipe['rigs']} for r in rig_ids):
            raise ValueError('invalid_rig_subset')
    checked_rigs=[r for r in recipe['rigs'] if rig_ids is None or r['rig_id'] in rig_ids]
    roads=map_data(recipe['scene'])['roads'];errors=[];warnings=[];states={};headings={}
    for p in recipe['people']:
        if not .1<=p['speed_m_s']<=4.:errors.append(f"p{p['person_id']}:speed_outside_0.1_to_4_m_s")
        states[p['person_id']]=[]
        for k in frames:
            try:
                xy,tangent,length=sample_person(p,k)
                states[p['person_id']].append((xy,tangent))
                headings[p['person_id'],k]=math.atan2(tangent[1],tangent[0])
                if not any(a<=xy[0]<=c and b<=xy[1]<=d for a,b,c,d in roads):
                    errors.append(f"p{p['person_id']}:frame{k}:outside_verified_road")
            except ValueError as exc:errors.append(f"p{p['person_id']}:frame{k}:{exc}")
    full_sequence=set(frames)==set(range(31))
    turn_frames=range(1,31) if full_sequence else sorted({k for f in frames for k in range(max(1,f-1),min(30,f+1)+1)})
    for p in recipe['people']:
        pid=p['person_id']
        for k in turn_frames:
            try:
                for tick in (k-1,k):
                    if (pid,tick) not in headings:
                        tangent=sample_person(p,tick)[1]
                        headings[pid,tick]=math.atan2(tangent[1],tangent[0])
                delta=headings[pid,k]-headings[pid,k-1]
                turn=abs(math.degrees(math.atan2(math.sin(delta),math.cos(delta))))
                if turn>60:
                    message=f'p{pid}:frame{k}:heading_jump_{turn:.1f}deg_in_0.1s'
                    (errors if full_sequence else warnings).append(message)
            except ValueError:
                # The selected frame's own route error is already reported above.
                pass
    for idx,k in enumerate(frames):
        for a in range(len(recipe['people'])):
            for b in range(a+1,len(recipe['people'])):
                x=states[a+1];y=states[b+1]
                if len(x)>idx and len(y)>idx and np.linalg.norm(x[idx][0]-y[idx][0])<.9:
                    errors.append(f'p{a+1}_p{b+1}:frame{k}:root_clearance_below_0.9m')
    for rig in checked_rigs:
        v=np.asarray(rig['origin_m'],float)
        if v.shape!=(3,) or not np.isfinite(v).all() or abs(v[2])>.001 or not math.isfinite(rig['yaw_deg']):
            errors.append(rig['rig_id']+':invalid_pose')
        else:
            for k in frames:
                for p in recipe['people']:
                    try:xy=sample_person(p,k)[0]
                    except ValueError:continue
                    if np.linalg.norm(xy-v[:2])<.5:errors.append(f"{rig['rig_id']}:p{p['person_id']}:frame{k}:camera_clearance")
    return dict(status='passed' if not errors else 'rejected',errors=errors[:100],warnings=warnings[:100],
                checked_frames=frames,qualification='CPU root and verified road polygons; exact body/ROI follows freeze')


def _matrices_world(recipe, frames=None):
    base=Path(recipe['base_prepared']);old=np.load(base/'scene_matrices.npy')[:,:len(recipe['people'])]
    old_world=np.load(base/'world_densekp.npy')[:,:len(recipe['people'])]
    selected=list(range(31)) if frames is None else frames
    result=np.array(old,copy=True)
    for j,p in enumerate(recipe['people']):
        o=old[:,j];old_roots=o[:,:2,3]
        for k in selected:
            xy,tangent,_=sample_person(p,k)
            before=old_roots[max(0,k-1)];after=old_roots[min(30,k+1)]
            old_yaw=math.atan2(after[1]-before[1],after[0]-before[0])
            new_yaw=math.atan2(tangent[1],tangent[0]);delta=new_yaw-old_yaw
            c,s=math.cos(delta),math.sin(delta)
            turn=np.array([[c,-s,0],[s,c,0],[0,0,1]])
            result[k,j]=o[k];result[k,j,:3,:3]=turn@o[k,:3,:3]
            result[k,j,:2,3]=xy
        checked_matrices(result[:,j],31)
    world=np.array(old_world,copy=True)
    for j in range(len(recipe['people'])):
        local=(old_world[:,j]-old[:,j,:3,3][:,None,:])@old[:,j,:3,:3]
        for k in selected:
            world[k,j]=local[k]@result[k,j,:3,:3].T+result[k,j,None,:3,3]
    return result,world


def freeze(recipe, output, *, frames=None, rig_ids=None):
    """Create fresh prepared dirs; exact projection/ROI is required before rendering."""
    check=validate(recipe,frames,rig_ids)
    if check['status']!='passed':raise ValueError(check['errors'])
    output=Path(output).resolve()
    if output.exists():raise FileExistsError(output)
    output.mkdir(parents=True)
    base=Path(recipe['base_prepared']);raw=read(base/'recipe.json')
    matrices,world=_matrices_world(recipe,frames)
    manifest=dict(schema='scene_episode_rig_list_v1',scene_episode_id=f"{recipe['recipe_id']}_r{recipe['revision']}",
                  maximum_budget_seconds=2700,rigs=[])
    for rig in (r for r in recipe['rigs'] if rig_ids is None or r['rig_id'] in rig_ids):
        rig_id=rig['rig_id'];source=SOURCES[recipe['scene']]['rigs'][rig.get('source_rig_id',rig_id)]
        child=output/rig_id;child.mkdir()
        for name in ('episode_timeline.json',):shutil.copy2(base/name,child/name)
        for j in range(len(recipe['people'])):
            for suffix in ('timeline.json','source_timeline.json'):
                name=f'person{j+1}_{suffix}';shutil.copy2(base/name,child/name)
        shutil.copytree(source/'calibration',child/'calibration')
        np.save(child/'scene_matrices.npy',matrices);np.save(child/'world_densekp.npy',world)
        prepared=copy.deepcopy(read(source/'recipe.json'))
        prepared['people']=copy.deepcopy(raw['people'][:len(recipe['people'])])
        # Keep each registered rig's ROI and box augmentation policy.  The
        # editor cannot weaken a producer's reserved crop support.
        if isinstance(prepared.get('component_provenance'),list):
            active={p['person_id'] for p in prepared['people']}
            prepared['component_provenance']=[entry for entry in prepared['component_provenance']
                                              if entry['person_id'] in active]
        prepared['scene_matrix_person_ids']=list(range(1,len(recipe['people'])+1))
        prepared.update(episode_id=manifest['scene_episode_id'],sequence_id=manifest['scene_episode_id'],
                        scene_episode_id=manifest['scene_episode_id'],rig_id=rig_id,
                        editor_recipe_ref=str(output/'editor_recipe.json'),integration_only=True,
                        scene_matrices_ref=str(child/'scene_matrices.npy'),
                        expected_in_view_ref=str(child/'expected_in_view.npy'),
                        episode_timeline_ref=str(child/'episode_timeline.json'))
        prepared['person_timeline_refs']={str(j+1):str(child/f'person{j+1}_timeline.json') for j in range(len(recipe['people']))}
        prepared['source_timeline_refs']={str(j+1):str(child/f'person{j+1}_source_timeline.json') for j in range(len(recipe['people']))}
        if 'person_timelines' in prepared:
            prepared['person_timelines']={str(j+1):prepared['person_timelines'][str(j+1)] for j in range(len(recipe['people']))}
        for j,p in enumerate(prepared['people']):
            p['scene_from_model']=matrices[:,j].tolist()
            p['road_start_m']=matrices[0,j,:3,3].tolist()
            p['speed_m_s']=recipe['people'][j]['speed_m_s']
            p.pop('model_to_world',None)
        prepared['rig']=dict(raw['rig'],origin_m=rig['origin_m'],yaw_deg=rig['yaw_deg'])
        prepared['roi_planning_policy']='per_person_square_union_v1'
        poses,lenses,_=rig_poses(rig['origin_m'],rig['yaw_deg'],.18)
        old_rig=read(source/'rig.json')
        new_rig=dict(old_rig,cameras=[dict(id=c,camera_to_world=poses[c].tolist(),physical_lens_to_world=lenses[c].tolist()) for c in ('cam2','cam4')])
        write(child/'rig.json',new_rig)
        for k in range(31):frame_record(prepared,k)
        write(child/'recipe.json',prepared)
        _exact_prepare(child, frames if frames is not None else list(range(31)))
        manifest['rigs'].append(dict(rig_id=rig_id,prepared=str(child),coverage_goal='ordinary_multiview'))
    write(output/'editor_recipe.json',recipe)
    write(output/'rig_list.json',manifest)
    write(output/'cpu_check.json',check)
    return manifest


def _exact_prepare(child, frames):
    """Use production's body, cloth, groom, physical domain and ROI planner."""
    from PIL import Image
    from densekp_retrain.camera.domain import point_in_view
    from densekp_retrain.camera.erp import project_erp
    from densekp_retrain.data.boxes import crop_from_box
    from densekp_retrain.imaging.roi_planning import plan_person_crop_union
    from densekp_retrain.data.scene_geometry import person_geometry as _person_geometry, transform as _transform
    recipe=read(child/'recipe.json');persons=_person_geometry(recipe)
    matrices=np.load(child/'scene_matrices.npy');world=np.load(child/'world_densekp.npy')
    poses={c['id']:np.asarray(c['camera_to_world']) for c in read(child/'rig.json')['cameras']}
    masks={c:np.asarray(Image.open(child/'calibration'/f'{c}_valid_mask.png'))>0 for c in poses}
    expected=np.zeros((31,len(persons),2,138),bool);regions=[{} for _ in range(31)]
    roads=map_data('city' if 'urban' in str(recipe['scene']) or recipe.get('scene_mode')=='t_junction_three_cyclists' else 'riverfront')['roads']
    for k in frames:
        for j,person in enumerate(persons):
            M=matrices[k,j]
            xy=np.concatenate((_transform(person['body'][k],M)[:,:2],_transform(person['cloth'][k],M)[:,:2]))
            support=np.zeros(len(xy),bool)
            for a,b,c,d in roads:
                low_y=b if recipe.get('scene_mode')=='t_junction_three_cyclists' and b==80 else b+.04
                high_y=d if recipe.get('scene_mode')=='t_junction_three_cyclists' and d==80 else d-.04
                support|=(xy[:,0]>=a+.04)&(xy[:,0]<=c-.04)&(xy[:,1]>=low_y)&(xy[:,1]<=high_y)
            if not support.all():raise ValueError(f'body_or_cloth_outside_road:person{j+1}:frame{k}')
        for ci,(camera,pose) in enumerate(poses.items()):
            grouped={}
            for j,person in enumerate(persons):
                M=matrices[k,j]
                parts=dict(body=_transform(person['body'][k],M),cloth=_transform(person['cloth'][k],M),
                           groom=_transform(_transform(person['hair'],person['head'][k]),M))
                grouped[str(j+1)]={name:project_erp(value,pose,5760,2880)[0] for name,value in parts.items()}
                xy,_,_,_,inside=point_in_view(world[k,j],pose,camera,masks[camera]);expected[k,j,ci]=inside
                proxy,_,_,_,visible=point_in_view(np.concatenate((parts['body'],parts['cloth'])),pose,camera,masks[camera])
                if visible.any():
                    support=proxy[visible];box=np.r_[support.min(0),support.max(0)]
                    crop_from_box(box,image_size_wh=(5760,2880),crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
            region=plan_person_crop_union(grouped,(5760,2880),box_augmentation_policy=recipe.get('box_augmentation_policy'))
            x0,y0,x1,y1=region['roi_xyxy']
            if (x1-x0)*(y1-y0)>recipe.get('max_union_roi_pixels',7_000_000):
                raise ValueError(f'ROI_pixel_budget:frame{k}:{camera}')
            regions[k][camera]=region
    np.save(child/'expected_in_view.npy',expected)
    write(child/'planned_regions.json',regions)
    write(child/'geometry_preflight.json',dict(status='passed',scope='all_31' if len(frames)==31 else 'selected_frames_only',
         frames=frames,physical_domain=True,production_ROI_planner=True,coverage_target='reported_only'))
