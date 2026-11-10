"""Bounded seeded trajectories for already adapted scene-owned road assets.

Uses the same timeline, transforms, projection and ROI planner as single-person
production. Hull overlap is a CPU screen, not depth/instance visibility GT.
"""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
from .dynamic_preflight import plan_scene_owned,write
from .dynamic_scene import geometry_paths,scene_trajectory
from .occlusion_events import three_stage_event

VERSION='road_pass_multiperson_scene_owned_v1'


def sample_candidates(config,seed):
    if config['person_count'] not in (2,3) or not 1<=config['attempt_limit']<=12:raise ValueError('invalid_bounded_people_or_attempts')
    rng=random.Random(seed);ranges=config['parameter_ranges'];result=[]
    for attempt in range(config['attempt_limit']):
        child=rng.getrandbits(63);r=random.Random(child)
        draw=lambda key:r.uniform(*ranges[key])
        target=[draw('target_x_m'),draw('target_y_m'),config['placement_height_m']]
        people=[dict(person_id=1,start_xyz_m=target,speed_m_s=draw('speed_m_s'))]
        for j in range(1,config['person_count']):
            people.append(dict(person_id=j+1,start_xyz_m=[target[0]+draw('lateral_spacing_m'),target[1]+draw('longitudinal_spacing_m'),target[2]],speed_m_s=draw('speed_m_s')))
        result.append(dict(attempt=attempt,child_seed=child,people=people))
    return result


def plan_multiperson(config):
    if len(config['planning_seeds'])!=3 or len(set(config['planning_seeds']))!=3:raise ValueError('three_distinct_planning_seeds_required')
    if config['master_seed']!=config['planning_seeds'][0]:raise ValueError('primary_seed_must_be_first')
    base=copy.deepcopy(config);base['placement_candidates']=[dict(start_xyz_m=[0,3,config['placement_height_m']],speed_m_s=2)]
    plan=plan_scene_owned(base);plan['config']=copy.deepcopy(config);plan['algorithm_version']=VERSION
    ep=plan['episodes'][0];prototype=ep['people'][0]
    ep['people']=[dict(copy.deepcopy(prototype),person_id=j+1,role='target' if j==0 else 'occluder') for j in range(config['person_count'])]
    ep.update(generator_version=VERSION,target_person_id=1,geometry_attempt_limit=config['attempt_limit'],
              occlusion_metric=config['occlusion_metric'],max_union_roi_pixels=config['max_union_roi_pixels'],
              scene_matrix_person_ids=[p['person_id'] for p in ep['people']],same_asset_instances_not_new_identities=True)
    plan['candidate_sets']=[dict(master_seed=s,candidates=sample_candidates(config,s)) for s in config['planning_seeds']]
    return plan


def trajectory_separation(bounds,roots,*,clearance,min_root_distance):
    """Disjoint per-frame body/cloth XY AABBs with a gap; conservative, no mesh collision claim."""
    minimum=float('inf');gap_min=float('inf')
    for j in range(bounds.shape[1]):
        for l in range(j+1,bounds.shape[1]):
            a,b=bounds[:,j],bounds[:,l]
            gaps=np.maximum(a[:,:2]-b[:,2:],b[:,:2]-a[:,2:]).max(1)
            distance=np.linalg.norm(roots[:,j,:2]-roots[:,l,:2],axis=1)
            minimum=min(minimum,float(distance.min()));gap_min=min(gap_min,float(gaps.min()))
            if (gaps<clearance).any() or (distance<min_root_distance).any():raise ValueError('neighbor_body_envelopes_not_separated')
    return dict(minimum_root_distance_m=minimum,minimum_separating_AABB_gap_m=gap_min)


def preflight_multiperson(plan,base):
    import cv2
    from PIL import Image
    from densekp_retrain.camera.rig import rig_poses
    from densekp_retrain.camera.domain import point_in_view
    from densekp_retrain.camera.erp import project_erp
    from densekp_retrain.imaging.roi_planning import plan_native_crop_region
    begin=time.monotonic();cfg=plan['config'];ep=plan['episodes'][0];asset=plan['asset_adapter']
    out=Path(base)/ep['episode_id'];out.mkdir(exist_ok=False)
    write(out/'asset_adapter.json',asset);write(out/'selected_timeline.json',ep['selected_timeline'])
    for p in ep['people']:p['asset_manifest']=str((out/'asset_adapter.json').resolve())
    ep['selected_timeline_ref']=str((out/'selected_timeline.json').resolve())
    paths=geometry_paths(asset);ix=ep['people'][0]['asset_frames'];rotation=np.asarray(cfg['fixed_scene_rotation'],float)
    body=np.load(paths['body_vertices'],mmap_mode='r')[ix];cloth=np.load(paths['cloth_vertices'],mmap_mode='r')[ix]
    dense=np.load(paths['densekp'])[ix];head=np.load(paths['head_transforms'])[ix];reg=np.load(asset['regressor'])
    for v,d in zip(body,dense):np.testing.assert_allclose(reg@v,d,atol=2e-6,rtol=0)
    if not all(np.isfinite(v).all() for v in [body,cloth,dense,head]):raise ValueError('nonfinite_asset')
    delta=np.max(np.abs(np.diff(body,axis=0)),axis=(1,2))
    if not (delta>1e-5).all():raise ValueError('static_body_frames')
    oriented_body=body@rotation.T;oriented_cloth=cloth@rotation.T
    min_bounds=np.minimum(oriented_body.min(1),oriented_cloth.min(1));max_bounds=np.maximum(oriented_body.max(1),oriented_cloth.max(1))
    ground=min_bounds[:,2]+cfg['placement_height_m']-cfg['road_height_m'];lo,hi=cfg['support_checks']['body_min_clearance_m']
    if (ground<lo).any() or (ground>hi).any():raise ValueError('cycling_clearance_outside_gate')
    with np.load(asset['parameters']) as params:
        np.testing.assert_array_equal(params['transl'][ix],np.repeat(params['transl'][ix[:1]],len(ix),axis=0))
        forwards=np.stack([cv2.Rodrigues(v)[0]@np.array([0.,0.,1.]) for v in params['global_orient'][ix]])@rotation.T
        heading_cos=forwards[:,:2]@np.array(cfg['path_direction_xy'])/np.linalg.norm(forwards[:,:2],axis=1)
        if (heading_cos<np.cos(np.deg2rad(35))).any():raise ValueError('body_heading_not_aligned')
    poses,lenses,_=rig_poses(ep['rig']['origin_m'],ep['rig']['yaw_deg'],ep['rig']['baseline_m'])
    masks={c:np.asarray(Image.open(p))>0 for c,p in cfg['physical_masks'].items()}
    signed={c:cv2.distanceTransform(m.astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform((~m).astype(np.uint8),cv2.DIST_L2,5) for c,m in masks.items()}
    for m in masks.values():
        if m.shape!=(2880,5760):raise ValueError('physical_mask_size')
    # Only a conservative small proxy is used for initial ROI screening. The
    # chosen candidate is subsequently checked against every active groom point.
    groom=Path(asset['groom_reference']);points=np.load(groom/'points.npy')
    points=points[np.repeat(np.load(groom/'active_curve_mask.npy'),np.diff(np.load(groom/'curve_offsets.npy')))]
    gmin=points.min(0);gmax=points.max(0)
    corners=np.array([[x,y,z] for x in [gmin[0],gmax[0]] for y in [gmin[1],gmax[1]] for z in [gmin[2],gmax[2]]])
    hair_proxy=(corners[None]@head[:,:3,:3].transpose(0,2,1)+head[:,None,:3,3])@rotation.T
    results=[];chosen=None
    for seedset in plan['candidate_sets']:
        seed_result=dict(master_seed=seedset['master_seed'],attempts=[],selected_attempt=None)
        assert sample_candidates(cfg,seedset['master_seed'])==seedset['candidates']
        for candidate in seedset['candidates']:
            attempt=dict(**candidate,rejections=[],frames=[]);matrices=None
            try:
                matrices=np.stack([scene_trajectory(ep['selected_timeline'],p['start_xyz_m'],cfg['path_direction_xy'],p['speed_m_s'],rotation) for p in candidate['people']],axis=1)
                roots=matrices[:,:,:3,3];bounds=np.concatenate([min_bounds[:,None,:2]+roots[:,:,:2],max_bounds[:,None,:2]+roots[:,:,:2]],axis=2)
                road=cfg['road_bounds_xyxy']
                if (bounds[:,:,:2]<road[:2]).any() or (bounds[:,:,2:]>road[2:]).any():raise ValueError('body_outside_road_corridor')
                attempt.update(trajectory_separation(bounds,roots,clearance=cfg['neighbor_envelope_clearance_m'],min_root_distance=cfg['minimum_root_distance_m']))
                curves={c:[] for c in poses};truncated={c:0 for c in poses}
                world=dense[:,None]@rotation.T+roots[:,:,None,:]
                for k in range(ep['frame_count']):
                    inside={};views={};predictions={}
                    for c,pose in poses.items():
                        xy,_,distance,_,inside[c]=point_in_view(world[k,0],pose,c,masks[c])
                        near=np.zeros(138,bool);parts={}
                        for j in range(cfg['person_count']):
                            b=oriented_body[k]+roots[k,j];cl=oriented_cloth[k]+roots[k,j]
                            uv=project_erp(b,pose,5760,2880)[0]
                            parts[f'body{j}']=uv;parts[f'cloth{j}']=project_erp(cl,pose,5760,2880)[0]
                            parts[f'groom_proxy{j}']=project_erp(hair_proxy[k]+roots[k,j],pose,5760,2880)[0]
                            if j and np.linalg.norm(b-pose[:3,3],axis=1).max()<distance.min():
                                hull=cv2.convexHull(uv.astype(np.float32))
                                near|=np.array([cv2.pointPolygonTest(hull,tuple(map(float,q)),False)>=0 for q in xy])
                        region=plan_native_crop_region(parts,(5760,2880));x0,y0,x1,y1=region['roi_xyxy']
                        if (x1-x0)*(y1-y0)>cfg['max_union_roi_pixels']:raise ValueError('union_ROI_pixel_budget')
                        yi=np.clip(np.floor(xy[:,1]).astype(int),0,2879);xi=np.floor(xy[:,0]).astype(int)%5760
                        dist=signed[c][yi,xi];truncated[c]+=int(not inside[c].all())
                        views[c]=dict(in_view=int(inside[c].sum()),outside_domain=int((~inside[c]).sum()),near_boundary_64px=int((np.abs(dist)<=64).sum()),roi_xyxy=region['roi_xyxy'],roi_pixels=(x1-x0)*(y1-y0))
                        predictions[c]=near
                    common=inside['cam2']&inside['cam4']
                    if common.sum()<cfg['coverage_thresholds']['min_common_points']:raise ValueError('target_insufficient_common_physical_domain')
                    for c in curves:
                        value=float((predictions[c]&common).sum()/common.sum());curves[c].append(value);views[c]['hull_overlap_common_fraction']=value
                    attempt['frames'].append(dict(frame_index=k,common_physical_points=int(common.sum()),views=views))
                if max(truncated.values())<cfg['coverage_thresholds']['min_truncated_times_any_view']:raise ValueError('insufficient_target_truncation')
                if any(sum(r['views'][c]['near_boundary_64px']>0 for r in attempt['frames'])<cfg['coverage_thresholds']['min_boundary_times_each_view'] for c in poses):raise ValueError('insufficient_each_view_boundary_opportunity')
                events={c:three_stage_event(v,delta=cfg['occlusion_metric']['delta'],consecutive=3) for c,v in curves.items()}
                attempt['projected_overlap_events']=events
                if not any(events.values()):raise ValueError('no_projected_overlap_rise_and_recovery')
                # Full actual hair points, not the bounding-box proxy, for the first
                # selected recipe. Same native-square union algorithm as renderer.
                if chosen is None:
                    regions=[]
                    for k in range(ep['frame_count']):
                        h=(points@head[k,:3,:3].T+head[k,:3,3])@rotation.T;per={}
                        for c,pose in poses.items():
                            parts={}
                            for j in range(cfg['person_count']):
                                for name,part in [('body',oriented_body[k]),('cloth',oriented_cloth[k]),('groom',h)]:parts[f'{j}_{name}']=project_erp(part+roots[k,j],pose,5760,2880)[0]
                            per[c]=plan_native_crop_region(parts,(5760,2880))
                            x0,y0,x1,y1=per[c]['roi_xyxy']
                            if (x1-x0)*(y1-y0)>cfg['max_union_roi_pixels']:raise ValueError('actual_groom_union_ROI_pixel_budget')
                        regions.append(per)
                    chosen=(candidate,seedset['master_seed'],matrices,world,regions,events)
                seed_result['selected_attempt']=candidate['attempt']
            except ValueError as error:attempt['rejections'].append(str(error))
            seed_result['attempts'].append(attempt)
            write(out/'candidate_checks.json',results+[seed_result])
            if not attempt['rejections']:break
        results.append(seed_result)
    write(out/'candidate_checks.json',results)
    if chosen is None:
        write(out/'geometry_preflight.json',dict(status='rejected',reason='no_bounded_candidate',seconds=time.monotonic()-begin))
        raise ValueError('no_bounded_multiperson_candidate')
    candidate,seed,matrices,world,regions,events=chosen
    for j,p in enumerate(ep['people']):
        trajectory=candidate['people'][j]
        p.update(scene_from_model=matrices[:,j].tolist(),placement_height_m=cfg['placement_height_m'],support_checks=cfg['support_checks'],road_start_m=trajectory['start_xyz_m'],path_direction_xy=cfg['path_direction_xy'],speed_m_s=trajectory['speed_m_s'],scene_yaw_deg=cfg['fixed_scene_yaw_deg'])
    ep.update(master_seed=seed,child_seed=candidate['child_seed'],selected_attempt=candidate['attempt'],parameter_ranges=cfg['parameter_ranges'],
              geometry_acceptance='passed',road_height_m=cfg['road_height_m'],road_bounds_xyxy=cfg['road_bounds_xyxy'],coverage_thresholds=cfg['coverage_thresholds'],
              scene_matrices_ref=str((out/'scene_matrices.npy').resolve()),predicted_overlap_events=events,
              random_algorithm='python.random.Random MT19937; per-attempt getrandbits(63), local uniform; version pinned by recipe',
              randomized_dimensions=['road_start_xy','lateral_spacing','longitudinal_spacing','constant_forward_speed'])
    np.save(out/'scene_matrices.npy',matrices);np.save(out/'world_densekp.npy',world)
    write(out/'rig.json',dict(calibration_target='20260320-riding-dataset3',baseline_kind='synthetic_not_measured',cameras=[dict(id=c,camera_to_world=p.tolist(),physical_lens_to_world=lenses[c].tolist()) for c,p in poses.items()]))
    (out/'calibration').mkdir()
    for c,m in masks.items():Image.fromarray(m.astype(np.uint8)*255).save(out/'calibration'/f'{c}_valid_mask.png')
    write(out/'planned_regions.json',regions)
    write(out/'geometry_preflight.json',dict(status='passed',seconds=time.monotonic()-begin,selected_seed=seed,selected_attempt=candidate['attempt'],
          body_adjacent_max_delta_m=delta.tolist(),body_cloth_min_clearance_m=ground.tolist(),heading_cosines=heading_cos.tolist(),
          checks='all native frames; strict shared asset timeline; disjoint XY body/cloth AABBs; exact selected groom ROI',
          limitations='hull overlap is only prescreen, not actual neighbor occlusion; envelope check is not a bicycle or full mesh collision certificate'))
    write(out/'recipe.json',ep)
    return ep
