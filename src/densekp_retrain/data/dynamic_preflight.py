"""CPU planning for explicit scene-owned assets in the public road_pass path."""
import copy
import json
from pathlib import Path
import numpy as np
from .dynamic_scene import adapt_proposal,select_timeline,scene_trajectory,geometry_paths,ROOT_MODE,SUPPORT


def write(path,obj):Path(path).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def plan_scene_owned(config):
    if config['mode']!='road_pass' or not 1<=len(config['placement_candidates'])<=4:raise ValueError('unsupported_dynamic_plan')
    asset=adapt_proposal(config['asset_proposal'])
    records=select_timeline(json.loads(Path(asset['timeline_ref']).read_text()),config['first_record'],config['frame_count'],asset['frame_count'])
    ep=dict(episode_id=config['episode_id'],sequence_id=config['episode_id'],child_seed=config['master_seed'],
        master_seed=config['master_seed'],integration_only=True,diagnostic_split='shared_source_diagnostic',
        scene=config['scene'],rig=config['rig'],render=config['render'],frame_count=len(records),time_base=records[0]['time_base'],
        selected_timeline=records,original_timeline_ref=asset['timeline_ref'],original_timeline_count=asset['frame_count'],
        unselected_asset_frames=[i for i in range(asset['frame_count']) if i not in {r['asset_frame'] for r in records}],
        time_origin=dict(kind='selected_first_source_tick',source_time_tick=records[0]['source_time_tick']),
        time_unit_status=asset['time_unit_status'],nominal_source_rate=asset['nominal_source_rate'],sampling_policy='native_observations_no_resampling',
        sync_status='synthetic_same_state',imaging_route='direct_erp_roi',storage_mode='roi_only',
        crop_policy='deployment',crop_expansion=1.0,geometry_acceptance='pending',geometry_attempt_limit=4,
        people=[dict(person_id=1,role='target',asset_id=asset['asset_id'],character_family=asset['character_family'],
            root_mode=ROOT_MODE,support_policy=SUPPORT,asset_frames=[r['asset_frame'] for r in records],
            source_frames=[r['source_frame'] for r in records],speed_policy='scene_constant_metres_per_second_not_source_speed',
            outfit_policy='fixed_accepted_combination_entire_episode')])
    return dict(algorithm_version='road_pass_scene_owned_v1',config=copy.deepcopy(config),asset_adapter=asset,episodes=[ep],
                seed_is_only_generation_provenance=True,source_independence=False)


def preflight_scene_owned(plan,base):
    import cv2
    from PIL import Image
    from densekp_retrain.camera.rig import rig_poses
    from densekp_retrain.camera.domain import point_in_view
    from densekp_retrain.camera.erp import project_erp
    from densekp_retrain.imaging.roi_planning import plan_native_crop_region
    from densekp_retrain.data.boxes import crop_from_box
    config=plan['config'];ep=plan['episodes'][0];asset=plan['asset_adapter'];out=Path(base)/ep['episode_id'];out.mkdir(exist_ok=False)
    write(out/'asset_adapter.json',asset);ep['people'][0]['asset_manifest']=str((out/'asset_adapter.json').resolve())
    write(out/'selected_timeline.json',ep['selected_timeline']);ep['selected_timeline_ref']=str((out/'selected_timeline.json').resolve())
    paths=geometry_paths(asset);ix=ep['people'][0]['asset_frames']
    body=np.load(paths['body_vertices'],mmap_mode='r')[ix];cloth=np.load(paths['cloth_vertices'],mmap_mode='r')[ix]
    head=np.load(paths['head_transforms'])[ix];dense=np.load(paths['densekp'])[ix];R=np.load(asset['regressor'])
    for v,d in zip(body,dense):np.testing.assert_allclose(np.einsum('kv,vc->kc',R,v,optimize=True),d,atol=2e-6,rtol=0)
    if not all(np.isfinite(x).all() for x in [body,cloth,head,dense]):raise ValueError('nonfinite_asset')
    delta=np.max(np.abs(np.diff(body,axis=0)),axis=(1,2))
    if not (delta>1e-5).all():raise ValueError('static_or_duplicate_body_frames')
    with np.load(asset['parameters']) as params:
        np.testing.assert_array_equal(params['transl'][ix],np.repeat(params['transl'][ix[:1]],len(ix),axis=0))
        forward=np.stack([cv2.Rodrigues(x)[0]@np.array([0.,0.,1.]) for x in params['global_orient'][ix]])
    # Root orientation is measured only to choose ONE scene yaw; it is already
    # baked into vertices/head and is never multiplied into them again.
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    f=forward@C.T;mean=f[:,:2].mean(0);direction=np.array(config['path_direction_xy'],float)
    yaw=np.arctan2(direction[1],direction[0])-np.arctan2(mean[1],mean[0])
    Y=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1]])
    orientation=Y@C;headings=forward@orientation.T;headings=headings[:,:2]/np.linalg.norm(headings[:,:2],axis=1)[:,None]
    heading_cos=headings@direction
    if (heading_cos<np.cos(np.deg2rad(35))).any():raise ValueError('body_heading_not_aligned_with_path')
    groom=Path(asset['groom_reference']);points=np.load(groom/'points.npy');offsets=np.load(groom/'curve_offsets.npy');active=np.load(groom/'active_curve_mask.npy')
    points=points[np.repeat(active,np.diff(offsets))]
    poses,lenses,_=rig_poses(ep['rig']['origin_m'],ep['rig']['yaw_deg'],ep['rig']['baseline_m'])
    write(out/'rig.json',dict(calibration_target='20260320-riding-dataset3',baseline_kind='synthetic_not_measured',cameras=[dict(id=c,camera_to_world=p.tolist(),physical_lens_to_world=lenses[c].tolist()) for c,p in poses.items()]))
    (out/'calibration').mkdir();masks={};signed={}
    for c,path in config['physical_masks'].items():
        masks[c]=np.asarray(Image.open(path))>0
        if masks[c].shape!=(2880,5760):raise ValueError('invalid_calibration_mask_size')
        Image.fromarray(masks[c].astype(np.uint8)*255).save(out/'calibration'/f'{c}_valid_mask.png')
        signed[c]=cv2.distanceTransform(masks[c].astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform((~masks[c]).astype(np.uint8),cv2.DIST_L2,5)
    attempts=[];accepted=None
    for attempt,candidate in enumerate(config['placement_candidates']):
        matrices=scene_trajectory(ep['selected_timeline'],candidate['start_xyz_m'],direction,candidate['speed_m_s'],orientation)
        rows=[];errors=[];world=[]
        try:
            for k,(v,cl,H,d,M) in enumerate(zip(body,cloth,head,dense,matrices)):
                apply=lambda p:p@M[:3,:3].T+M[:3,3]
                bv=apply(v);cv=apply(cl);hair=apply(points@H[:3,:3].T+H[:3,3]);dw=apply(d);world.append(dw)
                extent=np.r_[bv.min(0),bv.max(0)];road=config['road_bounds_xyxy']
                if extent[0]<road[0] or extent[1]<road[1] or extent[3]>road[2] or extent[4]>road[3]:raise ValueError('body_outside_road_corridor')
                clearance=float(min(bv[:,2].min(),cv[:,2].min())-config['road_height_m'])
                lo,hi=config['support_checks']['body_min_clearance_m']
                if not lo<=clearance<=hi:raise ValueError('cycling_body_clearance_outside_gate')
                views={};supports=[]
                for c,pose in poses.items():
                    xy,_,_,_,inside=point_in_view(dw,pose,c,masks[c]);uv,_,_,_,bi=point_in_view(bv,pose,c,masks[c]);visible=uv[bi]
                    if len(visible)==0:raise ValueError('no_projected_body_support')
                    box=np.r_[visible.min(0),visible.max(0)]
                    center,scale=crop_from_box(box,image_size_wh=(5760,2880),crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
                    support=inside&(np.abs(xy-center)<scale*100).all(1);supports.append(support)
                    parts={'body':uv,'cloth':project_erp(cv,pose,5760,2880)[0],'groom':project_erp(hair,pose,5760,2880)[0]}
                    region=plan_native_crop_region(parts,(5760,2880),
                        box_augmentation_policy=ep.get('box_augmentation_policy'))
                    yy=np.clip(np.floor(xy[:,1]).astype(int),0,2879);xx=np.floor(xy[:,0]).astype(int)%5760;dist=signed[c][yy,xx]
                    views[c]=dict(in_view=int(inside.sum()),outside_domain=int((~inside).sum()),projected_crop_support=int(support.sum()),
                        boundary_signed_distance_px=dist.tolist(),near_boundary_64px=int((np.abs(dist)<=64).sum()),
                        projected_body_box=box.tolist(),roi_xyxy=region['roi_xyxy'])
                rows.append(dict(frame_index=k,time=ep['selected_timeline'][k],views=views,common_projected_crop=int((supports[0]&supports[1]).sum()),
                    body_world_bounds=extent.tolist(),body_cloth_min_road_clearance_m=clearance))
            t=config['coverage_thresholds']
            if any(any(v['in_view']<t['min_each_view_points'] for v in r['views'].values()) or r['common_projected_crop']<t['min_common_points'] for r in rows):errors.append('insufficient_continuous_dual_support')
            for c in poses:
                if sum(r['views'][c]['outside_domain']>0 for r in rows)<t['min_truncated_times_each_view']:errors.append('insufficient_truncation_'+c)
        except ValueError as error:errors.append(str(error))
        attempts.append(dict(candidate_index=attempt,candidate=candidate,rejections=errors,frames=rows))
        write(out/'geometry_attempts.json',attempts)
        if not errors:accepted=(candidate,matrices,np.stack(world));break
    if accepted is None:raise ValueError('all_bounded_placements_rejected')
    candidate,matrices,world=accepted;p=ep['people'][0]
    p.update(scene_from_model=matrices.tolist(),placement_height_m=candidate['start_xyz_m'][2],support_checks=config['support_checks'],
        road_start_m=candidate['start_xyz_m'],path_direction_xy=config['path_direction_xy'],speed_m_s=candidate['speed_m_s'],scene_yaw_deg=float(np.rad2deg(yaw)))
    ep.update(geometry_acceptance='passed',geometry_attempts=[dict(candidate_index=x['candidate_index'],rejections=x['rejections']) for x in attempts],
        road_height_m=config['road_height_m'],road_bounds_xyxy=config['road_bounds_xyxy'],coverage_thresholds=config['coverage_thresholds'],
        scene_matrices_ref=str((out/'scene_matrices.npy').resolve()),calibration_mask_invariance='same normal ERP axes relative to physical lenses; unchanged rig and calibration')
    np.save(out/'scene_matrices.npy',matrices);np.save(out/'world_densekp.npy',world[:,None])
    write(out/'geometry_preflight.json',dict(status='passed',frames=len(ix),body_adjacent_max_delta_m=delta.tolist(),
        heading_path_cosines=heading_cos.tolist(),scene_yaw_deg=float(np.rad2deg(yaw)),root_once=True,
        selected_source_origin=ep['time_origin'],groom_active_points=len(points),road_actual_raycast='required before importing people in Blender',
        thresholds=config['coverage_thresholds'],accepted_candidate=len(attempts)-1))
    write(out/'recipe.json',ep)
    return ep
