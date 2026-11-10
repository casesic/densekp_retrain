"""Mixed-source road recipes: exact observation selection, independent root ownership."""
import copy
import json
import random
import time
from pathlib import Path
import numpy as np
from .person_time import map_timeline,SCHEMA
from .dynamic_scene import geometry_paths,scene_trajectory,person_matrices,frame_record
from .dynamic_preflight import write
from .multiperson_road_pass import trajectory_separation
from .occlusion_events import three_stage_event
from .viewpoint import classify_view,view_angle,view_coverage

VERSION='road_pass_mixed_sources_v1'
EDGE=dict(min_inside_each=8,min_outside_each=5,min_common_domain=8,min_common_crop=8,min_time_fraction=.5,min_continuous_seconds=.5,boundary_band_px=64,units='full ERP physical-mask pixels')
COVERAGE_GOALS=('edge_coverage','boundary_transition')


def coverage_goal(config,single):
    goal=config.get('coverage_goal','boundary_transition' if single else 'edge_coverage')
    if goal not in COVERAGE_GOALS:raise ValueError('invalid_coverage_goal')
    # A mixed episode has no inherited endpoint-crossing gate. Do not make an
    # implicit new requirement for historical two-person recipes.
    if not single and goal=='boundary_transition':raise ValueError('mixed_boundary_transition_unsupported')
    return goal


def crossing_transition(crossing,policy='enter_valid'):
    if policy not in ('enter_valid','exit_valid','either'):
        raise ValueError('invalid_crossing_transition')
    enter=all(v['first_outside']>=20 and v['last_outside']<=5 and v['first_distance_p10']<0<v['last_distance_p10'] for v in crossing.values())
    leave=all(v['first_outside']<=5 and v['last_outside']>=20 and v['first_distance_p10']>0>v['last_distance_p10'] for v in crossing.values())
    actual='enter_valid' if enter else 'exit_valid' if leave else None
    return actual if policy=='either' or policy==actual else None


def edge_summary(rows,clock,required_count=None):
    from .person_time import rational
    flags=[all(r['views'][c]['in_view']>=EDGE['min_inside_each'] and r['views'][c]['outside_domain']>=EDGE['min_outside_each'] for c in ['cam2','cam4']) and r['common_physical_points']>=EDGE['min_common_domain'] for r in rows]
    windows=[];start=None
    for i,flag in enumerate(flags+[False]):
        if flag and start is None:start=i
        if not flag and start is not None:
            end=i-1;duration=float(clock[end]['episode_time_tick']*rational(clock[end]['time_base'])-clock[start]['episode_time_tick']*rational(clock[start]['time_base']))
            windows.append(dict(first=start,last=end,duration_seconds=duration));start=None
    required=(len(clock)+1)//2 if required_count is None else required_count
    if not 1<=required<=len(clock):raise ValueError('invalid_dual_edge_required_count')
    return dict(flags=flags,count=sum(flags),required_count=required,fraction=sum(flags)/len(clock),windows=windows,
        passed=sum(flags)>=required and any(w['duration_seconds']>=EDGE['min_continuous_seconds'] for w in windows))


def candidates(config,seed):
    if not 1<=config['attempt_limit']<=12:raise ValueError('attempt_limit')
    rng=random.Random(seed);out=[]
    camera_rng=random.Random(config['camera_seed']) if 'camera_seed' in config else None
    def extra(i):
        if camera_rng is None:return {}
        child=camera_rng.getrandbits(63);r=random.Random(child)
        frozen=config.get('explicit_camera_parameters')
        if frozen is not None:
            if len(frozen)!=config['attempt_limit']:raise ValueError('explicit_camera_count')
            parameters=copy.deepcopy(frozen[i])
            if set(parameters)!=set(config['camera_parameter_ranges']) or any(
                not np.isfinite(value) or not config['camera_parameter_ranges'][key][0]<=value<=config['camera_parameter_ranges'][key][1]
                for key,value in parameters.items()):raise ValueError('invalid_explicit_camera')
        else:
            parameters={k:r.uniform(*v) for k,v in config['camera_parameter_ranges'].items()}
        return dict(camera_child_seed=child,
                    camera_parameters=parameters,
                    travel_direction_sign=rng.choice(config['travel_direction_options']))
    explicit=config.get('explicit_candidate_parameters')
    if explicit is not None:
        if len(explicit)!=config['attempt_limit']:raise ValueError('explicit_candidate_count')
        keys=set(config['parameter_ranges'])
        for i,parameters in enumerate(explicit):
            if set(parameters)!=keys or not all(np.isfinite(value) and config['parameter_ranges'][key][0]<=value<=config['parameter_ranges'][key][1]
                                                for key,value in parameters.items()):
                raise ValueError('invalid_explicit_candidate')
            out.append(dict(attempt=i,child_seed=rng.getrandbits(63),parameters=copy.deepcopy(parameters),**extra(i)))
        return out
    for i in range(config['attempt_limit']):
        child=rng.getrandbits(63);r=random.Random(child)
        out.append(dict(attempt=i,child_seed=child,parameters={k:r.uniform(*v) for k,v in config['parameter_ranges'].items()},**extra(i)))
    return out


def checked_mixed_separation(bounds,roots,worldparts,*,clearance=.1,min_root_distance=.9):
    """Use a vertex-distance narrow phase only for overlapping XY envelopes.

    This is a conservative sampled-surface check, not a full triangle collision
    test. It preserves the original root and 10 cm person separation gates.
    """
    from scipy.spatial import cKDTree
    if len(worldparts)!=2:raise ValueError('mixed_separation_requires_two_people')
    a,b=bounds[:,0],bounds[:,1]
    gaps=np.maximum(a[:,:2]-b[:,2:],b[:,:2]-a[:,2:]).max(1)
    distances=np.linalg.norm(roots[:,0,:2]-roots[:,1,:2],axis=1)
    if (distances<min_root_distance).any():raise ValueError('neighbor_root_distance_below_gate')
    exact=[]
    for k in np.flatnonzero(gaps<clearance):
        left=np.concatenate([worldparts[0][name][k] for name in ('body','cloth')])
        right=np.concatenate([worldparts[1][name][k] for name in ('body','cloth')])
        closest=float(cKDTree(right).query(left,k=1,workers=1)[0].min())
        exact.append((int(k),closest))
        if closest<clearance:raise ValueError('neighbor_surface_vertices_below_gate')
    return dict(minimum_root_distance_m=float(distances.min()),
                minimum_separating_AABB_gap_m=float(gaps.min()),
                narrow_phase_frame_count=len(exact),
                minimum_narrow_phase_vertex_distance_m=min((v for _,v in exact),default=None),
                separation_method='XY_AABB_then_full_body_cloth_vertex_3D_if_needed')


def plan_mixed(config):
    seeds=config['planning_seeds']
    limited=config.get('observation_policy')=='record_unavailable_v1'
    count=len(config['reference_recipes'])
    if count not in (1,2) or (count==1 and config.get('scene_mode')!='single_cyclist_boundary_pass'):
        raise ValueError('unsupported_person_count_or_scene_mode')
    goal=coverage_goal(config,count==1)
    if config.get('roi_planning_policy','legacy_collective_envelope') not in ('legacy_collective_envelope','per_person_square_union_v1'):
        raise ValueError('unknown_roi_planning_policy')
    if 'box_augmentation_policy' in config:
        from densekp_retrain.data.box_augmentation import validate_box_policy
        validate_box_policy(config['box_augmentation_policy'])
    if 'requested_view_class' in config:
        if config.get('viewpoint_policy_version')!='seeded_viewpoints_v1' or config.get('crossing_transition') not in ('enter_valid','exit_valid','either'):
            raise ValueError('invalid_viewpoint_policy')
    if (config['frame_count']!=31 or seeds[0]!=config['master_seed'] or len(set(seeds))!=len(seeds)
            or (limited and (len(seeds)!=1 or config['attempt_limit']>12))
            or (not limited and len(seeds)!=3)):
        raise ValueError('mixed_bounded_config')
    clock=[dict(frame_index=k,episode_time_tick=k,time_base=[1,10]) for k in range(config['frame_count'])]
    references=[json.loads(Path(p).read_text()) for p in config['reference_recipes']]
    ep=copy.deepcopy(references[0]);people=[copy.deepcopy(r['people'][0]) for r in references]
    for key in ['selected_timeline','selected_timeline_ref','time_origin','scene_matrices_ref','selection','original_timeline_count','original_timeline_ref','unselected_asset_frames','nominal_source_rate','parent_recipe_ref','geometry_attempts']:ep.pop(key,None)
    sources=[];maps={}
    for j,p in enumerate(people):
        asset=json.loads(Path(p['asset_manifest']).read_text())
        if j==0:source=json.loads(Path(asset['timeline_ref']).read_text())
        else:
            if asset.get('timeline_ref'):
                source=json.loads(Path(asset['timeline_ref']).read_text())
            else:
                with np.load(asset['parameters']) as pars:ids=pars['source_frame_ids'].tolist()
                source=[dict(asset_frame=i,source_frame=int(f),source_time_tick=int(f),time_base=[1,30]) for i,f in enumerate(ids)]
        mapped=map_timeline(source,clock,origin_record=config['source_origin_records'][j]);sources.append(source);maps[str(j+1)]=mapped
        p.update(person_id=j+1,role='target' if j==0 else 'occluder',asset_frames=[r['asset_frame'] for r in mapped],source_frames=[r['source_frame'] for r in mapped])
    if limited:
        people[0]['placement_height_m']=config['placement_height_m']
        people[0]['support_checks']=copy.deepcopy(config['support_checks'])
        people[0].pop('scene_from_model',None)
    edge_gate=copy.deepcopy(EDGE)
    if 'min_dual_edge_times' in config:
        edge_gate['required_count']=config['min_dual_edge_times']
        edge_gate['min_time_fraction']=config['min_dual_edge_times']/len(clock)
    ep.update(episode_id=config['episode_id'],sequence_id=config['episode_id'],scene=config['scene'],render=copy.deepcopy(config['render']),generator_version=VERSION,frame_count=len(clock),time_base=[1,10],
        episode_timeline=clock,person_timelines=maps,time_mapping_schema=SCHEMA,people=people,target_person_id=1,
        dual_edge_gate=edge_gate,occlusion_metric=config['occlusion_metric'],readback_policy='first_middle_last',integration_only=True,
        time_unit_status=('synthetic10Hz; target original PTS '+str(json.loads(Path(people[0]['asset_manifest']).read_text())['time_base'])+('; KIT published30Hz, original100Hz unknown' if count==2 else '')) if limited else 'synthetic10Hz; APD original microsecond unit inferred; KIT published30Hz, original100Hz unknown',
        sampling_policy='synthetic10Hz_nearest_original_record_ties_earlier_no_interpolation',coverage_thresholds=edge_gate,geometry_attempt_limit=12,reference_recipe_refs=config['reference_recipes'],
        same_asset_instances_not_new_identities=False,scene_matrix_person_ids=list(range(1,count+1)),max_union_roi_pixels=config['max_union_roi_pixels'])
    if config.get('scene_mode') in ('single_cyclist_boundary_pass','mixed_walk_cycle_crossing'):
        ep['scene_mode']=config['scene_mode']
    if 'coverage_goal' in config:ep['coverage_goal']=goal
    if 'component_provenance' in config:
        # A1 composition orchestration owns this descriptor. It is metadata,
        # while the actual body/cloth/groom paths still come from asset_manifest.
        ep['component_provenance']=copy.deepcopy(config['component_provenance'])
    if 'placement_policy_version' in config:
        ep['placement_policy_version']=config['placement_policy_version']
    if 'requested_view_class' in config:
        ep.update(requested_view_class=config['requested_view_class'],crossing_transition=config['crossing_transition'],
                  viewpoint_policy_version=config['viewpoint_policy_version'])
    if limited:ep['observation_policy']='record_unavailable_v1'
    if config.get('roi_planning_policy')=='per_person_square_union_v1':
        ep['roi_planning_policy']='per_person_square_union_v1'
    if 'box_augmentation_policy' in config:
        ep['box_augmentation_policy']=copy.deepcopy(config['box_augmentation_policy'])
    return dict(algorithm_version=VERSION,config=copy.deepcopy(config),episodes=[ep],source_timelines=sources,
        candidate_sets=[dict(master_seed=s,candidates=candidates(config,s)) for s in config['planning_seeds']])


def preflight_mixed(plan,base):
    import cv2
    from PIL import Image
    from densekp_retrain.camera.rig import rig_poses
    from densekp_retrain.camera.domain import point_in_view
    from densekp_retrain.camera.erp import project_erp
    from densekp_retrain.imaging.roi_planning import plan_native_crop_region,plan_person_crop_union
    from .boxes import crop_from_box
    begin=time.monotonic();cfg=plan['config'];ep=plan['episodes'][0];out=Path(base)/ep['episode_id'];out.mkdir(exist_ok=False)
    assets=[json.loads(Path(p['asset_manifest']).read_text()) for p in ep['people']]
    arrays=[]
    for j,(p,a) in enumerate(zip(ep['people'],assets)):
        paths=geometry_paths(a);ix=p['asset_frames'];d={n:np.load(paths[key],mmap_mode='r')[ix] for n,key in [('body','body_vertices'),('cloth','cloth_vertices'),('dense','densekp'),('head','head_transforms')]}
        if not all(np.isfinite(v).all() for v in d.values()):raise ValueError('nonfinite_asset')
        if not (np.max(np.abs(np.diff(d['body'],axis=0)),axis=(1,2))>1e-5).all():raise ValueError('static_articulation')
        groom=Path(a['groom_reference']);points=np.load(groom/'points.npy');points=points[np.repeat(np.load(groom/'active_curve_mask.npy'),np.diff(np.load(groom/'curve_offsets.npy')))]
        lo,hi=points.min(0),points.max(0);d['hair_corners']=np.array([[x,y,z] for x in [lo[0],hi[0]] for y in [lo[1],hi[1]] for z in [lo[2],hi[2]]]);d['hair']=points
        arrays.append(d)
        write(out/f'person{j+1}_source_timeline.json',plan['source_timelines'][j]);write(out/f'person{j+1}_timeline.json',ep['person_timelines'][str(j+1)])
    baseline_poses,baseline_lenses,_=rig_poses(ep['rig']['origin_m'],ep['rig']['yaw_deg'],ep['rig']['baseline_m'])
    masks={c:np.asarray(Image.open(p))>0 for c,p in cfg['physical_masks'].items()}
    def plan_region(parts):
        if cfg.get('roi_planning_policy')=='per_person_square_union_v1':
            grouped={str(j):{name:xy for name,xy in parts.items() if name.startswith(f'{j}_')} for j in range(len(assets))}
            return plan_person_crop_union(grouped,(5760,2880),
                box_augmentation_policy=cfg.get('box_augmentation_policy'))
        return plan_native_crop_region(parts,(5760,2880),
            box_augmentation_policy=cfg.get('box_augmentation_policy'))
    signed={c:cv2.distanceTransform(m.astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform((~m).astype(np.uint8),cv2.DIST_L2,5) for c,m in masks.items()}
    base_cycleR=np.array(cfg['fixed_scene_rotation']);single=len(assets)==1
    if 'rider_heading_gate_deg' in cfg or 'requested_view_class' in cfg:
        with np.load(assets[0]['parameters']) as pars:
            oriented=pars['global_orient'][ep['people'][0]['asset_frames']]
    else:oriented=None
    if not single:
        oldwalkR=np.array(ep['people'][1]['model_to_world'])[:3,:3]
        with np.load(assets[1]['parameters']) as pars:relative=pars['source_root_relative'][ep['people'][1]['asset_frames'][0]]
        lift=-float((arrays[1]['body'][:,:,1]-relative[1]).min())
    results=[];chosen=None
    for seedset in plan['candidate_sets']:
        result=dict(master_seed=seedset['master_seed'],attempts=[])
        assert seedset['candidates']==candidates(cfg,seedset['master_seed'])
        for candidate in seedset['candidates']:
            check=dict(**candidate,rejections=[],frames=[]);q=candidate['parameters']
            try:
                direction_sign=candidate.get('travel_direction_sign',1)
                if direction_sign not in (-1,1):raise ValueError('invalid_travel_direction')
                turn=np.deg2rad(180 if direction_sign==-1 else 0)
                yawturn=np.array([[np.cos(turn),-np.sin(turn),0],[np.sin(turn),np.cos(turn),0],[0,0,1]])
                cycleR=yawturn@base_cycleR
                direction=np.array([0.,float(direction_sign)])
                if oriented is not None:
                    headings=np.array([cycleR@cv2.Rodrigues(value)[0]@np.array([0.,0.,1.]) for value in oriented])
                    horizontal=headings[:,:2]/np.linalg.norm(headings[:,:2],axis=1)[:,None]
                    minimum=float((horizontal@direction).min())
                    if minimum<np.cos(np.deg2rad(cfg.get('rider_heading_gate_deg',25))):raise ValueError('rider_heading_not_aligned_with_road')
                camera_parameters=candidate.get('camera_parameters')
                if camera_parameters is None:poses,lenses=baseline_poses,baseline_lenses
                else:
                    origin=[camera_parameters['origin_x_m'],camera_parameters['origin_y_m'],ep['rig']['origin_m'][2]]
                    poses,lenses,_=rig_poses(origin,camera_parameters['yaw_deg'],ep['rig']['baseline_m'])
                    original_relative=np.linalg.inv(baseline_poses['cam2'])@baseline_poses['cam4']
                    if not np.allclose(np.linalg.inv(poses['cam2'])@poses['cam4'],original_relative,atol=1e-9):
                        raise ValueError('rig_relative_extrinsics_changed')
                    original_lens_relative=np.linalg.inv(baseline_lenses['cam2'])@baseline_lenses['cam4']
                    if not np.allclose(np.linalg.inv(lenses['cam2'])@lenses['cam4'],original_lens_relative,atol=1e-9):
                        raise ValueError('physical_lens_relative_extrinsics_changed')
                # A reversed episode starts at the old path's end, with time
                # still increasing. Rig Y translation carries the road layout
                # with it; rig X translation changes the observation angle.
                shift_y=(camera_parameters['origin_y_m']-ep['rig']['origin_m'][1]) if camera_parameters else 0.
                if cfg.get('camera_moves_without_scene_translation',False):
                    shift_y=0.
                start_y=q['target_y_m']+shift_y+(3*q['cycling_speed_m_s'] if direction_sign==-1 else 0.)
                check['actual_target_start_xy_m']=[q['target_x_m'],start_y]
                cm=scene_trajectory(ep['episode_timeline'],[q['target_x_m'],start_y,cfg.get('placement_height_m',1.2)],direction,q['cycling_speed_m_s'],cycleR)
                if single:mats=cm[:,None]
                else:
                    # Walker owns its published source-root progression. Its
                    # crossing direction is independent of the rider's route.
                    angle=np.deg2rad(cfg['walker_fixed_rotation_deg'])
                    rot=np.array([[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]])@oldwalkR
                    M=np.eye(4);M[:3,:3]=rot;M[:3,3]=np.array([q['walker_x_m'],q['walker_y_m']+shift_y,lift])-rot@relative
                    mats=np.stack([cm,np.repeat(M[None],ep['frame_count'],axis=0)],axis=1)
                worldparts=[];roots=[]
                for j,d in enumerate(arrays):
                    R=mats[:,j,:3,:3];t=mats[:,j,None,:3,3]
                    parts={n:v@R.transpose(0,2,1)+t for n,v in d.items() if n in ('body','cloth','dense')}
                    hair=d['hair_corners'][None]@d['head'][:,:3,:3].transpose(0,2,1)+d['head'][:,None,:3,3]
                    parts['groom']=hair@R.transpose(0,2,1)+t;worldparts.append(parts)
                    roots.append(parts['body'].mean(1))
                    clearance=parts['body'][:,:,2].min(1)
                    if j==0 and ((clearance<.05).any() or (clearance>.7).any()):raise ValueError('cycling_fixed_height_clearance')
                    if j==1 and ((clearance<-.02).any() or (clearance>.12).any()):raise ValueError('walking_source_root_clearance')
                bounds=np.stack([np.concatenate([np.minimum(p['body'].min(1),p['cloth'].min(1))[:,:2],np.maximum(p['body'].max(1),p['cloth'].max(1))[:,:2]],axis=1) for p in worldparts],axis=1)
                road=cfg['road_bounds_xyxy']
                if (bounds[:,:,:2]<road[:2]).any() or (bounds[:,:,2:]>road[2:]).any():raise ValueError('outside_road')
                if not single:
                    if cfg.get('separation_policy')=='aabb_then_vertex_3d_v1':
                        check.update(checked_mixed_separation(bounds,np.stack(roots,axis=1),worldparts))
                    else:
                        check.update(trajectory_separation(bounds,np.stack(roots,axis=1),clearance=.1,min_root_distance=.9))
                curves={c:[] for c in poses};regions=[];expected=[]
                for k in range(ep['frame_count']):
                    views={};ins={};supports=[];overlaps={};reg={};insidepeople=[]
                    for c,pose in poses.items():
                        xy,_,distance,_,ins[c]=point_in_view(worldparts[0]['dense'][k],pose,c,masks[c]);parts={};hull=None;people_inside=[]
                        for j,p in enumerate(worldparts):
                            for n in ['body','cloth','groom']:parts[f'{j}_{n}']=project_erp(p[n][k],pose,5760,2880)[0]
                            people_inside.append(point_in_view(p['dense'][k],pose,c,masks[c])[-1])
                            if j==1:
                                # Per point central foreground opportunity; body hull proxy only.
                                hull=cv2.convexHull(parts['1_body'].astype(np.float32));front=np.linalg.norm(p['body'][k]-pose[:3,3],axis=1).max()<distance
                                overlaps[c]=front&np.array([cv2.pointPolygonTest(hull,tuple(map(float,z)),False)>=0 for z in xy])
                        try:
                            reg[c]=plan_region(parts)
                        except ValueError as exc:
                            lo=np.min(np.concatenate(list(parts.values())),axis=0)
                            hi=np.max(np.concatenate(list(parts.values())),axis=0)
                            boxes={name:np.r_[xy.min(0),xy.max(0)].round(2).tolist() for name,xy in parts.items()}
                            raise ValueError(f'{exc}:frame={k}:camera={c}:projected_xyxy={[float(lo[0]),float(lo[1]),float(hi[0]),float(hi[1])]}:parts={boxes}') from exc
                        x0,y0,x1,y1=reg[c]['roi_xyxy']
                        if (x1-x0)*(y1-y0)>cfg['max_union_roi_pixels']:raise ValueError('union_ROI_pixel_budget')
                        bv,_,_,_,bi=point_in_view(np.concatenate([worldparts[0]['body'][k],worldparts[0]['cloth'][k]]),pose,c,masks[c]);b=bv[bi]
                        if len(b):
                            box=np.r_[b.min(0),b.max(0)]
                            center,scale=crop_from_box(box,image_size_wh=(5760,2880),crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
                            supports.append(ins[c]&(np.abs(xy-center)<scale*100).all(1))
                        elif cfg.get('observation_policy')=='record_unavailable_v1':
                            # No physical-domain body is an empty observation,
                            # not a corrupt frame. A native crop does not exist.
                            box=center=scale=None
                            supports.append(np.zeros(len(xy),dtype=bool))
                        else:raise ValueError('no_target_body_domain')
                        yi=np.clip(np.floor(xy[:,1]).astype(int),0,2879);xi=np.floor(xy[:,0]).astype(int)%5760;dist=signed[c][yi,xi]
                        views[c]=dict(in_view=int(ins[c].sum()),outside_domain=int((~ins[c]).sum()),signed_boundary_distance_px=dist.tolist(),body_domain_fraction=float(bi.mean()),body_bounds=np.r_[bv.min(0),bv.max(0)].tolist(),projected_box=box.tolist() if box is not None else None,crop_center=center.tolist() if center is not None else None,crop_scale=float(scale) if scale is not None else None,roi_xyxy=reg[c]['roi_xyxy'],roi_pixels=(x1-x0)*(y1-y0))
                        if 'requested_view_class' in cfg:
                            angle=view_angle(mats[k,0,:3,3],headings[k],pose[:3,3])
                            views[c].update(view_angle_deg=angle,view_class=classify_view(angle))
                        insidepeople.append(people_inside)
                    common=ins['cam2']&ins['cam4'];commoncrop=int(np.logical_and(*supports).sum())
                    row=dict(frame_index=k,common_physical_points=int(common.sum()),common_projected_crop_points=commoncrop,views=views)
                    if 'requested_view_class' in cfg:
                        midpoint=(poses['cam2'][:3,3]+poses['cam4'][:3,3])/2
                        mid_angle=view_angle(mats[k,0,:3,3],headings[k],midpoint)
                        row['rig_midpoint_view']=dict(angle_deg=mid_angle,view_class=classify_view(mid_angle))
                    for c in poses:
                        if not single:
                            curves[c].append(float((overlaps[c]&common).sum()/max(1,common.sum())));views[c]['hull_overlap_common_fraction']=curves[c][-1]
                    check['frames'].append(row);regions.append(reg);expected.append(np.array(insidepeople).transpose(1,0,2))
                check['dual_edge']=edge_summary(check['frames'],ep['episode_timeline'],cfg.get('min_dual_edge_times'))
                if cfg.get('require_dual_edge',True) and not check['dual_edge']['passed']:
                    raise ValueError('simultaneous_dual_edge_gate')
                if 'requested_view_class' in cfg:
                    view_support=(check['dual_edge']['flags'] if cfg.get('view_requires_dual_edge',True) else
                                  [r['common_physical_points']>=8 for r in check['frames']])
                    check['view_coverage']=view_coverage(check['frames'],view_support,cfg['requested_view_class'],
                        minimum_count=cfg.get('min_requested_view_count',16))
                    if not check['view_coverage']['passed']:raise ValueError('requested_view_class_gate')
                if single:
                    # An edge pass must move across the physical boundary, not
                    # merely remain partly truncated for the whole sequence.
                    first,last=check['frames'][0],check['frames'][-1]
                    crossing={c:dict(first_outside=first['views'][c]['outside_domain'],last_outside=last['views'][c]['outside_domain'],
                        first_distance_p10=float(np.percentile(first['views'][c]['signed_boundary_distance_px'],10)),
                        last_distance_p10=float(np.percentile(last['views'][c]['signed_boundary_distance_px'],10))) for c in poses}
                    check['boundary_crossing']=crossing
                    policy=cfg.get('crossing_transition','enter_valid')
                    check['crossing_transition']=crossing_transition(crossing,'either')
                    check['boundary_transition_passed']=crossing_transition(crossing,policy) is not None
                    if coverage_goal(cfg,True)=='boundary_transition' and not check['boundary_transition_passed']:
                        raise ValueError('boundary_crossing_not_proved')
                if cfg.get('observation_policy')!='record_unavailable_v1' and any(r['common_projected_crop_points']<8 for r,f in zip(check['frames'],check['dual_edge']['flags']) if f):raise ValueError('dual_edge_projected_crop_common_below8')
                events={c:three_stage_event(v,delta=.15,consecutive=3) for c,v in curves.items()} if not single else {};check['projected_overlap_events']=events
                if cfg.get('observation_policy')!='record_unavailable_v1' and not any(events.values()):raise ValueError('no_projected_foreground_rise_recovery')
                if chosen is None:
                    # Exact active groom geometry uses the same planner as production.
                    for k in range(ep['frame_count']):
                        for c,pose in poses.items():
                            parts={}
                            for j,d in enumerate(arrays):
                                for n in ['body','cloth']:parts[f'{j}_{n}']=project_erp(worldparts[j][n][k],pose,5760,2880)[0]
                                h=d['hair']@d['head'][k,:3,:3].T+d['head'][k,:3,3];h=h@mats[k,j,:3,:3].T+mats[k,j,:3,3]
                                parts[f'{j}_groom']=project_erp(h,pose,5760,2880)[0]
                            regions[k][c]=plan_region(parts);r=regions[k][c]['roi_xyxy']
                            if (r[2]-r[0])*(r[3]-r[1])>cfg['max_union_roi_pixels']:raise ValueError('exact_groom_ROI_budget')
                    chosen=(candidate,seedset['master_seed'],mats,worldparts,regions,np.array(expected),check,poses,lenses)
            except ValueError as e:check['rejections'].append(str(e))
            result['attempts'].append(check);write(out/'candidate_checks.json',results+[result])
            if not check['rejections']:break
        results.append(result)
    write(out/'candidate_checks.json',results)
    if chosen is None:
        write(out/'geometry_preflight.json',dict(status='rejected',reason='no_bounded_candidate',seconds=time.monotonic()-begin));return None
    candidate,seed,mats,worldparts,regions,expected,check,poses,lenses=chosen
    p=ep['people'][0];p.update(scene_from_model=mats[:,0].tolist(),road_start_m=mats[0,0,:3,3].tolist(),speed_m_s=candidate['parameters']['cycling_speed_m_s'])
    if not single:ep['people'][1].update(model_to_world=mats[0,1].tolist(),fixed_ground_lift_m=lift,source_start_frame=ep['people'][1]['source_frames'][0],source_nominal_seconds=[v/30 for v in ep['people'][1]['source_frames']],road_start_m=[candidate['parameters']['walker_x_m'],candidate['parameters']['walker_y_m']+(0. if cfg.get('camera_moves_without_scene_translation',False) else candidate.get('camera_parameters',{}).get('origin_y_m',ep['rig']['origin_m'][1])-ep['rig']['origin_m'][1]),0],direction_yaw_deg=cfg['walker_fixed_rotation_deg'],actual_ground_clearance_range_m=[float(worldparts[1]['body'][:,:,2].min()),float(worldparts[1]['body'][:,:,2].min(1).max())])
    for k in range(ep['frame_count']):frame_record(ep,k)
    ep.update(master_seed=seed,child_seed=candidate['child_seed'],selected_attempt=candidate['attempt'],parameter_ranges=cfg['parameter_ranges'],sampled_parameters=candidate['parameters'],geometry_acceptance='passed',predicted_overlap_events=check['projected_overlap_events'],
        scene_matrices_ref=str((out/'scene_matrices.npy').resolve()),episode_timeline_ref=str((out/'episode_timeline.json').resolve()),
        person_timeline_refs={str(j+1):str((out/f'person{j+1}_timeline.json').resolve()) for j in range(len(assets))},source_timeline_refs={str(j+1):str((out/f'person{j+1}_source_timeline.json').resolve()) for j in range(len(assets))},
        expected_in_view_ref=str((out/'expected_in_view.npy').resolve()),randomized_dimensions=list(cfg['parameter_ranges']),road_bounds_xyxy=cfg['road_bounds_xyxy'],road_height_m=0)
    if candidate.get('camera_parameters') is not None:
        ep['rig'].update(origin_m=[candidate['camera_parameters']['origin_x_m'],candidate['camera_parameters']['origin_y_m'],ep['rig']['origin_m'][2]],yaw_deg=candidate['camera_parameters']['yaw_deg'])
        ep['camera_child_seed']=candidate['camera_child_seed']
        ep['sampled_camera_parameters']=candidate['camera_parameters']
    if 'requested_view_class' in cfg:
        ep['rig'].update(origin_m=[candidate['camera_parameters']['origin_x_m'],candidate['camera_parameters']['origin_y_m'],ep['rig']['origin_m'][2]],yaw_deg=candidate['camera_parameters']['yaw_deg'])
        ep.update(actual_view_coverage=check['view_coverage'],view_angles_by_frame=[{c:dict(angle_deg=r['views'][c]['view_angle_deg'],view_class=r['views'][c]['view_class']) for c in poses} for r in check['frames']],
                  travel_direction_sign=candidate['travel_direction_sign'],road_direction_xy=[0,candidate['travel_direction_sign']],camera_child_seed=candidate['camera_child_seed'],
                  sampled_camera_parameters=candidate['camera_parameters'],actual_crossing_transition=check.get('crossing_transition'))
    if 'coverage_goal' in cfg:
        ep['coverage_goal']=coverage_goal(cfg,single)
        ep['boundary_transition_passed']=check.get('boundary_transition_passed')
    np.save(out/'scene_matrices.npy',mats);np.save(out/'world_densekp.npy',np.stack([p['dense'] for p in worldparts],axis=1));np.save(out/'expected_in_view.npy',expected)
    write(out/'episode_timeline.json',ep['episode_timeline']);write(out/'planned_regions.json',regions);write(out/'selected_projection.json',check)
    write(out/'rig.json',dict(calibration_target='20260320-riding-dataset3',baseline_kind='synthetic_not_measured',cameras=[dict(id=c,camera_to_world=p.tolist(),physical_lens_to_world=lenses[c].tolist()) for c,p in poses.items()]))
    (out/'calibration').mkdir()
    for c,m in masks.items():Image.fromarray(m.astype(np.uint8)*255).save(out/'calibration'/f'{c}_valid_mask.png')
    separation_check=('one scene-owned rider; no neighbor separation needed' if single else
        'XY AABB broad phase; full body/cloth vertex 3D narrow phase where overlapping' if cfg.get('separation_policy')=='aabb_then_vertex_3d_v1' else 'disjoint body/cloth AABBs')
    write(out/'geometry_preflight.json',dict(status='passed',seconds=time.monotonic()-begin,coverage_goal=coverage_goal(cfg,single),dual_edge=check['dual_edge'],boundary_crossing=check.get('boundary_crossing'),boundary_transition_passed=check.get('boundary_transition_passed'),view_coverage=check.get('view_coverage'),rider_heading_min_cos=minimum if 'rider_heading_gate_deg' in cfg else None,checks=f'per-person original records; {"one" if single else "two"} root policies; {separation_check}; exact groom union ROI; physical-domain counts; projected crop support',limitations='foreground hull is not raster occlusion GT; vertex spacing is not a full mesh-triangle intersection test'))
    write(out/'recipe.json',ep);return ep
