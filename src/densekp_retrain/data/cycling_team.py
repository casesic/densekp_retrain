"""Bounded, per-person cycling-team geometry for the shared ROI renderer.

This module selects no geometry assets and renders nothing.  It produces the
same recipe, matrices and projection files consumed by run_continuous_roi.py.
"""
from __future__ import annotations

import copy
import json
import random
from pathlib import Path

import numpy as np

from .dynamic_scene import geometry_paths, scene_trajectory, frame_record
from .mixed_road_pass import EDGE, edge_summary
from .occlusion_events import three_stage_event
from .person_time import map_timeline, SCHEMA as TIME_SCHEMA
from .dynamic_preflight import write

VERSION = 'cycling_team_boundary_pass_v1'
CLOCK = [dict(frame_index=k, episode_time_tick=k, time_base=[1, 10]) for k in range(31)]


def team_recipe(reference, people, origins, episode_id, environment):
    if not 2 <= len(people) <= 5 or len(people) != len(origins):
        raise ValueError('team_person_count_or_origin_conflict')
    if len({p['character_family'] for p in people}) != len(people):
        raise ValueError('duplicate_identity_family')
    if len({p['source_group'] for p in people}) < min(3, len(people)):
        raise ValueError('insufficient_distinct_motion_source_groups')
    ep = copy.deepcopy(reference)
    for key in ('selected_timeline', 'selected_timeline_ref', 'time_origin',
                'scene_matrices_ref', 'selection', 'original_timeline_count',
                'original_timeline_ref', 'unselected_asset_frames',
                'parent_recipe_ref', 'geometry_attempts'):
        ep.pop(key, None)
    records = {}; sources = {}
    for i, (person, origin) in enumerate(zip(people, origins), 1):
        source = json.loads(Path(json.loads(Path(person['asset_manifest']).read_text())['timeline_ref']).read_text())
        mapped = map_timeline(source, CLOCK, origin_record=origin)
        person.update(person_id=i, role='target' if i == 1 else 'occluder',
                      asset_frames=[r['asset_frame'] for r in mapped],
                      source_frames=[r['source_frame'] for r in mapped])
        person.pop('model_to_world', None)
        person.pop('scene_from_model', None)
        sources[str(i)] = source
        records[str(i)] = mapped
    ep.update(episode_id=episode_id, sequence_id=episode_id,
              generator_version=VERSION, scene_mode='cycling_team_boundary_pass',
              environment_id=environment, frame_count=31, time_base=[1, 10],
              episode_timeline=copy.deepcopy(CLOCK), person_timelines=records,
              time_mapping_schema=TIME_SCHEMA, people=people, target_person_id=1,
              scene_matrix_person_ids=list(range(1, len(people)+1)),
              reference_recipe_refs=[p['reference_recipe'] for p in people],
              same_asset_instances_not_new_identities=False,
              integration_only=True, observation_policy='record_unavailable_v1',
              roi_planning_policy='per_person_square_union_v1',
              coverage_goal='edge_coverage', dual_edge_gate=EDGE,
              coverage_thresholds=EDGE, readback_policy='first_middle_last',
              sampling_policy='synthetic10Hz_nearest_original_record_ties_earlier_no_interpolation',
              time_unit_status='synthetic10Hz; each person keeps original integer source time',
              source_group_policy='all_people_in_episode; seeds_are_not_independent_source_splits')
    for k in range(31):
        frame_record(ep, k)
    return ep, sources


def candidate_parameters(master_seed, episode_index, count, ranges, *, limit=12):
    if count not in (2, 3) or not 1 <= limit <= 12:
        raise ValueError('invalid_team_candidate_limit')
    rng = random.Random(f'{VERSION}|{master_seed}|{episode_index}|layout')
    result = []
    for attempt in range(limit):
        child = rng.getrandbits(63)
        local = random.Random(child)
        values = {key: local.uniform(*bounds) for key, bounds in sorted(ranges.items())}
        result.append(dict(attempt=attempt, child_seed=child, parameters=values))
    return result


def relative_layout_candidates(anchor, master_seed, *, limit=32, cells_config=None):
    """Freeze bounded person-2 offsets relative to an accepted person-1/rig path.

    The structured set keeps the target and rig together; the seed selects a
    reproducible order and small, bounded offsets within each cell.  No CPU
    projection or renderer output influences the candidate list.
    """
    if not 1 <= limit <= 32 or type(master_seed) is not int:
        raise ValueError('invalid_relative_layout_limit_or_seed')
    needed = ('rig_x_m','rig_y_m','rig_yaw_deg','p1_x_m','p1_y_m','p1_speed_m_s')
    if any(k not in anchor or not np.isfinite(anchor[k]) for k in needed):
        raise ValueError('invalid_relative_layout_anchor')
    spec = cells_config or dict(forward_gap_m=[1.4,1.7,2.0,2.3],
        lateral_offset_m=[-.34,-.18,.18,.34],speed_difference_m_s=[-.05,.05],
        jitter_half_widths=dict(gap_m=.035,lateral_m=.02,speed_difference_m_s=.01))
    jitter = spec['jitter_half_widths']
    cells = [(gap, lateral, dv) for gap in spec['forward_gap_m']
             for lateral in spec['lateral_offset_m'] for dv in spec['speed_difference_m_s']]
    if limit>len(cells) or any(not np.isfinite(v) for cell in cells for v in cell) or any(
            not np.isfinite(v) or v<0 for v in jitter.values()):
        raise ValueError('invalid_relative_layout_cells')
    rng = random.Random(f'cycling_team_relative_layout_v1|{master_seed}|order')
    rng.shuffle(cells)
    candidates = []
    for attempt, (gap, lateral, dv) in enumerate(cells[:limit]):
        child = rng.getrandbits(63)
        local = random.Random(child)
        actual_gap = gap + local.uniform(-jitter['gap_m'], jitter['gap_m'])
        actual_lateral = lateral + local.uniform(-jitter['lateral_m'], jitter['lateral_m'])
        actual_dv = dv + local.uniform(-jitter['speed_difference_m_s'], jitter['speed_difference_m_s'])
        q = dict(anchor, p2_x_m=anchor['p1_x_m']+actual_lateral,
                 p2_y_m=anchor['p1_y_m']+actual_gap,
                 p2_speed_m_s=anchor['p1_speed_m_s']+actual_dv)
        candidates.append(dict(attempt=attempt, child_seed=child, parameters=q,
              relative_parameters=dict(p2_minus_p1_start_x_m=actual_lateral,
              p2_minus_p1_start_y_m=actual_gap, p2_minus_p1_speed_m_s=actual_dv),
              structured_cell=dict(gap_m=gap,lateral_m=lateral,speed_difference_m_s=dv)))
    return candidates


def projected_occlusion_gate(events, policy='require_event'):
    if policy not in ('require_event', 'diagnostic_only'):
        raise ValueError('invalid_projected_occlusion_policy')
    return policy == 'diagnostic_only' or any(events.values())


def projected_support_diagnostic(grouped, image_size_wh=(5760, 2880)):
    """Describe a rejected conservative support region; no raster box is implied."""
    out = {}
    width, height = image_size_wh
    for pid, parts in grouped.items():
        xy = np.concatenate(list(parts.values()), axis=0)
        if xy.size == 0 or not np.isfinite(xy).all():
            out[str(pid)] = dict(invalid_projected_geometry=True)
            continue
        lo = xy.min(0)-2; hi = xy.max(0)+2
        extent = hi-lo; pad = extent[::-1]/2
        square = np.r_[np.floor(lo-pad-32), np.ceil(hi+pad+32)].astype(int)
        out[str(pid)] = dict(projected_bounds_xyxy=np.r_[xy.min(0),xy.max(0)].tolist(),
             conservative_square_xyxy=square.tolist(),
             seam_crossing=bool(np.ptp(xy[:,0])>=width/2),
             outside_full_raster=dict(left=bool(square[0]<0), top=bool(square[1]<0),
                                      right=bool(square[2]>width), bottom=bool(square[3]>height)))
    return out


def _transform(points, matrix):
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def heading_angle_degrees(asset, indices, rotation):
    from scipy.spatial.transform import Rotation
    with np.load(asset['parameters'], allow_pickle=False) as value:
        orient=value['global_orient'][indices]
    forward=Rotation.from_rotvec(orient).apply([0,0,1]) @ np.asarray(rotation)[:3,:3].T
    length=np.linalg.norm(forward[:,:2],axis=1)
    if not np.isfinite(forward).all() or (length<1e-9).any():
        raise ValueError('degenerate_cyclist_heading')
    return np.rad2deg(np.arccos(np.clip(forward[:,1]/length,-1,1)))


def preflight_team(recipe, source_timelines, assets, placements, candidates, scene, output,
                   *, max_roi_pixels=7_000_000, projected_occlusion_policy='require_event'):
    """Check all persons and all 31 frames; write one shared-renderer prepared episode.

    ``placements`` is one fixed rotation/height per person, from the actual
    selected body/motion.  The CPU hull overlap is only a screen; raster ID and
    range determine the eventual neighbor event.
    """
    import cv2
    from PIL import Image
    from densekp_retrain.camera.rig import rig_poses
    from densekp_retrain.camera.domain import point_in_view
    from densekp_retrain.camera.erp import project_erp
    from densekp_retrain.imaging.roi_planning import plan_person_crop_union
    from .multiperson_road_pass import trajectory_separation

    if projected_occlusion_policy not in ('require_event', 'diagnostic_only'):
        raise ValueError('invalid_projected_occlusion_policy')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    count = len(assets)
    arrays = []
    for i, (p, asset, placement) in enumerate(zip(recipe['people'], assets, placements), 1):
        paths = geometry_paths(asset); ix = p['asset_frames']
        arr = {name: np.load(paths[key], mmap_mode='r')[ix] for name, key in
               [('body', 'body_vertices'), ('cloth', 'cloth_vertices'),
                ('dense', 'densekp'), ('head', 'head_transforms')]}
        if any(not np.isfinite(v).all() for v in arr.values()):
            raise ValueError('nonfinite_asset_geometry')
        groom = Path(asset['groom_reference'])
        points = np.load(groom/'points.npy', mmap_mode='r')
        active = np.repeat(np.load(groom/'active_curve_mask.npy'), np.diff(np.load(groom/'curve_offsets.npy')))
        arr['hair'] = points[active]
        lo, hi = arr['hair'].min(0), arr['hair'].max(0)
        arr['hair_proxy'] = np.array([[x,y,z] for x in (lo[0],hi[0]) for y in (lo[1],hi[1]) for z in (lo[2],hi[2])])
        if not (np.max(np.abs(np.diff(arr['body'], axis=0)), axis=(1,2)) > 1e-5).all():
            raise ValueError('static_body_articulation')
        if np.asarray(placement['rotation']).shape != (3,3):
            raise ValueError('invalid_person_rotation')
        arrays.append(arr)
        write(output/f'person{i}_source_timeline.json', source_timelines[str(i)])
        write(output/f'person{i}_timeline.json', recipe['person_timelines'][str(i)])
    masks = {c: np.asarray(Image.open(path)) > 0 for c, path in scene['physical_masks'].items()}
    signed = {c: cv2.distanceTransform(m.astype('uint8'), cv2.DIST_L2, 5) -
                 cv2.distanceTransform((~m).astype('uint8'), cv2.DIST_L2, 5) for c,m in masks.items()}
    attempts = []; selected = None
    for candidate in candidates:
        q = candidate['parameters']; reasons = []; frames = []; regions = []; expected = []
        edge = None; events = None; overlap_curves = None
        failure_context = dict(stage='initial', frame_index=None, camera_id=None, person_id=None)
        try:
            origin = [q['rig_x_m'], q['rig_y_m'], scene['rig_origin_z_m']]
            poses, lenses, _ = rig_poses(origin, q['rig_yaw_deg'], scene['baseline_m'])
            matrices = []
            for j, placement in enumerate(placements):
                # Road-local forward is +Y.  Different speed and longitudinal
                # position create entry order; no fake pre-entry motion frame.
                start = [q[f'p{j+1}_x_m'], q[f'p{j+1}_y_m'], placement['height_m']]
                matrices.append(scene_trajectory(CLOCK, start, [0,1],
                                  q[f'p{j+1}_speed_m_s'], placement['rotation']))
            matrices = np.stack(matrices, axis=1)
            for j,(person,asset) in enumerate(zip(recipe['people'],assets)):
                if heading_angle_degrees(asset,person['asset_frames'],matrices[0,j]).max()>25:
                    raise ValueError(f'p{j+1}_heading_not_aligned_with_road')
            worldparts = []
            for j, arr in enumerate(arrays):
                R = matrices[:,j,:3,:3]; t = matrices[:,j,None,:3,3]
                parts = {key: values @ R.transpose(0,2,1) + t
                         for key, values in arr.items() if key in ('body','cloth','dense')}
                h = arr['hair_proxy'][None] @ arr['head'][:,:3,:3].transpose(0,2,1) + arr['head'][:,None,:3,3]
                parts['groom_proxy'] = h @ R.transpose(0,2,1) + t
                clearance = parts['body'][:,:,2].min(1)
                if (clearance < .05).any() or (clearance > .7).any():
                    raise ValueError(f'p{j+1}_fixed_cycling_clearance')
                worldparts.append(parts)
            bounds = np.stack([np.concatenate([np.minimum(p['body'].min(1), p['cloth'].min(1))[:,:2],
                           np.maximum(p['body'].max(1), p['cloth'].max(1))[:,:2]], axis=1)
                           for p in worldparts], axis=1)
            road = scene['road_bounds_xyxy']
            failure_context.update(stage='road_bounds', frame_index=None, camera_id=None)
            if (bounds[:,:,:2] < road[:2]).any() or (bounds[:,:,2:] > road[2:]).any():
                raise ValueError('body_or_cloth_outside_road')
            roots = matrices[:,:,:3,3]
            for a in range(count):
                for b in range(a+1,count):
                    failure_context.update(stage='pairwise_separation', person_id=[a+1,b+1])
                    pair = bounds[:,[a,b]]
                    trajectory_separation(pair, roots[:,[a,b]], clearance=.1, min_root_distance=.9)
            overlap_curves={c:[] for c in poses}
            for k in range(31):
                views = {}; inside = {}; per_camera = {}; per_inside = []; overlap={}
                for camera, pose in poses.items():
                    grouped = {}
                    person_inside = []
                    for j, parts in enumerate(worldparts):
                        grouped[str(j+1)] = {name: project_erp(parts[name][k], pose, 5760, 2880)[0]
                                             for name in ('body','cloth','groom_proxy')}
                        person_inside.append(point_in_view(parts['dense'][k], pose, camera, masks[camera])[-1])
                    failure_context.update(stage='proxy_union_ROI', frame_index=k, camera_id=camera)
                    try:
                        region = plan_person_crop_union(grouped, (5760,2880),
                            box_augmentation_policy=recipe.get('box_augmentation_policy'))
                    except ValueError:
                        failure_context['per_person_support'] = projected_support_diagnostic(grouped)
                        failure_context['rig_origin_xyz_m'] = pose[:3,3].tolist()
                        failure_context['person_root_xyz_m'] = {str(j+1):matrices[k,j,:3,3].tolist()
                                                                  for j in range(count)}
                        raise
                    x0,y0,x1,y1 = region['roi_xyxy']; pixels=(x1-x0)*(y1-y0)
                    if pixels > max_roi_pixels: raise ValueError('union_ROI_pixel_budget')
                    xy,_,distance,_, inside[camera] = point_in_view(worldparts[0]['dense'][k], pose, camera, masks[camera])
                    near=np.zeros(138,dtype=bool)
                    for j in range(1,count):
                        hull=cv2.convexHull(grouped[str(j+1)]['body'].astype(np.float32))
                        closer=np.linalg.norm(worldparts[j]['body'][k]-pose[:3,3],axis=1).max()<distance
                        near |= closer & np.array([cv2.pointPolygonTest(hull,tuple(map(float,point)),False)>=0 for point in xy])
                    overlap[camera]=near
                    ix=np.floor(xy[:,0]).astype(int)%5760; iy=np.clip(np.floor(xy[:,1]).astype(int),0,2879)
                    views[camera] = dict(in_view=int(inside[camera].sum()),
                                         outside_domain=int((~inside[camera]).sum()),
                                         signed_boundary_distance_p10_px=float(np.percentile(signed[camera][iy,ix],10)),
                                         roi_xyxy=region['roi_xyxy'], roi_pixels=pixels)
                    per_camera[camera]=region
                    per_inside.append(np.asarray(person_inside))
                common = inside['cam2'] & inside['cam4']
                for camera in poses:
                    overlap_curves[camera].append(float((overlap[camera]&common).sum()/common.sum()) if common.any() else 0.0)
                frames.append(dict(frame_index=k, views=views, common_physical_points=int(common.sum())))
                regions.append(per_camera)
                expected.append(np.stack(per_inside,axis=1))
            edge=edge_summary(frames,CLOCK)
            failure_context.update(stage='dual_edge_gate', frame_index=None, camera_id=None,
                                   person_id=recipe['target_person_id'])
            if not edge['passed']:raise ValueError('simultaneous_dual_edge_gate')
            events={c:three_stage_event(values,delta=.15,consecutive=3) for c,values in overlap_curves.items()}
            failure_context.update(stage='projected_occlusion_gate')
            if not projected_occlusion_gate(events, projected_occlusion_policy):
                raise ValueError('no_projected_neighbor_overlap_rise_recovery')
            # Replan the accepted candidate from actual active groom curves.
            for k in range(31):
                for camera, pose in poses.items():
                    grouped={}
                    for j, arr in enumerate(arrays):
                        M=matrices[k,j]; H=arr['head'][k]
                        h=_transform(_transform(arr['hair'],H),M)
                        grouped[str(j+1)] = dict(body=project_erp(worldparts[j]['body'][k],pose,5760,2880)[0],
                                                 cloth=project_erp(worldparts[j]['cloth'][k],pose,5760,2880)[0],
                                                 groom=project_erp(h,pose,5760,2880)[0])
                    failure_context.update(stage='exact_groom_union_ROI', frame_index=k, camera_id=camera)
                    try:
                        region=plan_person_crop_union(grouped,(5760,2880),
                            box_augmentation_policy=recipe.get('box_augmentation_policy'))
                    except ValueError:
                        failure_context['per_person_support'] = projected_support_diagnostic(grouped)
                        failure_context['rig_origin_xyz_m'] = pose[:3,3].tolist()
                        failure_context['person_root_xyz_m'] = {str(j+1):matrices[k,j,:3,3].tolist()
                                                                  for j in range(count)}
                        raise
                    x0,y0,x1,y1=region['roi_xyxy']
                    if (x1-x0)*(y1-y0)>max_roi_pixels:raise ValueError('exact_groom_union_ROI_budget')
                    regions[k][camera]=region
            selected=(candidate,matrices,worldparts,regions,np.asarray(expected),poses,lenses,edge,events)
        except ValueError as exc:
            reasons.append(str(exc))
        attempts.append(dict(**candidate, rejections=reasons, dual_edge=edge,
                             projected_overlap_events=events, projected_overlap_curves=overlap_curves,
                             frames=frames, failure_context=failure_context if reasons else None,
                             exact_groom_checked=selected is not None and not reasons))
        if not reasons:break
    write(output/'candidate_checks.json', dict(schema=VERSION,projected_occlusion_policy=projected_occlusion_policy,
          attempts=attempts,attempt_limit=len(candidates)))
    if selected is None:
        write(output/'geometry_preflight.json',dict(status='rejected',reason='no_bounded_team_layout'))
        return None
    candidate,matrices,worldparts,regions,expected,poses,lenses,edge,events=selected
    for j, (person, placement) in enumerate(zip(recipe['people'],placements)):
        person.update(scene_from_model=matrices[:,j].tolist(), placement_height_m=placement['height_m'],
                      support_checks=placement['support_checks'],
                      road_start_m=matrices[0,j,:3,3].tolist(),path_direction_xy=[0,1],
                      speed_m_s=candidate['parameters'][f'p{j+1}_speed_m_s'])
    recipe.update(rig=dict(recipe['rig'],origin_m=[candidate['parameters']['rig_x_m'],candidate['parameters']['rig_y_m'],scene['rig_origin_z_m']],
                                yaw_deg=candidate['parameters']['rig_yaw_deg']),
                  selected_attempt=candidate['attempt'],child_seed=candidate['child_seed'],
                  sampled_parameters=candidate['parameters'],
                  parameter_ranges=scene['parameter_ranges'],
                  randomized_dimensions=sorted(scene['parameter_ranges']),
                  scene_matrices_ref=str((output/'scene_matrices.npy').resolve()),
                  episode_timeline_ref=str((output/'episode_timeline.json').resolve()),
                  person_timeline_refs={str(i):str((output/f'person{i}_timeline.json').resolve()) for i in range(1,count+1)},
                  source_timeline_refs={str(i):str((output/f'person{i}_source_timeline.json').resolve()) for i in range(1,count+1)},
                  expected_in_view_ref=str((output/'expected_in_view.npy').resolve()),
                  max_union_roi_pixels=max_roi_pixels, geometry_acceptance='passed',
                  predicted_overlap_events=events, projected_occlusion_policy=projected_occlusion_policy,
                  environment_id=scene['environment_id'], road_bounds_xyxy=scene['road_bounds_xyxy'],
                  road_height_m=0)
    for k in range(31):frame_record(recipe,k)
    np.save(output/'scene_matrices.npy',matrices)
    np.save(output/'world_densekp.npy',np.stack([p['dense'] for p in worldparts],axis=1))
    np.save(output/'expected_in_view.npy',expected)
    write(output/'episode_timeline.json',CLOCK)
    write(output/'planned_regions.json',regions)
    write(output/'selected_projection.json',dict(frames=frames,dual_edge=edge,selected_attempt=candidate['attempt']))
    write(output/'rig.json',dict(calibration_target='20260320-riding-dataset3',baseline_kind='synthetic_not_measured',
         cameras=[dict(id=c,camera_to_world=p.tolist(),physical_lens_to_world=lenses[c].tolist()) for c,p in poses.items()]))
    (output/'calibration').mkdir()
    for c,m in masks.items():Image.fromarray(m.astype('uint8')*255).save(output/'calibration'/f'{c}_valid_mask.png')
    write(output/'geometry_preflight.json',dict(status='passed',dual_edge=edge,
          projected_occlusion_policy=projected_occlusion_policy, projected_overlap_events=events,
          checks='31 frames, every person original timeline, road, pairwise envelope, all-person square union and exact groom ROI',
          limitations='pairwise envelope is not continuous triangle collision; CPU overlap is not raster neighbor occlusion'))
    write(output/'recipe.json',recipe)
    return recipe
