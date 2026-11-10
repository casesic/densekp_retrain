"""Model/render-free recipe entry point; optional manifest geometry preflight."""
import argparse
import copy
import json
import math
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from densekp_retrain.data.road_pass import plan, longest_true_run

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')

def preflight(recipe, base):
    if recipe.get("algorithm_version")=="road_pass_mixed_sources_v1":
        from densekp_retrain.data.mixed_road_pass import preflight_mixed
        return preflight_mixed(recipe,base)
    if recipe.get('algorithm_version')=='road_pass_multiperson_scene_owned_v1':
        from densekp_retrain.data.multiperson_road_pass import preflight_multiperson
        return preflight_multiperson(recipe,base)
    if recipe.get('algorithm_version')=='road_pass_scene_owned_v1':
        from densekp_retrain.data.dynamic_preflight import preflight_scene_owned
        return preflight_scene_owned(recipe,base)
    import cv2
    from PIL import Image
    from densekp_retrain.camera.rig import rig_poses
    from densekp_retrain.camera.domain import point_in_view
    from densekp_retrain.data.boxes import crop_from_box
    masks={c:np.asarray(Image.open(ROOT/f'outputs/acceptance/ds_output_modes_20261008/attempt_4/compact/{c}/valid_mask.png'))>0 for c in ['cam2','cam4']}
    signed={c:cv2.distanceTransform(m.astype(np.uint8),cv2.DIST_L2,5)-cv2.distanceTransform((~m).astype(np.uint8),cv2.DIST_L2,5) for c,m in masks.items()}
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
    for ep in recipe['episodes']:
        out=base/ep['episode_id'];out.mkdir(exist_ok=False)
        poses,lenses,rigR=rig_poses(ep['rig']['origin_m'],ep['rig']['yaw_deg'],ep['rig']['baseline_m'])
        write(out/'rig.json',dict(calibration_target='20260320-riding-dataset3',baseline_kind='synthetic_not_measured',cameras=[dict(id=c,camera_to_world=p.tolist(),physical_lens_to_world=lenses[c].tolist()) for c,p in poses.items()]))
        people=[];allrecords=[]
        for person in ep['people']:
            asset=json.loads(Path(person['asset_manifest']).read_text());g=Path(asset['geometry']);frames=person['source_start_frame']+3*np.arange(31);ix=frames-asset['source_start_reference']
            v=np.load(g/'vertices.npy',mmap_mode='r')[ix];d=np.load(g/'densekp_3d.npy')[ix];p=np.load(asset['parameters'])
            relative=p['source_root_relative'][ix[0]]
            angle=math.radians(person['direction_yaw_deg']);yaw=np.array([[math.cos(angle),-math.sin(angle),0],[math.sin(angle),math.cos(angle),0],[0,0,1]])
            R=rigR@yaw@C
            # One constant ground lift for this complete clip; never per-frame min-Y.
            lift=-float((v[:,:,1]-relative[1]).min())
            t=np.array(ep['rig']['origin_m'])+rigR@(np.array(person['road_start_m'])+[0,0,lift])-R@relative
            M=np.eye(4);M[:3,:3]=R;M[:3,3]=t
            world=d@R.T+t;body=v@R.T+t
            person.update(asset_frames=ix.tolist(),source_frames=frames.tolist(),model_to_world=M.tolist(),fixed_ground_lift_m=lift,
                actual_ground_clearance_range_m=[float(body[:,:,2].min()),float(body[:,:,2].min(1).max())],
                source_nominal_seconds=(frames/30).tolist(),original_100fps_mapping=asset['original_100fps_mapping'])
            per=[]
            for k in range(31):
                views={};supports=[]
                for c,pose in poses.items():
                    xy,_,_,_,inside=point_in_view(world[k],pose,c,masks[c]);bv,_,_,_,binview=point_in_view(body[k],pose,c,masks[c])
                    q=bv[binview];box=np.r_[q.min(0),q.max(0)] if len(q) else None
                    if box is None:raise ValueError('Empty projected body before render')
                    center,scale=crop_from_box(box,image_size_wh=(5760,2880),crop_policy='deployment',boundary_policy='erp_reject_seam',min_side_px=2)
                    half=float(scale)*100;incrop=(np.abs(xy-center)<half).all(1);support=inside&incrop;supports.append(support)
                    yy=np.clip(np.floor(xy[:,1]).astype(int),0,2879);xx=np.floor(xy[:,0]).astype(int)%5760;distance=signed[c][yy,xx]
                    views[c]=dict(in_view_points=int(inside.sum()),out_of_view_fraction=float((~inside).mean()),
                        in_full_crop_points=int(support.sum()),projected_body_box=box.tolist(),crop_scale_pixels=float(scale*200),
                        valid_boundary_signed_distance_px=distance.tolist(),boundary_near_points_64px=int((np.abs(distance)<=64).sum()),
                        full_body_crop_truncated_points=int((~incrop).sum()),body_domain_fraction=float(binview.mean()))
                per.append(dict(frame_index=k,source_frame=int(frames[k]),time_tick=k,views=views,common_domain_and_projected_crop_points=int(np.logical_and(*supports).sum())))
            allrecords.append(dict(person_id=person['person_id'],frames=per));people.append(world)
        common=[r['common_domain_and_projected_crop_points'] for r in allrecords[0]['frames']]
        rejected=[]
        if longest_true_run(n>=8 for n in common)<3: rejected.append('target_fewer_than_3_consecutive_dual_supported_times')
        roots=[]
        for person in ep['people']:
            a=json.loads(Path(person['asset_manifest']).read_text());j=np.load(Path(a['geometry'])/'joints.npy')[person['asset_frames'],0];M=np.array(person['model_to_world']);roots.append(j@M[:3,:3].T+M[:3,3])
        separation=None
        if len(roots)>1:
            separation=min(float(np.linalg.norm(a[:,:2]-b[:,:2],axis=1).min()) for i,a in enumerate(roots) for b in roots[i+1:])
            if separation<.65:rejected.append('root_spacing_below_0.65m')
        # No resampling is currently needed. Failure is explicit and bounded.
        ep.update(geometry_acceptance='rejected' if rejected else 'passed',geometry_attempts=[dict(attempt=0,rejections=rejected)],minimum_root_spacing_m=separation,
            expected_occlusion_window_ticks=[0,30] if len(roots)>1 else None,
            expected_occlusion_status='geometric_alignment_candidate_not_visibility_verified')
        write(out/'geometry_preflight.json',dict(people=allrecords,thresholds=recipe['coverage_thresholds'],rejections=rejected,minimum_root_spacing_m=separation))
        np.save(out/'world_densekp.npy',np.stack(people,axis=1))
        write(out/'recipe.json',ep)
        if rejected:raise ValueError(f'{ep["episode_id"]}: {rejected}')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--geometry-preflight',action='store_true');a=ap.parse_args()
    config=json.loads(a.config.read_text());recipe=plan(config)
    recipe['coverage_thresholds']=config['coverage_thresholds'] if recipe.get('algorithm_version') in ('road_pass_scene_owned_v1','road_pass_multiperson_scene_owned_v1','road_pass_mixed_sources_v1') else dict(definition_frozen_before_inference=True,boundary_distance_units='final ERP raster pixels; not angle',edge_band_px=64,
        multi_view_target_min_points_each=8,multi_view_target_min_consecutive_times=3,
        truncation='at least one out-of-domain GT point, actual crop common GT >=1',
        occlusion='target visible/in_view fraction range >=0.10 in at least one view across multi-person episode',
        common_pair='reliable GT & in_view & inside_full_crop in each view, intersect by semantic ID',
        ground='one clip-wide constant lift; source vertical root motion preserved; measured foot clearance reviewed',
        min_root_separation_m=.65,actual_support_min_points=1,actual_target_mask_min_pixels=1,
        no_common_points='null; retain observation and denominator for coverage',prediction_nonfinite='failure; retain GT denominator')
    if a.geometry_preflight:preflight(recipe,a.output.parent)
    write(a.output,recipe)

if __name__=='__main__':main()
