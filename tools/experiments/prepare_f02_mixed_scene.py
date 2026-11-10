"""Freeze the accepted F02 + M04 asset inputs for the public mixed planner.

This prepares metadata only. Geometry preflight and rendering use existing
plan_road_pass.py and run_continuous_roi.py entry points.
"""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import cv2

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'outputs/experiments/mixed_walk_cycling_edge_a1_20261009'
F02=ROOT/'outputs/experiments/mtb_f02_dressed_asset_a2_20261009/a1_asset_adapter.json'


def prepare(out,seed):
    out=Path(out).resolve();out.mkdir(parents=True,exist_ok=False)
    adapter=json.loads(F02.read_text());old=json.loads((PARENT/'config.json').read_text())
    if adapter['asset_id']!='cycling_F02_v3_bikefit_XBLMidr4dps_8662_8782_simple_v1' or adapter['frame_count']!=120:
        raise ValueError('unaccepted_F02_asset_reference')
    with np.load(adapter['parameters']) as p:global_orient=p['global_orient']
    source=json.loads(Path(adapter['timeline_ref']).read_text())
    assert len(source)==120 and all(source[i]['source_time_tick']-source[i-1]['source_time_tick']==1001 for i in range(1,120))
    C=np.array([[1,0,0],[0,0,-1],[0,1,0]],float)
    # Fixed model +Z anatomical front transformed by already-baked global pose;
    # only model Y-up -> scene Z-up and a road yaw belong to this scene.
    selected=[0]+[min(range(120),key=lambda i:abs((source[i]['source_time_tick']-source[0]['source_time_tick'])/30000-k/10)) for k in range(1,31)]
    front=np.array([C@cv2.Rodrigues(global_orient[i])[0]@np.array([0.,0.,1.]) for i in selected])
    yaw=float(np.degrees(np.arctan2(np.median(front[:,0]),np.median(front[:,1]))))
    co,si=np.cos(np.deg2rad(yaw)),np.sin(np.deg2rad(yaw));R=np.array([[co,-si,0],[si,co,0],[0,0,1]])@C
    vertices=np.load(adapter['body_vertices'],mmap_mode='r');mins=np.array([vertices[i,:,1].min() for i in selected]);height=1.10
    clear=mins+height
    if not (.05<=clear.min() and clear.max()<=.7):raise ValueError('F02_fixed_height_body_clearance')
    base=json.loads(Path(old['reference_recipes'][0]).read_text());person=base['people'][0]
    person.clear();person.update(person_id=1,role='target',asset_id=adapter['asset_id'],asset_manifest=str(F02.resolve()),character_family=adapter['character_family'],
        root_mode=adapter['root_mode'],support_policy=adapter['support_policy'],placement_height_m=height,
        support_checks=copy.deepcopy(old['support_checks']),speed_policy='scene_constant_metres_per_second_not_source_speed',
        outfit_policy='fixed_accepted_combination_entire_episode')
    person['support_checks']['pelvis_height_above_road_m']=[1.05,1.15]
    base['people']=[person];base['episode_id']='f02_m04_prepared_reference_only';base['sequence_id']=base['episode_id']
    (out/'f02_base_recipe.json').write_text(json.dumps(base,indent=2,allow_nan=False)+'\n')
    cfg=copy.deepcopy(old);cfg.update(episode_id='f02_m04_road_pass_00',master_seed=seed,planning_seeds=[seed],attempt_limit=12,
        asset_proposal=str(F02.resolve()),reference_recipes=[str((out/'f02_base_recipe.json').resolve()),old['reference_recipes'][1]],
        observation_policy='record_unavailable_v1',roi_planning_policy='per_person_square_union_v1',placement_height_m=height,fixed_scene_rotation=R.tolist(),fixed_scene_yaw_deg=yaw,
        source_origin_records=[0,0],parameter_ranges=dict(target_x_m=[-.085,-.055],target_y_m=[.90,1.05],cycling_speed_m_s=[.44,.48],
            walker_x_m=[-1.80,-1.55],walker_y_m=[-.25,.05]),
        range_rationale='F02 model+Z body forward after its baked global_orient, fixed scene Y-up->Z-up then yaw; x/y tightened from physical-domain projection near old road/rig. 0.44..0.48m/s synthetic constant progression, walker source-root only; no model error selection.')
    cfg['support_checks']['pelvis_height_above_road_m']=[1.05,1.15]
    cfg['occlusion_metric']['window_rule']='three adjacent synthetic10Hz selected episode records per stage; same 15/15pp conservative rule'
    (out/'config.json').write_text(json.dumps(cfg,indent=2,allow_nan=False)+'\n')
    (out/'orientation_and_height.json').write_text(json.dumps(dict(asset_ref=str(F02.resolve()),adapter_status='asset adapter validated; scene use selected by A1',
        yaw_degrees=yaw,scene_rotation=R.tolist(),Y_up_to_Z_up=C.tolist(),
        source_front_model_Z_after_global_orient_in_scene_before_yaw=[v.tolist() for v in front],
        forward_rule='median source model+Z after already baked global_orient; align horizontal component with road +Y',
        camera_to_asset_rotation_reapplied=False,selected_asset_indices=selected,fixed_placement_height_m=height,
        body_model_minY_m=[float(mins.min()),float(mins.max())],fixed_scene_clearance_m=[float(clear.min()),float(clear.max())],
        M01_height_or_yaw_inherited=False),indent=2,allow_nan=False)+'\n')
    return cfg

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);parser.add_argument('--seed',type=int,default=20261010)
    args=parser.parse_args();prepare(args.output,args.seed)
