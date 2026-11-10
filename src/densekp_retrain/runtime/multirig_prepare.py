"""Pure file/CPU preparation for frozen multirig ROI runs."""
import copy
import json
from pathlib import Path
import re
import shutil
import sys
import time

import numpy as np
from densekp_retrain.imaging.output_modes import prepare_source_workspace

ROOT=Path(__file__).resolve().parents[3]
EDGE=ROOT/'outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront'
SIDE=ROOT/'outputs/experiments/multirig_scene_render_a1_20261010/prepared_side'
TRIAL=ROOT/'outputs/experiments/direct_erp_roi_trial_20261009/frozen_comparison.json'
MAMMA_TEMPLATE=Path('/home/yy/Repos/mamma/data/body_models/downsampled_verts/verts_512.pkl')


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def prepare(output, *, gpu_uuid, budget_seconds, edge_prepared=EDGE, side_prepared=SIDE,
            rig_list_manifest=None, frame_indices=None, selected_rig_ids=None, preview=False,label_probe=False,
            mamma_512=False, qualification_registry=None):
    output=Path(output).resolve()
    if rig_list_manifest is None:
        # Historical two-rig invocation stays byte-for-byte equivalent in its
        # source choices and readback cap.
        rig_inputs=[dict(rig_id='rig_edge',prepared=str(edge_prepared),coverage_goal='rig_edge'),
                    dict(rig_id='rig_side',prepared=str(side_prepared),coverage_goal='side_view_control')]
        manifest={}
    else:
        manifest=json.loads(Path(rig_list_manifest).read_text())
        if manifest.get('schema')!='scene_episode_rig_list_v1':raise ValueError('invalid_rig_list_schema')
        rig_inputs=manifest['rigs']
    if selected_rig_ids is not None:
        unknown=set(selected_rig_ids)-{r['rig_id'] for r in rig_inputs}
        if unknown:raise ValueError('unknown_selected_rig_id:'+','.join(sorted(unknown)))
        rig_inputs=[r for r in rig_inputs if r['rig_id'] in selected_rig_ids]
    frames=list(range(31)) if frame_indices is None else sorted(set(frame_indices))
    if not frames or any(type(k) is not int or not 0<=k<31 for k in frames):
        raise ValueError('invalid_frame_subset')
    if preview and (len(frames)!=1 or len(rig_inputs)!=1):
        raise ValueError('preview_requires_one_frame_and_one_rig')
    if label_probe and (preview or len(rig_inputs)!=1 or not 1<=len(frames)<=3):
        raise ValueError('label_probe_requires_one_rig_and_at_most_three_frames')
    if not preview and not label_probe and frames!=list(range(31)):
        raise ValueError('non_preview_requires_complete_31_frame_episode')
    if not 1<=len(rig_inputs)<=8 or len({r['rig_id'] for r in rig_inputs})!=len(rig_inputs):
        raise ValueError('invalid_rig_list')
    if any(not isinstance(r.get('rig_id'),str) or
           re.fullmatch(r'rig_[a-z0-9_]+',r['rig_id']) is None
           for r in rig_inputs):raise ValueError('invalid_rig_id')
    sources=[Path(r['prepared']).resolve() for r in rig_inputs]
    if output.exists() and any(output.iterdir()):raise FileExistsError('multirig run output is not empty')
    output.mkdir(parents=True,exist_ok=True)
    base=json.loads((sources[0]/'recipe.json').read_text())
    if not 1<=len(base['people'])<=5 or base['frame_count']!=31:
        raise ValueError('unsupported_scene_person_or_time_count')
    if mamma_512:
        if qualification_registry is None:raise ValueError('mamma_512_requires_qualification_registry')
        registry=json.loads(Path(qualification_registry).read_text())
        if registry.get('schema') not in ('mamma_manifest_reader_v1','mamma_manifest_reader_v2'):
            raise ValueError('unreviewed_qualification_registry_schema')
        registered=registry['qualified_assets']
        for person in base['people']:
            asset=json.loads(Path(person['asset_manifest']).read_text())
            body=str(Path(asset['body_vertices']).resolve())
            if body not in registered:raise ValueError('unqualified_asset:'+body)
    for source in sources:
        current=json.loads((source/'recipe.json').read_text())
        if (current['frame_count']!=base['frame_count'] or current['people']!=base['people'] or
            current['episode_timeline']!=base['episode_timeline'] or current['time_base']!=base['time_base'] or
            json.loads(Path(current['scene']).read_text())!=json.loads(Path(base['scene']).read_text()) or
            current['rig']['baseline_m']!=base['rig']['baseline_m']):
            raise ValueError('frozen_scene_mismatch')
        for array in ('scene_matrices.npy','world_densekp.npy'):
            if not np.array_equal(np.load(sources[0]/array),np.load(source/array)):
                raise ValueError('shared_person_scene_changed:'+array)
        expected=np.load(source/'expected_in_view.npy',mmap_mode='r')
        if expected.shape!=(31,len(base['people']),2,138):
            raise ValueError('expected_in_view_person_axis_conflict')
        geometry=json.loads((source/'geometry_preflight.json').read_text())
        if geometry['status']!='passed':
            raise ValueError('rig_geometry_not_preflighted')
        if preview:
            if current.get('editor_recipe_ref') and geometry.get('frames')!=frames:
                raise ValueError('preview_geometry_subset_mismatch')
        elif current.get('editor_recipe_ref') and (geometry.get('scope')!='all_31' or geometry.get('frames')!=list(range(31))):
            raise ValueError('full_episode_requires_all_31_geometry_preflight')
    scene_id=manifest.get('scene_episode_id',base['episode_id'])
    if not isinstance(scene_id,str) or not scene_id:raise ValueError('invalid_scene_episode_id')
    common=copy.deepcopy(base)
    common['render']['gpu_uuid']=gpu_uuid
    common['scene_episode_id']=scene_id
    common['rig_id']=rig_inputs[0]['rig_id']
    common['observation_policy']='record_unavailable_v1'
    write(output/'recipe.json',common)
    shutil.copy2(sources[0]/'world_densekp.npy',output/'world_densekp.npy')
    shutil.copy2(sources[0]/'rig.json',output/'rig.json')
    rigs=[]
    for input_rig,source in zip(rig_inputs,sources):
        rig_id=input_rig['rig_id']
        child=output/'rigs'/rig_id;child.mkdir(parents=True,exist_ok=False)
        recipe=copy.deepcopy(json.loads((source/'recipe.json').read_text()))
        recipe.update(episode_id=f'{scene_id}__{rig_id}',sequence_id=f'{scene_id}__{rig_id}',
                      scene_episode_id=scene_id,rig_id=rig_id,coverage_goal=input_rig['coverage_goal'],
                      observation_policy='record_unavailable_v1',imaging_route='direct_erp_roi',storage_mode='roi_only',
                      readback_row_cap=(0 if preview else (4*len(base['people']) if rig_list_manifest else 6)),
                      readback_frame_indices=None if preview else ([0,15] if rig_list_manifest else None),
                      integration_only=True)
        if 'camera_seed' in input_rig:
            recipe['camera_seed']=input_rig['camera_seed']
        if 'readback_observation_ids' in input_rig:
            recipe['readback_observation_ids']=input_rig['readback_observation_ids']
            recipe['readback_row_cap']=len(input_rig['readback_observation_ids'])
            recipe['readback_frame_indices']=None
        recipe['render']['gpu_uuid']=gpu_uuid
        for key in ('scene_matrices_ref','expected_in_view_ref','episode_timeline_ref'):
            src=Path(recipe[key]);dest=child/src.name;shutil.copy2(src,dest);recipe[key]=str(dest)
        for key in ('person_timeline_refs','source_timeline_refs'):
            recipe[key]={pid:str(child/Path(ref).name) for pid,ref in recipe[key].items()}
            for pid,ref in json.loads((source/'recipe.json').read_text())[key].items():shutil.copy2(ref,recipe[key][pid])
        shutil.copy2(source/'planned_regions.json',child/'planned_regions.json')
        shutil.copy2(source/'rig.json',child/'rig.json')
        shutil.copy2(source/'world_densekp.npy',child/'world_densekp.npy')
        shutil.copytree(source/'calibration',child/'calibration')
        (child/'camera_configs').mkdir()
        for camera in json.loads((child/'rig.json').read_text())['cameras']:
            write(child/'camera_configs'/f'{camera["id"]}.json',dict(cameras=[camera]))
        write(child/'recipe.json',recipe)
        rigs.append(dict(rig_id=rig_id,root=str(child),coverage_goal=recipe['coverage_goal'],
                         source_prepared_ref=str(source),
                         camera_seed=input_rig.get('camera_seed',recipe.get('camera_child_seed'))))
    write(output/'multirig.json',dict(schema='scene_episode_rig_list_v1',scene_episode_id=scene_id,
          source_grouping='one_scene_all_people_all_motions_all_rigs_shared_source_diagnostic',rigs=rigs,
          frame_order='frame -> ordered rig list(cam2,cam4) -> synchronous CPU consumption',
          maximum_budget_seconds=manifest.get('maximum_budget_seconds',2700)))
    source_dirs=[]
    for frame in frames:
        dummy=output/f'frame_{frame:03d}'/'_temporary_ds_source/source';dummy.mkdir(parents=True)
        source_dirs.append(str(dummy))
        for rig in rigs:prepare_source_workspace(Path(rig['root'])/f'frame_{frame:03d}','compact')
    write(output/'job.json',dict(recipe=str(output/'recipe.json'),rig=str(output/'rig.json'),
          frames=frames,source_dirs=source_dirs,remaining_budget_seconds=budget_seconds))
    frozen=json.loads(TRIAL.read_text());scene_cfg=json.loads(Path(base['scene']).read_text())
    settings=scene_cfg.get('scene_settings_after_configure_render') or frozen['scene_settings']
    now=time.monotonic()
    config=dict(schema='continuous_multirig_roi_v1',integration_only=True,storage_mode='roi_only',consumer_mode='persistent',
                consumer_python=sys.executable,gpu_uuid=gpu_uuid,scene_settings=settings,
                deadline_monotonic=now+budget_seconds,budget_seconds=budget_seconds,max_camera_calls=len(frames)*2*len(rigs),
                preview_only=preview,label_probe_only=label_probe,selected_frames=frames,
                full_image_size_wh=[5760,2880],roi_samples=64,background_samples=0,
                sampling=dict(seed=1009,animated_seed=False,adaptive=False,denoise=False,device='CUDA'))
    if label_probe or mamma_512:
        if not MAMMA_TEMPLATE.is_file():raise FileNotFoundError('registered_mamma_template_missing')
        config['mamma_512_label']=dict(schema='mamma_512_label_config_v1',template_id='MAMMA_verts_512_ordered_v1',
            template_path=str(MAMMA_TEMPLATE),visibility_policy='mamma512_conservative_visibility_v1_compatible_definition_pending',
            sidecar_policy='seven_box_probe_v1' if label_probe else 'mask_native_v2')
        if mamma_512:config['mamma_512_label']['qualification_registry_ref']=str(Path(qualification_registry).resolve())
    write(output/'run_config.json',config)
    for rig in rigs:
        child=Path(rig['root'])
        write(child/'run_config.json',dict(config,valid_masks={c:f'calibration/{c}_valid_mask.png' for c in ('cam2','cam4')},
             retain_reference_frames=[]))
    return config
