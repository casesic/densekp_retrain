"""One Blender process, explicit bounded timeline; synchronous CPU consumption."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
from densekp_retrain.runtime.roi_run_utils import BLENDER, write, disk_snapshot


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--storage-mode',choices=['full_image','roi_only'],default='full_image')
    ap.add_argument('--gpu-uuid')
    ap.add_argument('--consumer-mode',choices=['per_frame','persistent'],default='per_frame')
    ap.add_argument('--frame-count',type=int)
    ap.add_argument('--budget-seconds',type=int,default=1800,help='Render + required consumption budget; maximum2700')
    ap.add_argument('--prepared-episode',type=Path,help='Explicit preflighted episode; native selected timeline is not resampled')
    ap.add_argument('--resume-parent',type=Path,help='Read-only frozen parent; reuse successful prefix and consume rejected buffers before rendering missing suffix')
    ap.add_argument('--probe-first',type=int,help='Render only this many prefix frames; retain them for a later read-only resume')
    ap.add_argument('--observation-policy',choices=['strict_dual_crop_v1','record_unavailable_v1'],default='strict_dual_crop_v1')
    a=ap.parse_args();out=a.output.resolve()
    if a.probe_first is not None and (a.resume_parent or not a.prepared_episode or a.probe_first<1):
        ap.error('probe-first requires a prepared episode and no resume parent')
    if a.observation_policy=='record_unavailable_v1' and a.storage_mode!='roi_only':ap.error('record_unavailable_v1 currently supports roi_only')
    if a.resume_parent:
        if a.prepared_episode or a.observation_policy!='record_unavailable_v1':ap.error('resume requires explicit record_unavailable_v1 and no prepared-episode')
        a.prepared_episode=a.resume_parent
    resume=None
    base=a.prepared_episode.resolve() if a.prepared_episode else OLD
    if a.frame_count is None:a.frame_count=json.loads((base/'recipe.json').read_text())['frame_count'] if a.prepared_episode else 11
    if a.frame_count<2 or a.frame_count>76:ap.error('bounded run requires 2..76 frames')
    if a.probe_first is not None and a.probe_first>=a.frame_count:
        ap.error('probe-first must be shorter than the full episode')
    if not 1<=a.budget_seconds<=2700:ap.error('budget must be1..2700 seconds')
    if a.prepared_episode and (a.consumer_mode!='persistent' or a.storage_mode!='roi_only'):ap.error('prepared dynamic episode requires persistent roi_only')
    if (a.consumer_mode=='persistent' and a.storage_mode!='roi_only') or (a.frame_count!=11 and a.consumer_mode!='persistent'):
        ap.error('persistent/short delivery currently requires roi_only; legacy per_frame retains 11 frames')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'run_config.json').exists():raise RuntimeError('Never overwrite or rerender an existing run; inspect its progress/failure')
    queries=dict(gpus=['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],
        compute=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],
        processes=['ps','-eo','pid,comm,pcpu,pmem,args','--sort=-pcpu'],memory=['free','-h'],disk=['df','-h',str(out),'/tmp'])
    resources={k:subprocess.check_output(v,text=True) for k,v in queries.items()};write(out/'resources_before.json',resources)
    ep=json.loads((base/'recipe.json').read_text())
    if a.resume_parent:
        from densekp_retrain.data.sequence_resume import inspect_parent,prepare_reprocessing
        resume=inspect_parent(base)
    if ('selected_timeline' in ep or 'person_timelines' in ep) and a.frame_count!=ep['frame_count']:raise ValueError('select native timeline during planning, not inside render runner')
    if 'selected_timeline' not in ep and 'person_timelines' not in ep and a.frame_count>11:raise ValueError('legacy prefix limited to11')
    uuid=a.gpu_uuid or ep['render']['gpu_uuid']
    ep['render']['gpu_uuid']=uuid
    selected=next(x for x in resources['gpus'].splitlines()[1:] if uuid in x)
    if int(selected.split(',')[2].split()[0])>512 or int(selected.split(',')[4].split()[0])>5:
        raise RuntimeError('Selected GPU busy; leave all other tasks untouched')
    ignored_desktop=[]
    for row in resources['compute'].splitlines()[1:]:
        if uuid not in row:continue
        if resume:raise RuntimeError('Resume requires GPU with no other compute process')
        fields=[x.strip() for x in row.split(',')];pid=int(fields[1])
        # An idle desktop Blender window (no script/background command, <256MiB)
        # is not a render job. Utilization/free-memory gate above still applies.
        try:argv=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0');argv=[x.decode() for x in argv if x]
        except (FileNotFoundError,PermissionError):argv=[]
        if argv==[BLENDER] and int(fields[3].split()[0])<256:
            ignored_desktop.append(dict(pid=pid,command=argv,memory_MiB=int(fields[3].split()[0])))
        else:raise RuntimeError('Other compute process on selected GPU; do not interfere')
    write(out/'device_selection.json',dict(gpu_uuid=uuid,idle_desktop_processes=ignored_desktop,active_compute_jobs_allowed=False))
    from densekp_retrain.imaging.output_modes import prepare_source_workspace
    # Scope is this one-second prefix; source arrays retain their original indexing.
    ep.update(frame_count=a.frame_count,imaging_route='direct_erp_roi',storage_mode=a.storage_mode,parent_recipe_ref=str(base/'recipe.json'),
        selection=dict(frame_indices=list(range(a.frame_count)),original_frame_count=ep.get('original_timeline_count',31),first_last_span_seconds=(ep['selected_timeline'][-1]['episode_time_tick']*ep['time_base'][0]/ep['time_base'][1]) if 'selected_timeline' in ep else (a.frame_count-1)/10),integration_only=True)
    ep['render']=dict(ep['render'],render_budget_seconds=a.budget_seconds,background_resolution=None if a.storage_mode=='roi_only' else [1440,720],background_samples=0 if a.storage_mode=='roi_only' else 8)
    if 'selected_timeline' in ep:
        for name,key in [('selected_timeline.json','selected_timeline_ref'),('scene_matrices.npy','scene_matrices_ref')]:
            shutil.copy2(base/name,out/name);ep[key]=str(out/name)
    if 'person_timelines' in ep:
        for key in ['scene_matrices_ref','episode_timeline_ref','expected_in_view_ref']:
            src=Path(ep[key]);shutil.copy2(src,out/src.name);ep[key]=str(out/src.name)
        for key in ['person_timeline_refs','source_timeline_refs']:
            for pid,ref in ep[key].items():
                src=Path(ref);shutil.copy2(src,out/src.name);ep[key][pid]=str(out/src.name)
    if a.observation_policy!='strict_dual_crop_v1':ep['observation_policy']=a.observation_policy
    if resume:
        ep['readback_policy']='continuation_old_occluded_reappearance' if resume['reprocessed'] else 'first_middle_last'
        ep['parent_run_ref']=str(base)
    write(out/'recipe.json',ep);shutil.copy2(base/'rig.json',out/'rig.json');shutil.copy2(base/'world_densekp.npy',out/'world_densekp.npy')
    (out/'camera_configs').mkdir();(out/'calibration').mkdir()
    for c in json.loads((out/'rig.json').read_text())['cameras']:
        write(out/'camera_configs'/f'{c["id"]}.json',dict(cameras=[c]))
        shutil.copy2(base/'calibration'/f'{c["id"]}_valid_mask.png' if a.prepared_episode else base/f'frame_000/{c["id"]}/valid_mask.png',out/'calibration'/f'{c["id"]}_valid_mask.png')
    start=time.monotonic();deadline=start+a.budget_seconds
    render_frames=resume['new'] if resume else list(range(a.probe_first)) if a.probe_first is not None else list(range(a.frame_count))
    if resume:prepare_reprocessing(base,out,resume)
    sources=[str(prepare_source_workspace(out/f'frame_{k:03d}','compact')) for k in render_frames]
    write(out/'job.json',dict(recipe=str(out/'recipe.json'),rig=str(out/'rig.json'),frames=render_frames,source_dirs=sources,remaining_budget_seconds=a.budget_seconds))
    frozen=json.loads((TRIAL/'frozen_comparison.json').read_text())
    scene_cfg=json.loads(Path(ep['scene']).read_text())
    expected_scene_settings=scene_cfg.get('scene_settings_after_configure_render',frozen['scene_settings'])
    dependency_paths=[base/'recipe.json',base/'rig.json',base/'world_densekp.npy',Path(ep['scene']),ROOT/'assets/densekp_regressor_R.npy']
    if 'environment_blend_dependency' in ep:
        dependency_paths.append(ROOT/ep['environment_blend_dependency']['path'])
    for p in ep['people']:
        path=Path(p['asset_manifest']);dependency_paths.append(path);asset=json.loads(path.read_text())
        if asset.get('schema')=='densekp_scene_asset_v2':
            from densekp_retrain.data.dynamic_scene import geometry_paths
            dependency_paths.extend(geometry_paths(asset).values())
            dependency_paths.extend(Path(asset[key]) for key in ['parameters','timeline_ref','authority_manifest','proposal_ref','asset_evidence'])
            folders=['clothing_reference','groom_reference']
        else:folders=['geometry','clothing_reference','groom_reference']
        for key in folders:dependency_paths.extend(p for p in Path(asset[key]).iterdir() if p.is_file())
    write(out/'input_inventory.json',{str(p):[p.stat().st_size,p.stat().st_mtime_ns] for p in dependency_paths})
    write(out/'run_config.json',dict(schema='continuous_direct_erp_roi_v1',integration_only=True,scene_settings=expected_scene_settings,probe_first=a.probe_first,
        recipe_ref=str(out/'recipe.json'),old_recipe_ref=str(base/'recipe.json'),settings_evidence_ref=str(TRIAL/'frozen_comparison.json'),
        reprocess_frames_before_render=resume['reprocessed'] if resume else [],observation_policy=a.observation_policy,consumer_python=sys.executable,gpu_uuid=uuid,retain_reference_frames=[] if resume else sorted({0,a.frame_count//2,a.frame_count-1}) if 'selected_timeline' in ep or 'person_timelines' in ep else [0,5,10],deadline_monotonic=deadline,budget_seconds=a.budget_seconds,max_camera_calls=len(render_frames)*(2 if a.storage_mode=='roi_only' else 4),storage_mode=a.storage_mode,consumer_mode=a.consumer_mode,
        full_image_size_wh=[5760,2880],roi_samples=64,background_size_wh=None if a.storage_mode=='roi_only' else [1440,720],background_samples=0 if a.storage_mode=='roi_only' else 8,
        valid_masks={c:f'calibration/{c}_valid_mask.png' for c in ['cam2','cam4']},
        sampling=dict(seed=1009,animated_seed=False,adaptive=False,denoise=False,device='CUDA',dither_intensity=1.0),
        ROI_policy=('current live body/cloth/groom; bounded_native_box_v1 full-family reservation plus raster margin'
                    if 'box_augmentation_policy' in ep else
                    'current live body/cloth/groom geometry; conservative contained native-square envelope plus32px; no augmented crops'),
        ordering='single Blender scene; ROI2 then synchronous CPU consume' if a.storage_mode=='roi_only' else 'background2 then ROI2 then synchronous CPU consume',
        version=dict(git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),dirty=True)))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/densekp-roi-mpl')
    command=[BLENDER,'-b','-t','4','--python-exit-code','1','--python',str(ROOT/'tools/render/render_continuous_roi.py'),'--','--root',str(out)]
    samples=[];render_start=time.monotonic()
    with (out/'blender.log').open('w') as log:
        p=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        timed_out=False
        while p.poll() is None:
            if time.monotonic()>deadline:
                p.kill();timed_out=True;break  # Own child only.
            raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(seconds=time.monotonic()-start,gpu=raw,**disk_snapshot(out)));time.sleep(1)
        code=p.wait()
    blender_wall=time.monotonic()-render_start
    write(out/'blender_process.json',dict(command=command,pid=p.pid,returncode=code,timed_out=timed_out,wall_seconds=blender_wall,
        includes_synchronous_CPU_consumer=True,gpu_uuid=uuid,gpu_samples=samples,
        sampled_peak_GPU_MiB=max(int(x['gpu'].split(',')[0]) for x in samples),poll_seconds=1))
    if code or timed_out:
        write(out/'failure.json',dict(stage='Blender_or_frame_consumer',seconds=time.monotonic()-start,remaining_seconds=deadline-time.monotonic(),failed_site_preserved=True))
        raise SystemExit(code or 1)
    if a.probe_first is not None:
        write(out/'task_wall.json',dict(probe_only=True,rendered_frame_indices=render_frames,total_generation_wall_seconds=time.monotonic()-start,
            blender_process_wall_seconds=blender_wall,blender_start_count=1,finish_process_seconds=0))
        write(out/'resources_after.json',{k:subprocess.check_output(v,text=True) for k,v in queries.items() if k!='processes'})
        print('CONTINUOUS_ROI_PROBE_COMPLETE',out,flush=True)
        return
    t=time.monotonic();finish_command=[sys.executable,str(ROOT/'tools/experiments/consume_continuous_roi.py'),'finish','--root',str(out)]
    with (out/'finish.log').open('w') as log:
        finish=subprocess.run(finish_command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time.monotonic()))
    write(out/'task_wall.json',dict(total_generation_wall_seconds=time.monotonic()-start,blender_process_wall_seconds=blender_wall,
        finish_process_seconds=time.monotonic()-t,finish_returncode=finish.returncode,remaining_budget_seconds=deadline-time.monotonic(),
        budget_scope='all rendering, synchronous frame consumption, pack/readback/video; code and reading excluded',blender_start_count=1))
    if finish.returncode:raise SystemExit(finish.returncode)
    write(out/'resources_after.json',{k:subprocess.check_output(v,text=True) for k,v in queries.items() if k!='processes'})
    print('CONTINUOUS_ROI_COMPLETE',out,flush=True)

if __name__=='__main__':main()
