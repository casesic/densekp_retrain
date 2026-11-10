"""Bounded scene episode with a frozen list of paired ROI rigs in one frame loop."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
from densekp_retrain.runtime.multirig_prepare import prepare as prepare_multirig
from densekp_retrain.runtime.roi_run_utils import BLENDER,disk_snapshot,write

EDGE=ROOT/'outputs/experiments/second_environment_boundary_pass_a1_20261010/prepared_riverfront'
SIDE=ROOT/'outputs/experiments/multirig_scene_render_a1_20261010/prepared_side'
TRIAL=ROOT/'outputs/experiments/direct_erp_roi_trial_20261009/frozen_comparison.json'
MAMMA_TEMPLATE=Path('/home/yy/Repos/mamma/data/body_models/downsampled_verts/verts_512.pkl')


def prepare(*args, **kwargs):
    """Compatibility entry for historical CLI and editor callers."""
    return prepare_multirig(*args, **kwargs)


def resources(output,gpu_uuid):
    queries=dict(gpus=['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],
       compute=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],
       memory=['free','-h'],disk=['df','-h',str(output),'/tmp'])
    evidence={key:subprocess.check_output(cmd,text=True) for key,cmd in queries.items()}
    write(output/'resources_before.json',evidence)
    gpu=next((row for row in evidence['gpus'].splitlines()[1:] if gpu_uuid in row),None)
    if gpu is None or int(gpu.split(',')[2].split()[0])>512 or int(gpu.split(',')[4].split()[0])>5:
        raise RuntimeError('selected_GPU_unavailable')
    for row in evidence['compute'].splitlines()[1:]:
        if gpu_uuid not in row:continue
        fields=[x.strip() for x in row.split(',')];pid=int(fields[1])
        try:argv=[v.decode() for v in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if v]
        except (FileNotFoundError,PermissionError):argv=[]
        if argv!=[BLENDER] or int(fields[3].split()[0])>=256:
            raise RuntimeError('other_GPU_compute_process_present')
    return evidence


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--gpu-uuid',required=True);parser.add_argument('--budget-seconds',type=int,default=1800)
    parser.add_argument('--edge-prepared',type=Path,default=EDGE)
    parser.add_argument('--side-prepared',type=Path,default=SIDE)
    parser.add_argument('--rig-list-manifest',type=Path,
                        help='Versioned scene_episode_rig_list_v1; legacy edge/side flags remain supported')
    parser.add_argument('--frame-index',type=int,action='append',help='Render only selected 0..30 frames; repeatable')
    parser.add_argument('--rig-id',action='append',help='Render only selected rig IDs; repeatable')
    parser.add_argument('--preview',action='store_true',help='One frame and one paired rig; skip sequence finish')
    parser.add_argument('--label-probe',action='store_true',help='Sparse 1..3-frame MAMMA label probe; skip sequence pack')
    parser.add_argument('--mamma-512',action='store_true',help='Enable 512 labels for a complete episode, without seven-box diagnostics')
    parser.add_argument('--qualification-registry',type=Path,help='Reviewed qualified_assets JSON; required by --mamma-512')
    parser.add_argument('--prepare-only',action='store_true',help='Freeze run configuration and exit before resource or Blender checks')
    parser.add_argument('--run-prepared',action='store_true',
                        help='Run a previously prepared, never started task after static clearance checks')
    parser.add_argument('--resume-after-frame0-side-shape-fix',action='store_true',
                        help='Only the documented frame0 side label-shape failure; keep successful rig_edge data')
    args=parser.parse_args()
    bound=2700
    if args.rig_list_manifest is not None:
        requested=json.loads(args.rig_list_manifest.read_text())
        if len(requested.get('rigs',[]))==3 and requested.get('maximum_budget_seconds')==3600:
            bound=3600
    elif args.run_prepared and (args.output/'multirig.json').is_file():
        saved=json.loads((args.output/'multirig.json').read_text())
        if len(saved.get('rigs',[]))==3 and saved.get('maximum_budget_seconds')==3600:
            bound=3600
    if not 1<=args.budget_seconds<=bound:parser.error(f'budget must be 1..{bound} seconds')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    if args.run_prepared and (args.frame_index or args.rig_id or args.preview or args.label_probe or args.mamma_512 or args.prepare_only):
        parser.error('run-prepared uses the already frozen job subset')
    if args.mamma_512 and (args.preview or args.label_probe):parser.error('mamma-512 requires normal complete-episode mode')
    if args.mamma_512 and args.qualification_registry is None:parser.error('--mamma-512 requires --qualification-registry')
    if args.prepare_only:
        if args.resume_after_frame0_side_shape_fix:parser.error('prepare-only cannot resume a rendered run')
        config=prepare(out,gpu_uuid=args.gpu_uuid,budget_seconds=args.budget_seconds,
            edge_prepared=args.edge_prepared,side_prepared=args.side_prepared,
            rig_list_manifest=args.rig_list_manifest,frame_indices=args.frame_index,
            selected_rig_ids=args.rig_id,preview=args.preview,label_probe=args.label_probe,mamma_512=args.mamma_512,
            qualification_registry=args.qualification_registry)
        print('MULTIRIG_ROI_PREPARED',out,config['max_camera_calls'],flush=True)
        return
    if args.run_prepared and args.resume_after_frame0_side_shape_fix:
        parser.error('run-prepared and the historical targeted resume are incompatible')
    if (out/'run_config.json').exists() and not (args.resume_after_frame0_side_shape_fix or args.run_prepared):
        raise FileExistsError('existing multirig production output is read-only; choose a new directory')
    if args.resume_after_frame0_side_shape_fix and ((out/'multirig_render_result.json').exists()
                                                 or (out/'rigs/rig_side/frame_000/report.json').exists()):
        raise FileExistsError('documented resume unit is already complete; do not rerun it')
    evidence=resources(out,args.gpu_uuid)
    if args.run_prepared:
        config=json.loads((out/'run_config.json').read_text())
        if (config['gpu_uuid']!=args.gpu_uuid or (out/'render_calls.jsonl').exists() or
            (out/'multirig_render_result.json').exists() or (out/'failure.json').exists()):
            raise ValueError('prepared_task_started_or_GPU_changed')
        spec=json.loads((out/'multirig.json').read_text())
        if any((Path(r['root'])/'frame_000/report.json').exists() for r in spec['rigs']):
            raise ValueError('prepared_task_already_consumed')
        config['deadline_monotonic']=time.monotonic()+args.budget_seconds
        config['budget_seconds']=args.budget_seconds
        write(out/'run_config.json',config)
        for rig in spec['rigs']:
            child=Path(rig['root'])/'run_config.json';child_config=json.loads(child.read_text())
            child_config['deadline_monotonic']=config['deadline_monotonic']
            child_config['budget_seconds']=args.budget_seconds
            write(child,child_config)
    elif args.resume_after_frame0_side_shape_fix:
        config=json.loads((out/'run_config.json').read_text())
        failure=json.loads((out/'failure_attempt_1/diagnosis.json').read_text())
        if failure['issue']!='preflight_expected_in_view_missing_person_axis':
            raise ValueError('unsupported_multirig_resume_failure')
        if not (out/'rigs/rig_edge/frame_000/report.json').is_file() or (out/'rigs/rig_side/frame_000/report.json').exists():
            raise ValueError('unexpected_frame0_completion_state')
        if np.load(out/'rigs/rig_side/expected_in_view.npy').shape!=(31,1,2,138):
            raise ValueError('side_expected_mask_shape_not_fixed')
        job=json.loads((out/'job.json').read_text())
        job['frames']=list(range(1,31));job['source_dirs']=job['source_dirs'][1:]
        job['remaining_budget_seconds']=args.budget_seconds
        write(out/'job.json',job)
        config.update(deadline_monotonic=time.monotonic()+args.budget_seconds,
                      budget_seconds=args.budget_seconds,reprocess_units_before_render=[dict(frame_index=0,rig_id='rig_side')],
                      prior_completed_units=[dict(frame_index=0,rig_id='rig_edge')],max_camera_calls=124)
        write(out/'run_config.json',config)
        for rig_id in ('rig_edge','rig_side'):
            child=out/'rigs'/rig_id/'run_config.json';child_config=json.loads(child.read_text())
            child_config['deadline_monotonic']=config['deadline_monotonic'];child_config['budget_seconds']=args.budget_seconds
            write(child,child_config)
    else:
        config=prepare(out,gpu_uuid=args.gpu_uuid,budget_seconds=args.budget_seconds,
                       edge_prepared=args.edge_prepared,side_prepared=args.side_prepared,
                       rig_list_manifest=args.rig_list_manifest,frame_indices=args.frame_index,
                       selected_rig_ids=args.rig_id,preview=args.preview,label_probe=args.label_probe,mamma_512=args.mamma_512,
                       qualification_registry=args.qualification_registry)
    start=time.monotonic();deadline=config['deadline_monotonic']
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=args.gpu_uuid,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
             PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/densekp-multirig-mpl')
    command=[BLENDER,'-b','-t','4','--python-exit-code','1','--python',str(ROOT/'tools/render/render_multirig_roi.py'),
             '--','--root',str(out)]
    samples=[]
    with (out/'blender.log').open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        timed_out=False
        while process.poll() is None:
            if time.monotonic()>deadline:
                process.kill();timed_out=True;break
            gpu=subprocess.check_output(['nvidia-smi','-i',args.gpu_uuid,
                '--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(elapsed_seconds=time.monotonic()-start,gpu=gpu,**disk_snapshot(out)))
            time.sleep(1)
        code=process.wait()
    write(out/'blender_process.json',dict(pid=process.pid,returncode=code,timed_out=timed_out,
          wall_seconds=time.monotonic()-start,gpu_uuid=args.gpu_uuid,gpu_samples=samples))
    if code or timed_out:
        write(out/'failure.json',dict(stage='render_or_consume',returncode=code,timed_out=timed_out,
              elapsed_seconds=time.monotonic()-start,failed_site_preserved=True))
        raise SystemExit(code or 1)
    if config.get('preview_only'):
        write(out/'preview_result.json',dict(status='completed',frames=config['selected_frames'],
              rig_ids=[r['rig_id'] for r in json.loads((out/'multirig.json').read_text())['rigs']],
              camera_calls=config['max_camera_calls'],sequence_packed=False))
        print('MULTIRIG_ROI_PREVIEW_COMPLETE',out,flush=True)
        return
    if config.get('label_probe_only'):
        write(out/'label_probe_result.json',dict(status='render_and_frame_consumption_completed',frames=config['selected_frames'],
              rig_ids=[r['rig_id'] for r in json.loads((out/'multirig.json').read_text())['rigs']],
              camera_calls=config['max_camera_calls'],sequence_packed=False,sparse_frames_preserve_source_time=True))
        print('MULTIRIG_ROI_LABEL_PROBE_COMPLETE',out,flush=True)
        return
    finishes=[]
    for rig_id in [r['rig_id'] for r in json.loads((out/'multirig.json').read_text())['rigs']]:
        child=out/'rigs'/rig_id
        cmd=[sys.executable,str(ROOT/'tools/experiments/consume_continuous_roi.py'),'finish','--root',str(child)]
        t=time.monotonic()
        with (child/'finish.log').open('w') as log:
            done=subprocess.run(cmd,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1'),
                                stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,deadline-time.monotonic()))
        finishes.append(dict(rig_id=rig_id,returncode=done.returncode,seconds=time.monotonic()-t))
        if done.returncode:
            write(out/'finish_progress.json',finishes)
            raise SystemExit(done.returncode)
    write(out/'task_wall.json',dict(elapsed_seconds=time.monotonic()-start,finishes=finishes,
          total_camera_calls=config['max_camera_calls'],shared_scene_load_count=1,budget_seconds=args.budget_seconds,
          disk=disk_snapshot(out)))
    print('MULTIRIG_ROI_COMPLETE',out,flush=True)


if __name__=='__main__':main()
