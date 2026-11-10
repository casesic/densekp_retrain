"""C route callback for the existing single-process road_pass frame loop."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import resource
import bpy
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.render import render_road_pass_sources as assembly
from tools.render.render_direct_erp_roi_trial import scene_settings
from densekp_retrain.rendering.blender.assembly import render_cameras
from densekp_retrain.camera.erp import project_erp
from densekp_retrain.imaging.roi_planning import plan_native_crop_region,plan_person_crop_union,border_coordinates
from densekp_retrain.runtime.frame_consumer import FrameConsumer


def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')


def render_rig_roi(scene,cam,node,folder,rig_root,recipe,cameras,world_parts,part_owners,config,
                   frame,call_started):
    """Render one rig from the *already updated* scene geometry."""
    begin=time.monotonic();ids={p['person_id'] for p in recipe['people']};plans={}
    for camera in cameras:
        pose=np.array(camera['camera_to_world']);parts={n:project_erp(p,pose,5760,2880)[0] for n,p in world_parts.items()}
        if recipe.get('roi_planning_policy')=='per_person_square_union_v1':
            grouped={str(pid):{name:xy for name,xy in parts.items() if part_owners[name]==pid} for pid in ids}
            region=plan_person_crop_union(grouped,(5760,2880),
                box_augmentation_policy=recipe.get('box_augmentation_policy'))
        else:
            if recipe.get('roi_planning_policy','legacy_collective_envelope')!='legacy_collective_envelope':
                raise ValueError('unknown_roi_planning_policy')
            region=plan_native_crop_region(parts,(5760,2880),
                box_augmentation_policy=recipe.get('box_augmentation_policy'))
        plans[camera['id']]=dict(region,camera_to_world=camera['camera_to_world'])
        if 'max_union_roi_pixels' in recipe:
            x0,y0,x1,y1=region['roi_xyxy']
            if (x1-x0)*(y1-y0)>recipe['max_union_roi_pixels']:raise ValueError('union_ROI_pixel_budget')
    plan_seconds=time.monotonic()-begin
    write(folder/'roi_plan.json',dict(frame_index=frame,rig_id=recipe.get('rig_id'),cameras=plans,planning_seconds=plan_seconds))
    records=[];tasks=[('roi',5760,2880,64)] if config.get('storage_mode')=='roi_only' else [('background',1440,720,8),('roi',5760,2880,64)]
    for kind,w,h,samples in tasks:
        dest=folder/kind;dest.mkdir(exist_ok=False)
        for camera in cameras:
            roi=plans[camera['id']]['roi_xyxy'] if kind=='roi' else None
            scene.render.resolution_x=w;scene.render.resolution_y=h;scene.render.resolution_percentage=100;scene.cycles.samples=samples
            scene.render.use_border=roi is not None;scene.render.use_crop_to_border=roi is not None
            if roi:
                coords=border_coordinates(roi,(w,h))
                scene.render.border_min_x,scene.render.border_max_x,scene.render.border_min_y,scene.render.border_max_y=coords
            if hasattr(node,'directory'):node.directory=str(dest)
            else:node.base_path=str(dest)
            call=call_started(frame,camera['id'],kind,roi,[w,h],samples)
            t=time.monotonic();record=render_cameras(scene,cam,node,rig_root/'camera_configs'/f'{camera["id"]}.json',dest)[0]
            dt=time.monotonic()-t;record.update(kind=kind,full_grid=[w,h],samples=samples,roi_xyxy=roi,seconds=dt)
            records.append(record)
            call_started(frame,camera['id'],kind,roi,[w,h],samples,completed_call=call,seconds=dt)
    return dict(frame_index=frame,rig_id=recipe.get('rig_id'),roi_planning_seconds=plan_seconds,records=records)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',required=True,type=Path);ap.add_argument('--job-file',default='job.json');ap.add_argument('--run-tag',default='')
    args=ap.parse_args(sys.argv[sys.argv.index('--')+1:]);root=args.root.resolve()
    artifact_root=root/args.run_tag if args.run_tag else root
    if args.run_tag:artifact_root.mkdir(exist_ok=False)
    config=json.loads((root/'run_config.json').read_text());job=json.loads((root/args.job_file).read_text())
    recipe=json.loads((root/'recipe.json').read_text());cameras=json.loads((root/'rig.json').read_text())['cameras']
    ids={p['person_id'] for p in recipe['people']};started=time.monotonic();hair_cache={};object_signature=None;calls=sum(json.loads(line)['event']=='started' for line in (root/'render_calls.jsonl').read_text().splitlines()) if (root/'render_calls.jsonl').exists() else 0;frame_results=[]
    def remaining():return config['deadline_monotonic']-time.monotonic()
    def append(record):
        with (root/'render_calls.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
    def render(scene,cam,node,cameras_path,folder):
        nonlocal object_signature,calls
        frame=job['frames'][job['source_dirs'].index(str(folder))];begin=time.monotonic()
        assert scene_settings(scene)==config['scene_settings']
        assert scene.render.use_persistent_data and not scene.cycles.use_denoising and not scene.cycles.use_adaptive_sampling
        signature=[(o.name,o.as_pointer(),o.data.as_pointer() if o.data else None) for o in scene.objects]
        if object_signature is None:
            object_signature=signature
            write(artifact_root/'single_process_setup.json',dict(pid=os.getpid(),setup_callback_seconds=begin-started,
                object_signature=signature,person_instance_objects=[dict(name=o.name,person_id=o.pass_index,type=o.type) for o in scene.objects if o.pass_index in ids and not o.hide_render],
                settings=scene_settings(scene),persistent_data=True,blender_version=bpy.app.version_string))
        else:assert signature==object_signature,'scene/object/mesh replaced during frame loop'
        world_parts={};part_owners={}
        for obj in scene.objects:
            if obj.pass_index not in ids or obj.hide_render:continue
            if obj.type=='MESH':
                points=np.empty((len(obj.data.vertices),3),np.float32);obj.data.vertices.foreach_get('co',points.ravel())
            elif obj.type=='CURVES':
                if obj.name not in hair_cache:
                    points=np.empty((len(obj.data.points),3),np.float32);obj.data.position_data.foreach_get('vector',points.ravel());hair_cache[obj.name]=points
                points=hair_cache[obj.name];assert len(points)==len(obj.data.points)
            else:raise ValueError(f'unsupported person geometry: {obj.type}')
            M=np.asarray(obj.matrix_world);world_parts[obj.name]=points@M[:3,:3].T+M[:3,3]
            part_owners[obj.name]=int(obj.pass_index)
        def call_started(frame_index,camera_id,kind,roi,resolution,samples,*,completed_call=None,seconds=None):
            nonlocal calls
            if completed_call is not None:
                append(dict(event='completed',call=completed_call,frame_index=frame_index,camera=camera_id,kind=kind,seconds=seconds,
                    rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
                return completed_call
            if calls>=config['max_camera_calls'] or remaining()<1:raise RuntimeError('continuous C call/time budget exhausted')
            calls+=1
            append(dict(event='started',call=calls,frame_index=frame_index,camera=camera_id,kind=kind,roi_xyxy=roi,
                resolution=resolution,samples=samples,remaining_seconds=remaining(),pid=os.getpid()))
            return calls
        result=render_rig_roi(scene,cam,node,folder,root,recipe,cameras,world_parts,part_owners,config,frame,call_started)
        plan_seconds=result['roi_planning_seconds'];records=result['records']
        frame_results.append(dict(frame_index=frame,roi_planning_seconds=plan_seconds,records=records))
        write(artifact_root/'render_progress.json',dict(pid=os.getpid(),render_calls=calls,frames=frame_results))
        return records
    worker=None
    def consume(frame,folder,manifest):
        if worker is not None:
            response=worker.consume(frame)
            write(root/f'frame_{frame:03d}'/'consumer_process.json',dict(seconds=response['roundtrip_seconds'],mode='persistent',worker_response=response))
            return
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        command=[config['consumer_python'],str(ROOT/'tools/experiments/consume_continuous_roi.py'),'frame','--root',str(root),'--frame',str(frame)]
        t=time.monotonic()
        with (root/f'frame_{frame:03d}'/'consume.log').open('w') as log:
            subprocess.run(command,env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(.1,remaining()))
        write(root/f'frame_{frame:03d}'/'consumer_process.json',dict(seconds=time.monotonic()-t,command=command))
    sys.argv=[sys.argv[0],'--','--job',str(root/args.job_file)]
    try:
        if config.get('consumer_mode','per_frame')=='persistent':
            env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
            command=[config['consumer_python'],str(ROOT/'tools/experiments/consume_continuous_roi.py'),'serve','--root',str(root)]
            worker=FrameConsumer(command,env=env,cwd=ROOT,log_path=artifact_root/'consumer_worker.log',deadline=config['deadline_monotonic'])
            write(artifact_root/'consumer_worker_start.json',dict(pid=worker.process.pid,startup_seconds=worker.startup_seconds,command=command))
        for frame in config.get('reprocess_frames_before_render',[]):
            if worker is None:raise ValueError('resume_reprocess_requires_persistent_consumer')
            consume(frame,None,None)
        assembly.main(render_cameras_fn=render,frame_complete_fn=consume)
        if worker is not None:
            write(artifact_root/'consumer_worker_result.json',worker.close())
    except BaseException:
        if worker is not None:worker.abort()
        raise
    write(artifact_root/'continuous_render_result.json',dict(pid=os.getpid(),render_calls=calls,frames=len(frame_results),
        elapsed_seconds=time.monotonic()-started,object_identity_unchanged=True,scene_load_count=1,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

if __name__=='__main__':main()
