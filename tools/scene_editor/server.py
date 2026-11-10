"""Loopback-only scene editor with one bounded render worker."""
from __future__ import annotations

import argparse
import copy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from urllib.parse import urlparse, parse_qs

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.scene_editor.core import OUTPUT, ROOT, SOURCES, additional_person, default_recipe, map_data, read, validate, write, freeze

WEB=Path(__file__).with_name('index.html')
LOCK=threading.Lock()
TASKS={}
RUNNING=None
WARM=None
TASK_RECORD=OUTPUT/'tasks.json'


def episode_videos(task):
    """Expose only finished, registered episode videos under this task's output."""
    if task.get('kind')!='episode':return []
    base=OUTPUT/'episodes'/task['id']/'run'
    repaired=(base/'delivery_repair.json')
    done=task.get('state')=='completed' or (repaired.is_file() and
         read(repaired).get('status')=='completed_after_metadata_finish_repair')
    if not done or not (base/'multirig.json').is_file():return []
    spec=read(base/'multirig.json');videos=[]
    for entry in spec['rigs']:
        rig=entry['rig_id']
        if re.fullmatch(r'[a-z0-9_]{1,64}',rig) is None:continue
        path=base/'rigs'/rig/'pack/dual_roi.mp4'
        if path.is_file():videos.append(dict(rig_id=rig,url=f'/api/video/{task["id"]}/{rig}',
                                            path=str(path),size_bytes=path.stat().st_size))
    return videos


def record_tasks():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    write(TASK_RECORD,{tid:{k:v for k,v in task.items() if k not in ('recipe','command','pid')}
                       for tid,task in TASKS.items()})


def gpu_uuid():
    if WARM is not None and WARM['process'].poll() is None:
        uid=WARM['gpu_uuid'];pid=WARM['process'].pid
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
            '--format=csv,noheader,nounits'],text=True,timeout=5)
        for line in apps.splitlines():
            fields=[x.strip() for x in line.split(',')]
            if len(fields)==2 and fields[0]==uid and int(fields[1])!=pid:
                raise RuntimeError('常驻预览GPU上出现其它计算任务')
        free=subprocess.check_output(['nvidia-smi','-i',uid,'--query-gpu=memory.free',
            '--format=csv,noheader,nounits'],text=True,timeout=5)
        if int(free.strip())<1024:raise RuntimeError('常驻预览GPU剩余显存不足')
        return uid
    result=subprocess.run(['nvidia-smi','--query-gpu=uuid,memory.used,utilization.gpu',
                           '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
    if result.returncode:raise RuntimeError('GPU驱动不可用：'+result.stderr.strip())
    active=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid',
        '--format=csv,noheader,nounits'],text=True,timeout=5)
    occupied={line.split(',')[0].strip() for line in active.splitlines() if ',' in line}
    rows=[x.strip().split(',') for x in result.stdout.splitlines() if x.strip()]
    for row in rows:
        if len(row)==3 and row[0].strip() not in occupied and int(row[1])<=512 and int(row[2])<=5:return row[0].strip()
    raise RuntimeError('当前没有空闲GPU；请检查正在运行的任务')


def _resident_preview(task,base,run,uid):
    global WARM
    from densekp_retrain.runtime.multirig_prepare import prepare, write as run_write
    from densekp_retrain.runtime.roi_run_utils import BLENDER, disk_snapshot
    config=prepare(run,gpu_uuid=uid,budget_seconds=300,
                   rig_list_manifest=base/'prepared/rig_list.json',
                   frame_indices=[task['frame']],selected_rig_ids=[task['rig_id']],preview=True)
    snapshots={key:subprocess.check_output(cmd,text=True,timeout=10) for key,cmd in {
        'gpus':['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],
        'compute':['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],
        'memory':['free','-h'],'disk':['df','-h',str(run),'/tmp']}.items()}
    run_write(run/'resources_before.json',snapshots)
    if WARM is None or WARM['process'].poll() is not None:
        session_dir=OUTPUT/'warm_sessions'/('warm_'+uuid.uuid4().hex[:12]);session_dir.mkdir(parents=True)
        log=(session_dir/'blender.log').open('w')
        command=[BLENDER,'-b','-t','4','--python-exit-code','1','--python',
                 str(ROOT/'tools/render/serve_scene_editor_previews.py'),'--',
                 '--preview-root',str((OUTPUT/'previews').resolve()),'--gpu-uuid',uid]
        proc=subprocess.Popen(command,cwd=ROOT,stdin=subprocess.PIPE,stdout=log,
            stderr=subprocess.STDOUT,start_new_session=True,
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=uid,OMP_NUM_THREADS='4',
                     OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',
                     MPLCONFIGDIR='/tmp/densekp-multirig-mpl'))
        WARM=dict(process=proc,gpu_uuid=uid,log=log,log_path=str(session_dir/'blender.log'))
    proc=WARM['process'];task['pid']=proc.pid;task['warm_log']=WARM['log_path']
    proc.stdin.write((json.dumps(dict(task_id=task['id']))+'\n').encode());proc.stdin.flush()
    start=time.monotonic();deadline=min(config['deadline_monotonic'],start+360)
    while True:
        if (run/'preview_result.json').is_file():break
        if proc.poll() is not None:
            raise RuntimeError(f'常驻预览进程退出 {proc.returncode}；详见 {WARM["log_path"]}')
        if time.monotonic()>=deadline:raise subprocess.TimeoutExpired('resident_preview',360)
        time.sleep(.25)
    run_write(run/'blender_process.json',dict(pid=proc.pid,returncode=0,timed_out=False,
        wall_seconds=time.monotonic()-start,gpu_uuid=uid,resident=True,
        log=WARM['log_path'],disk=disk_snapshot(run)))
    task.pop('pid',None)


def _worker(task):
    global RUNNING,WARM
    with LOCK:
        RUNNING=task['id'];task['state']='preflight'
    try:
        recipe=task['recipe'];selection=[task['frame']] if task['kind']=='preview' else None
        result=validate(recipe,selection,[task['rig_id']] if selection is not None else None)
        if result['status']!='passed':raise ValueError('; '.join(result['errors'][:8]))
        task['state']='assembly'
        base=OUTPUT/('previews' if task['kind']=='preview' else 'episodes')/task['id']
        # The task owns only its fresh output. The saved recipe is copied before work.
        manifest=freeze(recipe,base/'prepared',frames=selection,
                        rig_ids=[task['rig_id']] if selection is not None else None)
        if task.get('cancelled'):task['state']='cancelled';return
        task['state']='queued_gpu'
        uid=gpu_uuid()
        run=base/'run';run.mkdir(parents=True,exist_ok=False)
        if task['kind']=='preview':
            task['state']='rendering';task['output']=str(base)
            _resident_preview(task,base,run,uid)
            if task.get('cancelled'):task['state']='cancelled';return
            task['state']='completed';task['result']=str(run)
            return
        command=[sys.executable,str(ROOT/'tools/experiments/run_multirig_roi_a1.py'),
                 '--output',str(run),'--gpu-uuid',uid,'--budget-seconds',str(300 if task['kind']=='preview' else 2400),
                 '--rig-list-manifest',str(base/'prepared/rig_list.json')]
        if task['kind']=='preview':command+=['--preview','--frame-index',str(task['frame']),'--rig-id',task['rig_id']]
        task['state']='rendering';task['command']=command;task['output']=str(base)
        with (base/'worker.log').open('w') as log:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1'),start_new_session=True)
            task['pid']=proc.pid
            code=proc.wait(timeout=360 if task['kind']=='preview' else 2700)
        if task.get('cancelled'):task['state']='cancelled';return
        if code:raise RuntimeError(f'生产入口退出码 {code}，详见 {base}/worker.log')
        task['state']='completed';task['result']=str(run)
    except subprocess.TimeoutExpired:
        if task.get('pid'):os.killpg(task['pid'],signal.SIGTERM)
        if task['kind']=='preview':WARM=None
        task['state']='failed';task['error']='任务超时；现场已保留'
    except Exception as exc:
        if task['kind']=='preview' and WARM is not None and WARM['process'].poll() is not None:WARM=None
        task['state']='failed';task['error']=str(exc)
    finally:
        task.pop('pid',None)
        record_tasks()
        with LOCK:RUNNING=None


def start_task(kind,recipe,frame=None,rig_id=None):
    if kind not in ('preview','episode'):raise ValueError('invalid_task_kind')
    if kind=='preview' and (type(frame) is not int or not 0<=frame<=30 or rig_id not in {r['rig_id'] for r in recipe['rigs']}):
        raise ValueError('invalid_preview_frame_or_rig')
    check=validate(recipe,[frame] if kind=='preview' else None,[rig_id] if kind=='preview' else None)
    if check['status']!='passed':raise ValueError('; '.join(check['errors'][:8]))
    if kind=='preview':
        with LOCK:
            for tid,older in TASKS.items():
                if (older.get('kind'),older.get('state'),older.get('frame'),older.get('rig_id'))!=(
                        'preview','completed',frame,rig_id):continue
                base=OUTPUT/'previews'/tid
                frozen=base/'prepared/editor_recipe.json';run=base/'run'
                if (frozen.is_file() and read(frozen)==recipe and
                    (run/'preview_result.json').is_file() and
                    read(run/'preview_result.json').get('status')=='completed' and
                    all((run/'rigs'/rig_id/f'frame_{frame:03d}'/cam/'roi.png').is_file()
                        for cam in ('cam2','cam4'))):
                    return dict(task_id=tid,reused=True)
    with LOCK:
        if RUNNING is not None or any(t['state'] in ('queued','preflight','assembly','queued_gpu','rendering') for t in TASKS.values()):
            raise ValueError('已有任务在运行；本服务同时只运行一个GPU任务')
        tid='a3_'+uuid.uuid4().hex[:12]
        task=dict(id=tid,kind=kind,recipe=copy.deepcopy(recipe),revision=recipe['revision'],
                  frame=frame,rig_id=rig_id,state='queued',created=time.time())
        TASKS[tid]=task
        record_tasks()
    threading.Thread(target=_worker,args=(task,),daemon=True).start()
    return dict(task_id=tid,reused=False)


class Handler(BaseHTTPRequestHandler):
    def _reply(self,value,status=200):
        data=json.dumps(value,ensure_ascii=False,allow_nan=False).encode()
        self.send_response(status);self.send_header('Content-Type','application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)

    def _video(self,tid,rig):
        task=TASKS.get(tid)
        if task is None or not any(v['rig_id']==rig for v in episode_videos(task)):
            raise ValueError('unregistered_episode_video')
        path=OUTPUT/'episodes'/tid/'run/rigs'/rig/'pack/dual_roi.mp4'
        size=path.stat().st_size;start=0;end=size-1;partial=False
        if 'Range' in self.headers:
            match=re.fullmatch(r'bytes=(\d*)-(\d*)',self.headers['Range'])
            if match is None or not any(match.groups()):raise ValueError('invalid_video_range')
            left,right=match.groups();partial=True
            if left:
                start=int(left);end=min(size-1,int(right)) if right else size-1
            else:start=max(0,size-int(right))
            if start>=size or end<start:
                self.send_response(416);self.send_header('Content-Range',f'bytes */{size}')
                self.send_header('Content-Length','0');self.end_headers();return
        self.send_response(206 if partial else 200)
        self.send_header('Content-Type','video/mp4')
        self.send_header('Accept-Ranges','bytes')
        self.send_header('Content-Length',str(end-start+1))
        if partial:self.send_header('Content-Range',f'bytes {start}-{end}/{size}')
        self.end_headers()
        with path.open('rb') as source:
            source.seek(start);remaining=end-start+1
            while remaining:
                chunk=source.read(min(1024*1024,remaining))
                if not chunk:break
                self.wfile.write(chunk);remaining-=len(chunk)

    def do_GET(self):
        route=urlparse(self.path);query=parse_qs(route.query)
        try:
            if route.path=='/':
                data=WEB.read_bytes();self.send_response(200);self.send_header('Content-Type','text/html; charset=utf-8')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            if route.path=='/api/default':return self._reply(default_recipe(query.get('scene',['city'])[0]))
            if route.path=='/api/map':return self._reply(map_data(query.get('scene',['city'])[0]))
            if route.path=='/api/person':
                return self._reply(additional_person(query.get('scene',['city'])[0],int(query.get('id',['0'])[0])))
            if route.path=='/api/task':
                tid=query.get('id',[''])[0]
                if tid not in TASKS:raise ValueError('unknown_task')
                task={k:v for k,v in TASKS[tid].items() if k!='recipe'}
                if task.get('kind')=='episode' and task.get('state')=='failed' and task.get('output'):
                    repair=Path(task['output'])/'run/delivery_repair.json'
                    if repair.is_file() and read(repair).get('status')=='completed_after_metadata_finish_repair':
                        task['state']='completed_after_repair';task['result']=str(repair.parent)
                        task['repair_ref']=str(repair)
                if task.get('output'):
                    log=Path(task['output'])/'worker.log'
                    if task.get('warm_log'):log=Path(task['warm_log'])
                    if log.exists():task['log_tail']='\n'.join(log.read_text(errors='replace').splitlines()[-12:])
                task['videos']=episode_videos(TASKS[tid])
                return self._reply(task)
            if route.path=='/api/episodes':
                episodes=[dict(id=tid,revision=task['revision'],created=task['created'],
                               videos=episode_videos(task)) for tid,task in TASKS.items()
                          if task.get('kind')=='episode' and episode_videos(task)]
                return self._reply(sorted(episodes,key=lambda x:x['created'],reverse=True))
            if route.path.startswith('/api/video/'):
                parts=route.path.split('/')
                if len(parts)!=5:raise ValueError('invalid_video_url')
                return self._video(parts[3],parts[4])
            if route.path=='/api/load':
                key=query.get('id',[''])[0]
                if re.fullmatch(r'[a-z0-9_]{1,64}_r[1-9][0-9]*',key) is None:raise ValueError('invalid_recipe_id')
                return self._reply(read(OUTPUT/'recipes'/f'{key}.json'))
            if route.path.startswith('/api/image/'):
                parts=route.path.split('/')
                if len(parts)!=7:raise ValueError('invalid_image_url')
                _,api,image,tid,rig,frame,cam=parts
                if tid not in TASKS or TASKS[tid]['kind']!='preview' or rig!=TASKS[tid]['rig_id'] or frame!=str(TASKS[tid]['frame']) or cam not in ('cam2','cam4'):
                    raise ValueError('unregistered_preview_image')
                path=OUTPUT/'previews'/tid/'run'/'rigs'/rig/f'frame_{int(frame):03d}'/cam/'roi.png'
                data=path.read_bytes();self.send_response(200);self.send_header('Content-Type','image/png')
                self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data);return
            raise ValueError('unknown_endpoint')
        except Exception as exc:self._reply(dict(error=str(exc)),400)

    def do_POST(self):
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 0<size<=300_000:raise ValueError('invalid_payload_size')
            data=json.loads(self.rfile.read(size))
            if self.path=='/api/check':return self._reply(validate(data['recipe'],data.get('frames'),data.get('rig_ids')))
            if self.path=='/api/save':
                recipe=data['recipe'];check=validate(recipe)
                name=recipe['recipe_id']
                if re.fullmatch(r'[a-z0-9_]{1,64}',name) is None:raise ValueError('invalid_recipe_id')
                revision=recipe['revision']
                if type(revision) is not int or not 1<=revision<=9999:raise ValueError('invalid_revision')
                path=OUTPUT/'recipes'/f'{name}_r{revision}.json';path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('x') as f:json.dump(recipe,f,indent=2,ensure_ascii=False,allow_nan=False)
                return self._reply(dict(id=f'{name}_r{revision}',path=str(path),check=check))
            if self.path=='/api/start':
                recipe=data['recipe'];answer=start_task(data['kind'],recipe,data.get('frame'),data.get('rig_id'))
                return self._reply(answer,200 if answer['reused'] else 202)
            if self.path=='/api/cancel':
                tid=data['task_id'];task=TASKS.get(tid)
                if task is None:raise ValueError('unknown_task')
                pid=task.get('pid')
                if pid is not None:
                    task['cancelled']=True;os.killpg(pid,signal.SIGTERM);task['state']='cancelled'
                elif task['state'] in ('queued','preflight','assembly','queued_gpu'):
                    task['cancelled']=True;task['state']='cancelling'
                else:raise ValueError('任务已结束')
                record_tasks()
                return self._reply(dict(id=tid,state='cancelled'))
            raise ValueError('unknown_endpoint')
        except Exception as exc:self._reply(dict(error=str(exc)),400)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args()
    if not 1024<=args.port<=65535:parser.error('invalid_port')
    if TASK_RECORD.exists():
        TASKS.update(read(TASK_RECORD))
        for task in TASKS.values():
            if task['state'] not in ('completed','failed','cancelled'):
                task['state']='interrupted';task['error']='服务重启；任务不自动恢复，请核对保留目录'
        record_tasks()
    server=ThreadingHTTPServer(('127.0.0.1',args.port),Handler)
    print(f'http://127.0.0.1:{args.port}',flush=True)
    server.serve_forever()


if __name__=='__main__':main()
