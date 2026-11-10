"""One bounded consumer-failure recovery; reuse frame0, never rerender it."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
from tools.experiments.run_continuous_roi import BLENDER,disk_snapshot,write


def main():
    root=Path(sys.argv[1]).resolve();config=json.loads((root/'run_config.json').read_text());uuid=config['gpu_uuid']
    if (root/'attempt_2').exists():raise RuntimeError('Only one targeted retry allowed')
    ledger=[json.loads(x) for x in (root/'render_calls.jsonl').read_text().splitlines()]
    assert len([r for r in ledger if r['event']=='started'])==2
    assert all(r['frame_index']==0 and r['kind']=='roi' for r in ledger)
    gpu=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
    if int(gpu.split(',')[0])>512 or int(gpu.split(',')[1])>5:raise RuntimeError('GPU busy; no retry launched')
    queries=dict(gpus=['nvidia-smi'],processes=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],memory=['free','-h'],disk=['df','-h',str(root),'/tmp'])
    resources={k:subprocess.check_output(v,text=True) for k,v in queries.items()}
    # Preserve the earlier resource rule: only the known idle desktop process.
    for row in resources['processes'].splitlines()[1:]:
        if uuid in row and int(row.split(',')[1])!=43491:raise RuntimeError('Other GPU task active')
    write(root/'retry_resources_before.json',resources)
    start=time.monotonic();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/densekp-roi-mpl')
    def remaining():return config['deadline_monotonic']-time.monotonic()
    command=[sys.executable,str(ROOT/'tools/experiments/consume_continuous_roi.py'),'frame','--root',str(root),'--frame','0']
    with (root/'frame_000/consume_retry.log').open('w') as log:
        subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=max(.1,remaining()))
    frame0_consume=time.monotonic()-start
    write(root/'frame_000/consumer_process_retry.json',dict(seconds=frame0_consume,command=command,reused_original_render=True))
    job=json.loads((root/'job.json').read_text());job['frames']=job['frames'][1:];job['source_dirs']=job['source_dirs'][1:];job['remaining_budget_seconds']=remaining()
    write(root/'job_attempt_2.json',job)
    command=[BLENDER,'-b','-t','4','--python-exit-code','1','--python',str(ROOT/'tools/render/render_continuous_roi.py'),'--','--root',str(root),'--job-file','job_attempt_2.json','--run-tag','attempt_2']
    begin=time.monotonic();samples=[]
    with (root/'blender_attempt_2.log').open('w') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=dict(env,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT)
        while proc.poll() is None:
            if remaining()<=0:proc.kill();break
            raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(seconds=time.monotonic()-begin,gpu=raw,**disk_snapshot(root)));time.sleep(1)
        code=proc.wait()
    wall=time.monotonic()-begin
    write(root/'blender_attempt_2_process.json',dict(returncode=code,pid=proc.pid,wall_seconds=wall,command=command,gpu_uuid=uuid,gpu_samples=samples,
        sampled_peak_GPU_MiB=max(int(r['gpu'].split(',')[0]) for r in samples)))
    if code:raise SystemExit(code)
    begin=time.monotonic();command=[sys.executable,str(ROOT/'tools/experiments/consume_continuous_roi.py'),'finish','--root',str(root)]
    with (root/'finish.log').open('w') as log:
        finish=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=max(.1,remaining()))
    prior=json.loads((root/'blender_process.json').read_text())['wall_seconds']
    write(root/'task_wall.json',dict(initial_failed_process_seconds=prior,frame0_reconsume_seconds=frame0_consume,
        second_blender_process_wall_seconds=wall,finish_process_seconds=time.monotonic()-begin,finish_returncode=finish.returncode,
        compute_wall_seconds_including_failure_and_retry=prior+(time.monotonic()-start),
        conservative_remaining_deadline_seconds=remaining(),blender_start_count=2,
        recovery='original two frame0 buffers reused; one targeted retry process rendered frames1..10 only'))
    write(root/'resources_after.json',{k:subprocess.check_output(v,text=True) for k,v in queries.items() if k!='processes'})
    raise SystemExit(finish.returncode)

if __name__=='__main__':main()
