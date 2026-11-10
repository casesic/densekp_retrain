"""Run only this trial's Blender child with budget/resource accounting."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/experiments/direct_erp_roi_trial_20261009'
BLENDER='/mnt/data1/khb/app/blender-5.1.1-linux-x64/blender'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['full','tiny','region']);args=ap.parse_args()
    f=json.loads((OUT/'frozen_comparison.json').read_text());uuid=f['source_render']['gpu_uuid']
    run_file=OUT/f'{args.stage}_process.json';assert not run_file.exists(),'Never overwrite a run'
    stages=[json.loads(p.read_text()) for p in OUT.glob('*_process.json')]
    used=sum(x['wall_seconds'] for x in stages)+json.loads((OUT/'scene_settings.json').read_text())['elapsed_seconds'];remaining=1200-used
    if remaining<30:raise RuntimeError('Insufficient remaining render budget')
    queries=dict(gpus=['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv'],
        processes=['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv'],memory=['free','-h'],disk=['df','-h',str(OUT),'/tmp'])
    before={k:subprocess.check_output(v,text=True) for k,v in queries.items()}
    own=[r for r in before['gpus'].splitlines()[1:] if uuid in r];assert len(own)==1
    used_mib=int(own[0].split(',')[2].strip().split()[0]);util=int(own[0].split(',')[4].strip().split()[0])
    if used_mib>512 or util>5:raise RuntimeError(f'Reserved GPU busy: {own}; do not interfere with other jobs')
    (OUT/f'{args.stage}_resources_before.json').write_text(json.dumps(dict(remaining_render_seconds=remaining,before=before),indent=2))
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=uuid,OMP_NUM_THREADS='4')
    cmd=[BLENDER,'-b','-t','4','--python-exit-code','1','--python',str(ROOT/'tools/render/render_direct_erp_roi_trial.py'),'--','--stage',args.stage,'--root',str(OUT)]
    started=time.monotonic();samples=[];timeout=False
    with (OUT/f'{args.stage}_blender.log').open('w') as log:
        p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            dt=time.monotonic()-started
            if dt>remaining:
                p.kill();timeout=True;break  # Only our own child, never A2/other processes.
            raw=subprocess.check_output(['nvidia-smi','-i',uuid,'--query-gpu=memory.used,utilization.gpu,power.draw,temperature.gpu','--format=csv,noheader,nounits'],text=True).strip()
            samples.append(dict(seconds=dt,gpu=raw))
            time.sleep(2)
        code=p.wait()
    result=dict(stage=args.stage,command=cmd,wall_seconds=time.monotonic()-started,returncode=code,timed_out=timeout,
        remaining_before_seconds=remaining,gpu_uuid=uuid,gpu_samples=samples,gpu_columns=['memory.used_MiB','utilization_percent','power_W','temperature_C'],
        gpu_memory_sampled_peak_MiB=max(int(x['gpu'].split(',')[0]) for x in samples) if samples else None)
    run_file.write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='gpu_samples'},indent=2))
    raise SystemExit(code or int(timeout))

if __name__=='__main__':main()
