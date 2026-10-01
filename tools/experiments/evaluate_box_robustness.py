"""Run missing fixed A/B/C predictions, leaving previous predictions untouched."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);p.add_argument('--physical-gpu',default='1')
    p.add_argument('--output-name',default='evaluation_v2');p.add_argument('--only',nargs='+');a=p.parse_args()
    root=a.experiment.resolve();trial=root/'training';out=root/a.output_name;out.mkdir(exist_ok=bool(a.only))
    report=json.loads((trial/'report.json').read_text())
    if report['actual_updates']!=400 or report['reload_max_abs_error']>1e-5:raise ValueError('Training/reload not accepted')
    data=ROOT/'outputs/experiments/source_isolated_pilot_20260930'
    plain=ROOT/'outputs/experiments/source_isolated_trial_400_gpu1_20261001'
    old=ROOT/'outputs/experiments/mixed_riders_pedestrians_20260930'
    riding=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930'
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=a.gpu_uuid,
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLCONFIGDIR='/tmp/densekp-box-mpl')
    commands=[]
    for name,checkpoint in [('A',ROOT/'assets_external/densekp.ckpt'),
            ('B',plain/'step_0400/checkpoint.ckpt'),('C',trial/'step_0400/checkpoint.ckpt')]:
        command=[sys.executable,str(ROOT/'tools/validate/evaluate_candidate_baseline.py'),
            '--indices',str(root/'data_v2/val_index.json'),'--labels','challenge',
            '--split-manifest',str(data/'split_v2.json'),'--checkpoint',str(checkpoint),
            '--output',str(out/f'challenge_{name}')]
        if name!='A':command+=['--reference-report',str(out/'challenge_A/report.json')]
        commands.append((f'challenge_{name}',command))
    commands.append(('val_C',[sys.executable,str(ROOT/'tools/validate/evaluate_candidate_baseline.py'),
        '--indices',str(data/'val_index.json'),'--labels','val_baseline','--split-manifest',str(data/'split_v2.json'),
        '--checkpoint',str(trial/'step_0400/checkpoint.ckpt'),'--reference-report',str(data/'original_baseline/report.json'),
        '--output',str(out/'val_C')]))
    commands.append(('regression_C',[sys.executable,str(ROOT/'tools/validate/evaluate_candidate_baseline.py'),
        '--indices',str(riding/'candidate_index.json'),str(old/'candidate_index.json'),'--labels','riding_A','mixed_B',
        '--checkpoint',str(trial/'step_0400/checkpoint.ckpt'),'--reference-report',str(old/'original_baseline/report.json'),
        '--output',str(out/'regression_C')]))
    for name,checkpoint in [('B',plain/'step_0400/checkpoint.ckpt'),('C',trial/'step_0400/checkpoint.ckpt')]:
        commands.append((f'history_{name}',[sys.executable,str(ROOT/'tools/validate/evaluate_ds_box_control.py'),
            '--control-root',str(ROOT/'outputs/experiments/ds_occlusion_control_20260928'),
            '--old-root',str(ROOT/'outputs/experiments/ds_scene_pair_20260928_final/urban_street_assets'),
            '--checkpoint',str(checkpoint),'--output',str(out/f'history_{name}')]))
    records=json.loads((out/'execution.json').read_text()) if a.only else []
    for name,command in commands:
        if a.only and name not in a.only:continue
        # Read-only preflight for each GPU inference process. Never reset another device.
        status=subprocess.run(['nvidia-smi','-i',a.physical_gpu,'--query-gpu=uuid,memory.used,memory.free,utilization.gpu','--format=csv'],capture_output=True,text=True)
        processes=subprocess.run(['nvidia-smi','-i',a.physical_gpu,'--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv'],capture_output=True,text=True)
        if status.returncode or processes.returncode or a.gpu_uuid not in status.stdout:raise RuntimeError('GPU preflight failed')
        started=time.monotonic()
        with (out/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(name=name,command=command,returncode=result.returncode,seconds=time.monotonic()-started,
            gpu=status.stdout,compute_processes=processes.stdout))
        (out/'execution.json').write_text(json.dumps(records,indent=2)+'\n')
        if result.returncode:raise RuntimeError(f'{name} failed; stop without retry')
        print(name,'complete',flush=True)


if __name__=='__main__':main()
