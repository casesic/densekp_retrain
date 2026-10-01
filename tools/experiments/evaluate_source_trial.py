"""Run only the frozen final-checkpoint evaluation batches, sequentially on one UUID."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trial',type=Path,required=True);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--gpu-uuid',required=True);a=p.parse_args()
    trial=a.trial.resolve();data=a.data_root.resolve();run=json.loads((trial/'report.json').read_text())
    if run['actual_updates']!=400 or run['reload_max_abs_error']>1e-5:raise ValueError('Final workflow gate failed')
    out=trial/'evaluation';out.mkdir(exist_ok=False)
    (out/'evaluation_entry.py').write_bytes(Path(__file__).read_bytes())
    (out/'metric_entry.py').write_bytes((ROOT/'tools/validate/evaluate_candidate_baseline.py').read_bytes())
    env=dict(os.environ,CUDA_DEVICE_ORDER='PCI_BUS_ID',CUDA_VISIBLE_DEVICES=a.gpu_uuid,
             OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MPLCONFIGDIR='/tmp/densekp-trial-mpl')
    old=ROOT/'outputs/experiments/mixed_riders_pedestrians_20260930'
    riding=ROOT/'outputs/experiments/cycling_ten_adults_candidates_20260930'
    batches=[('main',[data/'train_index.json',data/'val_index.json'],['train_diagnostic','val_baseline'],data/'original_baseline/report.json'),
             ('paired',[data/'paired_train_index.json',data/'paired_val_index.json'],['paired_train','paired_val'],data/'paired_baseline/report.json'),
             ('regression',[riding/'candidate_index.json',old/'candidate_index.json'],['riding_A','mixed_B'],old/'original_baseline/report.json')]
    records=[]
    for name,indices,labels,reference in batches:
        command=[sys.executable,str(ROOT/'tools/validate/evaluate_candidate_baseline.py'),
            '--indices',*map(str,indices),'--labels',*labels,'--output',str(out/name),
            '--checkpoint',str(trial/'step_0400/checkpoint.ckpt'),'--model-config',str(trial/'model_config.yaml'),
            '--reference-report',str(reference),'--device','cuda:0']
        if name!='regression':command+=['--split-manifest',str(data/'split_v2.json')]
        started=time.monotonic()
        with (out/f'{name}.log').open('w') as log:
            result=subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(batch=name,command=command,returncode=result.returncode,seconds=time.monotonic()-started))
        (out/'execution.json').write_text(json.dumps(dict(gpu_uuid=a.gpu_uuid,batches=records),indent=2)+'\n')
        if result.returncode:raise RuntimeError(f'{name} failed; preserved log, no skipping/retry')
        print(f'{name} evaluation complete',flush=True)
    subprocess.run([sys.executable,str(ROOT/'tools/validate/summarize_source_occlusion_pairs.py'),str(out/'paired')],cwd=ROOT,env=env,check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/validate/compare_source_trial.py'),'--data-root',str(data),'--trial',str(trial)],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
