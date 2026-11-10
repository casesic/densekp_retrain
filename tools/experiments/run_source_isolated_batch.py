"""Bounded layouts from a frozen split; pilots must be reviewed before expansion."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/"src")]
from densekp_retrain.data.source_split import validate_split


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def plan(root):
    split=json.loads((root/'split_v2.json').read_text());validate_split(split)
    assets=json.loads((root/'assets/report.json').read_text())
    rng=np.random.default_rng(930);layouts=[]
    for side,count,slots in [('train',36,[0,1,2,3]),('val',20,[0,2,3])]:
        ids=[c['character_id'] for c in split['characters'] if c['partition']==side]
        for i in range(count):
            chars=rng.permutation(ids)[:len(slots)].tolist()
            actions=['cycling','walking','standing']+(['cycling','walking','standing'][i%3:i%3+1] if side=='train' else [])
            actions=np.roll(actions,i%len(slots)).tolist()
            poses=[]
            for cid,action in zip(chars,actions):
                row=next(d for d in assets['derivatives'] if d['character_id']==cid and d['action']==action)
                if row['partition']!=side:raise ValueError('Asset partition mismatch')
                poses.append(row['pose_source_id'])
            # Discrete meaningful changes, not a frame-by-frame sweep.
            variation=dict(road_offset_m=[0.,1.5,3.,-.8][(i//3)%4],
                edge_distance_scale=[1.,1.15,.9,1.3][i%4],
                rear_distance_scale=[1.,1.15,1.25][(i//2)%3],
                person_yaw_delta_deg=[0.,-25.,25.,45.,-45.][(i//4)%5])
            layouts.append(dict(id=f'{side}_{i:03d}',partition=side,characters=chars,actions=actions,
                pose_source_ids=poses,slots=slots,scene='cycling_forest' if i%4==1 else 'urban_street_assets',
                control_distance_m=[3.8,5.,6.5][(i//2)%3],variation=variation,pilot=i==0))
    return dict(status='layout_recipe_frozen_before_render_and_prediction',seed=930,
        split_manifest=str(root/'split_v2.json'),samples=24,width=1536,layouts=layouts,
        note='Paired fixed-box diagnostics added separately within bounded scale; no full Cartesian expansion')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--phase',choices=['pilots','expand'],required=True);p.add_argument('--limit',type=int)
    a=p.parse_args();root=a.root.resolve()
    for name in ['body_clothing_readback','groom_binding_readback']:
        if json.loads((root/'assets'/f'{name}.json').read_text())['status']!='passed':raise ValueError(name)
    if not (root/'layout_recipe.json').exists():write(root/'layout_recipe.json',plan(root))
    recipe=json.loads((root/'layout_recipe.json').read_text())
    if a.phase=='expand':
        if json.loads((root/'pilot_review.json').read_text())['status']!='passed':raise ValueError('Pilot review required')
    jobs=[r for r in recipe['layouts'] if r['pilot']==(a.phase=='pilots')]
    if a.limit:jobs=jobs[:a.limit]
    (root/'layouts').mkdir(exist_ok=True);(root/'layout_configs').mkdir(exist_ok=True)
    records=[];status_file=root/f'{a.phase}_execution.json'
    if status_file.exists():records=json.loads(status_file.read_text())['layouts']
    for job in jobs:
        if any(r['id']==job['id'] for r in records):continue
        out=root/'layouts'/job['id'];config=root/'layout_configs'/f"{job['id']}.json"
        write(config,job['variation'])
        command=[sys.executable,str(ROOT/'tools/experiments/render_adult_candidates_ds.py'),
            '--assets',str(root/'assets'),'--output',str(out),'--characters',*job['characters'],
            '--pose-source-ids',*job['pose_source_ids'],'--layout-slots',*[str(v) for v in job['slots']],
            '--scene',job['scene'],'--pedestrian-layout','--control-distance',str(job['control_distance_m']),
            '--layout-config',str(config),'--width','1536','--samples','24','--device','CUDA',
            '--output-mode','full','--final-width','1536','--final-height','768']
        start=time.monotonic();print('START',job['id'],job['characters'],job['actions'],flush=True)
        with (root/f"{job['id']}.log").open('w') as log:
            done=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        records.append(dict(id=job['id'],directory=str(out),returncode=done.returncode,
            status='numerical_complete_visual_pending' if done.returncode==0 else 'failed_preserved',
            seconds=time.monotonic()-start,command=command))
        write(status_file,dict(layouts=records,phase=a.phase,visual_acceptance='separate_required'))
        print('DONE',records[-1]['id'],done.returncode,records[-1]['seconds'],flush=True)
        if a.phase=='pilots' and done.returncode:raise RuntimeError('Pilot failed; inspect before expansion')

if __name__=='__main__':main()
