"""Four preselected controls: remove only foreground person 1, retain all world transforms."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASES=['train_003','train_008','val_000','val_004']


def main():
    root=Path(sys.argv[1]).resolve();recipe=json.loads((root/'layout_recipe.json').read_text())
    record_file=root/'occlusion_pairs.json'
    rows=json.loads(record_file.read_text())['pairs'] if record_file.exists() else []
    environment=dict(os.environ)
    environment.setdefault('CUDA_VISIBLE_DEVICES','0')
    environment.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MPLCONFIGDIR='/tmp/densekp-source-mpl')
    for name in BASES:
        if any(row['id']==name for row in rows):continue
        base=root/'layouts'/name;job=next(r for r in recipe['layouts'] if r['id']==name)
        if not (base/'validation.json').exists() or json.loads((base/'validation.json').read_text())['status']!='passed':
            rows.append(dict(id=name,partition=job['partition'],occluded=str(base),returncode=1,
                seconds=0.,status='not_applicable_failed_base_layout',reason='Preselected source layout failed; no control rendered'))
            (root/'occlusion_pairs.json').write_text(json.dumps(dict(pairs=rows,status='rendered_controls_visual_review_pending'),indent=2)+'\n')
            continue
        people=json.loads((base/'people.json').read_text())['people'];keep=[p for p in people if p['id']!=1]
        out=root/'layouts'/(name+'_clear')
        command=[sys.executable,str(ROOT/'tools/experiments/render_adult_candidates_ds.py'),
            '--assets',str(root/'assets'),'--output',str(out),'--characters',*[p['character_id'] for p in keep],
            '--pose-source-ids',*[p['pose_source_id'] for p in keep],'--scene',job['scene'],
            '--reuse-layout',str(base),'--width','1536','--samples','24','--device','CUDA']
        started=time.monotonic()
        recovered=out.exists()
        if recovered:
            if not (out/'groom_validation.json').exists() or json.loads((out/'validation.json').read_text())['status']!='passed':
                raise ValueError('Interrupted control is incomplete; preserve it and inspect before resuming')
            returncode=0
            seconds=json.loads((out/'report.json').read_text())['seconds']
        else:
            with (root/f'{name}_clear.log').open('w') as log:
                returncode=subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT).returncode
            seconds=time.monotonic()-started
        row=dict(id=name,partition=job['partition'],occluded=str(base),clear=str(out),target_person_id=3,
                 removed_person_id=1,returncode=returncode,seconds=seconds,recovered_complete_render=recovered,
                 selection='fixed before predictions; male/female targets on both sides, cycling/standing/walking across pairs')
        if returncode==0:
            for pid in [p['id'] for p in keep]:
                for part in ['vertices','clothing','groom']:
                    np.testing.assert_array_equal(np.load(base/'source'/f'person_{pid}_world_{part}.npy'),np.load(out/'source'/f'person_{pid}_world_{part}.npy'))
            before=json.loads((base/'rig.json').read_text());after=json.loads((out/'rig.json').read_text())
            if before['cameras']!=after['cameras']:raise ValueError('Pair camera changed')
            row['world_target_and_camera']='exactly_equal'
        rows.append(row);(root/'occlusion_pairs.json').write_text(json.dumps(dict(pairs=rows,status='rendered_controls_visual_review_pending'),indent=2)+'\n')
        print(name,returncode,flush=True)

if __name__=='__main__':main()
