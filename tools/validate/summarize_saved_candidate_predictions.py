"""Finish aggregation of a fully saved batch after a report-only failure; no inference."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.validate.evaluate_candidate_baseline import aggregate, summary, write
from tools.validate.occlusion_metrics import evaluation_masks


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--label',required=True);a=p.parse_args()
    started=time.monotonic();out=a.directory.resolve()
    if (out/'report.json').exists():raise FileExistsError('Do not overwrite a completed report')
    baseline=json.loads(a.reference.read_text());refs={c['id']:c for c in baseline['cases'] if c['batch']==a.label}
    index=json.loads((out/f'{a.label}_frozen_index.json').read_text())
    cases=[json.loads((out/f'{a.label}_{i:03d}/case.json').read_text()) for i in range(len(index['rows']))]
    assert set(refs)=={c['id'] for c in cases}
    states={};pairs={};fixed={}
    for c in cases:
        z=np.load(c['prediction']);old=np.load(refs[c['id']]['prediction'])
        for key in ['target_crop_xy','target_erp_xy','target_world_xyz','affine','visible','in_view','supervised','network_input']:
            np.testing.assert_array_equal(z[key],old[key])
        assert z['prediction_raw'].shape==(138,3) and np.isfinite(z['prediction_raw']).all()
        masks=evaluation_masks(z['in_view'],z['visible'],z['target_crop_xy']);masks['visible_supervised']=masks['supervised']&z['visible']
        states[c['id']]=dict(crop=z['crop_error_px'],erp=z['erp_periodic_error_px'],masks=masks,visible=z['visible'])
        pairs.setdefault((c['batch'],c['scene'],c['person_id']),{})[c['camera']]=c['id']
        fixed.setdefault((c['action'],c['category']),c['id'])
    pair_results=[];buckets={}
    for (batch,scene,pid),views in pairs.items():
        result=dict(batch=batch,scene=scene,person_id=pid,views=views)
        if len(views)!=2:
            result['status']='missing accepted given-box view; not detector failure';pair_results.append(result);continue
        left,right=[states[views[n]] for n in ('cam2','cam4')]
        shared=left['masks']['supervised']&right['masks']['supervised']
        groups=dict(both_supervised=shared,both_visible=shared&left['visible']&right['visible'],
            one_visible=shared&(left['visible']^right['visible']),both_occluded=shared&~left['visible']&~right['visible'],
            both_in_view_diagnostic=left['masks']['all_in_view']&right['masks']['all_in_view'])
        result.update(status='scored',metrics={})
        for name,mask in groups.items():
            result['metrics'][name]={}
            for unit in ('crop','erp'):
                mean=(left[unit][mask]+right[unit][mask])/2;worst=np.maximum(left[unit][mask],right[unit][mask])
                result['metrics'][name][unit]=dict(mean_endpoint=summary(mean),worse_endpoint=summary(worst))
                for prefix in (batch,'combined'):
                    bucket=buckets.setdefault(f'{prefix}/{name}/{unit}',dict(mean=[],worst=[]));bucket['mean'].extend(mean);bucket['worst'].extend(worst)
        pair_results.append(result)
    write(out/'report.json',dict(status='completed',cases=cases,by_batch={a.label:aggregate(cases,states)},
        combined=aggregate(cases,states),checkpoint=json.loads((out/'checkpoint_load.json').read_text()),
        pairs=pair_results,paired_aggregate={k:dict(mean_endpoint=summary(v['mean']),worse_endpoint=summary(v['worst'])) for k,v in buckets.items()},
        prediction_failure_cases=[],fixed_controls=list(fixed.values()),
        recovery='79 predictions completed before overbroad train+val reference-set check failed; exact selected-batch denominator and arrays verified; CPU aggregation only',
        reference_report=str(a.reference),seconds=time.monotonic()-started))


if __name__=='__main__':main()
