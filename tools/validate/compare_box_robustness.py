"""Compare accuracy and cross-box drift on fixed common GT in original ERP units."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.validate.evaluate_candidate_baseline import summary


def read(path):return json.loads(path.read_text())
def write(path,x):path.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def distance(a,b,width=1536):
    delta=np.asarray(a,float)-np.asarray(b,float)
    delta[:,0]=(delta[:,0]+width/2)%width-width/2
    error=np.linalg.norm(delta,axis=1)
    error[~np.isfinite(delta).all(1)]=np.inf
    return error


def challenge(report):
    groups=defaultdict(list)
    for case in report['cases']:groups[case['parent_row']].append(case)
    rows=[]; targets=[];buckets=defaultdict(lambda:defaultdict(list))
    for parent,cases in sorted(groups.items()):
        arrays={c['id']:dict(np.load(c['prediction'])) for c in cases}
        reference=next(c for c in cases if c['box_variant']=='reference')
        ref=arrays[reference['id']]
        for z in arrays.values():
            np.testing.assert_array_equal(z['target_erp_xy'],ref['target_erp_xy'])
            np.testing.assert_array_equal(z['in_view'],ref['in_view'])
        common=np.logical_and.reduce([z['supervised'] for z in arrays.values()])
        multiplier=256/reference['reference_crop_side']
        target_values=[]
        for case in cases:
            z=arrays[case['id']];mask=z['supervised'];pair=mask&ref['supervised']
            error=distance(z['prediction_erp_xy'],z['target_erp_xy'])
            drift=distance(z['prediction_erp_xy'],ref['prediction_erp_xy'])
            nonfinite=not np.isfinite(z['prediction_raw']).all()
            scorable=int(common.sum())>=10
            common_mean=summary(error[common]*multiplier)['mean']
            severe=(nonfinite or common_mean is None or common_mean>32) if scorable else None
            record=dict(id=case['id'],parent_row=parent,variant=case['box_variant'],action=case['action'],
                category=case['category'],reference_crop_side=reference['reference_crop_side'],
                all_supervised_count=int(mask.sum()),in_view_count=int(z['in_view'].sum()),
                crop_truncation_fraction=float((z['in_view']&~mask).sum()/z['in_view'].sum()) if z['in_view'].any() else None,
                common_count=int(common.sum()),pair_common_count=int(pair.sum()),nonfinite_prediction=nonfinite,
                all_valid_erp=summary(error[mask]),all_valid_reference_px=summary(error[mask]*multiplier),
                common_erp=summary(error[common]),common_reference_px=summary(error[common]*multiplier),
                reference_pair_common_erp=summary(error[pair]),
                drift_erp=summary(drift[common]),drift_reference_px=summary(drift[common]*multiplier),
                severe=severe,box=case['given_bbox_xyxy'])
            rows.append(record);target_values.extend(error[common])
            names=['all',f'action/{case["action"]}',f'layout/{case["category"]}',f'variant/{case["box_variant"]}']
            for name in names:
                bucket=buckets[name]
                bucket['common_erp'].extend(error[common]);bucket['common_reference_px'].extend(error[common]*multiplier)
                bucket['all_valid_erp'].extend(error[mask])
                if common.any():bucket['equal_box_common_erp'].append(float(error[common].mean()))
                if case['box_variant']!='reference':bucket['changed_box_drift_erp'].extend(drift[common])
                if severe is not None:bucket['severe'].append(int(severe))
                else:bucket['unscorable'].append(case['id'])
        targets.append(dict(parent_row=parent,variants=len(cases),common_count=int(common.sum()),
            common_erp=summary(target_values),all_variants_severe=all(r['severe'] is True for r in rows if r['parent_row']==parent)))
    aggregate={name:{k:summary(v) for k,v in values.items() if k not in ('severe','unscorable')}|
        dict(severe_cases=sum(values['severe']),scorable_cases=len(values['severe']),unscorable_cases=values['unscorable'])
        for name,values in buckets.items()}
    transitions=[]
    for parent in sorted(groups):
        target_rows=[r for r in rows if r['parent_row']==parent]
        reference=next(r for r in target_rows if r['variant']=='reference')
        transitions.append(dict(parent_row=parent,reference_severe=reference['severe'],
            changed_boxes=sum(r['variant']!='reference' for r in target_rows),
            induced_severe=sum(reference['severe'] is False and r['severe'] is True for r in target_rows if r['variant']!='reference')))
    return dict(rows=rows,targets=targets,aggregate=aggregate,transitions=transitions,
        reference_severe_targets=sum(t['reference_severe'] is True for t in transitions),
        box_induced_severe_cases=sum(t['induced_severe'] for t in transitions),
        equal_target_common_erp=summary([t['common_erp']['mean'] if t['common_erp']['mean'] is not None else np.inf for t in targets if t['common_count']]),
        scope='18 held-out targets, fixed accepted box variants; common GT is intersection of all variants per target; no error-dependent filtering')


def delta(a,b):
    return dict(before=a,after=b,delta=b-a if a is not None and b is not None else None)


def report_comparison(reports):
    result={}
    batches=set().union(*(r['by_batch'] for r in reports.values()))
    for batch in batches:
        if not all(batch in r['by_batch'] for r in reports.values()):continue
        groups={}
        for name in reports['A']['by_batch'][batch]:
            groups[name]={model:r['by_batch'][batch][name] for model,r in reports.items()}
        result[batch]=dict(groups=groups,dual_view={model:{k:v for k,v in r['paired_aggregate'].items() if k.startswith(batch+'/')} for model,r in reports.items()})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True)
    p.add_argument('--output-name',default='comparison');a=p.parse_args()
    root=a.experiment.resolve();out=root/a.output_name;out.mkdir(exist_ok=False)
    evaluation=root/'evaluation_v2'
    data=ROOT/'outputs/experiments/source_isolated_pilot_20260930'
    plain=ROOT/'outputs/experiments/source_isolated_trial_400_gpu1_20261001'
    old=ROOT/'outputs/experiments/mixed_riders_pedestrians_20260930'
    val={'A':read(data/'original_baseline/report.json'),'B':read(plain/'evaluation/main/report.json'),
         'C':read(evaluation/'val_C/report.json')}
    regression={'A':read(old/'original_baseline/report.json'),'B':read(plain/'evaluation/regression/report.json'),
                'C':read(evaluation/'regression_C/report.json')}
    challenges={m:read(evaluation/f'challenge_{m}/report.json') for m in 'ABC'}
    statistics={m:challenge(r) for m,r in challenges.items()}
    gates={}
    for name,key,limit in [('rear','layout/rear_overlap/supervised',18.449216),
            ('occluded','all/occluded_in_crop',11.746464),('complete','layout/complete_control/supervised',4.156706),
            ('visible','all/visible_supervised',5.355821)]:
        values={m:r['by_batch']['val_baseline'][key]['point_weighted']['crop_256_px']['mean'] for m,r in val.items()}
        gates[name]=dict(limit=limit,values=values,C_passed=values['C'] is not None and values['C']<=limit)
    history={}
    for m,path in [('A',ROOT/'outputs/experiments/ds_occlusion_box_control_20260928'),('B',evaluation/'history_B'),('C',evaluation/'history_C')]:
        hist=read(path/'report.json');history[m]=hist
        # Verify identical historical GT and visible rows. Keep the original 22-point scoring contract.
        for case in hist['cases']:
            z=np.load(path/f'{case["image"]}_{case["box"]}.npz')
            base=np.load(ROOT/'outputs/experiments/ds_occlusion_box_control_20260928'/f'{case["image"]}_{case["box"]}.npz')
            for key in ['target_erp_xy','target_crop_xy','affine','visible']:np.testing.assert_array_equal(z[key],base[key])
    all_stats={m:s['aggregate']['all'] for m,s in statistics.items()}
    robust={m:dict(lower_error=all_stats['C']['common_erp']['mean']<all_stats[m]['common_erp']['mean'],
        lower_drift=all_stats['C']['changed_box_drift_erp']['mean']<all_stats[m]['changed_box_drift_erp']['mean'],
        fewer_severe=all_stats['C']['severe_cases']<all_stats[m]['severe_cases']) for m in 'AB'}
    # Fixed and worst examples: all were frozen before inference; ranking is presentation only.
    lookup={m:{c['id']:c for c in r['cases']} for m,r in challenges.items()}
    metrics={m:{c['id']:c for c in s['rows']} for m,s in statistics.items()}
    ids=list(lookup['A'])
    changed=[i for i in ids if lookup['A'][i]['box_variant']!='reference']
    scored=[i for i in changed if all(metrics[m][i]['common_reference_px']['mean'] is not None for m in 'ABC')]
    orders=dict(fixed=ids[:8],worst_C=sorted(scored,key=lambda i:metrics['C'][i]['common_reference_px']['mean'],reverse=True)[:6],
        improved=sorted(scored,key=lambda i:metrics['C'][i]['common_reference_px']['mean']-metrics['A'][i]['common_reference_px']['mean'])[:6],
        regressed=sorted(scored,key=lambda i:metrics['C'][i]['common_reference_px']['mean']-metrics['A'][i]['common_reference_px']['mean'],reverse=True)[:6])
    for label,chosen in orders.items():
        sheet=Image.new('RGB',(768,280*len(chosen)),'black');draw=ImageDraw.Draw(sheet)
        for row,cid in enumerate(chosen):
            for col,m in enumerate('ABC'):
                case=lookup[m][cid];panel=Image.open(Path(case['prediction']).parent/'prediction.jpg').resize((256,256))
                sheet.paste(panel,(col*256,row*280+24))
                value=metrics[m][cid]['common_reference_px']['mean']
                draw.text((col*256+2,row*280+4),f'{m} {cid} ref err={value:.2f}' if value is not None else f'{m} {cid} null',fill='white')
        sheet.save(out/f'{label}.jpg')
    changes=[]
    for cid in ids:
        changes.append(dict(id=cid,parent_row=lookup['A'][cid]['parent_row'],variant=lookup['A'][cid]['box_variant'],
            values={m:metrics[m][cid]['common_reference_px']['mean'] for m in 'ABC'}))
    result=dict(challenge=statistics,challenge_primary=all_stats,robustness_predeclared_checks=robust,
        original_val=report_comparison(val),regression=report_comparison(regression),original_four_gates=gates,
        history=history,challenge_changes=changes,visualization_selection=orders,
        severe_rule=read(root/'data_v2/config.json')['severe_failure'],
        denominators='All common sets frozen by GT only; raw nonfinite yields failure/null not denominator removal; equal-box is not equal-identity',
        limitations='Synthetic perturbations;18 targets; source-limited holdout; history/integration not independent val; no actual detector-error GT')
    write(out/'report.json',result)
    print(json.dumps(dict(primary=all_stats,gates=gates,robustness=robust),indent=2))


if __name__=='__main__':main()
