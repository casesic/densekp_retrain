"""Compare one predeclared final checkpoint against frozen baselines, without selection."""
import argparse
import json
from pathlib import Path
from PIL import Image, ImageDraw


def read(path):
    return json.loads(path.read_text())


def compare(before, after):
    if before['count'] != after['count']:
        raise ValueError('Evaluation denominator changed')
    a,b=before['mean'],after['mean']
    return dict(count=before['count'],before=a,after=b,
        delta=b-a if a is not None and b is not None else None,
        percent_change=(b/a-1)*100 if a not in (None,0) and b is not None else None,
        before_nonfinite=before['nonfinite_count'],after_nonfinite=after['nonfinite_count'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root',type=Path,required=True)
    parser.add_argument('--trial',type=Path,required=True)
    args=parser.parse_args();data=args.data_root.resolve();trial=args.trial.resolve()
    old=read(data/'regression_reference_audit.json')
    old_dir=Path(old['checks'][0]['predictions'][0]['path']).parent.parent
    sources=[('main',data/'original_baseline'),('paired',data/'paired_baseline'),('regression',old_dir)]
    results={};reports={};denominator=[];out=trial/'comparison';out.mkdir(exist_ok=False)
    for name,baseline_dir in sources:
        baseline=read(baseline_dir/'report.json');final=read(trial/'evaluation'/name/'report.json')
        before={c['id']:c for c in baseline['cases']};after={c['id']:c for c in final['cases']}
        if set(before)!=set(after):raise ValueError('Frozen case set changed')
        batches={}
        for label,groups in baseline['by_batch'].items():
            actual=final['by_batch'][label]
            batches[label]={key:{weight:{unit:compare(value,actual[key][weight][unit])
                for unit,value in values.items()} for weight,values in group.items() if weight!='rows'}
                for key,group in groups.items()}
        pairs={key:{endpoint:compare(value,final['paired_aggregate'][key][endpoint])
            for endpoint,value in endpoints.items()} for key,endpoints in baseline['paired_aggregate'].items()
            if not key.startswith('combined/')}
        changes=[]
        for cid,c in before.items():
            d=after[cid]
            metric=compare(c['metrics']['supervised']['crop_256_px'],d['metrics']['supervised']['crop_256_px'])
            changes.append(dict(id=cid,action=c['action'],category=c['category'],batch=c['batch'],**metric))
        finite=[c for c in changes if c['delta'] is not None]
        results[name]=dict(by_batch=batches,dual_view=pairs,case_changes=changes,
            largest_regressions=sorted(finite,key=lambda c:c['delta'],reverse=True)[:8],
            largest_improvements=sorted(finite,key=lambda c:c['delta'])[:8],
            prediction_failures=final['prediction_failure_cases'])
        reports[name]=(baseline,final,before,after)
        denominator.append(dict(batch=name,rows=len(before),unchanged=True,
            exact_input_check=final['frozen_input_comparison']))
    baseline_pairs=read(data/'paired_baseline/occlusion_control_metrics.json')
    final_pairs=read(trial/'evaluation/paired/occlusion_control_metrics.json')
    results['paired']['fixed_box_control']={key:{condition:compare(value,final_pairs['aggregate'][key][condition])
        for condition,value in group.items()} for key,group in baseline_pairs['aggregate'].items()}
    thresholds={'layout/rear_overlap/supervised':18.449216,'all/occluded_in_crop':11.746464,
                'layout/complete_control/supervised':4.156706,'all/visible_supervised':5.355821}
    criteria=[]
    for key,limit in thresholds.items():
        value=results['main']['by_batch']['val_baseline'][key]['point_weighted']['crop_256_px']
        criteria.append(dict(metric=key,limit=limit,**value,passed=value['after'] is not None and value['after']<=limit))
    run=read(trial/'report.json')
    process=run['actual_updates']==400 and run['reload_max_abs_error']<=1e-5 and run['validation_steps']==[100,200,300,400]
    process=process and not any(result['prediction_failures'] for result in results.values())
    report=dict(status='completed',workflow_passed=process,
        effect_passed=all(c['passed'] for c in criteria),criteria=criteria,denominators=denominator,
        batches=results,primary_checkpoint=str(trial/'step_0400/checkpoint.ckpt'),
        units='full 256-square px and actual 1536x768 periodic ERP px',
        averaging='point means and equal-box means; not identity-weighted',
        attribution='This joint data/loss/finetuning experiment cannot isolate any single cause.',
        limits='Nine synthetic identity families, four pose source groups, limited forest, unknown skin donors, no independent test.')
    (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    # Fixed before-selected val controls plus before/after worst and regression examples.
    baseline,final,before,after=reports['main']
    sets=dict(fixed_val=[cid for cid in baseline['fixed_controls'] if cid.startswith('val_')],
              baseline_worst=baseline['worst_samples'],final_worst=final['worst_samples'],
              largest_regressions=[r['id'] for r in results['main']['largest_regressions']])
    for name,ids in sets.items():
        for page,start in enumerate(range(0,len(ids),4)):
            selected=ids[start:start+4];sheet=Image.new('RGB',(1024,536*len(selected)),'white')
            for i,cid in enumerate(selected):
                for j,cases in enumerate([before,after]):
                    folder=Path(cases[cid]['prediction']).parent
                    sheet.paste(Image.open(folder/'prediction.jpg'),(j*512,i*536+24))
                    value=cases[cid]['metrics']['supervised']['crop_256_px']['mean']
                    ImageDraw.Draw(sheet).text((j*512+4,i*536+4),
                        f'{cid} {"original" if j==0 else "step400"} mean={value}',fill='black')
            sheet.save(out/f'{name}_{page}.jpg',quality=94)
    print(json.dumps(dict(workflow_passed=process,effect_passed=report['effect_passed'],criteria=criteria),indent=2))


if __name__=='__main__':main()
