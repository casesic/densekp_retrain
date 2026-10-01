"""Summarize frozen rows and measured failures; do not select data from predictions."""
import json
from pathlib import Path
import sys


def counts(rows):
    return dict(rows=len(rows), images=len({r['image'] for r in rows}),
        layouts=len({r['scene'] for r in rows}),
        identity_families=sorted({r['identity_family_id'] for r in rows}),
        source_groups=sorted({r['source_group'] for r in rows}),
        pose_ids=sorted({r['pose_source_id'] for r in rows}),
        supervised=sum(r['supervised_points'] for r in rows),
        occluded=sum(r['supervised_occluded_points'] for r in rows),
        visible=sum(r['supervised_points']-r['supervised_occluded_points'] for r in rows),
        in_view_outside_full_crop=sum(r['in_view_outside_full_crop'] for r in rows),
        lens_truncated_rows=sum(r['lens_truncated_gt'] for r in rows),
        neighbor_in_crop_rows=sum(r['neighbor_crop_pixels']>0 for r in rows))


def main():
    root=Path(sys.argv[1]).resolve()
    indices={s:json.loads((root/f'{s}_index.json').read_text()) for s in ['train','val']}
    result=dict(scope='initial source/identity holdout; no test set', partitions={})
    all_rows=[]
    for side,index in indices.items():
        rows=index['rows'];all_rows.extend(rows)
        groups={}
        for field in ['action','category','character_id','pose_source_id']:
            groups[field]={value:counts([r for r in rows if r[field]==value])
                           for value in sorted({r[field] for r in rows})}
        groups['scene_asset']={name:counts([r for r in rows if json.loads((Path(r['scene'])/'report.json').read_text())['scene']==name])
                               for name in ['cycling_forest','urban_street_assets']}
        groups['action_by_layout']={f'{action}/{category}':counts([r for r in rows if r['action']==action and r['category']==category])
            for action in ['cycling','standing','walking']
            for category in ['rear_overlap','complete_control','edge_partial']}
        result['partitions'][side]=dict(**counts(rows),groups=groups,
            excluded_rows=index['excluded_rows'],no_box_views=index['no_box_views'],
            excluded_layouts=index['excluded_layouts'])
    result['combined']=counts(all_rows)
    rendered=[]
    for directory in sorted((root/'layouts').iterdir()):
        images=[str(p) for p in sorted(directory.glob('cam*/normal_erp.png'))]
        if images:
            rendered.append(dict(layout=directory.name,images=images,
                primary_accepted=any(r['scene']==str(directory) for r in all_rows),
                paired_diagnostic=directory.name.endswith('_clear')))
    result['rendered_inventory']=dict(layouts=rendered,
        saved_erp_files=sum(len(r['images']) for r in rendered),
        meaning='Saved final ERP files, including failed/excluded layouts; reused intermediate EXR source is not another render.')
    result['derivative_coverage']=[dict(character_id=character,action=action,
        accepted_rows=sum(r['character_id']==character and r['action']==action for r in all_rows),
        status='accepted_in_some_layout' if any(r['character_id']==character and r['action']==action for r in all_rows)
               else 'no_accepted_row')
        for character in sorted({r['character_id'] for r in all_rows})
        for action in ['cycling','standing','walking']]
    result['independence_note']='Box/image counts are correlated derivatives; sources and pose IDs are counted separately.'
    result['layout_note']='Intent categories overlap actual truncation/occlusion/neighbor attributes; not mutually exclusive quotas.'
    acceptance=json.loads((root/'data_acceptance.json').read_text())
    result['geometry_maxima']={key:max(c['geometry_loader'][key] for c in acceptance['checks'])
        for key in ['crop_roundtrip_max_px','operator_max_error_m',
                    'rendered_mesh_operator_max_error_m','rendered_clothing_transform_max_error_m']}
    result['geometry_maxima']['rendered_groom_world_max_m']=max(
        c['world_max_m'] for scene in {r['scene'] for r in all_rows}
        for c in json.loads((Path(scene)/'groom_validation.json').read_text())['cases'])
    result['source_intersections']=acceptance['source_isolation']['train_val_intersections']
    times={}
    assets=json.loads((root/'assets/report.json').read_text())
    times['asset_derivatives']=dict(seconds_sum=sum(r['seconds'] for r in assets['compute_runs']),
        reused=sum(r['reused'] for r in assets['compute_runs']),
        newly_generated=sum(not r['reused'] for r in assets['compute_runs']))
    times['new_source_pose_audit_seconds']=json.loads((root/'pose_audit/report.json').read_text())['seconds']
    for filename in ['pilots_execution.json','expand_execution.json','ground_render_execution.json']:
        path=root/filename
        if path.exists():
            layouts=json.loads(path.read_text())['layouts']
            times[filename]=dict(attempts=len(layouts),
                successful=sum(not r['returncode'] for r in layouts),
                measured_job_seconds_sum=sum(r['seconds'] for r in layouts if r.get('seconds') is not None),
                jobs_without_complete_timing=[r['id'] for r in layouts if r.get('seconds') is None])
    for name in ['original_baseline','paired_baseline']:
        path=root/name/'report.json'
        if path.exists():
            report=json.loads(path.read_text())
            times[name]=dict(seconds=report['seconds'],rows=len(report['cases']),
                             checkpoint_load_seconds=report['checkpoint']['seconds'])
            result[name]=dict(by_batch=report['by_batch'],worst_samples=report['worst_samples'],
                nonfinite_xy_points=sum(c['nonfinite_xy_points'] for c in report['cases']),
                nonfinite_keypoint_value=sum(c['nonfinite_keypoint_value'] for c in report['cases']),
                supervised_prediction_outside_lens=sum(c['supervised_prediction_outside_lens'] for c in report['cases']))
    pairs=root/'occlusion_pairs.json'
    if pairs.exists():
        pairs=json.loads(pairs.read_text())['pairs']
        times['paired_control_render']=dict(jobs=len(pairs),seconds_sum=sum(r['seconds'] for r in pairs))
        result['separate_paired_controls']=dict(layouts=len(pairs),extra_erp_images=2*sum(not r['returncode'] for r in pairs),
            note='Clear images/fixed-box crops excluded from primary training NPZ and primary row counts.')
    times['note']='Job wall seconds include import/geometry/render/postprocessing; concurrent jobs overlap. Do not sum as active work-window duration. Overnight interruption excluded from compute.'
    result['timing']=times
    failures=[]
    for item in json.loads((root/'visual_acceptance.json').read_text())['layouts']:
        if item['status']!='rejected':continue
        directory=Path(item['directory']);ground=directory/'source/ground_contact.json'
        reason=item['reason'];detail=None
        if ground.exists():
            contact=json.loads(ground.read_text())
            if contact['status']=='failed':reason='native_ground_contact_failed';detail=contact
        failures.append(dict(layout=directory.name,reason=reason,evidence=detail or item.get('failure_log'),
            original_preserved=True,
            derived_replacement=next((str(p) for p in [directory.with_name(directory.name+'_ground_v2'),
                directory.with_name(directory.name+'_resume_v2')] if p.exists()),None)))
    result['layout_failures']=failures
    (root/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({s:counts(i['rows']) for s,i in indices.items()},indent=2))


if __name__=='__main__':main()
