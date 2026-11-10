"""Evidence-only all-frame accounting and bounded event-selected readback."""
import csv
import json
from pathlib import Path
import numpy as np
from densekp_retrain.data.frame_sources import frame_directory
from PIL import Image
from .dynamic_scene import frame_record,person_frame_record,person_matrices
from .occlusion_events import three_stage_event
from .observation_policy import RECORD,policy_name


def summarize_multiperson(root,index,coverage,pack_dir=None):
    root=Path(root);recipe=json.loads((root/'recipe.json').read_text());pack=Path(pack_dir) if pack_dir is not None else root/'pack'
    ids=[p['person_id'] for p in recipe['people']];target=recipe['target_person_id']
    cameras=[c['id'] for c in json.loads((root/'rig.json').read_text())['cameras']]
    world=np.load(root/'world_densekp.npy');rows=[];curves={c:[] for c in cameras}
    viewcoverage={(r['frame_index'],r['person_id'],r['camera_id']):r for r in coverage if not r.get('paired')}
    paircoverage={(r['frame_index'],r['person_id']):r for r in coverage if r.get('paired')}
    observations={o['observation_id']:o for o in index['observations']};mask_stats=[]
    available=[r['frame_index'] for r in index['frames'] if r['status']=='available']
    partial=available!=list(range(recipe['frame_count'])) or any(r.get('acceptance_status')=='rejected' for r in index['frames'])
    if available!=list(range(len(available))):raise ValueError('event_diagnostics_require_contiguous_prefix; gaps remain in sequence index')
    for k in available:
        timing=frame_record(recipe,k);annotations={};boxes={}
        render=json.loads((frame_directory(root,k)/'render_frame.json').read_text());states=render['source_manifest']['people_state']
        if [p['person_id'] for p in states]!=ids:raise ValueError('render_people_order_conflict')
        for j,p in enumerate(recipe['people']):
            np.testing.assert_array_equal(states[j]['scene_from_model'],person_matrices(p,recipe['frame_count'])[k])
            if states[j]['asset_frame']!=person_frame_record(recipe,k,p)['asset_frame']:raise ValueError('render_person_time_conflict')
        for c in cameras:
            folder=frame_directory(root,k)/c;ann=dict(np.load(folder/'annotations.npz',allow_pickle=False));annotations[c]=ann
            np.testing.assert_array_equal(ann['person_ids'],ids);np.testing.assert_array_equal(ann['world_densekp'],world[k])
            for value in ann.values():
                if value.dtype.kind in 'fc' and not np.isfinite(value).all():raise ValueError('nonfinite_annotations')
            meta=json.loads((folder/'raster_storage.json').read_text());valid=np.asarray(Image.open(folder/'valid_mask.png'))>0
            pm=[np.asarray(Image.open(folder/f'person_{pid}_mask.png'))>0 for pid in ids]
            if list(valid.shape[::-1])!=meta['stored_size_wh']:raise ValueError('local_mask_size')
            if any(m.shape!=valid.shape or (m&~valid).any() for m in pm):raise ValueError('person_mask_valid_support')
            if (np.stack(pm).sum(0)>1).any():raise ValueError('person_mask_instance_collision')
            with Image.open(folder/'roi.png') as image:
                if list(image.size)!=meta['stored_size_wh']:raise ValueError('ROI_header_size')
            boxes[c]=json.loads((folder/'boxes.json').read_text())
            mask_stats.append(dict(frame_index=k,camera=c,stored_size_wh=meta['stored_size_wh'],person_pixels=[int(m.sum()) for m in pm]))
        for j,pid in enumerate(ids):
            pr=person_frame_record(recipe,k,recipe['people'][j])
            common=np.logical_and(*(a['in_view'][j] for a in annotations.values()))
            for c,ann in annotations.items():
                code=ann['occlusion_evidence_code'][j];neighbor=common&(code==2)
                ratio=float(neighbor.sum()/common.sum()) if common.any() else None
                if pid==target:curves[c].append(ratio)
                obs=observations[f'f{k}_{c}_p{pid}'];cov=viewcoverage.get((k,pid,c),{})
                box=boxes[c][j]['bbox_xyxy_exclusive'];b=None if box is None else np.array(box)
                rows.append(dict(frame_index=k,time_tick=timing['episode_time_tick'],source_frame=pr['source_frame'],asset_frame=pr['asset_frame'],
                    person_id=pid,camera_id=c,status=obs['status'],reason=obs['reason'],in_view=int(ann['in_view'][j].sum()),visible=int(ann['visible'][j].sum()),
                    self_instance_occluded=int((code==3).sum()),neighbor_occluded=int((code==2).sum()),scene_occluded=int((code==4).sum()),unresolved_occluded=int((code==5).sum()),
                    outside_domain=int((~ann['in_view'][j]).sum()),common_physical_points=int(common.sum()),common_neighbor_occluded=int(neighbor.sum()),
                    common_neighbor_fraction=ratio,common_supervised_points=paircoverage.get((k,pid),{}).get('common_points'),
                    supervised=cov.get('supervised'),supervised_occluded=cov.get('supervised_occluded'),box_xyxy=box,
                    box_center=None if b is None else ((b[:2]+b[2:])/2).tolist(),box_width=None if b is None else float(b[2]-b[0]),
                    box_height=None if b is None else float(b[3]-b[1]),scale=None if b is None else float(max(b[2:]-b[:2])/200),
                    neighbor_semantic_indices=np.flatnonzero(neighbor).tolist()))
    metric=recipe['occlusion_metric'];events={c:three_stage_event(r,delta=metric['delta'],consecutive=metric['consecutive']) for c,r in curves.items()}
    event=next((e for e in events.values() if e),None);selection_basis='actual conservative neighbor-depth/ID evidence'
    if event is None:
        event=next((e for e in recipe['predicted_overlap_events'].values() if e),None)
        selection_basis='frozen CPU projected stages; actual event threshold NOT passed' if event else 'no projected neighbor event'
    selected=([event[key][len(event[key])//2] for key in ['before','during','after']] if event else [0,recipe['frame_count']//2,recipe['frame_count']-1])
    if recipe.get('readback_policy')=='first_middle_last':
        selected=[0,recipe['frame_count']//2,recipe['frame_count']-1];selection_basis='frozen first/middle/last; no model selection'
    box_events={}
    for c in cameras:
        rs=[r for r in rows if r['person_id']==target and r['camera_id']==c]
        flags=[r['status']=='observed' for r in rs]
        box_events[c]=dict(has_legal_input=flags,disappeared=[rs[i]['frame_index'] for i in range(1,len(rs)) if flags[i-1] and not flags[i]],reappeared=[dict(frame_index=rs[i]['frame_index'],time_tick=rs[i]['time_tick'],common_supervised_points=rs[i]['common_supervised_points']) for i in range(1,len(rs)) if not flags[i-1] and flags[i]])
    if recipe.get('readback_policy')=='continuation_old_occluded_reappearance':
        resume=json.loads((root/'resume_manifest.json').read_text())
        if resume['reprocessed']:
            returns=[r['frame_index'] for r in box_events['cam2']['reappeared'] if r['frame_index'] in resume['new']]
            selected=[resume['reused'][0],resume['reprocessed'][0],returns[0] if returns else available[-1]]
            selection_basis='old success, reprocessed full occlusion, first actual new cam2 target input reappearance (else last); no model selection'
        else:
            selected=[0,recipe['frame_count']//2,recipe['frame_count']-1]
            selection_basis='read-only successful probe prefix plus new suffix; frozen first/middle/last; no model selection'
    if partial:
        selected=json.loads((root/'partial_delivery.json').read_text())['readback_frame_indices'];selection_basis='planned first/middle plus rejected prefix endpoint; no model selection'
    summary=dict(pipeline_status='partial_stopped' if partial else 'passed',event_coverage_status='incomplete' if partial else ('not_applicable_no_neighbor' if len(ids)==1 else 'passed' if any(events.values()) else 'not_met'),metric=metric,target_person_id=target,
        observation_policy=policy_name(recipe),native_input_events=box_events,events=events,readback_frame_indices=selected,readback_selection_basis=selection_basis,people=ids,frames=recipe['frame_count'],
        observation_slots=len(index['observations']),packed_rows=index['packed']['row_count'],unique_RGB_paths=len({o['image'] for o in index['observations'] if o['image']}),
        observation_status_counts={s:sum(o['status']==s for o in index['observations']) for s in sorted({o['status'] for o in index['observations']})},
        evidence_codes={'0':'outside_domain','1':'visible','2':'neighbor','3':'own_instance_body_cloth_groom','4':'scene','5':'unresolved'},
        generated_frame_indices=available,missing_frame_indices=[k for k in range(recipe['frame_count']) if k not in available],rejected_frame_indices=[r['frame_index'] for r in index['frames'] if r.get('acceptance_status')=='rejected'],
        all_float_annotations_finite=True,all_person_masks_disjoint=True,all_world_time_and_matrix_correspondence=True)
    if 'coverage_goal' in recipe:
        summary['coverage_goal']=recipe['coverage_goal']
        summary['boundary_transition_passed']=recipe.get('boundary_transition_passed')
        summary['requested_view_class']=recipe.get('requested_view_class')
    if 'dual_edge_gate' in recipe:
        from .mixed_road_pass import edge_summary
        edge_rows=[]
        for k in available:
            rs={r['camera_id']:r for r in rows if r['person_id']==target and r['frame_index']==k}
            edge_rows.append(dict(views=rs,common_physical_points=rs['cam2']['common_physical_points']))
        edge=edge_summary(edge_rows,recipe['episode_timeline'])
        edge['common_crop_points']=[paircoverage.get((k,target),{}).get('common_points') for k in range(recipe['frame_count'])]
        edge['flags'] += [None]*(recipe['frame_count']-len(available))
        edge['unrendered_are_unknown']=True
        edge['complete_episode']=not partial
        edge['crop_gate_passed']=all(v is not None and v>=8 for v,f in zip(edge['common_crop_points'],edge['flags']) if f)
        if policy_name(recipe)==RECORD:
            edge['crop_gate_role']='diagnostic only; not generation stop or physical edge acceptance'
            flags=[bool(paircoverage.get((k,target),{}).get('sufficient_common_support',False)) for k in range(recipe['frame_count'])]
            windows=[];begin=None
            for k,flag in enumerate(flags+[False]):
                if flag and begin is None:begin=k
                if not flag and begin is not None:
                    windows.append(dict(first=begin,last=k-1,observations=k-begin,span_seconds=(k-1-begin)*recipe['time_base'][0]/recipe['time_base'][1]));begin=None
            summary['pair_input_availability']=dict(total_planned_times=recipe['frame_count'],common_at_least8_count=sum(flags),fraction=sum(flags)/recipe['frame_count'],flags=flags,windows=windows,per_time=[dict(frame_index=k,**{key:value for key,value in paircoverage[(k,target)].items() if key not in ('frame_index','person_id','paired')}) for k in available])
        summary['dual_edge']=edge
        (pack/'dual_edge.json').write_text(json.dumps(edge,indent=2)+'\n')
    for name,value in [('multiperson_acceptance' ,summary),('per_person_time_statistics',rows),('mask_statistics',mask_stats)]:
        (pack/f'{name}.json').write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    with (pack/'target_curves.csv').open('w') as file:
        fields=['frame_index','time_tick','source_frame','camera_id','in_view','visible','common_physical_points','common_supervised_points','common_neighbor_fraction','outside_domain','box_width','box_height','scale','status','reason']
        writer=csv.DictWriter(file,fieldnames=fields);writer.writeheader()
        for row in rows:
            if row['person_id']==target:writer.writerow({k:row[k] for k in fields})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    extended=policy_name(recipe)==RECORD
    fig,axes=plt.subplots(6 if extended else 4,1,figsize=(10,14 if extended else 10),sharex=True)
    for c in cameras:
        rs=[r for r in rows if r['person_id']==target and r['camera_id']==c];t=[r['time_tick']*recipe['time_base'][0]/recipe['time_base'][1] for r in rs]
        axes[0].plot(t,[r['common_neighbor_fraction'] for r in rs],label=c)
        axes[1].plot(t,[r['common_supervised_points'] for r in rs],label=c)
        axes[2].plot(t,[r['box_width'] for r in rs],label=c+' width');axes[2].plot(t,[r['box_height'] for r in rs],label=c+' height')
        for axis in range(2):axes[3].plot(t,[np.nan if r['box_center'] is None else r['box_center'][axis] for r in rs],label=c+(' cx' if axis==0 else ' cy'))
        if extended:
            axes[4].step(t,[r['status']=='observed' for r in rs],where='mid',label=c+' legal input')
            axes[5].plot(t,[r['common_physical_points'] for r in rs],label=c+' common physical')
    if extended:
        axes[4].step(t,summary['dual_edge']['flags'],where='mid',linestyle='--',label='simultaneous physical edge')
        axes[4].set_ylabel('Availability / edge');axes[4].legend();axes[4].grid(alpha=.2)
        axes[5].set_ylabel('Common physical GT');axes[5].legend();axes[5].grid(alpha=.2)
    for ax,title in zip(axes,['Neighbor fraction / common physical GT','Common crop supervision count','Native box dimensions (full ERP px)','Native box center (full ERP px)']):ax.set_ylabel(title);ax.legend();ax.grid(alpha=.2)
    axes[-1].set_xlabel('Synthetic episode seconds' if 'person_timelines' in recipe else 'Native episode seconds');fig.tight_layout();fig.savefig(pack/'target_curves.svg');plt.close(fig)
    return summary
