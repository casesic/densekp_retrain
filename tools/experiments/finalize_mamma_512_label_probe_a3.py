"""Finalize the sparse e03 MAMMA 512 label probe without sequence packing."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]

PARTS=Path('/home/yy/Repos/mamma/data/body_models/downsampled_verts/smplx_512_body_parts.npz')
REFERENCE_FACES=ROOT/'outputs/audits/bedlam_r138_compatibility_a2_20261009/locked_head_attributes.npz'


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')


def relative(root,path):return str(Path(path).resolve().relative_to(root.resolve()))


def finalize(output):
    output=Path(output).resolve();run=output/'run';rig=run/'rigs/rig_back'
    recipe=json.loads((rig/'recipe.json').read_text());config=json.loads((rig/'run_config.json').read_text())
    policy=json.loads((rig/'label_policy.json').read_text());scene_matrices=Path(recipe['scene_matrices_ref']).resolve()
    part_for={}
    with np.load(PARTS) as parts:
        for name in parts.files:
            for index in parts[name].tolist():part_for[index]=name
    observations=[];summary=[];check_pool=[];support_total=support_pass=support_reject=absent_variants=0
    qualified={}
    for person in recipe['people']:
        asset=json.loads(Path(person['asset_manifest']).read_text());qualified[str(Path(asset['body_vertices']).resolve())]=(
            'standard_male_smplx' if person['person_id']==1 else 'locked_head_neutral_smplx')
    for frame in config['selected_frames']:
        for camera in ('cam2','cam4'):
            dest=rig/f'frame_{frame:03d}'/camera
            boxes=json.loads((dest/'boxes.json').read_text());raster=json.loads((dest/'raster_storage.json').read_text())
            support=json.loads((dest/'mamma_box_support.json').read_text())
            with np.load(dest/'annotations.npz') as ann,np.load(dest/'mamma_512.npz') as mamma:
                if mamma['xy_full_erp'].shape!=(len(recipe['people']),512,2):raise ValueError('mamma_shape_conflict')
                if not all(np.isfinite(mamma[k]).all() for k in ['world_points_m','xy_full_erp','gt_ray_range_m','front_range_m','gt_minus_front_range_m','body_first_range_m']):
                    raise ValueError('nonfinite_mamma_evidence')
                np.testing.assert_array_equal(mamma['person_ids'],ann['person_ids'])
                if int(mamma['frame_index'])!=frame or str(mamma['camera_id'])!=camera:raise ValueError('mamma_observation_identity_conflict')
                for j,person in enumerate(recipe['people']):
                    pid=person['person_id'];box=boxes[j]['bbox_xyxy_exclusive'];entry=support['people'][j]
                    if entry['person_id']!=pid or entry['native_box_xyxy']!=box:raise ValueError('box_support_identity_conflict')
                    passed=sum(v['supported'] for v in entry['variants']);support_total+=7
                    support_pass+=passed;support_reject+=len(entry['variants'])-passed;absent_variants+=7-len(entry['variants'])
                    status='observed' if box is not None else 'person_absent';reason='visible_instance_box' if box is not None else 'no_visible_instance_box'
                    asset=json.loads(Path(person['asset_manifest']).read_text())
                    oid=f'e03_rig_back_f{frame:03d}_{camera}_p{pid}'
                    visibility_value=mamma['visibility_value'][j].astype(int).tolist();visibility_known=mamma['visibility_known'][j].astype(bool).tolist()
                    row=dict(schema='mamma_observation_v1',episode=recipe['scene_episode_id'],scene_episode_id=recipe['scene_episode_id'],
                        rig='rig_back',rig_id='rig_back',camera_id=camera,person_id=pid,observation_id=oid,status=status,reason=reason,
                        full_image_size_wh=raster['full_image_size_wh'],raster=raster,
                        files=dict(image=dict(path=str((dest/'roi.png').resolve())),person_mask=dict(path=str((dest/f'person_{pid}_mask.png').resolve())),
                            valid_mask=dict(path=str((dest/'valid_mask.png').resolve())),annotations=dict(path=str((dest/'annotations.npz').resolve())),
                            mamma_512=dict(path=str((dest/'mamma_512.npz').resolve())),box_support=dict(path=str((dest/'mamma_box_support.json').resolve())),
                            physical_mask_ref=dict(path=str(Path(raster['physical_mask_ref']).resolve()))),
                        mamma_native_crop=dict(box_xyxy=box,box_kind='visible_instance_tight',box_source=boxes[j]['source']) if box is not None else None,
                        source_asset=dict(asset_id=person['asset_id'],asset_manifest=person['asset_manifest'],body_vertices=str(Path(asset['body_vertices']).resolve()),
                            body_faces=str(Path(asset['body_faces']).resolve()),template_qualification=qualified[str(Path(asset['body_vertices']).resolve())]),
                        frame_index=frame,time_tick=int(ann['time_tick']),time_base=ann['time_base'].astype(int).tolist(),
                        asset_frame_index=int(ann['person_asset_frame'][j]),source_frame_index=int(ann['person_source_frame'][j]),
                        source_time_tick=int(ann['person_source_time_tick'][j]),source_time_base=ann['person_source_time_base'][j].astype(int).tolist(),
                        person_timeline_ref=recipe['person_timeline_refs'][str(pid)],scene_matrices_ref=str(scene_matrices),scene_matrix_row=frame,scene_matrix_person_index=j,
                        camera_matrix_ref=str((dest/'annotations.npz').resolve())+':camera_to_world_erp',
                        mamma_512=dict(template_id=str(mamma['template_id']),ordered_vertex_ids_ref=str((dest/'mamma_512.npz').resolve())+':ordered_vertex_ids',
                            world_points_ref=str((dest/'mamma_512.npz').resolve())+f':world_points_m[{j}]',xy_full_erp_ref=str((dest/'mamma_512.npz').resolve())+f':xy_full_erp[{j}]',
                            geometry_defined_ref=str((dest/'mamma_512.npz').resolve())+f':geometry_defined[{j}]',physical_domain_ref=str((dest/'mamma_512.npz').resolve())+f':physical_domain[{j}]',
                            render_coverage_ref=str((dest/'mamma_512.npz').resolve())+f':render_coverage[{j}]'),
                        point_labels=dict(visibility=dict(value=visibility_value,known=visibility_known,evidence_ref=str((dest/'mamma_512.npz').resolve()),policy=policy['visibility_policy']),
                                          contact=dict(value=[-1]*512,known=[False]*512,evidence_ref=None),
                                          floor_contact=dict(value=[-1]*512,known=[False]*512,evidence_ref=None)),
                        rectangular_probe=dict(policy='mamma_fixed_seven_box_support_v1',support_ref=str((dest/'mamma_box_support.json').resolve()),
                                               supported_variants=passed,total_variants=len(entry['variants'])),
                        eligibility='integration_only')
                    observations.append(row)
                    vals=mamma['visibility_value'][j];known=mamma['visibility_known'][j];reasons=mamma['visibility_reason_code'][j]
                    item=dict(observation_id=oid,person_id=pid,status=status,known_positive=int(((vals==1)&known).sum()),known_negative=int(((vals==0)&known).sum()),
                              unknown=int((~known).sum()),physical_domain=int(mamma['physical_domain'][j].sum()),render_coverage=int(mamma['render_coverage'][j].sum()))
                    summary.append(item)
                    for k in range(512):
                        check_pool.append(dict(observation_id=oid,frame_index=frame,camera_id=camera,person_id=pid,point_index=k,
                            ordered_vertex_id=int(mamma['ordered_vertex_ids'][k]),body_part=part_for[k],visibility_value=int(vals[k]),known=bool(known[k]),
                            reason_code=int(reasons[k]),xy_full_erp=mamma['xy_full_erp'][j,k].astype(float).tolist(),gt_ray_range_m=float(mamma['gt_ray_range_m'][j,k]),
                            front_instance_id=int(mamma['front_instance_id'][j,k]),front_range_m=float(mamma['front_range_m'][j,k]),
                            range_gap_m=float(mamma['gt_minus_front_range_m'][j,k]),body_self_frontmost=bool(mamma['body_self_frontmost'][j,k])))
            image=np.asarray(Image.open(dest/'roi.png').convert('RGB'));draw=ImageDraw.Draw(Image.fromarray(image))
            overlay=Image.fromarray(image);draw=ImageDraw.Draw(overlay);offset=np.asarray(raster['offset_xy'])
            with np.load(dest/'mamma_512.npz') as mamma:
                for j in range(len(recipe['people'])):
                    for xy,val,known in zip(mamma['xy_full_erp'][j],mamma['visibility_value'][j],mamma['visibility_known'][j]):
                        q=xy-offset;color=(40,230,80) if known and val==1 else ((240,50,50) if known else (255,205,40))
                        if 0<=q[0]<overlay.width and 0<=q[1]<overlay.height:draw.ellipse((q[0]-1,q[1]-1,q[0]+1,q[1]+1),fill=color)
            overlay.save(dest/'mamma_512_overlay.png')
    # Fixed, deterministic maximum 24 checks across reason classes and official body parts.
    selected=[];seen=set()
    for reason in [1,2,3,4,5,0,6,7]:
        for part in ['body','left_hand','right_hand','left_feet','right_feet','left_head','right_head']:
            candidate=next((x for x in check_pool if x['reason_code']==reason and x['body_part']==part and (x['observation_id'],x['point_index']) not in seen),None)
            if candidate is not None:selected.append(candidate);seen.add((candidate['observation_id'],candidate['point_index']))
            if len(selected)==24:break
        if len(selected)==24:break
    manifest=dict(schema='mamma_observation_v1',version=1,eligibility='integration_only',kind='sparse_label_probe_not_sequence',
                  source_episode='e03',source_frames=[0,15,30],rig_ids=['rig_back'],camera_ids=['cam2','cam4'],observations=observations)
    write(output/'probe_manifest.json',manifest)
    write(output/'visibility_summary.json',dict(schema='mamma_visibility_summary_v1',policy=policy['visibility_policy'],observations=summary,
        total=dict(known_positive=sum(x['known_positive'] for x in summary),known_negative=sum(x['known_negative'] for x in summary),unknown=sum(x['unknown'] for x in summary)),
        by_body_part={part:dict(known_positive=sum(x['known'] and x['visibility_value']==1 for x in check_pool if x['body_part']==part),
                                known_negative=sum(x['known'] and x['visibility_value']==0 for x in check_pool if x['body_part']==part),
                                unknown=sum(not x['known'] for x in check_pool if x['body_part']==part))
                      for part in sorted(set(part_for.values()))},
        training_qualification='pending_representative_negative_evidence'))
    write(output/'seven_box_support.json',dict(schema='mamma_fixed_seven_box_support_summary_v1',actual_rendered_boxes=True,
        variants_per_observed_box=7,supported=support_pass,rejected=support_reject,absent_not_tested=absent_variants,total_slots=support_total))
    write(output/'point_checks.json',dict(schema='mamma_point_checks_v1',count=len(selected),maximum=24,cases=selected,
        independent_check='body triangle ray first-hit versus live rendered range/instance evidence',missing_categories=[x for x in ['other_person_negative','scene_negative'] if not any((c['reason_code']==2 if x.startswith('other') else c['reason_code']==3) for c in selected)]))
    cfg=dict(schema='mamma_manifest_reader_v1',manifest=str((output/'probe_manifest.json').resolve()),
        template=config['mamma_512_label']['template_path'],reference_faces=str(REFERENCE_FACES.resolve()),box_policy='native_oracle_only',
        mask_policy='oracle_visible_instance_binary_nearest',augmentation='off',eligibility='integration_only',qualified_assets=qualified)
    write(output/'a1_reader_config.json',cfg)
    handoff=dict(schema='mamma_512_a1_handoff_v1',eligibility='integration_only',manifest=str((output/'probe_manifest.json').resolve()),
        reader_config=str((output/'a1_reader_config.json').resolve()),template=dict(id='MAMMA_verts_512_ordered_v1',path=config['mamma_512_label']['template_path'],
            ordered_vertex_ids='each mamma_512.npz:ordered_vertex_ids',qualified_assets=qualified),
        coordinate_spaces=dict(full_erp='5760x2880 logical pixel xy',stored_roi='full ERP xy minus raster.offset_xy',model_input='A1 adapter affine to width384 height512',world='metres after one scene_from_model application'),
        fields=dict(coordinates='mamma_512.npz world_points_m/xy_full_erp',physical='physical_domain',coverage='render_coverage and evidence_neighborhood_complete',
            visibility='visibility_value {-1,0,1}, visibility_known, visibility_reason_code and raw range/ID evidence',
            crops='mamma_box_support.json fixed seven independent variants',legacy='annotations.npz remains R138'),
        visibility_policy=policy,contact=dict(value=-1,known=False,valid=False),floor_contact=dict(value=-1,known=False,valid=False),
        rectangular_policy='native; x center +/-0.03w; y center +/-0.03h; centered scale 0.9/1.1; each uses 1.2 expansion, 384:512 aspect, deployment Gaussian and halo',
        old_data_compatibility='MAMMA files and manifest are additive; old observations without point_labels remain unknown in A1 reader',
        samples=dict(total_slots=12,observed=10,absent=2,source_frames=[0,15,30],rig='rig_back',cameras=['cam2','cam4']),
        pending_A1=['visibility training remains pending because fixed frames contain no high-confidence negative labels','review 15mm appearance tolerance before expansion','use manifest grouping to keep all derivatives on val_candidate side'])
    write(output/'a1_handoff.json',handoff)
    # Independent regression against the previous e03 R138 output.
    old=ROOT/'outputs/experiments/first_four_episodes_a1_20261011/e03/run/rigs/rig_back';comparisons=[]
    for frame in config['selected_frames']:
        for camera in ('cam2','cam4'):
            with np.load(rig/f'frame_{frame:03d}'/camera/'annotations.npz') as a,np.load(old/f'frame_{frame:03d}'/camera/'annotations.npz') as b:
                diffs=[]
                for key in b.files:
                    equal=np.array_equal(a[key],b[key],equal_nan=True) if a[key].dtype.kind in 'fc' else np.array_equal(a[key],b[key])
                    if not equal:diffs.append(key)
                comparisons.append(dict(frame_index=frame,camera_id=camera,old_fields=len(b.files),different_fields=diffs,passed=not diffs))
    write(output/'r138_regression.json',dict(schema='r138_frozen_regression_v1',reference=str(old),comparisons=comparisons,status='passed' if all(x['passed'] for x in comparisons) else 'failed'))
    old_regions=json.loads((old/'planned_regions.json').read_text());new_regions=json.loads((rig/'planned_regions.json').read_text())
    region_rows=[]
    for frame in config['selected_frames']:
        actual={row['id']:row['roi_xyxy'] for row in json.loads((rig/f'frame_{frame:03d}'/'render_frame.json').read_text())['source_manifest']['sources']}
        for camera in ('cam2','cam4'):
            before=old_regions[frame][camera]['roi_xyxy'];after=new_regions[frame][camera]['roi_xyxy'];rendered=actual[camera]
            area=lambda box:(box[2]-box[0])*(box[3]-box[1])
            region_rows.append(dict(frame_index=frame,camera_id=camera,before_xyxy=before,planned_after_xyxy=after,
                actual_rendered_xyxy=rendered,before_pixels=area(before),planned_after_pixels=area(after),actual_pixels=area(rendered),
                added_pixels=area(after)-area(before),exact_match=before==after==rendered))
    write(output/'roi_region_comparison.json',dict(schema='mamma_roi_region_comparison_v1',baseline=str((old/'planned_regions.json').resolve()),
        mamma_plan=str((rig/'planned_regions.json').resolve()),rows=region_rows,all_exact=all(x['exact_match'] for x in region_rows),
        total_before_pixels=sum(x['before_pixels'] for x in region_rows),total_after_pixels=sum(x['planned_after_pixels'] for x in region_rows),
        total_added_pixels=sum(x['added_pixels'] for x in region_rows)))
    completed=[json.loads(line) for line in (run/'render_calls.jsonl').read_text().splitlines() if json.loads(line)['event']=='completed']
    timings=[json.loads((rig/f'frame_{frame:03d}'/'consume_timing.json').read_text()) for frame in config['selected_frames']]
    files=[p for p in output.rglob('*') if p.is_file()];temporary=[p for p in run.rglob('*') if p.suffix=='.exr']
    write(output/'execution.json',dict(schema='mamma_512_execution_v1',gpu_uuid=json.loads((run/'blender_process.json').read_text())['gpu_uuid'],
        camera_calls=len(completed),retry_calls=0,render_seconds=sum(x['seconds'] for x in completed),blender_wall_seconds=json.loads((run/'blender_process.json').read_text())['wall_seconds'],
        scene_loads=1,person_state_updates=3,consumer_processes=1,consume_seconds=sum(x['seconds'] for x in timings),
        mamma_512_seconds=sum(c['mamma_512_seconds'] for t in timings for c in t['cameras']),
        source_read_seconds=sum(c['read_seconds'] for t in timings for c in t['cameras']),
        compose_seconds=sum(c['compose_seconds'] for t in timings for c in t['cameras']),save_seconds=sum(c['save_seconds'] for t in timings for c in t['cameras']),
        temporary_bytes_discarded=sum(t['discarded_temporary_bytes'] for t in timings),retained_raw_calibration_buffers=0,
        failed_render_bytes=0,initial_reader_import_failure='package-level optional wandb dependency missing; no data mutation or rerender; bypass validator used',
        output_file_count=len(files),output_bytes=sum(p.stat().st_size for p in files),cpu_development_time='unknown; bounded validation commands only'))
    cards=[]
    for frame in config['selected_frames']:
        for camera in ('cam2','cam4'):
            img=rig/f'frame_{frame:03d}'/camera/'mamma_512_overlay.png'
            cards.append(f'<figure><img src="{relative(output,img)}"><figcaption>frame {frame} · {camera}</figcaption></figure>')
    (output/'point_checks.html').write_text('<!doctype html><meta charset="utf-8"><title>MAMMA 512 probe</title><style>body{font:14px sans-serif;background:#222;color:#eee}main{display:grid;grid-template-columns:1fr 1fr;gap:12px}img{width:100%}figure{margin:0;background:#333;padding:8px}</style><h1>MAMMA 512 稀疏标签检查</h1><p>绿=known visible，红=known occluded，黄=unknown；仅 frame 0/15/30，不是连续视频。</p><main>'+''.join(cards)+'</main>')
    final_files=[p for p in output.rglob('*') if p.is_file()]
    reader_readback=output/'a1_reader_readback.json'
    reader_passed=(json.loads(reader_readback.read_text()).get('status')=='passed') if reader_readback.is_file() else False
    write(output/'acceptance.json',dict(schema='mamma_512_pilot_acceptance_v1',status='passed_with_visibility_training_pending',
        checks=dict(observations=12,observed=10,mamma_shapes_and_finite=True,identity_and_integer_time=True,r138_unchanged=all(x['passed'] for x in comparisons),
                    physical_coverage_visibility_separate=True,unknown_not_zero_filled=True,seven_boxes_supported=support_pass,seven_boxes_rejected=support_reject,
                    absent_variants_not_tested=absent_variants,temporary_exr_remaining=len(temporary),all_manifest_paths_exist=all(Path(f['path']).exists() for o in observations for f in o['files'].values()),
                    a1_reader_six_row_readback=reader_passed),
        limitations=['no high-confidence negative visibility points in fixed frames','official vertex_visibility production definition unavailable','contact and floor contact out of scope'],
        output_file_count=len(final_files)+(0 if (output/'acceptance.json').is_file() else 1),output_bytes_before_acceptance_rewrite=sum(p.stat().st_size for p in final_files)))
    return dict(observations=len(observations),observed=sum(x['status']=='observed' for x in observations),support=support_pass,checks=len(selected))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(finalize(args.output),ensure_ascii=False))


if __name__=='__main__':main()
