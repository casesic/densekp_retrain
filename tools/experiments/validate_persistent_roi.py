"""Bounded same-buffer process comparison and short ROI sequence delivery."""
import argparse
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
import numpy as np
from densekp_retrain.data.frame_sources import frame_directory
from PIL import Image,ImageDraw,ImageFont
from densekp_retrain.runtime.frame_consumer import FrameConsumer
from densekp_retrain.imaging.output_modes import prepare_source_workspace,complete_output
from densekp_retrain.imaging.roi_consumer import write
from densekp_retrain.data.sequence_packing import pack_rendered_sequence
from densekp_retrain.data.loading import load_box_dataset


def compare(root):
    root=Path(root).resolve();root.mkdir(exist_ok=False)
    prior=ROOT/'outputs/experiments/roi_only_render_a1_20261009'
    retained=prior/'failure_attempt_1/retained_source/source'
    start=time.monotonic();runs={}
    for mode in ['per_frame','persistent']:
        dest=root/mode;dest.mkdir()
        for name in ['rig.json','world_densekp.npy']:
            shutil.copy2(prior/name,dest/name)
        shutil.copytree(prior/'calibration',dest/'calibration')
        recipe=json.loads((prior/'recipe.json').read_text());recipe['frame_count']=1;write(dest/'recipe.json',recipe)
        config=json.loads((prior/'run_config.json').read_text());config['consumer_mode']=mode;write(dest/'run_config.json',config)
        source=prepare_source_workspace(dest/'frame_000','compact')
        shutil.copytree(retained,source,dirs_exist_ok=True)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
        command=[sys.executable,str(ROOT/'tools/experiments/consume_continuous_roi.py'),
                 'serve' if mode=='persistent' else 'frame','--root',str(dest)]
        t=time.monotonic()
        if mode=='per_frame':
            with (dest/'process.log').open('w') as log:
                subprocess.run(command+['--frame','0'],env=env,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=120)
            result={}
        else:
            worker=FrameConsumer(command,env=env,cwd=ROOT,log_path=dest/'consumer_worker.log',deadline=time.monotonic()+120)
            try:
                worker.consume(0);result=worker.close()
            finally:
                if not worker.closed:worker.abort()
        runs[mode]=dict(process_wall_seconds=time.monotonic()-t,**result)
    a=root/'per_frame';b=root/'persistent';checks=[]
    def normalized(value,base):
        if isinstance(value,dict):return {k:normalized(v,base) for k,v in value.items()}
        if isinstance(value,list):return [normalized(v,base) for v in value]
        return value.replace(str(base),'<run>') if isinstance(value,str) else value
    for camera in ['cam2','cam4']:
        left=a/'frame_000'/camera;right=b/'frame_000'/camera
        assert {p.name for p in left.iterdir()}=={p.name for p in right.iterdir()}
        for p in left.iterdir():
            q=right/p.name
            if p.suffix=='.npz':
                with np.load(p,allow_pickle=False) as x,np.load(q,allow_pickle=False) as y:
                    assert x.files==y.files
                    for k in x.files:np.testing.assert_array_equal(x[k],y[k])
            elif p.suffix=='.json':
                assert normalized(json.loads(p.read_text()),a)==normalized(json.loads(q.read_text()),b),p.name
            else:assert p.read_bytes()==q.read_bytes(),p.name
            checks.append(f'{camera}/{p.name}')
    np.testing.assert_array_equal(np.load(a/'frame_000/shared_world_densekp.npy'),np.load(b/'frame_000/shared_world_densekp.npy'))
    datasets=[]
    for dest in [a,b]:
        # This is one static buffer comparison, not a one-frame sequence.
        # Exercise the actual row packer without asserting a continuous timeline.
        from densekp_retrain.data.packing import pack_derivative_rows
        from densekp_retrain.data.raster_storage import validate_roi_box
        from densekp.core.datasets.image_layout import LAYOUT_FIELDS
        from densekp.core.keypoint_supervision import make_targets,SUPERVISION_POLICY
        rows=[]
        for c in ['cam2','cam4']:
            cd=dest/'frame_000'/c
            meta=json.loads((cd/'raster_storage.json').read_text())
            box=json.loads((cd/'boxes.json').read_text())[0]
            with np.load(cd/'annotations.npz') as ann:
                valid=np.asarray(Image.open(cd/'valid_mask.png'))>0
                person=np.asarray(Image.open(cd/'person_1_mask.png'))>0
                center,scale=validate_roi_box(box['bbox_xyxy_exclusive'],box,1,int(ann['person_ids'][0]),meta,valid,person)
                row=dict(imgname=str(cd/'roi.png'),center=center,scale=scale,
                    proj_verts=make_targets(ann['xy'][0],ann['in_view'][0],center,scale),
                    in_view=ann['in_view'][0],visible=ann['visible'][0],person_id=1,camera_id=c)
            row.update({key:meta[key] for key in LAYOUT_FIELDS});rows.append(row)
        pack=dest/'comparison_pack';pack.mkdir()
        np.savez(pack/'packed.npz',**pack_derivative_rows(rows),supervision_policy=SUPERVISION_POLICY,split='integration_only')
        datasets.append(load_box_dataset(pack/'packed.npz'))
    keys=[]
    for row in range(2):
        x,y=[d[row] for d in datasets]
        assert x.keys()==y.keys()
        for k in x:
            if k=='imgname':assert normalized(x[k],a)==normalized(y[k],b)
            else:np.testing.assert_array_equal(x[k],y[k],err_msg=k)
        keys=list(x)
    result=dict(status='passed_exact',same_original_buffer=str(retained),frames=1,cameras=2,
                compared_files=checks,Dataset_reads=4,Dataset_fields=keys,path_normalization_only=True,
                timings_and_ownership_timestamps_not_compared=True,runs=runs,elapsed_seconds=time.monotonic()-start)
    write(root/'comparison.json',result)
    for dest in [a,b]:
        out=dest/'frame_000';complete_output(out,'compact',out/'_temporary_ds_source/source',checks=dict(status='passed',same_buffer_exact=True))
    return result


def finish_short(root,output_name='pack'):
    root=Path(root);start=time.monotonic();recipe=json.loads((root/'recipe.json').read_text());count=recipe['frame_count']
    from densekp_retrain.data.dynamic_scene import frame_record
    records=[frame_record(recipe,k) for k in range(count)]
    cameras=[c['id'] for c in json.loads((root/'rig.json').read_text())['cameras']]
    pack,ep,index,coverage=pack_rendered_sequence(root,output_name=output_name);pack_seconds=time.monotonic()-start
    checks=[];t=time.monotonic()
    partial_path=root/'partial_delivery.json';partial=json.loads(partial_path.read_text()) if partial_path.exists() else None
    video_frames=partial['generated_frame_indices'] if partial else list(range(count))
    selected=sorted({0,count//2,count-1})
    if 'occlusion_metric' in recipe:
        from densekp_retrain.data.multiperson_acceptance import summarize_multiperson
        summary=summarize_multiperson(root,index,coverage,pack_dir=pack);selected=summary['readback_frame_indices']
    bounded_ids=None
    explicit_ids=recipe.get('readback_observation_ids')
    explicit_frames=recipe.get('readback_frame_indices')
    if explicit_ids is not None:
        valid={f'f{k}_{c}_p{p["person_id"]}' for k in range(count) for c in cameras for p in recipe['people']}
        if (not isinstance(explicit_ids,list) or not 1<=len(explicit_ids)<=24 or
            len(set(explicit_ids))!=len(explicit_ids) or any(x not in valid for x in explicit_ids) or
            recipe.get('readback_row_cap')!=len(explicit_ids)):
            raise ValueError('invalid_explicit_readback_observations')
        bounded_ids=set(explicit_ids)
        selected=sorted({int(x.split('_')[0][1:]) for x in explicit_ids})
        row_cap=len(explicit_ids)
    elif explicit_frames is not None:
        if (not isinstance(explicit_frames,list) or not explicit_frames or
            len(set(explicit_frames))!=len(explicit_frames) or
            any(type(k) is not int or not 0<=k<count for k in explicit_frames)):
            raise ValueError('invalid_explicit_readback_frames')
        selected=sorted(explicit_frames)
        row_cap=int(recipe['readback_row_cap'])
        planned=[f'f{k}_{c}_p{p["person_id"]}' for k in selected for c in cameras for p in recipe['people']]
        if row_cap!=len(planned) or row_cap>24:
            raise ValueError('explicit_readback_slot_limit')
        bounded_ids=set(planned)
    elif recipe.get('scene_mode') in ('single_cyclist_boundary_pass','mixed_walk_cycle_crossing'):
        # Freeze a small first/middle/last sample before opening any Dataset
        # row. Keep both cameras for the target, then include the other person.
        target=recipe['target_person_id'];others=[p['person_id'] for p in recipe['people'] if p['person_id']!=target]
        order=[f'f{k}_{c}_p{target}' for k in selected for c in cameras]
        order += [f'f{k}_{c}_p{pid}' for k in [selected[len(selected)//2],*selected] for pid in others for c in cameras]
        available={x['observation_id'] for x in index['boxes']}
        bounded_ids=set()
        row_cap=int(recipe.get('readback_row_cap',8))
        if not 1<=row_cap<=8:raise ValueError('invalid_readback_row_cap')
        for obs in order:
            if obs in available:bounded_ids.add(obs)
            if len(bounded_ids)>=row_cap:break
    write(pack/'readback_plan.json',dict(scene_mode=recipe.get('scene_mode'),coverage_goal=recipe.get('coverage_goal'),requested_view_class=recipe.get('requested_view_class'),
        row_cap=row_cap if bounded_ids is not None else None,selected_frame_indices=selected,
        selected_observation_ids=sorted(bounded_ids) if bounded_ids is not None else None,
        selection_before_Dataset_item_reads=True))
    dataset=load_box_dataset(pack/'packed.npz') if index['packed']['row_count'] else None
    for k in selected:
        for c in cameras:
            for target in [p['person_id'] for p in recipe['people']]:
                match=[x['packed_row_id'] for x in index['boxes'] if x['observation_id']==f'f{k}_{c}_p{target}']
                if not match:
                    observation=next(x for x in index['observations'] if x['observation_id']==f'f{k}_{c}_p{target}')
                    checks.append(dict(frame_index=k,person_id=target,camera=c,status=observation['status'],reason=observation['reason'],Dataset_read=False));continue
                if bounded_ids is not None and f'f{k}_{c}_p{target}' not in bounded_ids:
                    checks.append(dict(frame_index=k,person_id=target,camera=c,status='valid_not_sampled',Dataset_read=False));continue
                row=match[0];item=dataset[row]
                import cv2
                local=cv2.imread(item['imgname']);inverse=cv2.invertAffineTransform(item['_trans']);inverse[:,2]-=dataset.data['offset_xy'][row]
                expected=cv2.warpAffine(local,inverse,(256,256),flags=cv2.INTER_LINEAR|cv2.WARP_INVERSE_MAP)
                np.testing.assert_array_equal(expected,item['img_disp'])
                xy=(np.c_[item['proj_verts'][:,:2],np.ones(138)]@item['_trans'].T).astype(np.float32)
                xy=(xy/np.array([256,256])-.5).astype(np.float32)
                np.testing.assert_array_equal(xy,item['proj_verts_cropped'][:,:2])
                from densekp.core.keypoint_supervision import supervision_for_crop
                np.testing.assert_array_equal(supervision_for_crop(dataset.data['in_view'][row],xy),item['proj_verts_cropped'][:,2]>0)
                np.testing.assert_array_equal(item['img_size'],[2880,5760])
                assert np.isfinite(item['_trans']).all()
                assert np.isfinite(item['img']).all() and np.isfinite(item['proj_verts_cropped']).all()
                Image.fromarray(item['img_disp'][:,:,::-1]).save(pack/f'frame_{k:03d}_{c}_p{target}_crop.png')
                supervised=item['proj_verts_cropped'][:,2]>0
                checks.append(dict(frame_index=k,person_id=target,camera=c,row=row,Dataset_read=True,image_affine_exact=True,point_affine_exact=True,supervised=int(supervised.sum()),occluded_supervised=int((supervised&~item['visible']).sum())))
    readback_seconds=time.monotonic()-t;t=time.monotonic()
    import av
    from fractions import Fraction
    viewports=None
    if 'dual_edge_gate' in recipe:
        viewports={}
        for c in cameras:
            rectangles=np.array([json.loads((frame_directory(root,k)/c/'raster_storage.json').read_text())['roi_xyxy'] for k in video_frames])
            viewports[c]=np.r_[rectangles[:,:2].min(0),rectangles[:,2:].max(0)].tolist()
    with av.open(str(pack/'dual_roi.mp4'),'w') as video:
        time_base=Fraction(*recipe['time_base']);native='selected_timeline' in recipe
        stream=video.add_stream('libx264',rate=Fraction(*recipe['nominal_source_rate']) if native else 10);stream.width=1024;stream.height=544;stream.pix_fmt='yuv420p';stream.options={'crf':'24'}
        stream.time_base=time_base;stream.codec_context.time_base=time_base
        ticks=[records[k]['episode_time_tick'] for k in video_frames];last_hold=recipe.get('video_last_hold_ticks',ticks[-1]-ticks[-2])
        durations={tick:(ticks[k+1]-tick if k+1<len(ticks) else last_hold) for k,tick in enumerate(ticks)}
        def mux(packet):
            # Explicit last-frame display hold; it does not create an observation.
            if packet.pts in durations:packet.duration=durations[packet.pts]
            video.mux(packet)
        for k in video_frames:
            canvas=Image.new('RGB',(1024,544),(32,32,32));draw=ImageDraw.Draw(canvas)
            for i,c in enumerate(cameras):
                with Image.open(frame_directory(root,k)/c/'roi.png') as im:
                    preview=im.convert('RGB')
                    if 'dual_edge_gate' in recipe:
                        # Physical mask boundary in the same ROI coordinates, not its border.
                        import cv2
                        folder=frame_directory(root,k)/c
                        valid=np.asarray(Image.open(folder/'valid_mask.png'))>0
                        meta=json.loads((folder/'raster_storage.json').read_text());rx,ry,_,_=meta['roi_xyxy']
                        physical=np.asarray(Image.open(meta['physical_mask_ref']))>0
                        edge=cv2.morphologyEx(physical.astype(np.uint8),cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))[ry:ry+valid.shape[0],rx:rx+valid.shape[1]]>0
                        pixels=np.array(preview);pixels[edge]=[255,230,0];preview=Image.fromarray(pixels)
                        labels=ImageDraw.Draw(preview)
                        for box in json.loads((folder/'boxes.json').read_text()):
                            if box['bbox_xyxy_exclusive'] is not None:
                                x,y,x2,y2=box['bbox_xyxy_exclusive']
                                labels.rectangle((x-rx,y-ry,x2-rx,y2-ry),outline='cyan',width=2)
                                labels.text((x-rx,y-ry),f'p{box["person_id"]}',fill='cyan',stroke_width=1)
                    if viewports is not None:
                        vx,vy,ex,ey=viewports[c];factor=min(512/(ex-vx),490/(ey-vy))
                        width,height=round(preview.width*factor),round(preview.height*factor)
                        preview=preview.resize((width,height),Image.Resampling.BILINEAR)
                        left=i*512+(512-round((ex-vx)*factor))//2+round((rx-vx)*factor)
                        top=32+(490-round((ey-vy)*factor))//2+round((ry-vy)*factor)
                        canvas.paste(preview,(left,top))
                        # IDs are screen-sized; full-resolution default font would
                        # disappear when reducing a several-thousand-pixel ROI.
                        for box in json.loads((folder/'boxes.json').read_text()):
                            if box['bbox_xyxy_exclusive'] is not None:
                                bx,by,bx2,by2=box['bbox_xyxy_exclusive']
                                draw.rectangle((left+(bx-rx)*factor,top+(by-ry)*factor,left+(bx2-rx)*factor,top+(by2-ry)*factor),outline='cyan',width=1)
                                draw.text((left+(bx-rx)*factor,top+(by-ry)*factor),f'p{box["person_id"]}',fill='cyan',stroke_width=1,stroke_fill='black')
                    else:
                        preview.thumbnail((512,512));canvas.paste(preview,(i*512+(512-preview.width)//2,32+(512-preview.height)//2))
                view=recipe.get('view_angles_by_frame')
                draw.text((i*512+8,8),f'{c} tick={records[k]["episode_time_tick"]} base={recipe["time_base"]} source={records[k].get("source_frame",recipe["people"][0]["source_frames"][k])} ROI',fill='white')
                if view is not None:
                    draw.text((i*512+8,20),f'{recipe.get("coverage_goal","boundary_transition")}: {view[k][c]["view_class"]} {view[k][c]["angle_deg"]:.1f}deg',fill='white')
                if recipe.get('observation_policy')=='record_unavailable_v1':
                    slots=[o for o in index['observations'] if o['frame_index']==k and o['camera_id']==c]
                    for j,o in enumerate(slots):
                        state='input' if o['status']=='observed' else o['reason']
                        status_top=430 if len(slots)>3 else 478
                        status_step=12 if len(slots)>3 else 14
                        draw.text((i*512+8,status_top+j*status_step),f'p{o["person_id"]}: {state}',fill='cyan' if o['status']=='observed' else 'orange',stroke_width=1,stroke_fill='black')
            if 'person_timelines' in recipe:
                source=records[k]['person_records'][0]
                draw.text((8,510),f'10 Hz synthetic t={float(records[k]["episode_time_tick"]*time_base):.1f}s | p1 sourceTick={source["source_time_tick"]} base={source["source_time_base"]} | yellow=physical domain',fill='yellow')
            composition=recipe.get('component_provenance')
            component_label=''
            if composition:
                selections=composition if isinstance(composition,list) else [composition]
                def short_component(entry):
                    parts=entry.get('selection',entry.get('components'))
                    if parts is None:raise ValueError('missing_component_selection_in_video_provenance')
                    if len(recipe['people'])>3:
                        return f'p{entry.get("person_id",1)}:{parts["body_id"]}/{parts["motion_id"]}'
                    return (f'p{entry.get("person_id",1)}:{parts["body_id"]}/'
                            f'{parts["motion_id"]}/{parts["clothing_id"]}/{parts["hair_id"]}')
                component_label=' | '.join(short_component(entry) for entry in selections)
            if recipe.get('coverage_goal')=='edge_coverage' and recipe.get('requested_view_class')=='front':
                font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
                title='正面边缘覆盖（端点过界未通过）' if recipe.get('boundary_transition_passed') is False else '正面边缘覆盖'
                if component_label:title+=' | '+component_label
                if font_path.is_file():
                    draw.text((8,494),title,fill='yellow',font=ImageFont.truetype(str(font_path),13))
                else:
                    draw.text((8,494),'FRONT EDGE COVERAGE (endpoint transition not passed)',fill='yellow')
            elif component_label:
                goal=recipe.get('coverage_goal')
                if goal in ('back_view_and_edge_coverage','front_view_control','side_view_control'):
                    title={'back_view_and_edge_coverage':'BACK VIEW + EDGE',
                           'front_view_control':'FRONT VIEW',
                           'side_view_control':'SIDE VIEW (short window)'}[goal]
                else:
                    title='EDGE COVERAGE'
                draw.text((8,494),title+' | '+component_label,fill='yellow')
            draw.text((8,525),f'people={[p["person_id"] for p in recipe["people"]]} | grey = unrendered layout only',fill='white')
            if 'occlusion_metric' in recipe and k in selected:canvas.save(pack/f'event_frame_{k:03d}.png')
            frame=av.VideoFrame.from_ndarray(np.asarray(canvas),format='rgb24');frame.pts=records[k]['episode_time_tick'];frame.time_base=time_base
            for packet in stream.encode(frame):mux(packet)
        for packet in stream.encode():mux(packet)
    with av.open(str(pack/'dual_roi.mp4')) as video:decoded=[(f.pts,f.time_base) for f in video.decode(video=0)]
    expected=[Fraction(records[k]['episode_time_tick'])*Fraction(*records[k]['time_base']) for k in video_frames]
    assert [tick*tb for tick,tb in decoded]==expected,'video_integer_time_conflict'
    pts=[float(t) for t in expected]
    video_seconds=time.monotonic()-t
    for k in (partial['consumer_completed_frame_indices'] if partial else range(count)):
        out=root/f'frame_{k:03d}';source=out/'_temporary_ds_source/source'
        if source.exists():complete_output(out,'compact',source,checks=dict(status='passed',packed=True,Dataset_selected_frames=selected))
    if not partial:assert not list(root.rglob('_temporary_ds_source'))
    write(pack/'readback.json',dict(status='passed',scope='bounded valid Dataset rows; does not imply episode coverage acceptance',planned_frame_count=count,video_frame_indices=video_frames,preserved_rejected_frames=partial['rejected_frame_indices'] if partial else [],actual_Dataset_reads=sum(c.get('Dataset_read',False) for c in checks),checks=checks,
        pack_seconds=pack_seconds,readback_seconds=readback_seconds,video_seconds=video_seconds,
        elapsed_seconds=time.monotonic()-start,video_pts_seconds=pts,video_source_ticks=[r.get('source_time_tick') for r in records],video_person_source_records=[r.get('person_records') for r in records],video_fixed_full_erp_viewports=viewports,video_episode_ticks=ticks,video_time_base=recipe['time_base'],
        last_frame_display_hold_ticks=last_hold,last_frame_display_hold_seconds=float(last_hold*time_base),packed_rows=index['packed']['row_count'],
        observation_status_counts={status:sum(o['status']==status for o in index['observations']) for status in sorted({o['status'] for o in index['observations']})},model_forwards=0,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['compare','finish']);ap.add_argument('--root',type=Path,required=True)
    args=ap.parse_args();(compare if args.stage=='compare' else finish_short)(args.root)
