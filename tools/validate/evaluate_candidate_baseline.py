"""Strict checkpoint inference on frozen rows, with optional frozen-input comparison."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys
import time
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import cv2
import numpy as np
from PIL import Image, ImageDraw
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from densekp.core.configs import DATASET_FILES, DATASET_FOLDERS
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.densekp_trainer import DenseKP
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, points_in_mask
from tools.validate.validate_bedlam_densekp_tiny import audit_cfg
from tools.validate.occlusion_metrics import evaluation_masks, point_errors


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def summary(values):
    values=np.asarray(values,float);bad=int((~np.isfinite(values)).sum())
    valid=len(values)>0 and bad==0
    return dict(count=int(values.size),nonfinite_count=bad,
        status='nonfinite_failure' if bad else ('scored' if len(values) else 'empty'),
        mean=float(values.mean()) if valid else None,
        median=float(np.median(values)) if valid else None,
        p90=float(np.percentile(values,90)) if valid else None)


def render_panel(item,pred,supervised,visible,label):
    panel=Image.fromarray(cv2.cvtColor(item['img_disp'],cv2.COLOR_BGR2RGB)).resize((512,512))
    draw=ImageDraw.Draw(panel);gt=(item['proj_verts_cropped'][:,:2]+.5)*512;xy=(pred[:,:2]+.5)*512
    for k in np.flatnonzero(supervised):
        a=gt[k];b=xy[k];color=(40,255,70) if visible[k] else (255,175,40)
        if np.isfinite(b).all() and np.abs(b).max()<100000:
            draw.line([tuple(a),tuple(b)],fill=(230,190,70),width=1)
            draw.ellipse((b[0]-2,b[1]-2,b[0]+2,b[1]+2),fill=(255,60,70))
        draw.ellipse((a[0]-3,a[1]-3,a[0]+3,a[1]+3),fill=color)
    for x in (64,448):draw.line((x,25,x,512),fill=(180,180,180))
    draw.rectangle((0,0,512,24),fill='black');draw.text((5,5),label,fill='white')
    return panel


def aggregate(cases,states):
    groups={}
    for case in cases:
        state=states[case['id']]
        prefixes=['all',f"action/{case['action']}",f"layout/{case['category']}",f"scale/{case['scale_group']}"]
        if 'lens_truncated_gt' in case:
            prefixes.append('actual_lens_gt/'+('truncated' if case['lens_truncated_gt'] else 'complete'))
        if 'neighbor_crop_pixels' in case:
            prefixes.append('actual_crop_neighbor/'+('present' if case['neighbor_crop_pixels']>0 else 'absent'))
        if 'in_view_outside_full_crop' in case:
            prefixes.append('actual_crop_gt/'+('truncated' if case['in_view_outside_full_crop'] else 'complete'))
        for name,mask in state['masks'].items():
            for prefix in prefixes:
                group=groups.setdefault(prefix+'/'+name,dict(crop=[],erp=[],person_crop=[],person_erp=[],rows=0))
                group['crop'].extend(state['crop'][mask]);group['erp'].extend(state['erp'][mask]);group['rows']+=1
                if mask.any():
                    group['person_crop'].append(float(state['crop'][mask].mean()))
                    group['person_erp'].append(float(state['erp'][mask].mean()))
    return {key:dict(rows=g['rows'],point_weighted=dict(crop_256_px=summary(g['crop']),erp_periodic_px=summary(g['erp'])),
                     person_weighted=dict(crop_256_px=summary(g['person_crop']),erp_periodic_px=summary(g['person_erp'])))
            for key,g in groups.items()}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--indices',type=Path,nargs='+',required=True)
    parser.add_argument('--labels',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split-manifest',type=Path,help='Required for source-isolated train/val indices')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'assets_external/densekp.ckpt')
    parser.add_argument('--model-config',type=Path)
    parser.add_argument('--reference-report',type=Path,help='Require identical case IDs, images, GT, affine and masks to this baseline')
    parser.add_argument('--device',default='cuda:0');args=parser.parse_args()
    if len(args.labels)!=len(args.indices):raise ValueError('Each index needs a batch label')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    references={}
    if args.reference_report:
        references={c['id']:c for c in json.loads(args.reference_report.read_text())['cases']}
    sources=[];partitions={};partition_indices={}
    for label,path in zip(args.labels,args.indices):
        index=json.loads(path.read_text())
        if any(row['quality_state']!='candidate_accepted' for row in index['rows']):
            raise ValueError('Inference requires pre-frozen, accepted candidate rows')
        sources.append((label,index['rows']))
        partitions[label]=index.get('split','integration_only')
        if partitions[label] in ('train','val'):
            if partitions[label] in partition_indices:raise ValueError('Use one index per partition per baseline run')
            partition_indices[partitions[label]]=index
        shutil.copyfile(path,out/f'{label}_frozen_index.json')
    if partition_indices:
        if not args.split_manifest:raise ValueError('Source-isolated indices need their split manifest')
        from tools.validate.validate_source_isolated_split import validate_split
        write(out/'source_isolation.json',validate_split(json.loads(args.split_manifest.read_text()),partition_indices))
    split_scope='source_isolated_initial_holdout' if partition_indices else 'integration_only'
    write(out/'selection_frozen.json',dict(utc=datetime.now(timezone.utc).isoformat(),
        rows={label:len(rows) for label,rows in sources},selection='input indices frozen before model loading',
        scale_bins='exclusive oracle bbox height in synthetic ERP pixels: small <96, medium 96..191, large >=192',
        fixed_controls='first indexed row in each batch/action/layout category; determined before prediction',
        split=split_scope,partitions=partitions))
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    with initialize_config_dir(version_base='1.2',config_dir=str(ROOT/'configs/hydra')):
        cfg=compose(config_name='train',overrides=['experiment=densekp'])
    if args.model_config:cfg=OmegaConf.load(args.model_config)
    cfg.DATASETS.TARGET_VALUE_SEMANTICS='supervision_weight';OmegaConf.save(cfg,out/'model_config.yaml')
    (out/'loader_config.yaml').write_text(audit_cfg('supervision_weight').dump())
    checkpoint=args.checkpoint.resolve();stat=checkpoint.stat();load_start=time.monotonic()
    payload=torch.load(checkpoint,map_location='cpu');model=DenseKP(cfg=cfg)
    result=model.load_state_dict(payload['state_dict'],strict=True)
    load_audit=dict(checkpoint=str(checkpoint.resolve()),bytes=stat.st_size,mtime_ns=stat.st_mtime_ns,
        epoch=payload.get('epoch'),global_step=payload.get('global_step'),state_dict_keys=len(payload['state_dict']),
        missing_keys=result.missing_keys,unexpected_keys=result.unexpected_keys,strict=True,
        source='explicit strict checkpoint evaluation',with_var=bool(cfg.MODEL.with_var))
    del payload
    device=torch.device(args.device);model.to(device).eval();load_audit['seconds']=time.monotonic()-load_start
    write(out/'checkpoint_load.json',load_audit)
    datasets={};scenes={};loader_ids={};cases=[];states={};pairs={};fixed={};image_cache={}
    for batch,rows in sources:
        for position,row in enumerate(rows):
            directory=Path(row['scene']);key=str(Path(row['loader_input']).resolve())
            if key not in datasets:
                if json.loads((directory/'validation.json').read_text())['status']!='passed':raise ValueError('Scene numeric validation failed')
                with patch.dict(DATASET_FOLDERS,{'frozen-baseline':str(directory)}),patch.dict(DATASET_FILES[False],{'frozen-baseline':row['loader_input']}):
                    datasets[key]=DatasetKeypoints(audit_cfg('supervision_weight'),'frozen-baseline',is_train=False,use_augmentation=False)
                scenes[key]=json.loads((directory/'report.json').read_text())
                with np.load(row['loader_input']) as loader:
                    loader_ids[key]={name:loader[name] for name in ('camera_id','person_id')}
            dataset=datasets[key];i=row['loader_row'];item=dataset[i]
            if dataset.data.get('supervision_policy')!=SUPERVISION_POLICY:raise ValueError('Wrong supervision policy')
            if str(loader_ids[key]['camera_id'][i])!=row['camera'] or int(loader_ids[key]['person_id'][i])!=row['person_id']:
                raise ValueError('Frozen candidate index differs from loader row')
            person=next(p for p in scenes[key]['people'] if p['id']==row['person_id'])
            if person['character_id']!=row['character_id'] or person['pose_source_id']!=row['pose_source_id']:
                raise ValueError('Frozen character/pose differs from rendered person')
            data=np.load(directory/row['camera']/'annotations.npz');j=data['person_ids'].tolist().index(row['person_id'])
            target=item['proj_verts_cropped'][:,:2];visible=data['visible'][j]
            np.testing.assert_allclose(restore_crop_xy(target,item['_trans'],(256,256)),data['xy'][j],atol=2e-4,rtol=0)
            masks=evaluation_masks(data['in_view'][j],visible,target)
            masks['visible_supervised']=masks['supervised']&visible
            np.testing.assert_array_equal(masks['supervised'],item['proj_verts_cropped'][:,2]>0)
            if int(masks['supervised'].sum())!=row['supervised_points']:raise ValueError('Frozen supervision count changed')
            case_id=f'{batch}_{position:03d}';folder=out/case_id;folder.mkdir()
            if args.reference_report:
                previous=references[case_id]
                for field in ['scene','camera','person_id','character_id','pose_source_id','loader_row']:
                    if row[field]!=previous[field]:raise ValueError(f'Frozen case changed: {case_id}/{field}')
                with np.load(previous['prediction']) as old:
                    np.testing.assert_array_equal(item['img'][:,:,32:-32],old['network_input'])
                    np.testing.assert_array_equal(target,old['target_crop_xy'])
                    np.testing.assert_array_equal(data['xy'][j],old['target_erp_xy'])
                    np.testing.assert_array_equal(data['world_densekp'][j],old['target_world_xyz'])
                    np.testing.assert_array_equal(item['_trans'],old['affine'])
                    np.testing.assert_array_equal(visible,old['visible'])
                    np.testing.assert_array_equal(data['in_view'][j],old['in_view'])
                    np.testing.assert_array_equal(masks['supervised'],old['supervised'])
                np.testing.assert_array_equal(cv2.cvtColor(item['img_disp'],cv2.COLOR_BGR2RGB),
                    np.array(Image.open(Path(previous['prediction']).parent/'square.png')))
            with torch.inference_mode():
                pred=model({'img':torch.from_numpy(item['img'][None]).to(device)})['pred_keypoints'][0].float().cpu().numpy()
            if pred.shape!=(138,3):
                np.save(folder/'invalid_shape.npy',pred);raise ValueError('Model output shape is not 138 x 3')
            restored=restore_crop_xy(pred[:,:2],item['_trans'],(256,256))
            width,height=Image.open(row['image']).size
            crop_error,erp_error=point_errors(pred[:,:2],target,restored,data['xy'][j],width)
            invalid=~np.isfinite(pred[:,:2]).all(1);crop_error[invalid]=np.inf;erp_error[invalid]=np.inf
            valid_mask=np.array(Image.open(directory/row['camera']/'valid_mask.png'))>0
            predicted_in_lens=points_in_mask(restored,valid_mask)
            box=next(b for b in json.loads((directory/row['camera']/'boxes.json').read_text()) if b['person_id']==row['person_id'])['bbox_xyxy_exclusive']
            box=row.get('given_bbox_xyxy',box)
            if 'affine_flat' in row:
                np.testing.assert_allclose(item['_trans'].ravel(),row['affine_flat'],atol=1e-7,rtol=0)
            action=person.get('action','cycling');category=person['category'];bbox_height=box[3]-box[1]
            scale='small' if bbox_height<96 else ('medium' if bbox_height<192 else 'large')
            metrics={name:dict(crop_256_px=summary(crop_error[mask]),erp_periodic_px=summary(erp_error[mask])) for name,mask in masks.items()}
            case=dict(row)
            case.update(id=case_id,batch=batch,action=action,category=category,scale_group=scale,bbox_xyxy=box,
                prediction_status='finite' if np.isfinite(pred).all() else 'nonfinite_failure',
                bbox_height_erp_px=bbox_height,box_size=float(item['box_size']),metrics=metrics,
                nonfinite_xy_points=int(invalid.sum()),nonfinite_keypoint_value=int((~np.isfinite(pred[:,2])).sum()),
                supervised_prediction_outside_lens=int((masks['supervised']&~predicted_in_lens).sum()),
                prediction=str(folder/'prediction.npz'))
            np.savez(folder/'prediction.npz',prediction_raw=pred,prediction_erp_xy=restored,
                target_crop_xy=target,target_erp_xy=data['xy'][j],target_world_xyz=data['world_densekp'][j],
                affine=item['_trans'],bbox_xyxy=box,box_center=item['box_center'],box_size=item['box_size'],
                visible=visible,in_view=data['in_view'][j],supervised=masks['supervised'],
                crop_error_px=crop_error,erp_periodic_error_px=erp_error,prediction_in_lens=predicted_in_lens,
                character_id=row['character_id'],pose_source_id=row['pose_source_id'],camera_id=row['camera'],
                network_input=item['img'][:,:,32:-32])
            image_key=(key,row['camera'])
            if image_key not in image_cache:
                dest=out/'images'/f'{batch}_{directory.name}_{row["camera"]}.png';dest.parent.mkdir(exist_ok=True)
                shutil.copyfile(row['image'],dest);image_cache[image_key]=str(dest)
            case['saved_image']=image_cache[image_key]
            rgb=cv2.cvtColor(item['img_disp'],cv2.COLOR_BGR2RGB)
            Image.fromarray(rgb).save(folder/'square.png');Image.fromarray(rgb[:,32:-32]).save(folder/'network_center.png')
            panel=render_panel(item,pred,masks['supervised'],visible,f'{case_id} {action} {row["character_id"]} {row["camera"]}')
            panel.save(folder/'prediction.jpg');write(folder/'case.json',case)
            states[case_id]=dict(crop=crop_error,erp=erp_error,masks=masks,visible=visible)
            cases.append(case);pairs.setdefault((batch,str(directory),row['person_id']),{})[row['camera']]=case_id
            fixed.setdefault((batch,action,category),case_id)
            print(case_id,metrics['supervised']['crop_256_px'],flush=True)
    pair_results=[];pair_values={}
    for (batch,scene,pid),views in pairs.items():
        result=dict(batch=batch,scene=scene,person_id=pid,views=views)
        if len(views)!=2:
            result.update(status='missing accepted given-box view; not detector failure');pair_results.append(result);continue
        a,b=[states[views[name]] for name in ('cam2','cam4')];shared=a['masks']['supervised']&b['masks']['supervised']
        groups=dict(both_supervised=shared,both_visible=shared&a['visible']&b['visible'],
            one_visible=shared&(a['visible']^b['visible']),both_occluded=shared&~a['visible']&~b['visible'],
            both_in_view_diagnostic=a['masks']['all_in_view']&b['masks']['all_in_view'])
        result.update(status='scored',metrics={})
        for name,mask in groups.items():
            result['metrics'][name]={}
            for unit in ('crop','erp'):
                mean=(a[unit][mask]+b[unit][mask])/2;worst=np.maximum(a[unit][mask],b[unit][mask])
                result['metrics'][name][unit]=dict(mean_endpoint=summary(mean),worse_endpoint=summary(worst))
                for label in (batch,'combined'):
                    bucket=pair_values.setdefault(f'{label}/{name}/{unit}',dict(mean=[],worst=[]))
                    bucket['mean'].extend(mean);bucket['worst'].extend(worst)
        pair_results.append(result)
    by_batch={batch:aggregate([c for c in cases if c['batch']==batch],states) for batch,_ in sources}
    # Explicit empty actions remain null/count zero, never 0-pixel performance.
    for metrics in by_batch.values():
        for action in ('cycling','standing','walking'):
            metrics.setdefault('action/'+action+'/supervised',dict(rows=0,point_weighted=dict(crop_256_px=summary([]),erp_periodic_px=summary([])),
                person_weighted=dict(crop_256_px=summary([]),erp_periodic_px=summary([]))))
    worst=sorted(cases,key=lambda c:float('inf') if c['metrics']['supervised']['crop_256_px']['nonfinite_count'] else
                 (c['metrics']['supervised']['crop_256_px']['mean'] or -1),reverse=True)[:8]
    for name,ids in [('fixed_controls',list(fixed.values())),('worst_samples',[c['id'] for c in worst])]:
        sheet=Image.new('RGB',(1024,512*((len(ids)+1)//2)),(25,25,25))
        for i,cid in enumerate(ids):sheet.paste(Image.open(out/cid/'prediction.jpg'),(i%2*512,i//2*512))
        sheet.save(out/f'{name}.jpg')
    if checkpoint.stat().st_mtime_ns!=stat.st_mtime_ns or checkpoint.stat().st_size!=stat.st_size:raise ValueError('Checkpoint changed during inference')
    if args.reference_report and set(references)!={c['id'] for c in cases}:raise ValueError('Reference denominator changed')
    report=dict(status='completed',split=split_scope,partitions=partitions,checkpoint=load_audit,cases=cases,
        by_batch=by_batch,combined=aggregate(cases,states),pairs=pair_results,
        paired_aggregate={key:dict(mean_endpoint=summary(v['mean']),worse_endpoint=summary(v['worst'])) for key,v in pair_values.items()},
        fixed_controls=list(fixed.values()),worst_samples=[c['id'] for c in worst],seconds=time.monotonic()-start,
        prediction_failure_cases=[c['id'] for c in cases if c['prediction_status']=='nonfinite_failure'],
        metric_definition='Unweighted valid-GT point means and equal-person means, full 256-square px and periodic ERP px; no 32px shift',
        visibility='visible_supervised and occluded_in_crop partition supervised; all_in_view additionally diagnoses GT outside crop',
        nonfinite_policy='nonfinite XY is infinite error; count retained, aggregate mean/quantiles null and failure reported; raw third value never screens points',
        keypoint_value='all raw third values saved unchanged; not confidence, weight, or selection criterion',
        pair_definition='same world person/semantic row, mean and max of two per-view GT endpoint errors; both supervised required except named diagnostic group',
        reference_report=str(args.reference_report) if args.reference_report else None,
        frozen_input_comparison='all actual inputs, GT, affine, visibility and supervision equal' if references else None,
        limitation='few source groups; batches retain distinct purposes, combined is not a generalization score; no independent test or detector')
    write(out/'report.json',report)


if __name__=='__main__':main()
