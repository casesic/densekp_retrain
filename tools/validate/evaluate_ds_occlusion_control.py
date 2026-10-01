#!/usr/bin/env python3
"""Evaluate fixed oracle crops across a rendered three-level occlusion control."""
import argparse
import json
from pathlib import Path
import subprocess
import shutil
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
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
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, make_targets, points_in_mask
from tools.validate.validate_bedlam_densekp_tiny import audit_cfg
from tools.validate.occlusion_metrics import (
    summarize, point_errors, evaluation_masks, prediction_retention, pair_masks,
)

LEVELS = ('clear', 'mild', 'heavy')
CAMERAS = ('cam2', 'cam4')
FIXED_IDS = (2, 3, 4)


def read_data(root):
    data, reports, rigs, sources = {}, {}, {}, {}
    for level in LEVELS:
        directory = root/level
        reports[level] = json.loads((directory/'report.json').read_text())
        rigs[level] = json.loads((directory/'rig.json').read_text())
        source = json.loads((directory/'source/source_manifest.json').read_text())
        sources[level] = source
        if source['people'] != reports[level]['people']:
            raise ValueError('Annotation/render people mismatch')
        if source['seed'] != 0 or source['animated_seed'] or source['samples'] != 64:
            raise ValueError('Expected fixed seed and 64 samples')
        for camera in source['sources']:
            saved = next(c for c in rigs[level]['cameras'] if c['id'] == camera['id'])
            np.testing.assert_array_equal(camera['camera_to_world'], saved['camera_to_world'])
        for camera in CAMERAS:
            with np.load(directory/camera/'annotations.npz') as archive:
                data[level, camera] = {k: archive[k] for k in archive.files}
    for level in LEVELS:
        if rigs[level] != rigs['clear']:
            raise ValueError('Camera changed between conditions')
        for field in ('scene_blend','resolution','samples','seed','lighting','device'):
            if sources[level][field] != sources['clear'][field]:
                raise ValueError(f'Render setting changed: {field}')
        for pid in FIXED_IDS:
            people = {p['id']:p for p in reports[level]['people']}
            clear_people = {p['id']:p for p in reports['clear']['people']}
            if people[pid] != clear_people[pid]:
                raise ValueError('Target/control geometry changed')
            for camera in CAMERAS:
                current = data[level, camera]
                baseline = data['clear', camera]
                i, j = current['person_ids'].tolist().index(pid), baseline['person_ids'].tolist().index(pid)
                for key in ('world_densekp', 'xy', 'in_view'):
                    np.testing.assert_array_equal(current[key][i], baseline[key][j])
    return data, reports


def subset(data, level, camera, pid, key):
    row = data[level, camera]
    return row[key][row['person_ids'].tolist().index(pid)]


def draw_case(item, prediction, visible, in_view, common, instance, title):
    canvas = Image.fromarray(cv2.cvtColor(item['img_disp'], cv2.COLOR_BGR2RGB)).resize((512,512))
    draw = ImageDraw.Draw(canvas)
    mask = cv2.warpAffine(instance, item['_trans'], (256,256), flags=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        points = [tuple(map(int, p)) for p in contour[:,0]*2]
        if len(points)>1: draw.line(points+points[:1], fill=(0,220,255))
    gt = (item['proj_verts_cropped'][:,:2]+.5)*512
    pred = (prediction[:,:2]+.5)*512
    for i in range(138):
        x,y = pred[i]
        draw.ellipse((x-2,y-2,x+2,y+2), fill=(255,60,70))
        if in_view[i]:
            a,b = gt[i]
            if common[i]: draw.line([(a,b),(x,y)],fill=(230,190,70))
            draw.ellipse((a-3,b-3,a+3,b+3),fill=(30,255,80) if visible[i] else (255,170,40))
    draw.line((64,38,64,512),fill=(180,180,180))
    draw.line((448,38,448,512),fill=(180,180,180))
    draw.rectangle((0,0,512,37),fill='black')
    draw.text((6,4),title,fill='white')
    draw.text((6,21),f'GT green: visible / orange: occluded / red: prediction',fill='white')
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'assets_external/densekp.ckpt')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--reuse-predictions', type=Path,
                        help='Rescore saved raw outputs for the same source, fixed boxes and affine; no model run')
    args = parser.parse_args()
    args.root, args.output = args.root.resolve(), args.output.resolve()
    data, generated = read_data(args.root)
    device = torch.device('cpu' if args.reuse_predictions else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    cached_report = None
    if args.reuse_predictions:
        args.reuse_predictions = args.reuse_predictions.resolve()
        cached_report = json.loads((args.reuse_predictions/'report.json').read_text())
        if Path(cached_report['source']).resolve() != args.root:
            raise ValueError('Saved predictions refer to another source; run fresh inference')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(0); np.random.seed(0); torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.allow_tf32=False
    with initialize_config_dir(version_base='1.2',config_dir=str(ROOT/'configs/hydra')):
        cfg=compose(config_name='train',overrides=['experiment=densekp'])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS='supervision_weight'
    if args.reuse_predictions:
        shutil.copyfile(args.reuse_predictions/'model_config.yaml', args.output/'model_config.yaml')
    else:
        OmegaConf.save(cfg,args.output/'model_config.yaml')
    (args.output/'loader_config.yaml').write_text(audit_cfg('supervision_weight').dump())
    model = None
    if not args.reuse_predictions:
        print('Loading original DenseKP checkpoint',flush=True)
        model=DenseKP.load_from_checkpoint(str(args.checkpoint),cfg=cfg,map_location='cpu',strict=True).to(device).eval()
    fixed=[]
    for camera in CAMERAS:
        for box in json.loads((args.root/'clear'/camera/'boxes.json').read_text()):
            if box['person_id'] not in FIXED_IDS or box['bbox_xyxy_exclusive'] is None: continue
            bounds=np.asarray(box['bbox_xyxy_exclusive'])
            fixed.append(dict(camera=camera,person_id=box['person_id'],bbox_xyxy_exclusive=box['bbox_xyxy_exclusive'],
                              center=((bounds[:2]+bounds[2:])/2).tolist(), scale=float(max(bounds[2:]-bounds[:2])*1.2/200)))
    (args.output/'fixed_boxes.json').write_text(json.dumps(fixed,indent=2)+'\n')
    if args.reuse_predictions:
        if json.loads((args.reuse_predictions/'fixed_boxes.json').read_text()) != fixed:
            raise ValueError('Saved prediction boxes differ')
    cases, states, panels, raw_panels, affine_reference = [], {}, {}, {}, {}
    for level in LEVELS:
        output=args.output/level; output.mkdir()
        targets=[make_targets(subset(data,level,r['camera'],r['person_id'],'xy'),
                              subset(data,level,r['camera'],r['person_id'],'in_view'),
                              r['center'],r['scale']) for r in fixed]
        archive=output/'fixed_box_loader.npz'
        np.savez(archive,imgname=np.array([f"{r['camera']}/normal_erp.png" for r in fixed]),
                 center=np.array([r['center'] for r in fixed],dtype=np.float32),
                 scale=np.array([r['scale'] for r in fixed],dtype=np.float32),proj_verts=np.array(targets,dtype=np.float32),
                 supervision_policy=np.array(SUPERVISION_POLICY),
                 in_view=np.stack([subset(data,level,r['camera'],r['person_id'],'in_view') for r in fixed]),
                 visible=np.stack([subset(data,level,r['camera'],r['person_id'],'visible') for r in fixed]),
                 camera_id=np.array([r['camera'] for r in fixed]),
                 person_id=np.array([r['person_id'] for r in fixed]))
        with patch.dict(DATASET_FOLDERS,{'occlusion-control':str(args.root/level)}), \
             patch.dict(DATASET_FILES[False],{'occlusion-control':str(archive)}):
            dataset=DatasetKeypoints(audit_cfg('supervision_weight'),'occlusion-control',is_train=False,use_augmentation=False)
        for i,row in enumerate(fixed):
            item=dataset[i]; camera,pid=row['camera'],row['person_id']; key=(camera,pid)
            if level=='clear': affine_reference[key]=item['_trans'].copy()
            np.testing.assert_array_equal(item['_trans'],affine_reference[key])
            gt=subset(data,level,camera,pid,'xy')
            np.testing.assert_allclose(restore_crop_xy(item['proj_verts_cropped'][:,:2],item['_trans'],(256,256)),gt,atol=2e-4,rtol=0)
            if args.reuse_predictions:
                with np.load(args.reuse_predictions/level/f'{camera}_person_{pid}.npz') as saved:
                    np.testing.assert_array_equal(saved['target_erp_xy'], gt)
                    np.testing.assert_array_equal(saved['affine'], item['_trans'])
                    np.testing.assert_array_equal(saved['target_crop_xy'], item['proj_verts_cropped'][:,:2])
                    prediction = saved['prediction_raw'].copy()
            else:
                with torch.inference_mode():
                    raw=model({'img':torch.from_numpy(item['img'][None]).to(device)})['pred_keypoints']
                if tuple(raw.shape)!=(1,138,3): raise ValueError('Invalid prediction shape')
                prediction=raw[0].float().cpu().numpy()
            if prediction.shape != (138,3) or not np.isfinite(prediction).all():
                raise ValueError('Invalid prediction')
            restored=restore_crop_xy(prediction[:,:2],item['_trans'],(256,256))
            lens_mask=np.asarray(Image.open(args.root/level/camera/'valid_mask.png'))>0
            width=lens_mask.shape[1]
            crop_error,erp_error=point_errors(prediction[:,:2],item['proj_verts_cropped'][:,:2],restored,gt,width)
            prediction_valid=points_in_mask(restored,lens_mask)
            in_view=subset(data,level,camera,pid,'in_view')
            visible=subset(data,level,camera,pid,'visible')
            baseline=subset(data,'clear',camera,pid,'visible')
            common=np.logical_and.reduce([subset(data,l,camera,pid,'visible') for l in LEVELS])
            other='cam4' if camera=='cam2' else 'cam2'
            shared=visible & subset(data,level,other,pid,'visible')
            fixed_shared=common & np.logical_and.reduce([subset(data,l,other,pid,'visible') for l in LEVELS])
            crop_xy=(item['proj_verts_cropped'][:,:2]+.5)*256
            retained=(crop_xy[:,0]>=32)&(crop_xy[:,0]<224)&(crop_xy[:,1]>=0)&(crop_xy[:,1]<256)
            masks=dict(current_visible=visible,common_visible_all_levels=common,
                       baseline_visible_including_newly_occluded=baseline,shared_current=shared,shared_all_levels=fixed_shared,
                       common_visible_retained=common&retained)
            masks.update(evaluation_masks(in_view,visible,item['proj_verts_cropped'][:,:2]))
            scores={k:dict(crop_px=summarize(crop_error[m]),erp_periodic_px=summarize(erp_error[m])) for k,m in masks.items()}
            instance=np.array(Image.open(args.root/level/camera/f'person_{pid}_mask.png'))
            base_instance=np.array(Image.open(args.root/'clear'/camera/f'person_{pid}_mask.png'))
            nearest=np.full(138,np.inf)
            a=data[level,camera]
            for j,other_id in enumerate(a['person_ids']):
                if other_id==pid: continue
                delta=restored-a['xy'][j]; delta[:,0]=(delta[:,0]+width/2)%width-width/2
                nearest=np.minimum(nearest,np.where(a['visible'][j],np.linalg.norm(delta,axis=1),np.inf))
            hint=nearest+5*float(item['box_size'])/256<erp_error
            case=dict(level=level,**row,metrics=scores,visible_instance_pixels=int((instance>0).sum()),
                      prediction_retention={name:prediction_retention(mask,prediction_valid) for name,mask in masks.items()},
                      fully_occluded_target=not bool((instance>0).any()),
                      baseline_instance_pixels=int((base_instance>0).sum()),
                      remaining_instance_fraction=float((instance>0).sum()/(base_instance>0).sum()),
                      common_visible_nearer_neighbor_hint=int((hint&common).sum()),
                      baseline_visible_outside_retained=int((baseline&~retained).sum()))
            cases.append(case)
            np.savez(output/f'{camera}_person_{pid}.npz',prediction_raw=prediction,prediction_erp_xy=restored,
                     target_crop_xy=item['proj_verts_cropped'][:,:2],target_erp_xy=gt,affine=item['_trans'],
                     crop_error_px=crop_error,erp_periodic_error_px=erp_error,retained_input=retained,
                     nearer_neighbor_hint=hint,prediction_valid=prediction_valid,**masks)
            panels[level,camera,pid]=draw_case(item,prediction,visible,in_view,common,instance,f'{level} / {camera} / target {pid}')
            raw_panels[level,camera,pid]=Image.fromarray(cv2.cvtColor(item['img_disp'],cv2.COLOR_BGR2RGB)).resize((512,512))
            panels[level,camera,pid].save(output/f'{camera}_person_{pid}.jpg')
            states[level,camera,pid]=dict(error=crop_error,shared=shared,fixed_shared=fixed_shared,
                                         in_view=in_view,visible=visible,prediction_valid=prediction_valid)
            print(f'{level} {camera} target {pid}: visible {visible.sum()}, occluded {masks["occluded"].sum()}, '
                  f'all-in-view error {scores["all_in_view"]["crop_px"]["mean"]}',flush=True)
    pairs=[]
    for level in LEVELS:
        for pid in FIXED_IDS:
            a,b=states.get((level,'cam2',pid)),states.get((level,'cam4',pid))
            if a is None or b is None: continue
            pair=dict(level=level,person_id=pid)
            for name, mask in pair_masks(a['in_view'],a['visible'],b['in_view'],b['visible']).items():
                pair[name]=dict(points=int(mask.sum()),
                                mean_two_view_crop_px=summarize((a['error'][mask]+b['error'][mask])/2),
                                worse_view_crop_px=summarize(np.maximum(a['error'][mask],b['error'][mask])),
                                prediction_retention=prediction_retention(mask,a['prediction_valid']&b['prediction_valid']))
            for key in ('shared','fixed_shared'):
                np.testing.assert_array_equal(a[key],b[key])
                m=a[key]
                pair[key]=dict(points=int(m.sum()),mean_two_view_crop_px=summarize((a['error'][m]+b['error'][m])/2),
                               worse_view_crop_px=summarize(np.maximum(a['error'][m],b['error'][m])))
            pairs.append(pair)
    for pid in FIXED_IDS:
        for panels_dict,name in ((panels,'prediction'),(raw_panels,'input')):
            sheet=Image.new('RGB',(1536,1024),(25,25,25))
            for j,level in enumerate(LEVELS):
                for i,camera in enumerate(CAMERAS):
                    key=(level,camera,pid)
                    if key in panels_dict:
                        sheet.paste(panels_dict[key],(512*j,512*i))
                        if name=='input':
                            d=ImageDraw.Draw(sheet);d.rectangle((512*j,512*i,512*j+512,512*i+22),fill='black')
                            d.text((512*j+6,512*i+5),f'{level} / {camera} / target {pid}',fill='white')
                    else: ImageDraw.Draw(sheet).text((512*j+6,512*i+6),f'{level} / {camera}: no baseline box',fill='white')
            sheet.save(args.output/f'person_{pid}_{name}_review.jpg')
    result=dict(status='completed',source=str(args.root),checkpoint=str(args.checkpoint.resolve()),
                git_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                torch=torch.__version__,device=str(device),precision='float32, no autocast, TF32 disabled',
                samples=64,seed=0,baseline_m=.18,baseline_status='synthetic, not measured',
                fixed_target_ids=list(FIXED_IDS),primary_target=3,paired_edge_control=2,
                evaluated_views=len(cases),finite_output_values=len(cases)*414,
                invariants='same target pose/shape/location, cameras, projected xy, in_view and exact crop affine across levels',
                masks='5x5 neighborhood and 4cm ray-range tolerance; approximate visibility',
                metric_policy='GT in-view points, visible/occluded/crop groups; predictions rejected by lens mask remain scored; empty subsets null',
                supervision_policy=SUPERVISION_POLICY,
                fully_occluded_policy='fixed clear-box diagnostic only; not a detector-recall or observable-person claim',
                baseline_metric='baseline_visible includes newly occluded points, a diagnostic extrapolation test, not visible localization',
                value_semantics='third output saved unchanged, never used as confidence or weight',
                neighbor_hint='same-row visible neighbor closer by >5 crop pixels; not proof of identity swap',
                cases=cases,pairs=pairs,scope='one street scene, repeated identity/poses; synthetic fixed boxes; no training/detector/reconstruction')
    if cached_report is not None:
        result.update(prediction_mode='rescore saved predictions; no inference',
                      reused_predictions=str(args.reuse_predictions),
                      checkpoint=cached_report['checkpoint'],
                      inference_device=cached_report['device'],
                      inference_precision=cached_report['precision'],
                      reuse_checks='same source path, boxes, GT and affine; source image contents must remain unchanged')
    (args.output/'report.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(f'Completed: {args.output}',flush=True)


if __name__=='__main__': main()
