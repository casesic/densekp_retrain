#!/usr/bin/env python3
"""Cross two saved heavy-occlusion images and two fixed box definitions."""
import argparse
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.validate.evaluate_ds_occlusion_control import (
    np, torch, Image, DatasetKeypoints, DATASET_FILES, DATASET_FOLDERS,
    audit_cfg, DenseKP, initialize_config_dir, compose, restore_crop_xy,
    point_errors, summarize, draw_case,
)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control-root',type=Path,required=True)
    parser.add_argument('--old-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'assets_external/densekp.ckpt')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    box_sources={'clear_fixed':args.control_root/'clear','heavy_visible':args.control_root/'heavy'}
    boxes={name:next(b['bbox_xyxy_exclusive'] for b in json.loads((path/'cam2/boxes.json').read_text()) if b['person_id']==3)
           for name,path in box_sources.items()}
    images={'old_8_samples':args.old_root,'new_64_samples':args.control_root/'heavy'}
    original_people=json.loads((args.old_root/'people.json').read_text())['people']
    current_people=json.loads((args.control_root/'heavy/people.json').read_text())['people']
    if original_people!=current_people: raise ValueError('Heavy scene geometry changed')
    if json.loads((args.old_root/'rig.json').read_text())!=json.loads((args.control_root/'heavy/rig.json').read_text()):
        raise ValueError('Camera changed')
    torch.manual_seed(0);np.random.seed(0);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.allow_tf32=False
    with initialize_config_dir(version_base='1.2',config_dir=str(ROOT/'configs/hydra')):
        cfg=compose(config_name='train',overrides=['experiment=densekp'])
    cfg.DATASETS.TARGET_VALUE_SEMANTICS='supervision_weight'
    model=DenseKP.load_from_checkpoint(str(args.checkpoint),cfg=cfg,map_location='cpu',strict=True).to('cuda:0').eval()
    cases=[];sheet=Image.new('RGB',(1024,1024));reference=None;affines={}
    for i,(image_name,directory) in enumerate(images.items()):
        with np.load(directory/'cam2/annotations.npz') as a:
            index=a['person_ids'].tolist().index(3)
            gt,visible=a['xy'][index],a['visible'][index]
        if reference is None: reference=(gt,visible)
        else:
            np.testing.assert_array_equal(gt,reference[0]);np.testing.assert_array_equal(visible,reference[1])
        for j,(box_name,bounds) in enumerate(boxes.items()):
            bounds=np.asarray(bounds);center=(bounds[:2]+bounds[2:])/2;scale=float(max(bounds[2:]-bounds[:2])*1.2/200)
            name=f'{image_name}_{box_name}'
            archive=args.output/f'{name}_loader.npz'
            np.savez(archive,imgname=np.array(['cam2/normal_erp.png']),center=center[None].astype(np.float32),
                     scale=np.array([scale],dtype=np.float32),proj_verts=np.column_stack([gt,visible])[None].astype(np.float32))
            with patch.dict(DATASET_FOLDERS,{'box-control':str(directory)}),patch.dict(DATASET_FILES[False],{'box-control':str(archive)}):
                dataset=DatasetKeypoints(audit_cfg('supervision_weight'),'box-control',is_train=False,use_augmentation=False)
            item=dataset[0]
            if box_name not in affines: affines[box_name]=item['_trans'].copy()
            np.testing.assert_array_equal(item['_trans'],affines[box_name])
            np.testing.assert_allclose(restore_crop_xy(item['proj_verts_cropped'][:,:2],item['_trans'],(256,256)),gt,atol=2e-4,rtol=0)
            with torch.inference_mode(): raw=model({'img':torch.from_numpy(item['img'][None]).to('cuda:0')})['pred_keypoints']
            if tuple(raw.shape)!=(1,138,3) or not torch.isfinite(raw).all():raise ValueError('Invalid output')
            raw=raw[0].float().cpu().numpy();restored=restore_crop_xy(raw[:,:2],item['_trans'],(256,256))
            crop,erp=point_errors(raw[:,:2],item['proj_verts_cropped'][:,:2],restored,gt,1536)
            case=dict(image=image_name,image_path=str((directory/'cam2/normal_erp.png').resolve()),box=box_name,
                      bounds=bounds.tolist(),center=center.tolist(),crop_side_original_px=scale*200,
                      visible_crop_error_px=summarize(crop[visible]),visible_erp_error_px=summarize(erp[visible]))
            cases.append(case)
            np.savez(args.output/f'{name}.npz',prediction_raw=raw,prediction_erp_xy=restored,
                     affine=item['_trans'],target_erp_xy=gt,target_crop_xy=item['proj_verts_cropped'][:,:2],visible=visible,crop_error_px=crop)
            mask=np.array(Image.open(directory/'cam2/person_3_mask.png'))
            panel=draw_case(item,raw,visible,visible,visible,mask,f'{image_name} / {box_name} / mean {crop[visible].mean():.2f}px')
            sheet.paste(panel,(512*j,512*i))
            print(case,flush=True)
    sheet.save(args.output/'box_comparison.jpg')
    report=dict(checkpoint=str(args.checkpoint.resolve()),precision='float32, no autocast, TF32 disabled',
                cases=cases,person_id=3,camera='cam2',invariants='same heavy layout, camera, 22 visible semantic rows; box affines equal across image versions',
                limitation='old image seed not recorded; between-image effects are rendering-version effects, not proven sampling-only effects',
                value_semantics='third output preserved, never weighted',scope='given-box sensitivity; no detector changes')
    (args.output/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
