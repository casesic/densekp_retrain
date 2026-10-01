"""Audit completed frozen box experiment and correct image-copy metadata without re-inference."""
import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from tools.experiments.prepare_box_robustness import write


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment',type=Path,required=True);a=p.parse_args()
    root=a.experiment.resolve();trial=root/'training';data=root/'data_v2';evaluation=root/'evaluation_v2'
    run=read(trial/'report.json');rows=read(data/'train_index.json')['rows']
    updates=[json.loads(s) for s in (trial/'completed_updates.jsonl').read_text().splitlines()]
    assert [r['completed_update'] for r in updates]==list(range(1,401))
    assert [r['row_index'] for r in updates]==list(range(400))
    before=ROOT/'outputs/experiments/source_isolated_trial_400_gpu1_20261001'
    old_order=[json.loads(s)['row_index'] for s in (before/'completed_updates.jsonl').read_text().splitlines()]
    assert [r['parent_row'] for r in rows]==old_order
    fixed_root=ROOT/'outputs/experiments/source_isolated_pilot_20260930'
    for side in ('train','val'):
        assert (trial/f'frozen_{side}_index.json').read_bytes()==(fixed_root/f'{side}_index.json').read_bytes()
    assert (trial/'frozen_split_v2.json').read_bytes()==(fixed_root/'split_v2.json').read_bytes()
    checkpoint=ROOT/'assets_external/densekp.ckpt';s=checkpoint.stat()
    assert (s.st_size,s.st_mtime_ns)==(run['settings']['checkpoint_bytes'],run['settings']['checkpoint_mtime_ns'])
    validations=[]
    for step in (100,200,300,400):
        folder=trial/f'step_{step:04d}';v=read(folder/'validation.json');z=np.load(folder/'val_predictions.npz')
        assert v['rows']==79 and v['parameters_steps_rng_unchanged'] and z['prediction_raw'].shape==(79,138,3)
        assert np.isfinite(z['prediction_raw']).all() and (folder/'checkpoint.ckpt').stat().st_size>0
        validations.append(dict(step=step,rows=79,checkpoint_bytes=(folder/'checkpoint.ckpt').stat().st_size))
    counts={};corrections=[];raw_shapes=[];neighbor_audit={};instance_cache={}
    for side in ('train','val'):
        for row in read(data/f'{side}_index.json')['rows']:
            camera=Path(row['scene'])/row['camera'];key=str(camera)
            if key not in instance_cache:
                instance=np.zeros((768,1536),np.float32)
                for path in camera.glob('person_*_mask.png'):
                    pid=int(path.stem.split('_')[1]);instance[np.array(Image.open(path))>0]=pid
                instance_cache[key]=instance
            instance=instance_cache[key]
            affine=np.asarray(row['affine_flat']).reshape(2,3)
            crop=cv2.warpAffine(instance,affine,(256,256),flags=cv2.INTER_NEAREST)
            with np.load(row['geometry']) as geometry:
                np.testing.assert_array_equal(crop==row['person_id'],geometry['target_support_crop']>0)
            neighbor_audit[(side,row['loader_row'])]=dict(split=side,row=row['loader_row'],
                parent_neighbor_crop_pixels=row['neighbor_crop_pixels'],
                actual_neighbor_crop_pixels=int(((crop>0)&(crop!=row['person_id'])).sum()),
                target_network_center_pixels=int((crop[:,32:-32]==row['person_id']).sum()))
    write(root/'actual_crop_context_v2.json',list(neighbor_audit.values()))
    for name in ['challenge_A','challenge_B','challenge_C','val_C','regression_C']:
        folder=evaluation/name;path=folder/'report.json';report=read(path);changed=False;images={}
        for case in report['cases']:
            z=np.load(case['prediction']);assert z['prediction_raw'].shape==(138,3)
            assert np.isfinite(z['prediction_raw']).all()
            raw_shapes.append(z['prediction_raw'].shape)
            if name.startswith('challenge_'):
                context=neighbor_audit[('val',case['loader_row'])]
                # Frozen selection/crops remain unchanged; this field was inadvertently inherited from parent.
                case['parent_neighbor_crop_pixels']=context['parent_neighbor_crop_pixels']
                case['neighbor_crop_pixels']=context['actual_neighbor_crop_pixels']
                case['target_network_center_pixels']=context['target_network_center_pixels']
                changed=True
            expected=Image.open(case['image']).convert('RGB');actual=Image.open(case['saved_image']).convert('RGB')
            if not np.array_equal(expected,actual):
                # One shared loader now spans many scenes: an early image-copy cache was too coarse.
                # Predictions used the correct absolute image paths throughout. Preserve original metadata.
                old_image=case['saved_image'];dest=folder/'correct_images'/f'{Path(case["scene"]).name}_{case["camera"]}.png'
                dest.parent.mkdir(exist_ok=True)
                if not dest.exists():shutil.copyfile(case['image'],dest)
                case['saved_image']=str(dest);changed=True
                corrections.append(dict(case=case['id'],batch=name,old_saved_image=old_image,correct_saved_image=str(dest),prediction_unchanged=True))
            images[case['image']]=True
        if changed:
            backup=folder/'report_before_image_copy_metadata_fix.json'
            if backup.exists():backup=folder/'report_before_float_instance_context_fix.json'
            if backup.exists():raise FileExistsError(backup)
            shutil.copyfile(path,backup)
            if name.startswith('challenge_'):
                from tools.validate.evaluate_candidate_baseline import aggregate
                from tools.validate.occlusion_metrics import evaluation_masks
                states={}
                for case in report['cases']:
                    z=np.load(case['prediction']);masks=evaluation_masks(z['in_view'],z['visible'],z['target_crop_xy'])
                    masks['visible_supervised']=masks['supervised']&z['visible']
                    states[case['id']]=dict(crop=z['crop_error_px'],erp=z['erp_periodic_error_px'],masks=masks)
                report['by_batch']['challenge']=aggregate(report['cases'],states)
                report['combined']=aggregate(report['cases'],states)
            write(path,report)
        counts[name]=dict(rows=len(report['cases']),unique_images=len(images),nonfinite_predictions=0)
    write(root/'final_audit_v2.json',dict(status='passed',training_updates=400,parent_targets=198,
        exposure_variants=dict(Counter(r['box_variant'] for r in rows)),target_order_identical_to_B=True,
        original_indices_split_checkpoint_unchanged=True,validations=validations,reload_max_abs=run['reload_max_abs_error'],
        predictions=counts,prediction_records=len(raw_shapes),image_copy_metadata_corrections=corrections,
        qualification='Image copy cache and inherited parent-neighbor metadata corrected; actual crop counts saved separately, raw predictions/GT/input crops/selection not changed; before-fix reports retained'))
    print(counts)


if __name__=='__main__':main()
