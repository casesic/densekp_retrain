"""Limited CPU readback of shared native parents with box augmentation off/on.

Observation IDs are frozen by the caller. Missing boxes are recorded, never
replaced with a more convenient row. No model, renderer or optimizer is used.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from yacs.config import CfgNode

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp_retrain.data.box_augmentation import DEFAULT_POLICY, sample_box_variant
from tests.dataset_crop_fixture import audit_cfg


def audit(pack, observation_ids, output, seed):
    pack=Path(pack).resolve();output=Path(output)
    index=json.loads((pack/'sequence_index.json').read_text())
    observations={o['observation_id']:o for o in index['observations']}
    rows={b['observation_id']:b['packed_row_id'] for b in index['boxes']}
    with np.load(pack/'packed.npz') as saved:
        arrays={key:saved[key] for key in saved.files}
    support=arrays['box_augmentation_family_supported'].astype(bool)
    off_cfg=audit_cfg('supervision_weight')
    on_cfg=audit_cfg('supervision_weight')
    on_cfg.DATASETS.BOX_AUGMENTATION=CfgNode(dict(enabled=True,seed=seed,policy=DEFAULT_POLICY))
    source=dict(annotation_path=pack/'packed.npz',image_root='/')
    off=DatasetKeypoints(off_cfg,'explicit',is_train=True,use_augmentation=False,**source)
    evaluation=DatasetKeypoints(on_cfg,'explicit',is_train=False,use_augmentation=False,**source)
    if len(off)!=len(evaluation) or not np.array_equal(off.imgname,evaluation.imgname):
        raise ValueError('off_on_parent_order_changed')
    prior=json.loads((pack/'readback.json').read_text())
    native_checks={(r['frame_index'],r['camera'],r['person_id']):r
                   for r in prior['checks'] if r.get('Dataset_read')}
    enabled=None
    if bool(support.all()):
        enabled=DatasetKeypoints(on_cfg,'explicit',is_train=True,use_augmentation=False,**source)
    records=[];reads=0
    for oid in observation_ids:
        if oid not in observations:raise ValueError('unknown_frozen_observation_id:'+oid)
        obs=observations[oid]
        if oid not in rows:
            records.append(dict(observation_id=oid,status='no_native_training_row',reason=obs['reason']))
            continue
        row=rows[oid]
        if not support[row]:
            records.append(dict(observation_id=oid,status='augmentation_family_unsupported',
                                reason=str(arrays['box_augmentation_family_reason'][row]),native_row_retained=True))
            continue
        native_key=(obs['frame_index'],obs['camera_id'],obs['person_id'])
        if native_key not in native_checks or native_checks[native_key]['row']!=row:
            raise ValueError('native_Dataset_readback_not_previously_checked')
        fixed=evaluation[row];reads+=1
        np.testing.assert_array_equal(fixed['proj_verts'],off.proj_verts[row])
        np.testing.assert_array_equal(fixed['img_size'],[2880.,5760.])
        variants=[]
        for visit in (0,1):
            enabled.set_box_augmentation_visit_index(visit)
            variant=sample_box_variant(arrays['parent_bbox_xyxy'][row],policy=DEFAULT_POLICY,seed=seed,
                observation_key=(0,row,visit),roi_xyxy=arrays['roi_xyxy'][row],variant_index=0)
            item=enabled[row];reads+=1
            np.testing.assert_allclose(item['box_center'],variant['center'],atol=2e-4,rtol=0)
            np.testing.assert_allclose(item['_scale'],variant['scale'],atol=1e-6,rtol=0)
            np.testing.assert_array_equal(item['img_size'],fixed['img_size'])
            np.testing.assert_array_equal(item['proj_verts'],fixed['proj_verts'])
            replay=sample_box_variant(arrays['parent_bbox_xyxy'][row],policy=DEFAULT_POLICY,seed=seed,
                observation_key=(0,row,visit),roi_xyxy=arrays['roi_xyxy'][row],variant_index=0)
            if replay!=variant:raise ValueError('deterministic_box_variant_replay_failed')
            variants.append(dict(visit_index=visit,mode=variant['mode'],parameters=variant['parameters'],
                                 center=variant['center'],scale=variant['scale']))
        records.append(dict(observation_id=oid,status='readback_passed',packed_row=row,
                            same_parent_image_and_GT=True,fixed_evaluation_uses_native_box=True,
                            native_readback_ref=str(pack/'readback.json'),
                            native_supervised=native_checks[native_key]['supervised'],
                            variants=variants))
    result=dict(schema='first_four_box_readback_v1',pack=str(pack),requested_observations=observation_ids,
                parent_row_count=len(arrays['imgname']),whole_family_supported=int(support.sum()),
                whole_family_unsupported=int((~support).sum()),on_dataset_constructed=enabled is not None,
                earlier_native_Dataset_item_reads=prior['actual_Dataset_reads'],
                actual_additional_Dataset_item_reads=reads,records=records)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pack',type=Path,required=True)
    parser.add_argument('--observations',nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args()
    result=audit(args.pack,args.observations,args.output,args.seed)
    print(json.dumps(dict(output=str(args.output),supported=result['whole_family_supported'],
                          unsupported=result['whole_family_unsupported'],reads=result['actual_additional_Dataset_item_reads'])))


if __name__=='__main__':main()
