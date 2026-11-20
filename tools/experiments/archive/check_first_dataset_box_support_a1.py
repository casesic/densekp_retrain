"""Bounded read-only audit of old ROI packs; writes only small metadata overlays."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp_retrain.data.box_augmentation import DEFAULT_POLICY, require_full_box_family_support
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.configuration import audit_cfg
from densekp.core.keypoint_supervision import supervision_for_crop
from yacs.config import CfgNode

# Fixed before any Dataset access; at most 12 historical parents and 3 draws each.
SLOTS=[
 ('riverfront_single','second_environment_boundary_pass_a1_20261010/riverfront_complete/pack_corrected',[0,30]),
 ('riverfront_team','cycling_team_relative_layout_a1_20261010/riverfront_2/run/pack_correction1',[1,3,15,80]),
 ('city_junction','t_junction_local_rig_delivery_a1_20261011/production_retry3/rigs/rig_front/pack',[1,3,15,50]),
 ('city_single','seeded_scene_modes_a1_20261010/runs/single_cyclist_boundary_pass_s20261010_e00/pack',[0,30]),
]

def main():
    out=ROOT/'outputs/acceptance/first_dataset_preparation_a1_20261011'
    out.mkdir(parents=True,exist_ok=True)
    frozen=[dict(group=g,parent_pack=str(ROOT/'outputs/experiments'/r),rows=ids)
            for g,r,ids in SLOTS]
    (out/'frozen_readback_plan.json').write_text(json.dumps(frozen,indent=2)+'\n')
    results=[];reads=0
    for group,relative,rows in SLOTS:
        pack=ROOT/'outputs/experiments'/relative
        index=json.loads((pack/'sequence_index.json').read_text())
        boxes={b['packed_row_id']:b for b in index['boxes'] if b['packed_row_id'] is not None}
        with np.load(pack/'packed.npz',allow_pickle=False) as archive:
            data={k:archive[k] for k in archive.files}
        original_dataset=DatasetKeypoints(audit_cfg('supervision_weight'),'explicit',
            is_train=True,use_augmentation=False,annotation_path=pack/'packed.npz',image_root='/')
        count=len(data['imgname'])
        for row_id in rows:
            if not 0<=row_id<count:raise ValueError('frozen_parent_row_missing')
            parent=boxes[row_id]['bbox_xyxy'];roi=data['roi_xyxy'][row_id]
            record=dict(group=group,parent_pack=str(pack),packed_row_id=row_id,
                parent_box_id=boxes[row_id]['box_id'],parent_bbox_xyxy=parent,
                roi_xyxy=roi.tolist(),variants=[])
            one={k:(v[row_id:row_id+1] if np.asarray(v).ndim else v) for k,v in data.items()}
            one['parent_bbox_xyxy']=np.asarray([parent],np.float32)
            one['box_augmentation_reserved_policy']=np.asarray([json.dumps(DEFAULT_POLICY,sort_keys=True)])
            overlay=out/f'{group}_row{row_id:03d}.npz'
            np.savez(overlay,**one)
            off_cfg=audit_cfg('supervision_weight')
            off=DatasetKeypoints(off_cfg,'explicit',is_train=True,use_augmentation=False,
                annotation_path=overlay,image_root='/')
            off_item=off[0];reads+=1
            original_item=original_dataset[row_id];reads+=1
            for key in ('img','_trans','proj_verts_cropped','img_size','box_center'):
                np.testing.assert_array_equal(off_item[key],original_item[key])
            record['native_readback']=dict(supervised=int((off_item['proj_verts_cropped'][:,2]>0).sum()),
                full_img_size=off_item['img_size'].tolist(),same_parent_row_count=len(off))
            try:
                required=require_full_box_family_support(parent,roi,data['full_image_size_wh'][row_id],DEFAULT_POLICY)
                record['family_support']='passed';record['required_xyxy']=required
            except ValueError as error:
                record['family_support']=str(error)
                results.append(record);continue
            on_cfg=audit_cfg('supervision_weight');on_cfg.defrost()
            on_cfg.DATASETS.BOX_AUGMENTATION=CfgNode(dict(enabled=True,seed=20261041,policy=DEFAULT_POLICY))
            on=DatasetKeypoints(on_cfg,'explicit',is_train=True,use_augmentation=False,
                annotation_path=overlay,image_root='/')
            if len(on)!=len(off):raise ValueError('box_toggle_changed_parent_count')
            np.testing.assert_array_equal(on.imgname,off.imgname)
            fixed_eval=DatasetKeypoints(on_cfg,'explicit',is_train=False,use_augmentation=False,
                annotation_path=overlay,image_root='/')[0];reads+=1
            for key in ('img','_trans','proj_verts_cropped'):
                np.testing.assert_array_equal(fixed_eval[key],off_item[key])
            for visit in range(3):
                on.set_box_augmentation_visit_index(visit)
                item=on[0];repeat=on[0];reads+=2
                for key in ('img','_trans','proj_verts_cropped'):
                    np.testing.assert_array_equal(item[key],repeat[key])
                xy=data['proj_verts'][row_id,:,:2]
                projected=np.column_stack((xy,np.ones(len(xy))))@item['_trans'].T
                normalized=projected/256-.5
                np.testing.assert_allclose(item['proj_verts_cropped'][:,:2],normalized,atol=2e-6)
                expected=supervision_for_crop(data['in_view'][row_id],normalized)
                np.testing.assert_array_equal(item['proj_verts_cropped'][:,2]>0,expected)
                record['variants'].append(dict(visit_index=visit,supervised=int((item['proj_verts_cropped'][:,2]>0).sum()),
                    box_center=item['box_center'].tolist(),scale=float(item['_scale']),
                    affine_finite=bool(np.isfinite(item['_trans']).all()),
                    full_img_size=item['img_size'].tolist(),replay_equal=True))
            results.append(record)
    summary=dict(schema='first_dataset_box_support_acceptance_v1',frozen_parent_count=sum(len(s[2]) for s in SLOTS),
        supported_family_count=sum(r['family_support']=='passed' for r in results),
        rejected_family_count=sum(r['family_support']!='passed' for r in results),
        actual_dataset_reads=reads,rows=results)
    (out/'box_support_readback.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('frozen_parent_count','supported_family_count','rejected_family_count','actual_dataset_reads')}))

if __name__=='__main__':main()
