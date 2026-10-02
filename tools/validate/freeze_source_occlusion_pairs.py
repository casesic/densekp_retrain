"""Use each clear instance box unchanged in both images; recompute actual crop supervision."""
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from densekp.core.keypoint_supervision import in_view_targets
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.keypoint_supervision import SUPERVISION_POLICY
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.configuration import audit_cfg
from densekp_retrain.data.source_split import validate_split


def main():
    root=Path(sys.argv[1]).resolve();split=json.loads((root/'split_v2.json').read_text())
    pairs=json.loads((root/'occlusion_pairs.json').read_text())['pairs'];chars={c['character_id']:c for c in split['characters']}
    reviewed={Path(v['directory']).name:v for v in json.loads((root/'visual_acceptance.json').read_text())['layouts']}
    out=root/'paired_crops';out.mkdir(exist_ok=False);all_rows={'train':[],'val':[]};checks=[]
    for pair in pairs:
        if pair['returncode']:continue
        clear=Path(pair['clear']);occluded=Path(pair['occluded']);pid=pair['target_person_id'];side=pair['partition']
        if any(reviewed[d.name]['status']!='accepted' for d in [clear,occluded]):continue
        saved_gt={};affines={};masks={};count={}
        for condition,directory in [('occluded',occluded),('clear',clear)]:
            people=json.loads((directory/'people.json').read_text())['people'];person=next(p for p in people if p['id']==pid)
            identity=chars[person['character_id']];rig=json.loads((directory/'rig.json').read_text())
            folder=out/(pair['id']+'_'+condition);folder.mkdir();raw=[];boxes=[];cameras=[];visible=[];in_view=[]
            for camera in ['cam2','cam4']:
                box=next(b for b in json.loads((clear/camera/'boxes.json').read_text()) if b['person_id']==pid)['bbox_xyxy_exclusive']
                if box is None:continue
                z=np.load(directory/camera/'annotations.npz');j=z['person_ids'].tolist().index(pid)
                if camera in saved_gt:np.testing.assert_array_equal(z['xy'][j],saved_gt[camera])
                else:saved_gt[camera]=z['xy'][j]
                raw.append(in_view_targets(z['xy'][j], z['in_view'][j]));boxes.append(box);cameras.append(camera)
                visible.append(z['visible'][j]);in_view.append(z['in_view'][j])
            if not boxes:continue
            boxes=np.array(boxes);archive=folder/'loader.npz'
            np.savez(archive,imgname=[str(directory/c/'normal_erp.png') for c in cameras],
                center=((boxes[:,:2]+boxes[:,2:])/2).astype(np.float32),scale=(np.max(boxes[:,2:]-boxes[:,:2],axis=1)*1.2/200).astype(np.float32),
                proj_verts=np.array(raw,np.float32),in_view=in_view,visible=visible,supervision_policy=SUPERVISION_POLICY,
                camera_id=cameras,person_id=np.full(len(cameras),pid),split=side)
            dataset = DatasetKeypoints(
                audit_cfg("supervision_weight"),
                "paired-source",
                is_train=False,
                use_augmentation=False,
                annotation_path=str(archive),
                image_root=str(directory),
            )
            for i,camera in enumerate(cameras):
                item=dataset[i];supervised=item['proj_verts_cropped'][:,2]>0
                if camera in affines:
                    np.testing.assert_array_equal(item['_trans'],affines[camera]);np.testing.assert_array_equal(supervised,masks[camera])
                else:affines[camera]=item['_trans'];masks[camera]=supervised
                np.testing.assert_allclose(restore_crop_xy(item['proj_verts_cropped'][:,:2],item['_trans'],(256,256)),saved_gt[camera],atol=2e-4,rtol=0)
                rgb=cv2.cvtColor(item['img_disp'],cv2.COLOR_BGR2RGB);square=folder/f'{camera}_person_{pid}_square.png'
                center=folder/f'{camera}_person_{pid}_center.png';Image.fromarray(rgb).save(square);Image.fromarray(rgb[:,32:-32]).save(center)
                np.save(folder/f'{camera}_person_{pid}_input.npy',item['img'][:,:,32:-32])
                count[(condition,camera)]=int((supervised&~visible[i]).sum())
                matrix=json.loads((directory/'source'/f'person_{pid}_geometry.json').read_text())['object_to_world']
                cam=next(c['camera_to_world'] for c in rig['cameras'] if c['id']==camera)
                all_rows[side].append(dict(scene=str(directory),camera=camera,person_id=pid,character_id=person['character_id'],
                    identity_family_id=identity['identity_family_id'],pose_source_id=person['pose_source_id'],source_group=person['source_group'],
                    action=person['action'],category=person['category'],loader_row=i,loader_input=str(archive),
                    image=str(directory/camera/'normal_erp.png'),square=str(square),network_center=str(center),
                    supervised_points=int(supervised.sum()),supervised_occluded_points=count[(condition,camera)],
                    in_view_outside_full_crop=int((in_view[i]&~supervised).sum()),given_bbox_xyxy=boxes[i].tolist(),
                    affine_flat=item['_trans'].ravel().tolist(),target_object_to_world_flat=np.array(matrix).ravel().tolist(),
                    camera_to_world_flat=np.array(cam).ravel().tolist(),occlusion_pair_id=pair['id'],occlusion_condition=condition,
                    quality_state='diagnostic_pending_actual_paired_crop_review',split=side,
                    purpose='paired given-box diagnostic, not an extra independent training source'))
        checks.append(dict(pair=pair['id'],target_camera_gt_box_affine_supervision='equal',
            actual_occluded_supervised={f'{condition}/{camera}':value for (condition,camera),value in count.items()},
            meaning='only foreground presence changes; visible state recomputed, self/other occlusion not separated'))
    indices={s:dict(split=s,rows=rows,created_utc=datetime.now(timezone.utc).isoformat(),
                scope='separate paired diagnostic; not appended to training NPZ') for s,rows in all_rows.items()}
    isolation=validate_split(split)
    for side,index in indices.items():(root/f'paired_{side}_pending.json').write_text(json.dumps(index,indent=2)+'\n')
    (root/'paired_acceptance.json').write_text(json.dumps(dict(status='numeric_passed_visual_pending',checks=checks,source_isolation=isolation),indent=2)+'\n')

if __name__=='__main__':main()
