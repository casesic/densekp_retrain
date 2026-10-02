"""Freeze only reviewed, numerically valid new rows; never use model errors."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import cv2
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'assets_external/python_deps')]
from densekp_retrain.io import read_combined_depth_index
from densekp_retrain.camera.double_sphere import DS_20260320, build_erp_to_fisheye_lut, build_fisheye_to_source_erp_lut
from densekp_retrain.imaging.remap import remap_periodic_erp
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.configuration import audit_cfg
from densekp_retrain.data.source_split import validate_split


def write(path,data):path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def freeze_layout(directory, side, character_map, accepted):
    validation=json.loads((directory/'validation.json').read_text())
    if validation['status']!='passed':raise ValueError('Numerical scene gate failed')
    if json.loads((directory/'source/ground_contact.json').read_text())['status']!='passed':raise ValueError('Ground failed')
    if json.loads((directory/'groom_validation.json').read_text())['status']!='world_transform_and_instance_assignment_passed':raise ValueError('Groom gate failed')
    report=json.loads((directory/'report.json').read_text());people={p['id']:p for p in report['people']}
    instances={};boxes_by_camera={};missing=[];buffer_checks=[]
    for camera in ['cam2','cam4']:
        _,depth,instance,_=read_combined_depth_index(directory/'source'/f'{camera}.exr')
        h,w=depth.shape;mask=np.array(Image.open(directory/camera/'valid_mask.png'))>0
        lens=np.array(Image.open(directory/camera/'reference_lens_mask.png'))>0
        x,y=build_fisheye_to_source_erp_lut(DS_20260320[camera],camera,w,h)
        bx,by=build_erp_to_fisheye_lut(DS_20260320[camera],camera,w,h)
        fd=remap_periodic_erp(depth,x,y,cv2.INTER_NEAREST);fd[~lens]=0
        fi=remap_periodic_erp(np.rint(instance).astype(np.float32),x,y,cv2.INTER_NEAREST);fi[~lens]=0
        actual_depth=cv2.remap(fd,bx,by,cv2.INTER_NEAREST);actual_depth[~mask]=np.nan
        ids=cv2.remap(fi,bx,by,cv2.INTER_NEAREST).astype(np.int16);ids[~mask]=0
        foreground=ids>0
        if not np.isfinite(actual_depth[foreground]).all() or (actual_depth[foreground]<=0).any():raise ValueError('Bad depth')
        np.savez(directory/camera/'rendered_buffers.npz',depth_range_m=actual_depth,instance_id=ids,valid_mask=mask)
        instances[camera]=ids;boxes=json.loads((directory/camera/'boxes.json').read_text());boxes_by_camera[camera]={}
        for box in boxes:
            pid=box['person_id'];selected=ids==pid
            np.testing.assert_array_equal(selected,np.array(Image.open(directory/camera/f'person_{pid}_mask.png'))>0)
            yy,xx=np.where(selected);bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None
            if bbox!=box['bbox_xyxy_exclusive']:raise ValueError('Instance bbox mismatch')
            boxes_by_camera[camera][pid]=bbox
            if bbox is None:missing.append(dict(scene=str(directory),camera=camera,person_id=pid,reason='no valid rendered instance box'))
        buffer_checks.append(dict(camera=camera,foreground_pixels=int(foreground.sum()),mask_instance_box_exact=True,
            depth_range_m=[float(actual_depth[foreground].min()),float(actual_depth[foreground].max())]))
    z=np.load(directory/'diagnostic_loader.npz')
    data = DatasetKeypoints(
        audit_cfg("supervision_weight"),
        "source-freeze",
        is_train=False,
        use_augmentation=False,
        annotation_path=str(directory / "diagnostic_loader.npz"),
        image_root=str(directory),
    )
    rows=[];excluded=[]
    rejected={r['key']:r['reason'] for r in accepted.get('excluded_rows',[])}
    actual_keys={f'{cam}_person_{int(pid)}' for cam,pid in zip(z['camera_id'],z['person_id'])}
    if set(rejected)-actual_keys:raise ValueError('Review exclusion does not identify an actual loader row')
    for i,(pid,camera) in enumerate(zip(z['person_id'],z['camera_id'])):
        pid=int(pid);camera=str(camera);person=people[pid];c=character_map[person['character_id']]
        item=data[i];supervised=item['proj_verts_cropped'][:,2]>0;visible=z['visible'][i]
        ids=cv2.warpAffine(instances[camera].astype(np.float32),item['_trans'],(256,256),flags=cv2.INTER_NEAREST)
        name=f'{camera}_person_{pid}';rgb=cv2.cvtColor(item['img_disp'],cv2.COLOR_BGR2RGB)
        np.testing.assert_array_equal(rgb,np.array(Image.open(directory/'actual_inputs'/f'{name}_square.png')))
        np.testing.assert_array_equal(item['img'][:,:,32:-32],np.load(directory/'actual_inputs'/f'{name}_input.npy'))
        row=dict(scene=str(directory),camera=camera,person_id=pid,character_id=person['character_id'],
            identity_family_id=c['identity_family_id'],face_source_group=c['face_source_group'],shape_source_group=c['shape_source_group'],
            source_group=person['source_group'],pose_source_id=person['pose_source_id'],action=person['action'],category=person['category'],
            loader_row=i,loader_input=str(directory/'diagnostic_loader.npz'),image=str(directory/camera/'normal_erp.png'),
            square=str(directory/'actual_inputs'/f'{name}_square.png'),network_center=str(directory/'actual_inputs'/f'{name}_center.png'),
            given_bbox_xyxy=boxes_by_camera[camera][pid],affine_flat=item['_trans'].ravel().tolist(),
            supervised_points=int(supervised.sum()),supervised_occluded_points=int((supervised&~visible).sum()),
            in_view_points=int(z['in_view'][i].sum()),
            lens_truncated_gt=bool(not z['in_view'][i].all()),
            in_view_outside_full_crop=int((z['in_view'][i]&~supervised).sum()),
            neighbor_crop_pixels=int(((ids>0)&(ids!=pid)).sum()),target_crop_pixels=int((ids==pid).sum()),
            quality_state='candidate_accepted',split=side)
        if name in rejected:
            excluded.append(dict(row,quality_state='excluded_before_inference',reason=rejected[name]))
        elif not supervised.any():
            excluded.append(dict(row,quality_state='excluded_before_inference',
                reason='Given instance box has no valid in-view GT inside the full crop; no training/evaluation supervision.'))
        else:rows.append(row)
    return rows,excluded,missing,dict(scene=str(directory),geometry_loader=validation,buffers=buffer_checks)


def main():
    root=Path(sys.argv[1]).resolve();split=json.loads((root/'split_v2.json').read_text());validate_split(split)
    review=json.loads((root/'visual_acceptance.json').read_text())
    if review['status']!='review_complete':raise ValueError('Explicit actual-image review required')
    chars={c['character_id']:c for c in split['characters']};indices={};checks=[]
    for side in ['train','val']:
        path=root/f'{side}_index.json'
        if path.exists():raise FileExistsError('Frozen index cannot be overwritten')
        rows=[];excluded=[];missing=[];failed=[]
        for layout in review['layouts']:
            if layout['partition']!=side:continue
            if layout.get('use')=='paired_diagnostic_only':continue
            directory=Path(layout['directory'])
            if layout['status']!='accepted':failed.append(layout);continue
            r,e,m,c=freeze_layout(directory,side,chars,layout)
            rows.extend(r);excluded.extend(e);missing.extend(m);checks.append(c)
        indices[side]=dict(split=side,rows=rows,excluded_rows=excluded,no_box_views=missing,excluded_layouts=failed,
            frozen_utc=datetime.now(timezone.utc).isoformat(),split_manifest=str(root/'split_v2.json'),
            selection='actual geometry/loader and human image review before prediction; never model-error based',
            scope='initial identity/pose-source holdout, not independent test or broad generalization')
    validation=validate_split(split,indices)
    for side,index in indices.items():
        write(root/f'{side}_index.json',index)
        for scene in sorted({row['scene'] for row in index['rows']}):
            write(Path(scene)/'frozen_candidate_assignment.json',dict(
                partition=side,index=str(root/f'{side}_index.json'),split_manifest=str(root/'split_v2.json'),
                accepted_rows=[dict(camera=r['camera'],person_id=r['person_id'],loader_row=r['loader_row'])
                               for r in index['rows'] if r['scene']==scene],
                meaning='This frozen assignment supersedes the low-level renderer diagnostic scope; only indexed rows are candidates.'))
    write(root/'data_acceptance.json',dict(status='passed',source_isolation=validation,checks=checks,
        counts={s:dict(rows=len(i['rows']),supervised=sum(r['supervised_points'] for r in i['rows']),
            occluded=sum(r['supervised_occluded_points'] for r in i['rows'])) for s,i in indices.items()}))
    print(validation)

if __name__=='__main__':main()
