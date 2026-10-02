"""Audit actual buffers and freeze visually reviewed rows before any inference."""
import json
from datetime import datetime, timezone
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
from densekp_retrain.io.json import write_json
from densekp_retrain.body.parameters import PARAMETERS
from tools.experiments.audit_pedestrian_poses import SOURCES, parameters
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.configuration import audit_cfg


def main():
    root=Path(sys.argv[1]).resolve();review=json.loads((root/'visual_acceptance.json').read_text())
    if review['status']!='passed_for_candidate_use':raise ValueError('Explicit visual acceptance required')
    if (root/'candidate_index.json').exists():raise ValueError('Do not overwrite frozen selection')
    for name in ('body_clothing_readback','groom_binding_readback'):
        if json.loads((root/f'{name}.json').read_text())['status']!='passed':raise ValueError(name)
    report=json.loads((root/'report.json').read_text());source=np.load(SOURCES[1]);rows=[];missing=[];checks=[]
    for d in report['derivatives']:
        if d['action']=='cycling':continue
        saved=np.load(Path(d['geometry'])/'motion_params.npz')
        original=parameters(source['poses'][d['source_frame']],saved['betas'],np.zeros(3))
        for key in PARAMETERS:
            if key not in ('betas','transl'):np.testing.assert_array_equal(saved[key],original[key])
    for layout in review['layouts']:
        directory=root/layout['directory']
        if layout['status']!='accepted':raise ValueError('Handle rejected layouts explicitly before freezing')
        validation=json.loads((directory/'validation.json').read_text())
        if validation['status']!='passed':raise ValueError('DS numeric gate failed')
        if json.loads((directory/'source/ground_contact.json').read_text())['status']!='passed':raise ValueError('Ground check failed')
        if json.loads((directory/'groom_validation.json').read_text())['status']!='world_transform_and_instance_assignment_passed':raise ValueError('Hair transform failed')
        generated=json.loads((directory/'report.json').read_text());people={p['id']:p for p in generated['people']}
        instances={};camera_checks=[]
        for camera in ('cam2','cam4'):
            _,depth,instance,_=read_combined_depth_index(directory/'source'/f'{camera}.exr')
            h,w=depth.shape;mask=np.array(Image.open(directory/camera/'valid_mask.png'))>0
            lens=np.array(Image.open(directory/camera/'reference_lens_mask.png'))>0
            x,y=build_fisheye_to_source_erp_lut(DS_20260320[camera],camera,w,h)
            bx,by=build_erp_to_fisheye_lut(DS_20260320[camera],camera,w,h)
            fish_depth=remap_periodic_erp(depth,x,y,cv2.INTER_NEAREST);fish_depth[~lens]=0
            fish_instance=remap_periodic_erp(np.rint(instance).astype(np.float32),x,y,cv2.INTER_NEAREST);fish_instance[~lens]=0
            final_depth=cv2.remap(fish_depth,bx,by,cv2.INTER_NEAREST);final_depth[~mask]=np.nan
            final_instance=cv2.remap(fish_instance,bx,by,cv2.INTER_NEAREST).astype(np.int16);final_instance[~mask]=0
            instances[camera]=final_instance
            foreground=final_instance>0
            if not np.isfinite(final_depth[foreground]).all() or (final_depth[foreground]<=0).any():raise ValueError('Invalid foreground depth')
            np.savez(directory/camera/'rendered_buffers.npz',depth_range_m=final_depth,instance_id=final_instance,valid_mask=mask)
            boxes=json.loads((directory/camera/'boxes.json').read_text())
            for box in boxes:
                pid=box['person_id'];selected=final_instance==pid
                np.testing.assert_array_equal(selected,np.array(Image.open(directory/camera/f'person_{pid}_mask.png'))>0)
                yy,xx=np.where(selected);bbox=[int(xx.min()),int(yy.min()),int(xx.max())+1,int(yy.max())+1] if len(xx) else None
                if bbox!=box['bbox_xyxy_exclusive']:raise ValueError('Actual instance bbox differs')
                if bbox is None:missing.append(dict(scene=str(directory),camera=camera,person_id=pid,character_id=people[pid]['character_id'],
                    reason='no rendered valid instance pixels; no oracle box, not a detector failure'))
            camera_checks.append(dict(camera=camera,person_pixels=int(foreground.sum()),
                person_depth_range_m=[float(final_depth[foreground].min()),float(final_depth[foreground].max())],
                instance_mask_and_bbox_exact=True,depth_units='camera ray range metres, nearest-neighbor DS roundtrip',
                invalid_depth='NaN outside explicit mask; background sky depth is renderer sentinel'))
        z=np.load(directory/'diagnostic_loader.npz')
        dataset = DatasetKeypoints(
            audit_cfg("supervision_weight"),
            "mixed-freeze",
            is_train=False,
            use_augmentation=False,
            annotation_path=str(directory / "diagnostic_loader.npz"),
            image_root=str(directory),
        )
        for i,(pid,camera) in enumerate(zip(z['person_id'],z['camera_id'])):
            pid=int(pid);camera=str(camera);person=people[pid];item=dataset[i]
            supervise=item['proj_verts_cropped'][:,2]>0;visible=z['visible'][i]
            cropped_instance=cv2.warpAffine(instances[camera].astype(np.float32),item['_trans'],(256,256),flags=cv2.INTER_NEAREST)
            neighbor=(cropped_instance>0)&(cropped_instance!=pid)
            row=dict(scene=str(directory),camera=camera,person_id=pid,character_id=person['character_id'],
                pose_source_id=person['pose_source_id'],action=person['action'],category=person['category'],
                loader_row=i,loader_input=str(directory/'diagnostic_loader.npz'),image=str(directory/camera/'normal_erp.png'),
                square=str(directory/'actual_inputs'/f'{camera}_person_{pid}_square.png'),
                network_center=str(directory/'actual_inputs'/f'{camera}_person_{pid}_center.png'),
                supervised_points=int(supervise.sum()),supervised_occluded_points=int((supervise&~visible).sum()),
                in_view_outside_full_crop=int((z['in_view'][i]&~supervise).sum()),neighbor_crop_pixels=int(neighbor.sum()),
                target_crop_pixels=int((cropped_instance==pid).sum()),quality_state='candidate_accepted',split='integration_only')
            rows.append(row)
        checks.append(dict(directory=str(directory),geometry_loader=validation,buffers=camera_checks))
    counts=dict(characters=len({r['character_id'] for r in rows}),new_pedestrian_poses=len({r['pose_source_id'] for r in rows if r['action']!='cycling'}),
        layouts=len(checks),erp_images=2*len(checks),candidate_boxes=len(rows),no_box_views=len(missing),
        supervised_points=sum(r['supervised_points'] for r in rows),occluded_supervised_points=sum(r['supervised_occluded_points'] for r in rows),
        in_view_outside_full_crop=sum(r['in_view_outside_full_crop'] for r in rows),
        rows_with_neighbor_pixels=sum(r['neighbor_crop_pixels']>0 for r in rows))
    write_json(root/'candidate_index.json',dict(rows=rows,excluded_rows=[],no_box_views=missing,split='integration_only',
        selection='explicit visual and numerical acceptance before DenseKP inference; no predictions used',
        frozen_utc=datetime.now(timezone.utc).isoformat(),not_a_train_val_split=True))
    write_json(root/'acceptance.json',dict(status='passed_for_bounded_candidate_use',checks=checks,counts=counts,
        visual_review=str(root/'visual_acceptance.json'),pose_source_selection=str(root/'pose_selection.json'),
        limitations=['one standing pose instead of requested two','all new pedestrian poses share one KIT source clip',
        'sampled contacts, not zero intersections','finite-view visual acceptance; low-sample rendering noise remains'],split='integration_only'))
    print(json.dumps(counts,indent=2))


if __name__=='__main__':main()
