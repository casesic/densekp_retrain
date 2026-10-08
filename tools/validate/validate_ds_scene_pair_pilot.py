#!/usr/bin/env python3
"""Validate a rendered DS scene-pair pilot and exercise the actual DenseKP loader.

Writes diagnostic-only NPZ and crop review; oracle instance boxes, no detector.
No train/validation split is assigned to these few repeated-identity samples.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
import cv2
import numpy as np
from PIL import Image, ImageDraw
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, make_targets
from densekp.core.datasets.configuration import audit_cfg
from densekp_retrain.body.transforms import world_points
from densekp_retrain.camera.domain import point_in_view


def validate(directory, *, source_dir=None, output_mode="full"):
    if output_mode not in ("compact", "full"):
        raise ValueError(f"Invalid output mode: {output_mode}")
    source_dir = directory / "source" if source_dir is None else Path(source_dir)
    report = json.loads((directory / 'report.json').read_text())
    source = json.loads((source_dir / 'source_manifest.json').read_text())
    rig = json.loads((directory / 'rig.json').read_text())
    if source['people'] != report['people']:
        raise ValueError('People differ between render and annotations')
    cameras = {row['id']: row for row in rig['cameras']}
    physical = [np.asarray(cameras[n]['physical_ds_camera_to_world']) for n in ('cam2', 'cam4')]
    np.testing.assert_allclose(physical[0][:3, 2], -physical[1][:3, 2], atol=1e-12)
    np.testing.assert_allclose(physical[0][2, 3], physical[1][2, 3], atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(physical[0][:3, 3]-physical[1][:3, 3]), rig['baseline_m'], atol=1e-12)
    arrays, records = [], []
    projection_max_error = 0.
    coordinate_tolerance = 2e-4
    for name in ('cam2','cam4'):
        with np.load(directory / name / 'annotations.npz') as archive:
            data = {key:archive[key] for key in archive.files}
        arrays.append(data)
        mask=np.asarray(Image.open(directory/name/'valid_mask.png'))>0
        rgb=np.asarray(Image.open(directory/name/'normal_erp.png'))
        coordinate_tolerance=max(coordinate_tolerance,rgb.shape[1]*np.finfo(np.float32).eps)
        if rgb.shape[:2]!=mask.shape:
            raise ValueError('Normal ERP RGB and explicit mask raster sizes differ')
        if np.any(rgb[~mask]):
            raise ValueError('Nonzero pixels outside the explicit valid mask')
        if output_mode == 'full':
            lens=np.asarray(Image.open(directory/name/'reference_lens_mask.png'))>0
            fish=np.asarray(Image.open(directory/name/'fisheye.png'))
            if fish.shape[:2]!=lens.shape:
                raise ValueError('Fisheye RGB and lens mask raster sizes differ')
            if np.any(fish[~lens]):
                raise ValueError('Nonzero fisheye pixels outside explicit lens mask')
        projected,_,_,_,valid=point_in_view(data['world_densekp'].reshape(-1,3),
                                          np.asarray(cameras[name]['camera_to_world']),name,mask)
        projection_error=float(np.max(np.abs(projected.reshape(data['xy'].shape)-data['xy'])))
        projection_max_error=max(projection_max_error,projection_error)
        np.testing.assert_allclose(projected.reshape(data['xy'].shape),data['xy'],atol=coordinate_tolerance,rtol=0)
        np.testing.assert_array_equal(valid.reshape(data['in_view'].shape),data['in_view'])
        np.testing.assert_array_equal(data['visible'], data['surface_visible'] & data['in_view'])
        if np.any(data['occluded_in_view'] & data['out_of_view']):
            raise ValueError('Occluded-in-view and out-of-view categories overlap')
        for row in json.loads((directory / name / 'boxes.json').read_text()):
            if row['bbox_xyxy_exclusive'] is None: continue
            person_index = data['person_ids'].tolist().index(row['person_id'])
            x1,y1,x2,y2 = row['bbox_xyxy_exclusive']
            size = max(x2-x1,y2-y1)*1.2
            center = [(x1+x2)/2, (y1+y2)/2]
            xy = data['xy'][person_index]
            in_view = data['in_view'][person_index]
            records.append(dict(camera=name, person_id=row['person_id'], imgname=f'{name}/normal_erp.png',
                                center=center, scale=size/200,
                                in_view=in_view, visible=data['visible'][person_index],
                                target=make_targets(xy, in_view, center, size/200)))
    np.testing.assert_array_equal(arrays[0]['world_densekp'], arrays[1]['world_densekp'])
    with np.load(directory / 'pair_annotations.npz') as pair:
        np.testing.assert_array_equal(pair['shared_visible'], arrays[0]['visible'] & arrays[1]['visible'])
        np.testing.assert_array_equal(pair['world_densekp'], arrays[0]['world_densekp'])
    # Preserve a diagnostic adapter input; never present it as an independent split.
    annotation_path = directory / 'diagnostic_loader.npz'
    np.savez(annotation_path, imgname=np.array([r['imgname'] for r in records]),
             center=np.array([r['center'] for r in records], dtype=np.float32),
             scale=np.array([r['scale'] for r in records], dtype=np.float32),
             proj_verts=np.stack([r['target'] for r in records]),
             supervision_policy=np.array(SUPERVISION_POLICY),
             in_view=np.stack([r['in_view'] for r in records]),
             visible=np.stack([r['visible'] for r in records]),
             split=np.array(report.get('split','diagnostic_only')),
             person_id=np.array([r['person_id'] for r in records]),
             camera_id=np.array([r['camera'] for r in records]))
    dataset_name = 'scene-pair-diagnostic'
    cfg = audit_cfg('supervision_weight')
    crop_panels = {}
    max_error = 0.
    dataset = DatasetKeypoints(
        cfg,
        dataset_name,
        is_train=False,
        use_augmentation=False,
        annotation_path=str(annotation_path),
        image_root=str(directory),
    )
    for i, row in enumerate(records):
        item = dataset[i]
        restored = restore_crop_xy(item['proj_verts_cropped'][:, :2], item['_trans'], (256, 256))
        error = float(np.max(np.abs(restored - item['proj_verts'][:, :2])))
        max_error = max(max_error, error)
        if error > coordinate_tolerance:
            raise ValueError(f'Crop inverse error {error}')
        if item['img'].shape != (3, 256, 256) or not np.isfinite(item['img']).all():
            raise ValueError('Invalid actual loader model input')
        np.testing.assert_array_equal(item['proj_verts_cropped'][:, 2], row['target'][:, 2])
        panel = Image.fromarray(cv2.cvtColor(item['img_disp'], cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(panel)
        draw.rectangle((0, 0, 256, 19), fill=(0, 0, 0))
        draw.text((5, 4), f"{row['camera']} / person {row['person_id']} / oracle crop", fill='white')
        points = (item['proj_verts_cropped'][:, :2] + 0.5) * 256
        for j, (x, y) in enumerate(points):
            if 0 <= x < 256 and 20 <= y < 256:
                color = (0, 255, 100) if row['visible'][j] else (255, 170, 40)
                if row['target'][j, 2] == 0:
                    color = (170, 170, 170)
                draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=color)
        crop_panels[row['person_id'], row['camera']] = panel
    ids=arrays[0]['person_ids'].tolist()
    if output_mode == 'full':
        sheet=Image.new('RGB',(512,256*len(ids)),(25,25,25))
        for i, person_id in enumerate(ids):
            for j,name in enumerate(('cam2','cam4')):
                if (person_id,name) in crop_panels:
                    sheet.paste(crop_panels[(person_id,name)],(256*j,256*i))
                else:
                    ImageDraw.Draw(sheet).text((256*j+8,256*i+8),f'{name} / person {person_id}: no box',fill='white')
        sheet.save(directory/'crop_review.jpg')
    # Confirm all referenced 138-row model points against the repository operator.
    regressor = np.load(ROOT/'assets/densekp_regressor_R.npy')
    geom=ROOT/'outputs/baselines/temporal_case_aaron_1095/geometry'
    body_bounds=[]; operator_errors=[]; rendered_errors=[]; clothing_errors=[]
    for person_index, person in enumerate(report['people']):
        person_geometry=Path(person.get('geometry',geom))
        vertices=np.load(person_geometry/'vertices.npy',mmap_mode='r')
        dense=np.load(person_geometry/'densekp_3d.npy',mmap_mode='r')
        body=world_points(vertices[person['frame']],vertices[person['frame']],person)
        bounds_body=body
        if 'clothing_geometry' in person:
            clothing_dir=Path(person['clothing_geometry'])
            metadata=json.loads((clothing_dir/'clothing_manifest.json').read_text())
            if Path(metadata['body_geometry']).resolve()!=person_geometry.resolve():
                raise ValueError('Clothing provenance differs from the target body')
            clothing=np.load(clothing_dir/'clothing_vertices.npy')[person['frame']]
            expected_clothing=world_points(clothing,vertices[person['frame']],person)
            rendered_clothing=np.load(source_dir/f"person_{person['id']}_world_clothing.npy")
            error=float(np.abs(expected_clothing-rendered_clothing).max())
            if error>5e-6: raise ValueError('Clothing must use the body lift, yaw and location')
            clothing_errors.append(error)
            bounds_body=np.concatenate((body,rendered_clothing))
        body_bounds.append((bounds_body.min(0),bounds_body.max(0)))
        operator_errors.append(float(np.abs(regressor@vertices[person['frame']]-dense[person['frame']]).max()))
        expected=world_points(dense[person['frame']],vertices[person['frame']],person)
        np.testing.assert_allclose(expected,arrays[0]['world_densekp'][person_index],atol=2e-6,rtol=0)
        rendered_path=source_dir/f"person_{person['id']}_world_vertices.npy"
        if 'geometry' in person or rendered_path.exists():
            metadata=json.loads(rendered_path.with_name(f"person_{person['id']}_geometry.json").read_text())
            if Path(metadata['geometry']).resolve()!=person_geometry.resolve() or metadata['frame']!=person['frame']:
                raise ValueError('Rendered geometry provenance differs from labels')
            rendered_body=np.load(rendered_path)
            np.testing.assert_allclose(rendered_body,body,atol=5e-6,rtol=0)
            discrepancy=float(np.abs(regressor@rendered_body-expected).max())
            if discrepancy>5e-6: raise ValueError('Rendered mesh and 138 world points disagree')
            rendered_errors.append(discrepancy)
    for i,(low,high) in enumerate(body_bounds):
        for other_low,other_high in body_bounds[i+1:]:
            if not ((high+.05<other_low)|(other_high+.05<low)).any():
                raise ValueError('Body AABB overlap or insufficient clearance')
    operator_error=max(operator_errors)
    if operator_error>1e-6: raise ValueError('DenseKP operator mismatch')
    result=dict(scene=report['scene'], status='passed', people=len(ids), loader_rows=len(records),
                same_world_densekp=True, physical_axes_opposite=True, equal_height=True,
                body_aabbs_separated_with_5cm_margin=True,
                baseline_m=rig['baseline_m'], baseline_status='synthetic value; not a measured baseline',
                crop_roundtrip_max_px=max_error, operator_max_error_m=operator_error,
                projection_roundtrip_max_px=projection_max_error,
                coordinate_tolerance_px=coordinate_tolerance,
                rendered_mesh_operator_max_error_m=max(rendered_errors) if rendered_errors else None,
                rendered_clothing_transform_max_error_m=max(clothing_errors) if clothing_errors else None,
                target_value_semantics='explicit binary supervision mask, not prediction confidence',
                supervision_policy=SUPERVISION_POLICY,
                supervised_points=sum(int((r['target'][:,2]>0).sum()) for r in records),
                supervised_occluded_points=sum(int(((r['target'][:,2]>0) & ~r['visible']).sum()) for r in records),
                annotation_scope='diagnostic only; no independent train/validation split',
                boxes='oracle masks; no detector was run', shared_visible=[p['shared_visible'] for p in report['pairs']])
    (directory/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('pilot_root',type=Path)
    args=parser.parse_args()
    for name in ('cycling_forest','urban_street_assets'):
        print(json.dumps(validate(args.pilot_root/name),indent=2))


if __name__=='__main__': main()
