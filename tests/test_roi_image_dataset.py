"""Synthetic_test_only local-image/full-image coordinate compatibility, no model."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import cv2
import numpy as np
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.configuration import audit_cfg
from densekp.core.datasets.augmentation import Augmentation
from densekp.core.datasets.image_layout import parse_image_layout,LAYOUT_FIELDS
from densekp.core.keypoint_supervision import SUPERVISION_POLICY,make_targets

class ROIImageTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.image=np.random.default_rng(9).integers(0,256,(480,640,3),dtype=np.uint8)
        cv2.imwrite(str(self.root/'full.png'),self.image);cv2.imwrite(str(self.root/'roi.png'),self.image[50:430,80:560])
        xy=np.column_stack([np.linspace(240,360,138),np.linspace(180,280,138)]).astype(np.float32)
        self.data=dict(imgname=np.array([str(self.root/'full.png')]),center=np.array([[300,230]],np.float32),scale=np.array([1.],np.float32),
            proj_verts=make_targets(xy,np.ones(138,bool),[300,230],1)[None],in_view=np.ones((1,138),bool),visible=np.zeros((1,138),bool),supervision_policy=SUPERVISION_POLICY)
        self.meta=dict(image_layout=np.array(['roi_image']),image_layout_version=np.array([1]),full_image_size_wh=np.array([[640,480]]),
            roi_xyxy=np.array([[80,50,560,430]]),offset_xy=np.array([[80,50]]),stored_size_wh=np.array([[480,380]]))
    def tearDown(self):self.temp.cleanup()
    def dataset(self,*,roi=False,change=None,cfg=None,train=False):
        data=dict(self.data)
        if roi:data.update(self.meta,imgname=np.array([str(self.root/'roi.png')]))
        if change:data.update(change)
        p=self.root/('roi.npz' if roi else 'full.npz');np.savez(p,**data)
        return DatasetKeypoints(cfg or audit_cfg('supervision_weight'),'explicit',is_train=train,use_augmentation=train,annotation_path=p,image_root='/')
    def equal(self,a,b):
        for k in ['img','img_disp','proj_verts','proj_verts_cropped','visible','visibility_known','_trans','img_size','_scale','box_center','box_size']:
            np.testing.assert_array_equal(a[k],b[k],err_msg=k)
    def test_nonzero_offset_and_occluded_supervision(self):
        a=self.dataset()[0];b=self.dataset(roi=True)[0];self.equal(a,b)
        self.assertEqual(int((b['proj_verts_cropped'][:,2]>0).sum()),138)
        np.testing.assert_array_equal(b['img_size'],[480,640])
    def test_fixed_rotated_scaled_translated_plan_exact(self):
        aug=Augmentation(.95,27,False,False,0,[.9,1.,1.1],.12,-.08)
        with patch('densekp.core.datasets.crop_pipeline.sample_augmentation',return_value=aug):
            self.equal(self.dataset(train=True)[0],self.dataset(roi=True,train=True)[0])
    def test_actual_augmentation_outside_rejected(self):
        aug=Augmentation(2.,45,False,False,0,[1,1,1],.5,0)
        with patch('densekp.core.datasets.crop_pipeline.sample_augmentation',return_value=aug):
            with self.assertRaisesRegex(ValueError,'outside_roi_support'):self.dataset(roi=True,train=True)[0]
    def test_metadata_and_decoded_size_rejected(self):
        row={k:v[0] for k,v in self.meta.items()}
        for key in LAYOUT_FIELDS:
            bad=dict(row);bad.pop(key)
            with self.assertRaises(ValueError):parse_image_layout(bad)
        for change in [dict(offset_xy=[81,50]),dict(stored_size_wh=[481,380]),dict(roi_xyxy=[80,50,660,430]),dict(image_layout_version=2),dict(full_image_size_wh=[640.,480.])]:
            with self.assertRaises(ValueError):parse_image_layout(dict(row,**change))
        with self.assertRaisesRegex(ValueError,'stored_image_size_conflict'):
            self.dataset(roi=True,change=dict(imgname=self.data['imgname']))[0]
    def test_unsupported_prefilter_alb_and_flip(self):
        cfg=audit_cfg('supervision_weight');cfg.DATASETS.USE_SKIMAGE_ANTIALIAS=True
        with self.assertRaisesRegex(ValueError,'prefilter_not_supported'):self.dataset(roi=True,cfg=cfg)[0]
        cfg=audit_cfg('supervision_weight');cfg.DATASETS.CONFIG.USE_ALB=True
        with self.assertRaisesRegex(ValueError,'albumentations_not_supported'):self.dataset(roi=True,cfg=cfg,train=True)[0]
        cfg=audit_cfg('supervision_weight');cfg.DATASETS.CONFIG.DO_FLIP=True;cfg.DATASETS.CONFIG.FLIP_AUG_RATE=1
        with self.assertRaisesRegex(ValueError,'verified 138-point'):self.dataset(roi=True,cfg=cfg,train=True)
    def test_explicit_full_and_legacy_same(self):
        meta=dict(image_layout=np.array(['full_image']),image_layout_version=np.array([1]),full_image_size_wh=np.array([[640,480]]),roi_xyxy=np.array([[0,0,640,480]]),offset_xy=np.array([[0,0]]),stored_size_wh=np.array([[640,480]]))
        self.equal(self.dataset()[0],self.dataset(change=meta)[0])
    def test_pack_retains_layout_for_shared_image_boxes(self):
        from densekp_retrain.data.packing import pack_derivative_rows,pack_index_rows
        data={key:value[0] for key,value in self.meta.items()}
        rows=[dict(imgname='shared.png',center=[300,230],**data),dict(imgname='shared.png',center=[310,230],**data)]
        packed=pack_derivative_rows(rows)
        np.testing.assert_array_equal(packed['offset_xy'],[[80,50],[80,50]])
        with self.assertRaisesRegex(ValueError,'inconsistent_row_fields'):
            pack_derivative_rows([dict(imgname='shared.png',center=[300,230]),rows[1]])
        arrays=dict(self.data,**self.meta,person_id=[1],camera_id=['cam2'])
        path=self.root/'index_input.npz';np.savez(path,**arrays)
        result=pack_index_rows(dict(rows=[dict(loader_input=str(path),loader_row=0,image='shared.png')]*2))
        for key in LAYOUT_FIELDS:np.testing.assert_array_equal(result[key],np.repeat(self.meta[key],2,axis=0))

    def test_sequence_logical_and_stored_sizes_and_packed_conflict(self):
        import json
        from PIL import Image
        from tests.test_sequence_index import sequence_fixture,write_fixture_files
        from densekp_retrain.data.sequence_index import validate_sequence_index,SequenceIndexError
        from densekp_retrain.data.raster_storage import roi_metadata
        ep,table=sequence_fixture();write_fixture_files(self.root,ep,table)
        full_valid=np.asarray(Image.open(self.root/'valid.png'));Image.fromarray(full_valid).save(self.root/'physical.png')
        for name in ['image.png','valid.png','person.png']:
            pixels=np.asarray(Image.open(self.root/name));Image.fromarray(pixels[2:58,5:65]).save(self.root/name)
        meta=roi_metadata([5,2,65,58],[80,60],str(self.root/'physical.png'))
        (self.root/'storage.json').write_text(json.dumps(meta))
        (self.root/'coverage.json').write_text(json.dumps(dict(high_quality_region_xyxy=meta['roi_xyxy'])))
        for obs in table['observations']:
            obs.update({key:meta[key] for key in LAYOUT_FIELDS},raster_storage_ref='storage.json',physical_mask_ref=str(self.root/'physical.png'),render_coverage_ref='coverage.json')
        with np.load(self.root/'packed.npz') as z:arrays=dict(z)
        arrays.update({key:np.array([meta[key]]*6) for key in LAYOUT_FIELDS});np.savez(self.root/'packed.npz',**arrays)
        result=validate_sequence_index(ep,table,base_dir=self.root,check_image_headers=True,check_packed=True)
        self.assertEqual(result['packed_rows'],6)
        arrays['offset_xy'][1,0]+=1;np.savez(self.root/'packed.npz',**arrays)
        with self.assertRaises(SequenceIndexError):validate_sequence_index(ep,table,base_dir=self.root,check_packed=True)

    def test_saved_roi_native_box_offset_and_json_types(self):
        import json
        from densekp_retrain.data.raster_storage import roi_metadata,save_roi_observation
        dest=self.root/'saved';dest.mkdir();meta=roi_metadata([80,50,560,430],[640,480],'physical.png')
        rgb=self.image[50:430,80:560];valid=np.ones(rgb.shape[:2],bool);instance=np.zeros(valid.shape,np.float32);instance[40:200,50:200]=1
        save_roi_observation(dest,rgb,instance,dict(person_ids=np.array([1]),xy=np.tile([300,230],(1,138,1))),valid,meta)
        box=json.loads((dest/'boxes.json').read_text())[0]['bbox_xyxy_exclusive']
        self.assertEqual(box,[130,90,280,250])
        self.assertEqual(json.loads((dest/'render_coverage.json').read_text())['full_image_size_wh'],[640,480])

    def test_legal_guard_boundary(self):
        # Square [81,51..279,249], exactly one source pixel inside ROI edges.
        change=dict(center=np.array([[180,150]],np.float32),scale=np.array([.99],np.float32))
        self.equal(self.dataset(change=change)[0],self.dataset(roi=True,change=change)[0])

if __name__=='__main__':unittest.main()
