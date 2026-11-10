import json
from pathlib import Path
import unittest
import tempfile
import numpy as np

from densekp_retrain.data.box_augmentation import (DEFAULT_POLICY,
    sample_box_variant, require_full_box_family_support)
from densekp_retrain.data.first_dataset_plan import plan, read_catalog
from densekp_retrain.imaging.roi_planning import plan_native_crop_region
from densekp_retrain.imaging.regions import require_crop_in_region

ROOT=Path(__file__).resolve().parents[1]


class FirstDatasetPreparationTests(unittest.TestCase):
    def test_seed_replay_and_source_filter(self):
        cfg=json.loads((ROOT/'configs/first_dataset_preparation_a1_20261011.json').read_text())
        catalog,proposals=read_catalog(ROOT)
        a=plan(cfg,catalog,proposals)
        self.assertEqual(a,plan(cfg,catalog,proposals))
        self.assertEqual(len(a['episodes']),4)
        self.assertEqual([e['source_eligible'] for e in a['episodes']],[True,True,False,True])
        self.assertTrue(all(not e['geometry_checked'] and not e['rendered'] for e in a['episodes']))
        cfg['master_seed']+=1
        b=plan(cfg,catalog,proposals)
        self.assertNotEqual(a['episodes'][0]['layout'],b['episodes'][0]['layout'])
        self.assertNotEqual(a['episodes'][0]['rigs'],b['episodes'][0]['rigs'])
        self.assertEqual(a['episodes'][0]['persons'][0]['combo_id'],b['episodes'][0]['persons'][0]['combo_id'])

    def test_mode_and_pool_rejections(self):
        cfg=json.loads((ROOT/'configs/first_dataset_preparation_a1_20261011.json').read_text())
        catalog,proposals=read_catalog(ROOT)
        cfg['episodes'][0]['mode']='city_t_junction'
        with self.assertRaisesRegex(ValueError,'person_count_outside_mode'):plan(cfg,catalog,proposals)
        cfg['episodes'][0]['mode']='single_cyclist_boundary_pass'
        cfg['episodes'][0]['allowed_combinations']=['p1_comfortable_M02']
        result=plan(cfg,catalog,proposals)
        self.assertFalse(result['episodes'][0]['plan_valid'])

    def test_component_pool_can_plan_unbuilt_cross_combination(self):
        cfg=json.loads((ROOT/'configs/first_dataset_preparation_a1_20261011.json').read_text())
        catalog,proposals=read_catalog(ROOT)
        registry=json.loads((ROOT/'configs/asset_composition_a2_20261010/components.json').read_text())
        spec=cfg['episodes'][0]
        spec['selection_mode']='component_pool'
        spec['component_pools']=dict(body_ids=['M01'],motion_ids=['p2_straight'],
            clothing_ids=['M02_simple'],hair_ids=['M01_groom'])
        episode=plan(cfg,catalog,proposals,registry=registry)['episodes'][0]
        selected=episode['persons'][0]
        self.assertEqual(selected['combination_status'],'build_required')
        self.assertEqual(selected['clothing_id'],'M02_simple')
        self.assertEqual(selected['motion_id'],'p2_straight')
        self.assertTrue(selected['appearance_parent_review_required'])
        self.assertFalse(episode['source_eligible'])

    def test_augmentation_envelope_and_legacy_default(self):
        geometry={'body':np.array([[1000.,800.],[1100.,800.],[1000.,900.],[1100.,900.]])}
        old=plan_native_crop_region(geometry,(5760,2880))
        augmented=plan_native_crop_region(geometry,(5760,2880),box_augmentation_policy=DEFAULT_POLICY)
        self.assertLessEqual(augmented['roi_xyxy'][0],old['roi_xyxy'][0])
        self.assertGreaterEqual(augmented['roi_xyxy'][2],old['roi_xyxy'][2])
        parent=[1010.,810.,1090.,890.]
        require_full_box_family_support(parent,augmented['roi_xyxy'],(5760,2880),DEFAULT_POLICY)
        for visit in range(100):
            item=sample_box_variant(parent,policy=DEFAULT_POLICY,seed=8,
                observation_key=('scene','rig',0,1,'cam2',visit),
                roi_xyxy=augmented['roi_xyxy'])
            require_crop_in_region(item['center'],item['scale'],augmented['roi_xyxy'],(5760,2880))
        with self.assertRaisesRegex(ValueError,'family_outside'):
            require_full_box_family_support(parent,[1000,800,1100,900],(5760,2880),DEFAULT_POLICY)

    def test_sample_stable_and_invalid_roi_rejected(self):
        args=dict(policy=DEFAULT_POLICY,seed=1,observation_key=('s','r',0,1,'cam2'),
                  roi_xyxy=[100,100,500,450],image_size_wh=(640,480))
        a=sample_box_variant([250,200,300,260],**args)
        self.assertEqual(a,sample_box_variant([250,200,300,260],**args))
        self.assertEqual(a['parent_bbox_xyxy'],[250.,200.,300.,260.])
        with self.assertRaisesRegex(ValueError,'outside_high_quality_support'):
            sample_box_variant([250,200,300,260],**dict(args,roi_xyxy=[250,200,300,260]))

    def test_dataset_toggle_same_parent_and_fixed_eval(self):
        from PIL import Image
        from yacs.config import CfgNode
        from densekp.core.datasets.configuration import audit_cfg
        from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
        from densekp.core.keypoint_supervision import SUPERVISION_POLICY,make_targets
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);yy,xx=np.mgrid[0:480,0:640]
            full=np.stack((xx%256,yy%256,(xx+yy)%256),axis=2).astype(np.uint8)
            Image.fromarray(full).save(root/'full.png')
            Image.fromarray(full[50:430,80:560]).save(root/'roi.png')
            xy=np.tile([300.,240.],(138,1)).astype(np.float32)
            parent=[260,200,340,280]
            arrays=dict(imgname=np.array([str(root/'roi.png')]),center=np.array([[300.,240.]],np.float32),
                scale=np.array([.4],np.float32),proj_verts=make_targets(xy,np.ones(138,bool),[300,240],.4)[None],
                in_view=np.ones((1,138),bool),visible=np.zeros((1,138),bool),
                supervision_policy=SUPERVISION_POLICY,image_layout=np.array(['roi_image']),
                image_layout_version=np.array([1]),full_image_size_wh=np.array([[640,480]]),
                roi_xyxy=np.array([[80,50,560,430]]),offset_xy=np.array([[80,50]]),
                stored_size_wh=np.array([[480,380]]),parent_bbox_xyxy=np.array([parent],np.float32),
                box_augmentation_reserved_policy=np.array([json.dumps(DEFAULT_POLICY,sort_keys=True)]))
            np.savez(root/'parent.npz',**arrays)
            off_cfg=audit_cfg('supervision_weight')
            on_cfg=audit_cfg('supervision_weight')
            on_cfg.DATASETS.BOX_AUGMENTATION=CfgNode(dict(enabled=True,seed=99,policy=DEFAULT_POLICY))
            def dataset(cfg,train=True):
                return DatasetKeypoints(cfg,'explicit',is_train=train,use_augmentation=False,
                    annotation_path=root/'parent.npz',image_root='/')
            old=dataset(off_cfg);on=dataset(on_cfg)
            self.assertEqual(len(old),len(on))
            self.assertEqual(old.imgname.tolist(),on.imgname.tolist())
            np.testing.assert_array_equal(old[0]['img'],dataset(on_cfg,False)[0]['img'])
            a=on[0];b=on[0]
            for key in ('img','_trans','proj_verts_cropped'):
                np.testing.assert_array_equal(a[key],b[key])
            full_arrays=dict(arrays,imgname=np.array([str(root/'full.png')]),
                image_layout=np.array(['full_image']),roi_xyxy=np.array([[0,0,640,480]]),
                offset_xy=np.array([[0,0]]),stored_size_wh=np.array([[640,480]]))
            np.savez(root/'full.npz',**full_arrays)
            full_dataset=DatasetKeypoints(on_cfg,'explicit',is_train=True,use_augmentation=False,
                annotation_path=root/'full.npz',image_root='/')
            full_item=full_dataset[0]
            for key in ('img','_trans','proj_verts_cropped','img_size'):
                np.testing.assert_array_equal(a[key],full_item[key])
            on.set_box_augmentation_epoch(1)
            c=on[0];d=on[0]
            np.testing.assert_array_equal(c['_trans'],d['_trans'])
            arrays['roi_xyxy']=np.array([[250,190,350,290]])
            arrays['offset_xy']=np.array([[250,190]])
            arrays['stored_size_wh']=np.array([[100,100]])
            np.savez(root/'parent.npz',**arrays)
            dataset(off_cfg)  # off remains loadable; on rejects before reading pixels
            with self.assertRaisesRegex(ValueError,'family_outside_roi_support'):
                dataset(on_cfg)

    def test_index_pack_retains_parent_box_reservation(self):
        from densekp_retrain.data.packing import pack_index_rows
        from densekp.core.keypoint_supervision import SUPERVISION_POLICY
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'one.npz'
            np.savez(p,imgname=np.array(['unused.png']),center=np.array([[300,200]]),
                scale=np.array([.5]),proj_verts=np.zeros((1,138,3)),
                in_view=np.zeros((1,138),bool),visible=np.zeros((1,138),bool),
                person_id=np.array([1]),camera_id=np.array(['cam2']),
                image_layout=np.array(['roi_image']),image_layout_version=np.array([1]),
                full_image_size_wh=np.array([[640,480]]),roi_xyxy=np.array([[50,40,550,440]]),
                offset_xy=np.array([[50,40]]),stored_size_wh=np.array([[500,400]]),
                parent_bbox_xyxy=np.array([[260,160,340,240]]),
                box_augmentation_reserved_policy=np.array([json.dumps(DEFAULT_POLICY)]),
                supervision_policy=SUPERVISION_POLICY)
            packed=pack_index_rows(dict(rows=[dict(loader_input=str(p),loader_row=0,image='unused.png')]))
            np.testing.assert_array_equal(packed['parent_bbox_xyxy'],[[260,160,340,240]])
            self.assertEqual(json.loads(packed['box_augmentation_reserved_policy'][0]),DEFAULT_POLICY)

if __name__=='__main__':unittest.main()
