"""synthetic_test_only policy/resume tests; no render, models or parameter updates."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from densekp_retrain.data.observation_policy import *
from densekp_retrain.data.raster_storage import roi_metadata,save_roi_observation
from densekp_retrain.data.sequence_packing import pack_rendered_sequence
from densekp_retrain.data.frame_sources import frame_directory


class ObservationContinuationTests(unittest.TestCase):
    def test_explicit_policy_and_pair_null_not_zero(self):
        a=np.ones(138,bool)
        one=pair_availability(dict(cam4=a))
        self.assertIsNone(one['common_points']);self.assertEqual(one['evaluable_point_count'],0)
        self.assertTrue(one['single_side_only']);self.assertFalse(one['has_dual_input'])
        with self.assertRaisesRegex(ValueError,'below8'):enforce_crop_policy({},True,one)
        enforce_crop_policy(dict(observation_policy=RECORD),True,one)
        both=pair_availability(dict(cam2=~a,cam4=a))
        self.assertEqual(both['common_points'],0);self.assertTrue(both['has_dual_input'])
        self.assertEqual(both['reason'],'empty_common_crop_support')
        with self.assertRaises(ValueError):policy_name(dict(observation_policy='unknown'))

    def test_occluded_and_domain_states(self):
        inside=np.ones(138,bool);visible=np.zeros(138,bool)
        self.assertEqual(no_box_state(inside,visible,np.full(138,2)),('empty','fully_neighbor_occluded_no_native_box'))
        self.assertEqual(no_box_state(~inside,visible)[0],'person_absent')
        self.assertEqual(no_box_state(inside,visible,np.full(138,5))[1],'no_visible_instance_box')
        with self.assertRaises(ValueError):no_box_state(~inside,inside)

    def test_hard_mask_identity_and_coverage_errors_not_availability(self):
        meta=roi_metadata([100,100,356,356],[5760,2880],'test-only')
        valid=np.ones((256,256),bool);pm=np.zeros_like(valid);entry=dict(person_id=1,source='synthetic_test_only')
        with self.assertRaisesRegex(BoxRejected,'no_visible'):validate_native_roi(None,entry,1,1,meta,valid,pm)
        pm[5,5]=True
        with self.assertRaisesRegex(ValueError,'null_box'):validate_native_roi(None,entry,1,1,meta,valid,pm)
        with self.assertRaisesRegex(ValueError,'identity'):validate_native_roi([105,105,106,106],entry,1,2,meta,valid,pm)
        with self.assertRaisesRegex(BoxRejected,'below_min_side'):validate_native_roi([105,105,106,106],entry,1,1,meta,valid,pm)
        with self.assertRaisesRegex(BoxRejected,'tight_mismatch'):validate_native_roi([104,104,106,106],entry,1,1,meta,valid,pm)
        pm[:]=False;pm[1:151,1:11]=True
        with self.assertRaises(ValueError) as error:validate_native_roi([101,101,111,251],entry,1,1,meta,valid,pm)
        self.assertNotIn(str(error.exception),EXPECTED_UNAVAILABLE)

    def test_multibox_pack_keeps_empty_GT_and_camera_missing(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);write=lambda p,v:p.write_text(json.dumps(v))
            write(root/'asset.json',dict(source_ref=str(root/'asset.json'),synthetic_test_only=True));write(root/'rig.json',{})
            write(root/'resume_manifest.json',dict(synthetic_test_only=True))
            people=[dict(person_id=j,asset_id='test',asset_manifest=str(root/'asset.json'),character_family='test',asset_frames=[0,1,2],source_frames=[30,31,32]) for j in (1,2)]
            recipe=dict(episode_id='synthetic_test_only',sequence_id='synthetic_test_only',frame_count=3,time_base=[1,10],people=people,child_seed=7,diagnostic_split='synthetic_test_only',storage_mode='roi_only',imaging_route='direct_erp_roi',observation_policy=RECORD)
            write(root/'recipe.json',recipe);Image.new('L',(5760,2880),255).save(root/'physical.png')
            for k in (0,1):
                frame=root/f'frame_{k:03d}';frame.mkdir();write(frame/'report.json',dict(final_erp_resolution=[5760,2880]))
                for c in ('cam2','cam4'):
                    cd=frame/c;cd.mkdir();instance=np.zeros((256,256),np.float32);instance[100:150,80:105]=1;instance[100:150,160:185]=2
                    if k==1 and c=='cam2':instance[instance==1]=0
                    xy=np.tile(np.array([[2792.,1325.],[2872.,1325.]])[:,None],(1,138,1)).astype(np.float32)
                    ann=dict(person_ids=np.array([1,2]),xy=xy,in_view=np.ones((2,138),bool),visible=np.zeros((2,138),bool),occlusion_evidence_code=np.full((2,138),2,np.uint8))
                    save_roi_observation(cd,np.zeros((256,256,3),np.uint8),instance,ann,np.ones((256,256),bool),roi_metadata([2700,1200,2956,1456],[5760,2880],root/'physical.png'))
            _,ep,table,coverage=pack_rendered_sequence(root,output_name='pack')
            self.assertEqual(table['packed']['row_count'],7)
            o=next(o for o in table['observations'] if o['observation_id']=='f1_cam2_p1')
            self.assertEqual(o['status'],'empty');self.assertEqual(o['reason'],'fully_neighbor_occluded_no_native_box')
            self.assertEqual(o['annotation_person_index'],0);self.assertTrue(Path(o['annotations']).exists())
            self.assertEqual(sum(o['status']=='camera_missing' for o in table['observations']),4)
            self.assertEqual(len({o['image'] for o in table['observations'] if o['image']}),4)
            self.assertIsNone(next(r for r in coverage if r.get('paired') and r['frame_index']==1 and r['person_id']==1)['common_points'])
            annpath=root/'frame_001/cam2/annotations.npz';ann=dict(np.load(annpath));ann['xy'][1]=[2701,1201];np.savez(annpath,**ann)
            _,_,zero_table,zero_cov=pack_rendered_sequence(root,output_name='zero_pack')
            self.assertEqual(zero_table['packed']['row_count'],6)
            pair=next(r for r in zero_cov if r.get('paired') and r['frame_index']==1 and r['person_id']==2)
            self.assertEqual(pair['common_points'],0);self.assertTrue(pair['common_set_defined'])
            ann['xy'][0,0,0]=np.nan;np.savez(annpath,**ann)
            with self.assertRaisesRegex(ValueError,'nonfinite'):pack_rendered_sequence(root,output_name='bad_pack')

    def test_readonly_frame_resolution_no_cleanup_alias(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);parent=root/'parent';out=root/'continued';parent.mkdir();out.mkdir()
            mapping={'18':dict(directory=str(parent/'frame_018'),ownership='readonly_parent'),'19':dict(directory=str(out/'frame_019'),ownership='this_run')}
            (out/'frame_sources.json').write_text(json.dumps(dict(schema='readonly_frame_sources_v1',frames=mapping)))
            self.assertEqual(frame_directory(out,18),parent/'frame_018');self.assertEqual(frame_directory(out,19),out/'frame_019')
            mapping['19']['directory']=str(parent/'frame_019')
            (out/'frame_sources.json').write_text(json.dumps(dict(schema='readonly_frame_sources_v1',frames=mapping)))
            with self.assertRaisesRegex(ValueError,'ownership'):frame_directory(out,19)

if __name__=='__main__':unittest.main()
