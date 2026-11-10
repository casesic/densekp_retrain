"""synthetic_test_only: bounded CPU tests, no models, renderer or real motion."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from densekp_retrain.data.multiperson_road_pass import sample_candidates,trajectory_separation
from densekp_retrain.data.occlusion_events import occlusion_evidence,three_stage_event
from densekp_retrain.data.dynamic_scene import frame_record,select_timeline
from densekp_retrain.imaging.roi_planning import plan_native_crop_region
from densekp_retrain.imaging.regions import require_crop_in_region
from densekp_retrain.data.raster_storage import roi_metadata,save_roi_observation
from densekp_retrain.data.sequence_packing import pack_rendered_sequence
from densekp_retrain.data.sequence_index import validate_sequence_index,SequenceIndexError


class MultiRoadPassTests(unittest.TestCase):
    def test_seed_local_and_replay(self):
        config=dict(person_count=2,attempt_limit=12,placement_height_m=1.2,parameter_ranges={k:[1,2] for k in ['target_x_m','target_y_m','speed_m_s','lateral_spacing_m','longitudinal_spacing_m']})
        a=sample_candidates(config,31)
        import random
        random.seed(9);random.random()
        self.assertEqual(a,sample_candidates(config,31));self.assertNotEqual(a,sample_candidates(config,32))
        self.assertEqual(len(a),12);self.assertEqual([p['person_id'] for p in a[0]['people']],[1,2])

    def test_shared_time_phase_not_relaxed(self):
        original=[dict(asset_frame=k,source_frame=30+k,source_time_tick=100000+t,time_base=[1,1000000],episode_time_tick=t) for k,t in enumerate([0,39999,80001])]
        rows=select_timeline(original,0,3,3)
        ep=dict(selected_timeline=rows,frame_count=3,time_base=[1,1000000],time_origin=dict(source_time_tick=100000),people=[dict(asset_frames=[0,1,2],source_frames=[30,31,32]) for _ in range(2)])
        self.assertEqual(frame_record(ep,1)['episode_time_tick'],39999)
        ep['people'][1]['asset_frames'][1]=2
        with self.assertRaises(ValueError):frame_record(ep,1)

    def test_neighbor_envelopes_reject_overlap(self):
        bounds=np.array([[[0,0,1,1],[0,2,1,3]],[[0,1,1,2],[0,3,1,4]]],float)
        roots=np.array([[[.5,.5,1.2],[.5,2.5,1.2]],[[.5,1.5,1.2],[.5,3.5,1.2]]])
        self.assertEqual(trajectory_separation(bounds,roots,clearance=.1,min_root_distance=.9)['minimum_root_distance_m'],2)
        bounds[1,1]=bounds[1,0]
        with self.assertRaises(ValueError):trajectory_separation(bounds,roots,clearance=.1,min_root_distance=.9)

    def test_union_covers_both_native_squares(self):
        parts={'body1':np.array([[2000,1200],[2200,1600]]),'groom1':np.array([[2030,1190],[2150,1250]]),
               'body2':np.array([[2300,1250],[2500,1580]])}
        r=plan_native_crop_region(parts,(5760,2880))['roi_xyxy']
        require_crop_in_region([2100,1400],2,r,(5760,2880));require_crop_in_region([2400,1415],1.65,r,(5760,2880))
        with self.assertRaises(ValueError):require_crop_in_region([3000,1400],2,r,(5760,2880))

    def test_depth_ID_evidence_does_not_change_visibility(self):
        xy=np.full((2,138,2),5.2);distance=np.full((2,138),3.);visible=np.zeros((2,138),bool);inside=np.ones_like(visible)
        depth=np.full((12,12),2.);ids=np.full((12,12),2.)
        out=occlusion_evidence(xy,distance,visible,inside,[1,2],depth,ids)
        self.assertTrue((out['occlusion_evidence_code'][0]==2).all());self.assertTrue((out['occlusion_evidence_code'][1]==3).all())
        self.assertFalse(visible.any())
        depth[4:7,4:7]=2.99
        self.assertTrue((occlusion_evidence(xy,distance,visible,inside,[1,2],depth,ids)['occlusion_evidence_code']==5).all())
        depth[5,5]=np.nan
        with self.assertRaises(ValueError):occlusion_evidence(xy,distance,visible,inside,[1,2],depth,ids)

    def test_event_requires_three_consecutive_with_fixed_denominator(self):
        event=three_stage_event([.05]*3+[.3]*3+[.1]*3)
        self.assertEqual(event['during'],[3,4,5])
        self.assertIsNone(three_stage_event([.05]*3+[.3,None,.3]+[.1]*3))
        self.assertIsNone(three_stage_event([.05]*3+[.3]*3+[.2]*3))

    def test_two_people_shared_image_empty_slot_and_matrix_identity(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);write=lambda p,x:p.write_text(json.dumps(x))
            write(root/'asset.json',dict(source_ref=str(root/'asset.json'),synthetic_test_only=True))
            write(root/'rig.json',dict(synthetic_test_only=True))
            original=[dict(asset_frame=k,source_frame=30+k,source_time_tick=100000+t,time_base=[1,1000000],episode_time_tick=t) for k,t in enumerate([0,39999,80001])]
            records=select_timeline(original,0,3,3);write(root/'original.json',original);write(root/'selected.json',records)
            matrices=np.tile(np.eye(4),(3,2,1,1));matrices[:,1,0,3]=2;np.save(root/'matrices.npy',matrices)
            people=[dict(person_id=j+1,asset_id='test',asset_manifest=str(root/'asset.json'),character_family='test',asset_frames=[0,1,2],source_frames=[30,31,32],scene_from_model=matrices[:,j].tolist()) for j in range(2)]
            recipe=dict(episode_id='synthetic_test_only',sequence_id='synthetic_test_only',frame_count=3,time_base=[1,1000000],people=people,child_seed=1,
                diagnostic_split='synthetic_test_only',storage_mode='roi_only',imaging_route='direct_erp_roi',selected_timeline=records,
                original_timeline_ref=str(root/'original.json'),selected_timeline_ref=str(root/'selected.json'),scene_matrices_ref=str(root/'matrices.npy'),scene_matrix_person_ids=[1,2],generator_version='synthetic_test_only',
                time_origin=dict(source_time_tick=100000),time_unit_status='test only',sampling_policy='test',unselected_asset_frames=[])
            write(root/'recipe.json',recipe)
            Image.fromarray(np.full((2880,5760),255,np.uint8)).save(root/'physical.png')
            for k,row in enumerate(records):
                frame=root/f'frame_{k:03d}';frame.mkdir();write(frame/'report.json',dict(final_erp_resolution=[5760,2880]))
                for c in ['cam2','cam4']:
                    folder=frame/c;folder.mkdir();instance=np.zeros((256,256),np.float32)
                    instance[100:150,80:105]=1
                    if k!=1 or c!='cam2':instance[100:150,160:185]=2
                    meta=roi_metadata([2700,1200,2956,1456],[5760,2880],root/'physical.png')
                    xy=np.tile(np.array([[2792.,1325.],[2872.,1325.]])[:,None],(1,138,1)).astype(np.float32)
                    ann=dict(person_ids=np.array([1,2]),xy=xy,in_view=np.ones((2,138),bool),visible=np.zeros((2,138),bool),
                        time_tick=np.array(row['episode_time_tick']),source_time_tick=np.array(row['source_time_tick']),asset_frame_index=np.array(k),source_frame_index=np.array(30+k),time_base=np.array([1,1000000]))
                    save_roi_observation(folder,np.zeros((256,256,3),np.uint8),instance,ann,np.ones((256,256),bool),meta)
            pack,ep,index,coverage=pack_rendered_sequence(root,output_name='pack')
            self.assertEqual(index['packed']['row_count'],11);self.assertEqual(len(index['observations']),12)
            self.assertEqual(sum(o['status']=='empty' for o in index['observations']),1)
            with np.load(pack/'packed.npz') as rows:
                self.assertEqual(len(set(rows['imgname'])),6);self.assertEqual(set(rows['person_id']),{1,2})
                np.testing.assert_array_equal(rows['offset_xy'],np.tile([2700,1200],(11,1)))
            index['observations'][0]['scene_matrix_person_index']=1
            with self.assertRaises(SequenceIndexError):validate_sequence_index(ep,index,base_dir=pack,check_packed=True)


if __name__=='__main__':unittest.main()
