"""synthetic_test_only mixed clock/root/dual-domain contracts; no network/models."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
from PIL import Image
from densekp_retrain.data.person_time import map_timeline,SCHEMA,validate_record
from densekp_retrain.data.dynamic_scene import frame_record,person_frame_record,person_matrices,ROOT_MODE,SUPPORT,LEGACY_ROOT
from densekp_retrain.data.mixed_road_pass import candidates,edge_summary,checked_mixed_separation
from densekp_retrain.data.multiperson_road_pass import trajectory_separation
from densekp_retrain.imaging.roi_planning import plan_person_crop_union,plan_native_crop_region


def clock(n=3):return [dict(frame_index=k,episode_time_tick=k,time_base=[1,10]) for k in range(n)]
def source(times,base,shift=0):return [dict(asset_frame=k,source_frame=k+shift,source_time_tick=t,time_base=base) for k,t in enumerate(times)]

class MixedRoadPassTests(unittest.TestCase):
    def test_exact_nonuniform_nearest_tie_and_error(self):
        s=source([0,40001,80001,120000,160001,200000],[1,1000000],30)
        rows=map_timeline(s,clock(),origin_record=0)
        self.assertEqual([r['asset_frame'] for r in rows],[0,2,5]);self.assertEqual(rows[1]['quantization_error_seconds'],[-19999,1000000])
    def test_tie_selection_rejects_repetition(self):
        with self.assertRaisesRegex(ValueError,'repeated'):map_timeline(source([0,2,4],[1,10]),clock(),origin_record=0)
        c=[dict(frame_index=k,episode_time_tick=t,time_base=[1,10]) for k,t in enumerate([0,3])]
        rows=map_timeline(source([0,2,4],[1,10]),c,origin_record=0)
        self.assertEqual(rows[1]['asset_frame'],1);self.assertEqual(rows[1]['quantization_error_seconds'],[-1,10])
    def test_outside_missing_nonmonotonic_and_tampered_error(self):
        for s in [source([0,1],[1,10]),source([0,2,1],[1,10])]:
            with self.assertRaises(ValueError):map_timeline(s,clock(),origin_record=0)
        s=source([0,1,2],[1,10]);s[1].pop('source_time_tick')
        with self.assertRaises(ValueError):map_timeline(s,clock(),origin_record=0)
        r=map_timeline(source([0,1,2],[1,10]),clock(),origin_record=0)[1];r['quantization_error_seconds']=[1,10]
        with self.assertRaises(ValueError):validate_record(r,clock()[1],dict(asset_frames=[0,1,2],source_frames=[0,1,2]))
    def test_independent_person_records_and_roots(self):
        mappings=[map_timeline(source(list(range(7)),[1,30],9),clock(),origin_record=0),map_timeline(source([0,39999,80000,120001,160000,200000],[1,1000000],30),clock(),origin_record=0)]
        people=[dict(person_id=j+1,asset_frames=[r['asset_frame'] for r in rs],source_frames=[r['source_frame'] for r in rs]) for j,rs in enumerate(mappings)]
        ep=dict(frame_count=3,time_base=[1,10],episode_timeline=clock(),time_mapping_schema=SCHEMA,person_timelines={str(j+1):r for j,r in enumerate(mappings)},people=people)
        self.assertNotEqual(frame_record(ep,1)['person_records'][0]['source_frame'],person_frame_record(ep,1,people[1])['source_frame'])
        bad=copy.deepcopy(ep);bad['people'][0]['asset_frames'][1]=2
        with self.assertRaises(ValueError):frame_record(bad,1)
        M=np.eye(4);M[:3,3]=[.2,1.1,.95];legacy=dict(root_mode=LEGACY_ROOT,model_to_world=M.tolist())
        np.testing.assert_array_equal(person_matrices(legacy,3),np.repeat(M[None],3,axis=0))
        new=dict(root_mode=ROOT_MODE,support_policy=SUPPORT,placement_height_m=1.2,scene_from_model=np.repeat(np.eye(4)[None],3,axis=0).tolist())
        for k in range(3):new['scene_from_model'][k][2][3]=1.2;new['scene_from_model'][k][1][3]=.5*k/10
        self.assertAlmostEqual(person_matrices(new,3)[2,1,3],.1)
        new['model_to_world']=M.tolist()
        with self.assertRaises(ValueError):person_matrices(new,3)
    def test_dual_edge_is_same_time_intersection_with_duration(self):
        rows=[dict(views={c:dict(in_view=130,outside_domain=8) for c in ['cam2','cam4']},common_physical_points=122) for _ in range(11)]
        self.assertTrue(edge_summary(rows,clock(11))['passed'])
        for k,r in enumerate(rows):r['views']['cam2' if k<6 else 'cam4']['outside_domain']=0
        self.assertEqual(edge_summary(rows,clock(11))['count'],0)
        rows=rows[:5]
        for r in rows:
            for v in r['views'].values():v['outside_domain']=8
        self.assertFalse(edge_summary(rows,clock(5))['passed'])
    def test_new_candidate_edge_quota_keeps_old_default(self):
        rows=[dict(views={c:dict(in_view=130,outside_domain=8 if k<7 else 0)
                          for c in ('cam2','cam4')},common_physical_points=120)
              for k in range(31)]
        self.assertFalse(edge_summary(rows,clock(31))['passed'])
        self.assertTrue(edge_summary(rows,clock(31),required_count=7)['passed'])
        with self.assertRaisesRegex(ValueError,'required_count'):
            edge_summary(rows,clock(31),required_count=32)
    def test_seed_local_bounds(self):
        c=dict(attempt_limit=12,parameter_ranges=dict(x=[-.1,.1],speed=[.45,.6]))
        a=candidates(c,20261009);self.assertEqual(a,candidates(c,20261009));self.assertNotEqual(a,candidates(c,20261010))
        self.assertTrue(all(-.1<=r['parameters']['x']<=.1 for r in a))
    def test_directed_candidates_are_frozen_and_bounded(self):
        c=dict(attempt_limit=2,parameter_ranges={'x':[-2.,0.]},explicit_candidate_parameters=[{'x':-1.6},{'x':-1.1}])
        self.assertEqual([x['parameters']['x'] for x in candidates(c,7)],[-1.6,-1.1])
        self.assertEqual(candidates(c,7),candidates(c,7))
        c['explicit_candidate_parameters'].append({'x':-1.0})
        with self.assertRaisesRegex(ValueError,'explicit_candidate_count'):candidates(c,7)
    def test_frozen_camera_stream_does_not_change_layout_stream(self):
        base=dict(attempt_limit=2,parameter_ranges={'x':[-2.,0.]},
                  explicit_candidate_parameters=[{'x':-1.6},{'x':-1.1}],
                  camera_seed=91,camera_parameter_ranges={'yaw_deg':[80.,100.]},
                  travel_direction_options=[1],
                  explicit_camera_parameters=[{'yaw_deg':82.},{'yaw_deg':98.}])
        first=candidates(base,7)
        self.assertEqual(first,candidates(base,7))
        other=copy.deepcopy(base);other['explicit_camera_parameters'][0]['yaw_deg']=84.
        self.assertEqual([x['parameters'] for x in first],[x['parameters'] for x in candidates(other,7)])
        self.assertNotEqual(first[0]['camera_parameters'],candidates(other,7)[0]['camera_parameters'])
    def test_narrow_phase_distinguishes_aabb_false_positive_from_close_surfaces(self):
        bounds=np.array([[[0.,0.,1.,1.],[.9,0.,1.9,1.]]])
        roots=np.array([[[0.,0.,0.],[1.,0.,0.]]])
        far=[dict(body=np.array([[[0.,0.,0.]]]),cloth=np.array([[[0.,0.,.2]]])),
             dict(body=np.array([[[.9,0.,.2]]]),cloth=np.array([[[.9,0.,.4]]]))]
        with self.assertRaisesRegex(ValueError,'neighbor_body_envelopes'):trajectory_separation(bounds,roots,clearance=.1,min_root_distance=.9)
        check=checked_mixed_separation(bounds,roots,far)
        self.assertEqual(check['narrow_phase_frame_count'],1)
        self.assertGreater(check['minimum_narrow_phase_vertex_distance_m'],.1)
        near=copy.deepcopy(far);near[1]['body'][0,0]=[.05,0.,0.]
        with self.assertRaisesRegex(ValueError,'neighbor_surface_vertices'):checked_mixed_separation(bounds,roots,near)
    def test_f02_integer_pts_remain_separate_from_walk_clock(self):
        bike=source([8670662+1001*k for k in range(120)],[1,30000],8662)
        walk=source(list(range(127)),[1,30],2200)
        b=map_timeline(bike,clock(31),origin_record=0)
        w=map_timeline(walk,clock(31),origin_record=0)
        self.assertEqual([b[k]['asset_frame'] for k in (0,1,15,30)],[0,3,45,90])
        self.assertEqual([w[k]['asset_frame'] for k in (0,1,15,30)],[0,3,45,90])
        self.assertEqual(b[1]['source_time_tick'],8673665)
        self.assertEqual(b[1]['quantization_error_seconds'],[1,10000])
        self.assertEqual(w[1]['quantization_error_seconds'],[0,1])
        self.assertEqual(b[30]['source_frame'],8752)
    def test_union_covers_individual_squares_without_cross_person_padding(self):
        a={'body':np.array([[4200.,1100.],[4550.,1950.]])}
        b={'body':np.array([[2700.,1000.],[3100.,2100.]])}
        with self.assertRaisesRegex(ValueError,'region_outside'):
            plan_native_crop_region({'a':a['body'],'b':b['body']},(5760,2880))
        union=plan_person_crop_union({1:a,2:b},(5760,2880))
        x0,y0,x1,y1=union['roi_xyxy']
        for item in union['person_regions'].values():
            u0,v0,u1,v1=item['roi_xyxy']
            self.assertLessEqual(x0,u0);self.assertLessEqual(y0,v0)
            self.assertGreaterEqual(x1,u1);self.assertGreaterEqual(y1,v1)
        with self.assertRaisesRegex(ValueError,'region_outside'):
            plan_person_crop_union({1:a,2:{'body':np.array([[2700.,100.],[3100.,2800.]])}},(5760,2880))
    def test_real_packer_two_clocks_and_missing_observation(self):
        from densekp_retrain.data.sequence_packing import pack_rendered_sequence
        from densekp_retrain.data.sequence_index import validate_sequence_index,SequenceIndexError
        from densekp_retrain.data.raster_storage import roi_metadata,save_roi_observation
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);write=lambda p,x:p.write_text(json.dumps(x));people=[];timelines={};sources={};refs={};srefs={}
            records=[source(list(range(7)),[1,30],9),source([0,39999,80000,120001,160000,200000],[1,1000000],30)]
            matrices=np.repeat(np.eye(4)[None,None],6,axis=0).reshape(3,2,4,4);matrices[:,1,2,3]=1.2
            for j,src in enumerate(records):
                pid=str(j+1);rs=map_timeline(src,clock(),origin_record=0);timelines[pid]=rs
                refs[pid]=str(root/f'person{pid}.json');srefs[pid]=str(root/f'source{pid}.json');write(Path(refs[pid]),rs);write(Path(srefs[pid]),src)
                af=root/f'asset{pid}.json';write(af,dict(source_ref=str(af),synthetic_test_only=True))
                p=dict(person_id=j+1,asset_id=pid,asset_manifest=str(af),character_family='family'+pid,asset_frames=[r['asset_frame'] for r in rs],source_frames=[r['source_frame'] for r in rs])
                if j:p.update(root_mode=ROOT_MODE,support_policy=SUPPORT,placement_height_m=1.2,scene_from_model=matrices[:,j].tolist())
                else:p.update(root_mode=LEGACY_ROOT,model_to_world=matrices[0,j].tolist())
                people.append(p)
            np.save(root/'matrices.npy',matrices);write(root/'clock.json',clock());write(root/'rig.json',dict(synthetic_test_only=True))
            recipe=dict(episode_id='synthetic_test_only',sequence_id='synthetic_test_only',frame_count=3,time_base=[1,10],people=people,child_seed=1,diagnostic_split='synthetic_test_only',storage_mode='roi_only',imaging_route='direct_erp_roi',time_mapping_schema=SCHEMA,person_timelines=timelines,episode_timeline=clock(),episode_timeline_ref=str(root/'clock.json'),person_timeline_refs=refs,source_timeline_refs=srefs,scene_matrices_ref=str(root/'matrices.npy'),scene_matrix_person_ids=[1,2],generator_version='synthetic_test_only',time_unit_status='synthetic_test_only')
            write(root/'recipe.json',recipe);write(root/'partial_delivery.json',dict(rejected_frame_indices=[1],stop_reason='synthetic_test_only quality rejection'));Image.fromarray(np.full((2880,5760),255,np.uint8)).save(root/'physical.png')
            for k in range(3):
                f=root/f'frame_{k:03d}';f.mkdir();write(f/('rejected_frame.json' if k==1 else 'report.json'),dict(final_erp_resolution=[5760,2880],acceptance_status='rejected' if k==1 else 'passed',reason='synthetic_test_only quality rejection'))
                for c in ['cam2','cam4']:
                    d=f/c;d.mkdir();instance=np.zeros((256,256),np.float32);instance[100:150,80:105]=1
                    if k!=1 or c!='cam2':instance[100:150,160:185]=2
                    ann=dict(person_ids=np.array([1,2]),xy=np.tile(np.array([[2792.,1325.],[2872.,1325.]])[:,None],(1,138,1)).astype(np.float32),in_view=np.ones((2,138),bool),visible=np.zeros((2,138),bool),time_tick=np.array(k),time_base=np.array([1,10]))
                    for key in ['asset_frame','source_frame','source_time_tick','source_time_base','quantization_error_seconds']:ann['person_'+key]=np.array([timelines[str(j+1)][k][key] for j in range(2)])
                    save_roi_observation(d,np.zeros((256,256,3),np.uint8),instance,ann,np.ones((256,256),bool),roi_metadata([2700,1200,2956,1456],[5760,2880],root/'physical.png'))
            pack,ep,table,cov=pack_rendered_sequence(root,output_name='pack');self.assertEqual(table['packed']['row_count'],11)
            self.assertEqual(table['frames'][1]['acceptance_status'],'rejected');self.assertEqual(ep['execution_status'],'partial_stopped')
            self.assertEqual(len({o['image'] for o in table['observations']}),6)
            bad=copy.deepcopy(table);bad['observations'][1]['source_time_tick']+=1
            with self.assertRaises(SequenceIndexError):validate_sequence_index(ep,bad,base_dir=pack,check_packed=True)
            bad=copy.deepcopy(table);bad['observations'][0]['scene_matrix_person_index']=1
            with self.assertRaises(SequenceIndexError):validate_sequence_index(ep,bad,base_dir=pack,check_packed=True)

if __name__=='__main__':unittest.main()
