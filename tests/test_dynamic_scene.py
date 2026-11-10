"""synthetic_test_only native time/root/support checks; no model or renderer."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from densekp_retrain.data.dynamic_scene import select_timeline,scene_trajectory,person_matrices,support_limits,ROOT_MODE,LEGACY_ROOT,SUPPORT


class DynamicSceneTests(unittest.TestCase):
    def records(self):
        return [dict(asset_frame=k,source_frame=k+30,source_time_tick=100000000+t,episode_time_tick=t,time_base=[1,1000000]) for k,t in enumerate([0,40010,79989,120004])]

    def test_nonuniform_ticks_and_selection_origin(self):
        rows=select_timeline(self.records(),1,3,4)
        self.assertEqual([r['asset_frame'] for r in rows],[1,2,3])
        self.assertEqual([r['episode_time_tick'] for r in rows],[0,39979,79994])
        self.assertEqual(rows[0]['asset_episode_time_tick'],40010)

    def test_missing_repeated_and_out_of_range_rejected(self):
        for mutation in [lambda r:r[1].pop('source_time_tick'),lambda r:r[1].update(source_time_tick=r[0]['source_time_tick']),lambda r:r[1].update(asset_frame=9),lambda r:r[1].update(source_frame=1.5)]:
            rows=self.records();mutation(rows)
            with self.assertRaises(ValueError):select_timeline(rows,0,4,4)

    def test_trajectory_uses_ticks_not_nominal_rate(self):
        rows=select_timeline(self.records(),0,4,4);C=np.array([[1,0,0],[0,0,-1],[0,1,0]])
        matrices=scene_trajectory(rows,[0,2,1.2],[0,1],2.,C)
        np.testing.assert_array_equal(matrices[:,2,3],np.full(4,1.2))
        np.testing.assert_allclose(matrices[:,1,3],2+np.array([0,40010,79989,120004])*2/1000000,atol=1e-15,rtol=0)
        p=dict(root_mode=ROOT_MODE,support_policy=SUPPORT,scene_from_model=matrices.tolist(),placement_height_m=1.2)
        np.testing.assert_array_equal(person_matrices(p,4),matrices)
        # Body and groom share one scene matrix; the head transform is local.
        H=np.eye(4);H[:3,3]=[.2,.3,.1];point=np.array([.1,.2,.3,1.])
        np.testing.assert_allclose(matrices[2]@(H@point),(matrices[2]@H)@point)

    def test_double_root_unknown_mode_and_nonrigid_matrix_rejected(self):
        p=dict(root_mode=ROOT_MODE,support_policy=SUPPORT,scene_from_model=[np.eye(4).tolist()]*2,placement_height_m=0)
        for mutation in [lambda q:q.update(model_to_world=np.eye(4).tolist()),lambda q:q.update(root_mode='unknown'),lambda q:q['scene_from_model'][0][0].__setitem__(0,2),lambda q:q.update(placement_height_m=.2)]:
            q=copy.deepcopy(p);mutation(q)
            with self.assertRaises(ValueError):person_matrices(q,2)

    def test_support_policies_remain_distinct(self):
        self.assertEqual(support_limits(dict(root_mode=LEGACY_ROOT)),(-.02,.12))
        p=dict(root_mode=ROOT_MODE,support_policy=SUPPORT,support_checks=dict(body_min_clearance_m=[.05,.7]))
        self.assertEqual(support_limits(p),(.05,.7))
        p['support_policy']='ignore_ground'
        with self.assertRaises(ValueError):support_limits(p)

    def test_legacy_fixed_matrix_is_unchanged(self):
        M=np.eye(4);M[:3,3]=[1,2,3]
        np.testing.assert_array_equal(person_matrices(dict(root_mode=LEGACY_ROOT,model_to_world=M.tolist()),3),np.repeat(M[None],3,axis=0))

    def test_native_sidecar_rejects_source_time_conflict(self):
        from tests.test_sequence_index import sequence_fixture,write_fixture_files
        from densekp_retrain.data.sequence_index import validate_sequence_index,SequenceIndexError
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);ep,table=sequence_fixture();rows=select_timeline(self.records(),0,3,4)
            ep.update(time_base=[1,1000000],selected_timeline_ref='selected.json',original_timeline_ref='original.json',scene_matrices_ref='matrices.npy')
            ep['timeline']['expected_tick_step']=None
            for k,frame in enumerate(table['frames']):frame['time_tick']=rows[k]['episode_time_tick']
            for obs in table['observations']:
                k=obs['frame_index'];row=rows[k]
                obs.update(time_tick=row['episode_time_tick'],source_time_tick=row['source_time_tick'],source_time_base=row['time_base'],source_frame_index=row['source_frame'],
                           selected_timeline_ref='selected.json',scene_matrices_ref='matrices.npy',timeline_row=k,scene_matrix_row=k)
            write_fixture_files(root,ep,table)
            (root/'selected.json').write_text(json.dumps(rows));(root/'original.json').write_text(json.dumps(self.records()))
            np.save(root/'matrices.npy',np.repeat(np.eye(4)[None],3,axis=0))
            self.assertEqual(validate_sequence_index(ep,table,base_dir=root,check_image_headers=True,check_packed=True)['status'],'passed')
            table['observations'][1]['source_time_tick']+=1
            with self.assertRaises(SequenceIndexError):validate_sequence_index(ep,table,base_dir=root,check_packed=True)


if __name__=='__main__':unittest.main()
