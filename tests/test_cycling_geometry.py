"""Independent image poses must retain their own meshes, labels and lift."""
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.fisheye import run_ds_scene_pair_pilot as pilot
from tools.fisheye.run_fisheye_multiperson_pilot import world_points
from tools.experiments.prepare_cycling_clothing import shell


class CyclingGeometryTests(unittest.TestCase):
    def test_clothing_shell_preserves_body_and_uses_body_lift(self):
        vertices = np.array([[1., 1., 1.], [-1., -1., 1.],
                             [-1., 1., -1.], [1., -1., -1.]])
        faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        original = vertices.copy()
        clothing, clothing_faces, ids, separation = shell(vertices, faces, np.ones(4, bool), .004)
        np.testing.assert_array_equal(vertices, original)
        np.testing.assert_array_equal(ids[clothing_faces], faces)
        self.assertTrue((separation > 0).all())
        person = dict(location=[0, 0, .25], yaw_deg=37)
        body_world = world_points(vertices[ids], vertices, person)
        clothing_world = world_points(clothing, vertices, person)
        np.testing.assert_allclose(np.linalg.norm(clothing_world-body_world, axis=1),
                                   np.linalg.norm(clothing-vertices[ids], axis=1), atol=2e-7, rtol=0)
        # Lifting clothing by its own minimum would move it relative to the body.
        self.assertGreater(np.abs(world_points(clothing, clothing, person)-clothing_world).max(), .001)

    def test_clothing_rejects_invalid_offset(self):
        for offset in (0, -.001, .1, np.nan):
            with self.assertRaises(ValueError):
                shell(np.zeros((3, 3)), np.array([[0, 1, 2]]), np.array([True]), offset)

    def test_axis_lift_yaw_translation_matches_homogeneous_mesh_matrix(self):
        rng = np.random.default_rng(7)
        vertices = rng.normal(size=(20, 3))
        operator = rng.uniform(size=(138, 20))
        operator /= operator.sum(1, keepdims=True)
        person = dict(location=[13.676, -4.25, .25], yaw_deg=72.8)
        angle = np.deg2rad(person['yaw_deg'])
        axis = np.array([[1,0,0,0],[0,0,-1,0],[0,1,0,0],[0,0,0,1.]])
        lift = np.eye(4); lift[2,3] = -vertices[:,1].min()
        yaw = np.eye(4)
        yaw[:2,:2] = [[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        translation = np.eye(4); translation[:3,3] = person['location']
        matrix = translation @ yaw @ lift @ axis
        rendered = np.column_stack((vertices,np.ones(len(vertices)))) @ matrix.T
        labels = world_points(operator@vertices,vertices,person)
        np.testing.assert_allclose(labels,operator@rendered[:,:3],atol=1e-12)

    def test_annotation_reads_each_person_asset_without_default_geometry(self):
        intrinsics = {name:replace(k,width=64,height=64,fx=k.fx/45,fy=k.fy/45,
                                  cx=k.cx/45,cy=k.cy/45) for name,k in pilot.DS_20260320.items()}
        poses,_,_ = pilot.rig_poses([0,0,0],0,.18)
        with TemporaryDirectory() as temporary:
            root=Path(temporary); (root/'source').mkdir()
            people=[]; expected=[]
            for i in range(2):
                geometry=root/f'asset_{i}'; geometry.mkdir()
                vertices=np.array([[0.,-1-i,0.],[.2,1.,.2]])
                dense=np.tile([[.1,.3+i,.1]],(138,1))
                np.save(geometry/'vertices.npy',vertices[None])
                np.save(geometry/'densekp_3d.npy',dense[None])
                person=dict(id=i+1,geometry=str(geometry),frame=0,
                            location=[i*2.,2.,.25],yaw_deg=10.*i)
                people.append(person); expected.append(world_points(dense,vertices,person))
            for name in poses:
                Image.fromarray(np.full((32,64,3),150,np.uint8)).save(root/'source'/f'{name}_preview.png')
            with patch.object(pilot,'DS_20260320',intrinsics), \
                 patch.object(pilot,'read_combined_depth_index',return_value=(None,np.ones((32,64)),np.ones((32,64)),None)), \
                 patch.object(pilot,'erp_visibility',return_value=(None,np.ones(276,bool))):
                pilot.process_scene(root,people,poses,{n:np.ones((32,64),bool) for n in poses},
                                    None,None,lens_masks={n:np.ones((64,64),bool) for n in poses},
                                    lens_metadata={n:{} for n in poses})
            np.testing.assert_allclose(np.load(root/'shared_world_densekp.npy'),expected)
            for name in poses:
                with np.load(root/name/'annotations.npz') as z:
                    np.testing.assert_allclose(z['world_densekp'],expected)


if __name__=='__main__': unittest.main()
