"""Clipped clothing retains exact body correspondence and has intentional hems."""
import unittest
import numpy as np
from tools.experiments.refine_cycling_character_clothing import clipped_shell


class CharacterClothingTests(unittest.TestCase):
    def test_clipped_boundary_and_barycentric_rigid_transform(self):
        rest=np.array([[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]],np.float32)
        faces=np.array([[a,b,c] for a in [0,1] for b in [2,3] for c in [4,5]])
        for face in faces:
            v=rest[face]
            if np.cross(v[1]-v[0],v[2]-v[0])@v.mean(0)<0:face[[1,2]]=face[[2,1]]
        joints=np.zeros((55,3),np.float32)
        joints[12,1]=.3
        joints[[1,2],1]=-.15;joints[[4,5],1]=-.6
        joints[16,0]=.5;joints[18,0]=.8
        joints[17,0]=-.5;joints[19,0]=-.8
        angle=.37
        rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]],np.float32)
        body=rest@rotation.T+np.array([.3,-.2,.7],np.float32)
        before=body.copy()
        arrays,meta=clipped_shell(rest,joints,body,faces)
        np.testing.assert_array_equal(before,body)
        indices=arrays['body_barycentric_indices']; weights=arrays['body_barycentric_weights']
        np.testing.assert_allclose(weights.sum(1),1,atol=1e-6)
        self.assertTrue((weights>=0).all())
        reference=arrays['reference_body_vertices'][0]
        np.testing.assert_allclose(reference,arrays['clothing_rest_vertices']@rotation.T+[.3,-.2,.7],atol=2e-7,rtol=0)
        np.testing.assert_array_equal(reference,(body[indices]*weights[:,:,None]).sum(1))
        self.assertLessEqual(meta['max_vertex_offset_m'],.01)
        self.assertGreaterEqual(meta['minimum_face_normal_separation_m'],.0004-2e-7)
        cf=arrays['clothing_faces']
        edges=np.sort(np.concatenate([cf[:,[0,1]],cf[:,[1,2]],cf[:,[2,0]]]),axis=1)
        unique,counts=np.unique(edges,axis=0,return_counts=True)
        self.assertLessEqual(counts.max(),2)
        endpoints=arrays['clothing_rest_vertices'][unique[counts==1]]
        deliberate=np.zeros(len(endpoints),bool)
        for axis,value in [(0,meta['sleeve_left_x']),(0,meta['sleeve_right_x']),
                           (1,meta['hem_y']),(1,meta['collar_y'])]:
            deliberate|=(np.abs(endpoints[:,:,axis]-value)<2e-6).all(1)
        self.assertTrue(deliberate.all(),'Unexpected open boundary in clothing interior')


if __name__=='__main__':unittest.main()
