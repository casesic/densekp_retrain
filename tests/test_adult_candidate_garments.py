"""Coverage changes must retain posed-body correspondence and legacy shell limits."""
import unittest
import numpy as np
from tools.experiments.refine_cycling_character_clothing import clipped_shell

class GarmentCoverageTest(unittest.TestCase):
    def test_clipping_interpolates_the_same_posed_triangle(self):
        rest=np.array([[-.3,-.5,0],[.3,-.5,0],[0,.5,0]],np.float32)
        joints=np.zeros((25,3),np.float32)
        joints[16,0]=.1;joints[17,0]=-.1;joints[20,0]=.4;joints[21,0]=-.4
        joints[1:3,1]=0;joints[7:9,1]=-.6;joints[12,1]=.5
        body=rest@np.array([[0,-1,0],[1,0,0],[0,0,1]],np.float32).T+[1,2,3]
        arrays,_=clipped_shell(rest,joints,body,np.array([[0,1,2]]),dict(sleeve=.5,leg=.7))
        ids=arrays['body_barycentric_indices'];w=arrays['body_barycentric_weights']
        np.testing.assert_allclose(w.sum(1),1,atol=1e-6)
        np.testing.assert_allclose(arrays['reference_body_vertices'][0],(body[ids]*w[:,:,None]).sum(1),atol=1e-7)
        self.assertLess(np.linalg.norm(arrays['clothing_vertices']-arrays['reference_body_vertices'],axis=-1).max(),.01)
        self.assertTrue((arrays['clothing_rest_vertices'][:,1]>=-.420001).all())

if __name__=='__main__':unittest.main()
