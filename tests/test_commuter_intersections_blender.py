"""Run with Blender --python-exit-code 1 --python this_file.py."""
from pathlib import Path
import sys
import unittest
import numpy as np

try:
    import bpy
except ImportError:
    raise unittest.SkipTest('These geometry checks require Blender mathutils/bpy')

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from densekp_retrain.rendering.blender.intersections import crossings
from tests.blender_geometry_fixture import dual_quaternion_binding, scan
from mathutils.bvhtree import BVHTree


class TriangleCrossings(unittest.TestCase):
    def test_edge_pierces_face_while_vertices_are_outside(self):
        a=np.array([[-1,-1,0],[1,-1,0],[0,1,0]],float)
        b=np.array([[0,0,-1],[0,0,1],[2,0,1]],float)
        pairs,points,_=crossings(a,np.array([[0,1,2]]),b,np.array([[0,1,2]]))
        self.assertEqual(len(pairs),1)
        self.assertAlmostEqual(points[0,2],0)

    def test_disjoint(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        self.assertEqual(len(crossings(a,np.array([[0,1,2]]),a+[0,0,1],np.array([[0,1,2]]))[0]),0)

    def test_shared_edge_is_not_self_crossing(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],float)
        self.assertEqual(len(crossings(a,np.array([[0,1,2],[0,1,3]]))[0]),0)

    def test_coplanar_overlap_remains_unresolved(self):
        a=np.array([[0,0,0],[1,0,0],[0,1,0]],float)
        pairs,_,unresolved=crossings(a,np.array([[0,1,2]]),a+[.1,.1,0],np.array([[0,1,2]]))
        self.assertEqual(len(pairs),0)
        self.assertIsNone(unresolved)

    def test_oriented_scan_keeps_overlap_instead_of_xor(self):
        bpy.ops.mesh.primitive_cube_add(size=1)
        obj=bpy.context.object;obj.data.calc_loop_triangles()
        v=np.array([v.co[:] for v in obj.data.vertices]);f=np.array([t.vertices[:] for t in obj.data.loop_triangles])
        v2=np.concatenate([v,v+[.4,0,0]]);f2=np.concatenate([f,f+len(v)])
        tree=BVHTree.FromPolygons(v2.tolist(),f2.tolist(),all_triangles=True)
        origin=np.array([-.6,-.42,-.38]);occupancy,unclosed=scan(tree,origin,(9,5,5),.2)
        self.assertTrue(occupancy[4,2,2])  # x=.2 is inside both cubes.
        self.assertFalse(occupancy[0,2,2]);self.assertEqual(unclosed,0)
        bpy.data.objects.remove(obj,do_unlink=True)

    def test_dqs_exact_rigid_transform(self):
        theta=.7;c,s=np.cos(theta),np.sin(theta)
        tf=np.eye(4)[None];tf[0,:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]];tf[0,:3,3]=[.1,.2,.3]
        points=np.array([[1.,0,0],[0,1.,1.]])
        result,_,_=dual_quaternion_binding(points,np.ones((2,1)),tf)
        np.testing.assert_allclose(result,points@tf[0,:3,:3].T+tf[0,:3,3],atol=2e-7)

    def test_dqs_quaternion_sign_alignment(self):
        tf=np.repeat(np.eye(4)[None],2,axis=0)
        for i,theta in enumerate(np.deg2rad([179,-179])):
            c,s=np.cos(theta),np.sin(theta);tf[i,:3,:3]=[[c,-s,0],[s,c,0],[0,0,1]]
        result,_,_=dual_quaternion_binding(np.array([[1.,0,0]]),np.array([[.5,.5]]),tf)
        np.testing.assert_allclose(result,[[-1,0,0]],atol=2e-7)


if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TriangleCrossings))
    if not result.wasSuccessful():raise RuntimeError('Triangle intersection regression failed')
