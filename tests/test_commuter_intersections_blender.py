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





if __name__=='__main__':
    result=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromTestCase(TriangleCrossings))
    if not result.wasSuccessful():raise RuntimeError('Triangle intersection regression failed')
