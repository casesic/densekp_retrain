import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from densekp_retrain.camera import (OmniRadtanIntrinsics, build_erp_to_omni_lut,
                                    omni_pixels_to_erp, project_points_omni_radtan,
                                    unproject_pixels_omni_radtan)

class OmniRadtanTest(unittest.TestCase):
    def setUp(self): self.k=OmniRadtanIntrinsics(); self.pose=np.eye(4)
    def test_optical_axis(self):
        p=project_points_omni_radtan(np.array([[0.,0.,2.]]),self.pose,self.k)
        np.testing.assert_allclose(p,[[1500,1500]],atol=1e-12)
    def test_axis_symmetry(self):
        p=project_points_omni_radtan(np.array([[1.,0.,1.],[-1.,0.,1.],[0.,1.,1.],[0.,-1.,1.]]),self.pose,self.k)
        self.assertAlmostEqual(p[0,0]-1500,1500-p[1,0],places=10); self.assertAlmostEqual(p[2,1]-1500,1500-p[3,1],places=10)
    def test_roundtrip(self):
        rng=np.random.default_rng(17); rays=rng.normal(size=(1000,3)); rays[:,2]=np.abs(rays[:,2])+1; rays/=np.linalg.norm(rays,axis=1)[:,None]
        px=project_points_omni_radtan(rays,self.pose,self.k); back,valid=unproject_pixels_omni_radtan(px,self.k,return_valid=True)
        error=np.arccos(np.clip(np.sum(back[valid]*rays[valid],axis=1),-1,1)); self.assertLess(float(error.max()),3e-8)
    def test_fisheye_erp_lut_and_point_roundtrip(self):
        mx,my=build_erp_to_omni_lut(self.k,360,180)
        self.assertAlmostEqual(float(mx[90,180]),self.k.cx,places=3)
        self.assertAlmostEqual(float(my[90,180]),self.k.cy,places=3)
        target=np.array([[180.,90.],[200.,100.],[155.,75.]])
        fish=np.column_stack((mx[target[:,1].astype(int),target[:,0].astype(int)],
                              my[target[:,1].astype(int),target[:,0].astype(int)]))
        recovered,valid=omni_pixels_to_erp(fish,self.k,360,180)
        self.assertTrue(valid.all())
        np.testing.assert_allclose(recovered,target,atol=1e-4)

if __name__=='__main__': unittest.main()
