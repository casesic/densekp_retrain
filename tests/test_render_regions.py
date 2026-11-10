"""Small synthetic_test_only region fixtures; no renderer, model or updates."""
import unittest
import numpy as np
from densekp_retrain.imaging.regions import (checked_region,embed_region,region_mask,
    require_crop_in_region,require_point_neighborhoods,apply_physical_mask)
from densekp.core.keypoint_supervision import supervision_for_crop
from densekp_retrain.imaging.direct_erp import direct_annotations


class RegionTests(unittest.TestCase):
    def test_offset_and_unknown_outside(self):
        a=np.arange(6,dtype=np.float32).reshape(2,3)
        out=embed_region(a,[4,3,7,5],[12,8],unknown_value=np.nan)
        np.testing.assert_array_equal(out[3:5,4:7],a)
        self.assertTrue(np.isnan(out[0,0]));self.assertEqual(np.isfinite(out).sum(),6)

    def test_bad_rectangle_and_shape(self):
        for r in [[-1,0,4,3],[0,0,13,3],[4,2,3,5],[0,0,np.nan,3],[0,.5,2,3]]:
            with self.assertRaises(ValueError):checked_region(r,[12,8])
        with self.assertRaises(ValueError):embed_region(np.zeros((2,3)),[0,0,4,2],[12,8],unknown_value=np.nan)

    def test_crop_margin_and_outside_rejected(self):
        self.assertTrue(require_crop_in_region([50,50],.2,[28,28,72,72],[100,100]))
        with self.assertRaisesRegex(ValueError,'outside_high_quality'):
            require_crop_in_region([50,50],.21,[28,28,72,72],[100,100])
        with self.assertRaises(ValueError):require_crop_in_region([np.nan,50],.2,[0,0,100,100],[100,100])

    def test_visibility_neighborhood(self):
        self.assertTrue(require_point_neighborhoods(np.array([[6.2,5.1]]),[4,3,9,8],[12,10]))
        with self.assertRaisesRegex(ValueError,'neighborhood_outside'):
            require_point_neighborhoods(np.array([[6.2,5.1]]),[4,3,8,8],[12,10])

    def test_physical_domain_is_not_render_coverage(self):
        coverage=region_mask([2,2,6,6],[8,8]);valid=np.ones((8,8),bool);valid[3,3]=False
        rgb=np.full((8,8,3),25,np.uint8);out=apply_physical_mask(rgb,valid)
        self.assertTrue((out[0,0]==25).all());self.assertFalse(coverage[0,0]);self.assertTrue((out[3,3]==0).all())

    def test_invalid_mask(self):
        for valid in [np.ones((8,8),np.uint8),np.ones((7,8),bool)]:
            with self.assertRaises(ValueError):apply_physical_mask(np.zeros((8,8,3),np.uint8),valid)

    def test_occluded_reliable_point_stays_supervised(self):
        xy=np.array([[0,0],[.5,0],[0,0]],np.float32);inside=np.array([True,True,False])
        np.testing.assert_array_equal(supervision_for_crop(inside,xy),[True,False,False])

    def test_direct_local_visibility_matches_full_with_unknown_exterior(self):
        world=np.tile([1.,0.,2.],(1,138,1));valid=np.ones((8,16),bool);pose=np.eye(4)
        depth=np.full((8,16),np.sqrt(5),np.float32);ids=np.ones((8,16),np.float32)
        a=direct_annotations(world,[1],pose,'cam2',valid,depth,ids)
        roi=[7,2,13,7];rd=embed_region(depth[2:7,7:13],roi,(16,8),unknown_value=np.nan);ri=embed_region(ids[2:7,7:13],roi,(16,8),unknown_value=np.nan)
        b=direct_annotations(world,[1],pose,'cam2',valid,rd,ri,coverage_region=roi)
        np.testing.assert_array_equal(a['xy'],b['xy']);np.testing.assert_array_equal(a['visible'],b['visible'])
        self.assertTrue(b['visible'].all());self.assertTrue(np.isnan(ri[0,0]))
        with self.assertRaisesRegex(ValueError,'neighborhood_outside'):
            direct_annotations(world,[1],pose,'cam2',valid,rd,ri,coverage_region=[8,2,13,7])

if __name__=='__main__':unittest.main()
