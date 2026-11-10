"""Small synthetic_test_only CPU checks for the new conservative ROI planner."""
import unittest
import numpy as np
from densekp_retrain.imaging.roi_planning import plan_native_crop_region,border_coordinates
from densekp_retrain.imaging.regions import require_crop_in_region
from densekp_retrain.data.boxes import crop_from_box

class ContinuousROITests(unittest.TestCase):
    def test_visible_subboxes_have_full_square_support(self):
        parts={'body':np.array([[3900,1100],[4250,1850]]),'hair':np.array([[4050,1080],[4160,1160]])}
        region=plan_native_crop_region(parts,[5760,2880])['roi_xyxy']
        rng=np.random.default_rng(7)
        for _ in range(40):
            xy=rng.uniform([3900,1100],[4250,1850],(2,2));lo=xy.min(0);hi=xy.max(0)
            if (hi-lo<2).any():continue
            center,scale=crop_from_box(np.r_[lo,hi],image_size_wh=(5760,2880),boundary_policy='erp_reject_seam',crop_policy='deployment',min_side_px=2)
            self.assertTrue(require_crop_in_region(center,scale,region,(5760,2880)))
    def test_region_follows_actual_geometry(self):
        xy=np.array([[3900,1100],[4250,1850]])
        a=plan_native_crop_region({'body':xy},[5760,2880])['roi_xyxy']
        b=plan_native_crop_region({'body':xy+[100,20]},[5760,2880])['roi_xyxy']
        np.testing.assert_array_equal(np.array(b)-a,[100,20,100,20])
    def test_seam_and_nonfinite_and_border_rejection(self):
        for xy in [np.array([[1,1000],[5759,1300]]),np.array([[np.nan,0],[30,40]]),np.array([[10,10],[30,500]])]:
            with self.assertRaises(ValueError):plan_native_crop_region({'body':xy},[5760,2880])
    def test_integer_border_mapping(self):
        r=[3855,1091,4795,2031];b=border_coordinates(r,[5760,2880])
        np.testing.assert_array_equal(np.floor(np.array(b)*[5760,5760,2880,2880]),[3855,4795,849,1789])

if __name__=='__main__':unittest.main()
