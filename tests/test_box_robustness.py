import unittest
import numpy as np
from tools.experiments.prepare_box_robustness import change_box
from densekp.core.crop_geometry import gen_trans_from_patch_cv, restore_crop_xy, trans_points2d_parallel
from densekp.core.keypoint_supervision import inside_full_crop


class BoxGeometryTest(unittest.TestCase):
    def test_shift_changes_crop_membership_not_erp_target(self):
        box = [100.,100.,200.,300.]
        xy = np.array([[150.,200.],[35.,200.],[265.,200.]])
        masks = []
        for changed in [np.array(box),change_box(box,dx=.15)]:
            center=(changed[:2]+changed[2:])/2;side=max(changed[2:]-changed[:2])*1.2
            affine=gen_trans_from_patch_cv(*center,side,side,256,256,1,0)
            target=trans_points2d_parallel(xy,affine)/256-.5
            np.testing.assert_allclose(restore_crop_xy(target,affine,(256,256)),xy,atol=2e-4)
            masks.append(inside_full_crop(target))
        self.assertTrue(masks[0][1]);self.assertFalse(masks[1][1])
        self.assertTrue(masks[1][0])

    def test_deployment_tight_and_asymmetric_boundary(self):
        box=np.array([100.,100.,200.,300.])
        tight=change_box(box,factor=1/1.2)
        self.assertAlmostEqual(max(tight[2:]-tight[:2])*1.2,200.)
        edge=change_box(box,edge=3,fraction=.35)
        np.testing.assert_array_equal(edge,[100.,100.,200.,230.])
        np.testing.assert_allclose((edge[:2]+edge[2:])/2,[150.,165.])


if __name__=='__main__':unittest.main()
