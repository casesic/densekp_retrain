import unittest
import numpy as np
from densekp_retrain.evaluation.candidates import summary, aggregate


class CandidateMetricsTest(unittest.TestCase):
    def test_failure_and_empty_groups_preserve_denominators(self):
        self.assertEqual(summary([]),dict(count=0,nonfinite_count=0,status='empty',mean=None,median=None,p90=None))
        failed=summary([2,np.inf,4])
        self.assertEqual(failed['count'],3)
        self.assertEqual(failed['nonfinite_count'],1)
        self.assertIsNone(failed['mean'])

    def test_point_and_person_means_are_distinct(self):
        cases=[dict(id='a',action='walking',category='control',scale_group='small'),
               dict(id='b',action='walking',category='edge',scale_group='large')]
        states={'a':dict(crop=np.array([1.,1.]),erp=np.array([2.,2.]),masks={'supervised':np.array([True,True])}),
                'b':dict(crop=np.array([10.,100.]),erp=np.array([20.,200.]),masks={'supervised':np.array([True,False])})}
        result=aggregate(cases,states)['all/supervised']
        self.assertEqual(result['point_weighted']['crop_256_px']['mean'],4)
        self.assertEqual(result['person_weighted']['crop_256_px']['mean'],5.5)
        states['b']['crop'][0]=np.inf
        result=aggregate(cases,states)['all/supervised']
        self.assertEqual(result['point_weighted']['crop_256_px']['count'],3)
        self.assertIsNone(result['person_weighted']['crop_256_px']['mean'])


if __name__=='__main__':unittest.main()
