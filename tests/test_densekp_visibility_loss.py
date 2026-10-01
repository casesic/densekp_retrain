from types import SimpleNamespace
import unittest

import torch

from densekp.core.densekp_trainer import DenseKP


class VisibilityLossTests(unittest.TestCase):
    def test_coordinate_only_trial_does_not_weight_by_prediction_value(self):
        dummy = SimpleNamespace(cfg=SimpleNamespace(MODEL=SimpleNamespace(with_var=False)))
        target = torch.tensor([[[0.2, 0.1, 1.0], [-0.1, 0.3, 1.0]]])
        first = torch.zeros((1, 2, 3), requires_grad=True)
        second = first.detach().clone()
        second[..., 2] = torch.tensor([-20.0, 20.0])
        second.requires_grad_()
        values = []
        for prediction in [first, second]:
            loss, _ = DenseKP.compute_loss(dummy, {'proj_verts_cropped': target},
                                          {'pred_keypoints': prediction})
            loss.backward()
            values.append(loss.detach())
            self.assertTrue(torch.equal(prediction.grad[..., 2], torch.zeros((1, 2))))
        self.assertTrue(torch.equal(values[0], values[1]))
        self.assertTrue(torch.equal(first.grad[..., :2], second.grad[..., :2]))

    def test_zero_weight_points_do_not_affect_keypoint_loss(self):
        dummy = SimpleNamespace(cfg=SimpleNamespace(MODEL=SimpleNamespace(with_var=False)))
        prediction = torch.zeros((1, 2, 3), dtype=torch.float32)
        ground_truth = torch.tensor([[[1.0, 0.0, 1.0], [100.0, 100.0, 0.0]]])
        loss, _ = DenseKP.compute_loss(dummy, {"proj_verts_cropped": ground_truth},
                                       {"pred_keypoints": prediction})
        self.assertTrue(torch.isclose(loss, torch.tensor(60.0)))


    def test_legacy_all_visible_loss_is_unchanged(self):
        dummy = SimpleNamespace(cfg=SimpleNamespace(MODEL=SimpleNamespace(with_var=False)))
        prediction = torch.zeros((1, 2, 3), dtype=torch.float32)
        ground_truth = torch.tensor([[[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]]])
        loss, _ = DenseKP.compute_loss(dummy, {"proj_verts_cropped": ground_truth},
                                       {"pred_keypoints": prediction})
        self.assertTrue(torch.isclose(loss, torch.tensor(150.0)))

    def test_zero_weight_points_are_masked_with_variance_head(self):
        dummy = SimpleNamespace(cfg=SimpleNamespace(MODEL=SimpleNamespace(with_var=True)))
        prediction = torch.zeros((1, 2, 3), dtype=torch.float32)
        ground_truth = torch.tensor([[[1.0, 0.0, 1.0], [100.0, 100.0, 0.0]]])
        loss, _ = DenseKP.compute_loss(dummy, {"proj_verts_cropped": ground_truth},
                                       {"pred_keypoints": prediction})
        self.assertTrue(torch.isclose(loss, torch.tensor(30000.0)))
