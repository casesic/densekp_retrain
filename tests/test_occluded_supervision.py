"""Occluded GT is supervised; off-lens GT is not; rejected predictions still score."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase, mock

import numpy as np
from PIL import Image
import torch
from torch.utils.data import default_collate

from densekp.core.configs import DATASET_FILES, DATASET_FOLDERS
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.densekp_trainer import DenseKP
from densekp.core.keypoint_contract import validate_annotations
from densekp.core.keypoint_objective import keypoint_loss
from densekp.core.keypoint_supervision import SUPERVISION_POLICY, make_targets, points_in_mask
from tools.annotation.build_omni_multiperson_tiny import build
from densekp.core.evaluation import evaluation_masks, pair_masks, prediction_retention, summarize
from densekp.core.datasets.augmentation import Augmentation
from densekp.core.datasets.configuration import audit_cfg


class OccludedSupervisionTests(TestCase):
    def test_loader_recomputes_eligibility_after_augmentation_and_mixes_old_rows(self):
        xy = np.full((138, 2), 128., dtype=np.float32)
        xy[:4] = [[128, 128], [20, 20], [300, 128], [148, 128]]
        in_view = np.zeros(138, dtype=bool)
        in_view[:3] = True
        visible = np.zeros(138, dtype=bool)
        visible[1] = True
        target = make_targets(xy, in_view, [128, 128], 1.28)
        data = dict(imgname=np.array(['image.png']), center=np.array([[128., 128.]]),
                    scale=np.array([1.28], dtype=np.float32), proj_verts=target[None])
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.new('RGB', (384, 256), 'black').save(root/'image.png')
            np.savez(root/'old.npz', **data)
            np.savez(root/'new.npz', **data, supervision_policy=SUPERVISION_POLICY,
                     in_view=in_view[None], visible=visible[None])
            cfg = audit_cfg('supervision_weight')
            with mock.patch.dict(DATASET_FOLDERS, {'old': str(root), 'new': str(root)}), \
                 mock.patch.dict(DATASET_FILES[True], {'old': str(root/'old.npz'), 'new': str(root/'new.npz')}):
                old = DatasetKeypoints(cfg, 'old', use_augmentation=False)[0]
                ds = DatasetKeypoints(cfg, 'new', use_augmentation=False)
                new = ds[0]
                np.testing.assert_array_equal(new['proj_verts_cropped'][:4, 2], [1, 1, 0, 0])
                self.assertFalse(new['visible'][0])  # hidden, but supervised
                batch = default_collate([old, new])
                self.assertFalse(batch['visibility_known'][0].any())
                self.assertTrue(batch['visibility_known'][1].all())
                np.testing.assert_array_equal(old['proj_verts_cropped'][:, 2], target[:, 2])
                ds.use_augmentation = True
                sampled = (1., 0., False, False, 0, [1., 1., 1.], .25, 0.)
                with mock.patch('densekp.core.datasets.crop_pipeline.sample_augmentation', return_value=Augmentation(*sampled)):
                    augmented = ds[0]
                np.testing.assert_array_equal(augmented['proj_verts_cropped'][:4, 2], [1, 0, 1, 0])
                for with_var in (False, True):
                    prediction = torch.full((1, 138, 3), .1, requires_grad=True)
                    loss, _ = keypoint_loss(prediction, torch.from_numpy(new['proj_verts_cropped'][None]), with_var)
                    loss.backward()
                    self.assertGreater(prediction.grad[0, 0, :2].abs().sum().item(), 0)
                    self.assertEqual(prediction.grad[0, 2:].abs().sum().item(), 0)

    def test_ds_adapter_uses_lens_mask_instead_of_visibility_or_rgb_color(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root/'source'; source.mkdir()
            (source/'source').mkdir()
            camera = source/'cam2'; camera.mkdir()
            Image.new('RGB', (64, 64), 'black').save(source/'source/cam2_preview.png')
            lens = np.zeros((64, 64), dtype=np.uint8); lens[:, 24:] = 255
            Image.fromarray(lens).save(camera/'erp_valid_mask.png')
            person = np.zeros_like(lens); person[16:48, 16:48] = 255
            for pid in (1, 2):
                Image.fromarray(person).save(camera/f'erp_person_{pid}_mask.png')
            xy = np.full((2, 138, 2), 32., dtype=np.float32)
            xy[:, 0] = [20, 32]  # inside person bbox but outside lens footprint
            visible = np.ones((2, 138), dtype=bool); visible[:, :2] = False
            np.savez(camera/'erp_annotations.npz', person_ids=[1, 2], xy=xy,
                     visible=visible, valid=np.ones_like(visible))
            (source/'report.json').write_text(json.dumps(dict(
                camera_model='Double Sphere', people=[dict(id=1), dict(id=2)], results=[dict(camera='cam2')])) )
            build(source, root/'dataset', val_person_id=2)
            with np.load(root/'dataset/train.npz') as archive:
                data = dict(archive)
            validate_annotations(data, 'supervision_weight')
            self.assertEqual(data['proj_verts'][0, 0, 2], 0)
            self.assertEqual(data['proj_verts'][0, 1, 2], 1)
            self.assertFalse(data['visible'][0, 1])

    def test_invalid_predictions_do_not_remove_gt_from_metric_denominator(self):
        in_view = np.array([True, True, False])
        visible = np.array([True, False, False])
        groups = evaluation_masks(in_view, visible, np.zeros((3, 2)))
        lens = np.zeros((8, 8), dtype=bool); lens[2:6, 2:6] = True
        valid_prediction = points_in_mask(np.array([[3., 3.], [0., 0.], [np.nan, 4.]]), lens)
        np.testing.assert_array_equal(valid_prediction, [True, False, False])
        errors = np.array([1., 99., 1000.])
        self.assertEqual(summarize(errors[groups['all_in_view']])['mean'], 50.)
        self.assertEqual(summarize(errors[groups['occluded']])['count'], 1)
        self.assertEqual(prediction_retention(in_view, valid_prediction)['rejection_rate'], .5)
        self.assertIsNone(prediction_retention(np.zeros(3, bool), valid_prediction)['rejection_rate'])

    def test_pair_visibility_groups_partition_gt_eligible_points(self):
        groups = pair_masks(np.array([1, 1, 1, 0], bool), np.array([1, 1, 0, 0], bool),
                            np.array([1, 1, 1, 1], bool), np.array([1, 0, 0, 1], bool))
        np.testing.assert_array_equal(groups['both_visible'], [1, 0, 0, 0])
        np.testing.assert_array_equal(groups['one_visible'], [0, 1, 0, 0])
        np.testing.assert_array_equal(groups['both_occluded'], [0, 0, 1, 0])

    def test_training_validation_reports_visible_and_occluded_separately(self):
        prediction = torch.zeros((1, 3, 3))
        target = torch.tensor([[[1., 0., 1.], [3., 0., 1.], [999., 0., 0.]]])
        batch = dict(proj_verts_cropped=target, visible=torch.tensor([[True, False, False]]),
                     visibility_known=torch.ones((1, 3), dtype=torch.bool))
        logged = {}
        dummy = SimpleNamespace(_evaluation_totals={},
                                forward_step=lambda *a, **k: dict(pred_keypoints=prediction),
                                log=lambda name, value, **k: logged.update({name: value}))
        DenseKP._evaluation_step(dummy, batch, 0, 'val')
        DenseKP._evaluation_epoch_end(dummy, 'val')
        self.assertEqual(logged['val_loss'].item(), 2.)
        self.assertEqual(logged['val/crop_l2_visible_0'].item(), 1.)
        self.assertEqual(logged['val/crop_l2_occluded_0'].item(), 3.)

    def test_policy_requires_explicit_visibility_and_in_view_metadata(self):
        data = dict(imgname=np.array(['image.png']), center=np.array([[128.,128.]]),
                    scale=np.array([1.28]), proj_verts=np.ones((1,138,3), dtype=np.float32),
                    supervision_policy=np.array(SUPERVISION_POLICY))
        with self.assertRaisesRegex(ValueError, 'in_view'):
            validate_annotations(data, 'supervision_weight')
