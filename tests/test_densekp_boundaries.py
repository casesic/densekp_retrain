"""Asset-free correctness regressions and a real CPU Lightning lifecycle.

Only the heavy network is substituted. The NPZ loader, crop, objective,
optimizer, entry point, callbacks, checkpoint save and resume are real.
"""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, mock

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from densekp.core.configs import DATASET_FILES, DATASET_FOLDERS, dataset_config
from densekp.core.crop_geometry import gen_trans_from_patch_cv, normalize_crop_xy, restore_crop_xy, trans_points2d_parallel
from densekp.core.datasets.dataset_keypoints import DatasetKeypoints
from densekp.core.datasets.datamodule_keypoints import DataModule
from densekp.core.datasets.crop_pipeline import CropBox, crop_example
from densekp.core.keypoint_contract import validate_annotations
from densekp.core.keypoint_objective import keypoint_loss, keypoint_error_totals
from densekp.core.training_config import validate_training_config
from densekp.core.datasets.augmentation import Augmentation, sample_augmentation
from densekp.core.datasets.configuration import audit_cfg

ROOT = Path(__file__).resolve().parents[1]


def annotations(n=3):
    points = np.ones((n, 138, 3), dtype=np.float32)
    points[..., :2] = (60, 40)
    return dict(imgname=np.array(['image.png'] * n), center=np.tile([60., 40.], (n, 1)),
                scale=np.full(n, .5, dtype=np.float32), proj_verts=points)


def config(directory):
    with initialize_config_dir(version_base='1.2', config_dir=str(ROOT / 'configs/hydra')):
        cfg = compose(config_name='train', overrides=['data=ds_multiperson_tiny', 'trainer=cpu'])
    OmegaConf.set_struct(cfg, False)
    cfg.paths.output_dir = str(directory)
    cfg.paths.root_dir = str(ROOT)
    cfg.paths.work_dir = str(ROOT)
    cfg.DATASETS.DATASETS_AND_RATIOS = 'test-fixture'
    cfg.DATASETS.VAL_DATASETS = 'test-fixture_test-fixture'
    cfg.DATASETS.CONFIG = OmegaConf.create(audit_cfg('supervision_weight').DATASETS.CONFIG.dump())
    cfg.TRAIN.BATCH_SIZE = 2
    cfg.GENERAL.NUM_WORKERS = 0
    cfg.GENERAL.CHECKPOINT_STEPS = 1
    cfg.GENERAL.LOG_STEPS = 1
    cfg.GENERAL.VAL_STEPS = 1
    cfg.GENERAL.TOTAL_STEPS = 2
    cfg.MODEL.with_var = False
    cfg.trainer.enable_model_summary = False
    return cfg


class GeometryContractTests(TestCase):
    def test_rotated_rectangular_crop_and_inverse_keep_third_field(self):
        center = (107., 51.)
        width, height, scale, angle = 160., 80., 1.2, 33.
        theta = np.deg2rad(angle)
        right = np.array([np.cos(theta), np.sin(theta)]) * width * scale / 2
        down = np.array([-np.sin(theta), np.cos(theta)]) * height * scale / 2
        xy = np.array([center, center + right, center + down, center - right])
        affine = gen_trans_from_patch_cv(*center, width, height, 192, 256, scale, angle)
        crop = trans_points2d_parallel(xy, affine)
        np.testing.assert_allclose(crop, [[96, 128], [192, 128], [96, 256], [0, 128]], atol=2e-5)
        normalized = normalize_crop_xy(crop, (192, 256))
        np.testing.assert_allclose(normalized, [[0, 0], [.5, 0], [0, .5], [-.5, 0]], atol=2e-7)
        np.testing.assert_allclose(restore_crop_xy(normalized, affine, (192, 256)), xy, atol=1e-10)
        points = np.column_stack([xy, [1, 0, .25, 1]]).astype(np.float32)
        result = crop_example(np.zeros((120, 240, 3), np.uint8), CropBox(*center, width, height),
                              points.copy(), points.copy(), patch_size=(192, 256))
        np.testing.assert_array_equal(result.dense_points[:, 2], points[:, 2])
        np.testing.assert_allclose(restore_crop_xy(result.dense_points[:, :2], result.affine, (192, 256)), xy, atol=1e-5)

    def test_image_and_label_agree_after_scale_rotation_translation(self):
        image = np.zeros((128, 192, 3), dtype=np.uint8)
        cv2.circle(image, (80, 60), 3, (255, 255, 255), -1)
        points = np.tile([80., 60., 1.], (138, 1)).astype(np.float32)
        cfg = audit_cfg('supervision_weight').DATASETS.CONFIG
        sampled = (1.2, 30., False, False, 0, [1., 1., 1.], .1, -.1)
        with mock.patch('densekp.core.datasets.crop_pipeline.sample_augmentation', return_value=Augmentation(*sampled)):
            result = crop_example(image, CropBox(96, 64, 128, 128), np.zeros((44, 3)),
                                  points, patch_size=(256, 256), do_augment=True, augm_config=cfg)
        intensity = result.display[..., 0].astype(float)
        yy, xx = np.indices(intensity.shape)
        centroid = np.array([(xx * intensity).sum(), (yy * intensity).sum()]) / intensity.sum()
        label_pixel = (result.dense_points[0, :2] + .5) * 256
        np.testing.assert_allclose(centroid, label_pixel, atol=.2)

    def test_enabled_flip_fails_before_it_can_corrupt_semantics(self):
        cfg = audit_cfg('supervision_weight').DATASETS.CONFIG
        cfg.DO_FLIP = True
        cfg.FLIP_AUG_RATE = 1.
        with self.assertRaisesRegex(ValueError, '138-point'):
            crop_example(np.zeros((64, 96, 3), np.uint8), CropBox(48, 32, 64, 64),
                         np.zeros((44, 3)), np.zeros((138, 3)), patch_size=(256, 256),
                         do_augment=True, augm_config=cfg)

    def test_translation_rate_zero_and_one(self):
        cfg = audit_cfg().DATASETS.CONFIG
        cfg.TRANS_FACTOR = .2
        with mock.patch('numpy.random.randn', return_value=1.):
            augmentation = sample_augmentation(cfg)
            self.assertEqual((augmentation.translation_x, augmentation.translation_y), (0., 0.))
            cfg.TRANS_AUG_RATE = 1.
            augmentation = sample_augmentation(cfg)
            self.assertEqual((augmentation.translation_x, augmentation.translation_y), (.2, .2))

    def test_antialias_only_blurs_downsampling(self):
        args = (np.zeros((120, 240, 3), np.uint8), 100, 60)
        for source_size, expected in [(512, True), (64, False)]:
            with mock.patch('densekp.core.datasets.image_processing.gaussian', side_effect=lambda x, **kw: x) as blur:
                crop_example(args[0], CropBox(*args[1:], source_size, source_size), np.zeros((44, 3)),
                             np.zeros((138, 3)), patch_size=(256, 256), use_skimage_antialias=True)
                self.assertEqual(blur.called, expected)

    def test_annotation_contract_requires_semantics_and_rejects_invalid_data(self):
        data = annotations()
        with self.assertRaisesRegex(ValueError, 'Declare'):
            validate_annotations(data, None)
        validate_annotations(data, 'supervision_weight')
        for key, value in [('proj_verts', np.ones((3, 137, 3))), ('center', np.ones((2, 2))),
                           ('scale', np.zeros(3))]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_annotations({**data, key: value}, 'supervision_weight')
        data['proj_verts'][0, 0] = (np.nan, np.inf, 0.)
        validate_annotations(data, 'supervision_weight')
        data['proj_verts'][0, 0, 2] = 1.
        with self.assertRaisesRegex(ValueError, 'Supervised xy'):
            validate_annotations(data, 'supervision_weight')


class ObjectiveTests(TestCase):
    def test_masked_undefined_targets_have_finite_loss_and_zero_gradient(self):
        for with_var in (False, True):
            prediction = torch.zeros((1, 2, 3), requires_grad=True)
            target = torch.tensor([[[.1, .2, 1.], [float('nan'), float('inf'), 0.]]])
            loss, _ = keypoint_loss(prediction, target, with_var)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(prediction.grad).all())
            self.assertEqual(prediction.grad[0, 1].abs().sum().item(), 0.)
        target[..., 2] = 0
        loss, _ = keypoint_loss(prediction, target, True)
        self.assertEqual(loss.item(), 0.)

    def test_half_precision_log_sigma_does_not_underflow_variance(self):
        prediction = torch.tensor([[[.1, .2, -10.]]], dtype=torch.float16, requires_grad=True)
        target = prediction.detach().float().clone()
        target[..., 2] = 1.
        loss, _ = keypoint_loss(prediction, target, True)
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_nonfinite_prediction_or_weight_is_an_error(self):
        pred = torch.zeros((1, 2, 3))
        target = torch.ones_like(pred)
        pred[0, 0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'prediction'):
            keypoint_loss(pred, target, False)
        pred.zero_()
        target[..., 2] = 2
        with self.assertRaisesRegex(ValueError, 'weights'):
            keypoint_loss(pred, target, False)

    def test_metric_is_person_mean_and_counts_empty_people(self):
        prediction = torch.zeros((3, 2, 3))
        target = torch.tensor([[[1., 0., 1.], [100., 0., 0.]],
                               [[3., 0., .1], [3., 0., .1]],
                               [[float('nan'), 0., 0.], [9., 0., 0.]]])
        torch.testing.assert_close(keypoint_error_totals(prediction, target), torch.tensor([4., 2., 1.]))

    def test_finite_legacy_variance_formula_is_preserved(self):
        generator = torch.Generator().manual_seed(18)
        prediction = torch.randn((3, 138, 3), generator=generator) * .2
        target = torch.randn((3, 138, 3), generator=generator)
        target[..., 2] = torch.randint(0, 2, (3, 138), generator=generator)
        weights = target[..., 2]
        sigma_squared = prediction[..., 2].exp().square()
        old = 60 * (((1000 * (prediction[..., :2] - target[..., :2]).square().sum(-1)
                     / (2 * sigma_squared) + sigma_squared.log()) * weights).sum() / weights.sum())
        actual, _ = keypoint_loss(prediction, target, True)
        torch.testing.assert_close(actual, old)


class TinyHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(3, 138 * 3)

    def forward(self, x):
        return {'pred_keypoints': self.linear(x).reshape(-1, 138, 3)}


def tiny_backbone():
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 3))


class LifecycleTests(TestCase):
    def test_configuration_requires_data_and_rejects_unsupported_controls(self):
        with initialize_config_dir(version_base='1.2', config_dir=str(ROOT / 'configs/hydra')):
            default = compose(config_name='train')
        with self.assertRaisesRegex(ValueError, 'semantics|SEMANTICS'):
            validate_training_config(default)
        cfg = config('/tmp/unused-densekp-tests')
        validate_training_config(cfg)
        cfg.trainer.accumulate_grad_batches = 2
        with self.assertRaisesRegex(ValueError, 'accumulation'):
            validate_training_config(cfg)
        cfg.trainer.accumulate_grad_batches = 1
        cfg.init_checkpoint = 'weights.ckpt'
        cfg.ckpt_path = 'resume.ckpt'
        with self.assertRaisesRegex(ValueError, 'OR'):
            validate_training_config(cfg)

    def test_loader_and_training_entrypoint_save_and_resume(self):
        from densekp.train_keypoints import train
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            cv2.imwrite(str(folder / 'image.png'), np.full((80, 120, 3), 100, np.uint8))
            np.savez(folder / 'train.npz', **annotations(4))
            np.savez(folder / 'val.npz', **annotations(3))
            backbone_file = folder / 'backbone.pt'
            torch.save({'state_dict': tiny_backbone().state_dict()}, backbone_file)
            cfg = config(folder / 'run')
            with mock.patch.dict(DATASET_FOLDERS, {'test-fixture': str(folder)}), \
                 mock.patch.dict(DATASET_FILES[True], {'test-fixture': str(folder / 'train.npz')}), \
                 mock.patch.dict(DATASET_FILES[False], {'test-fixture': str(folder / 'val.npz')}), \
                 mock.patch('densekp.core.densekp_trainer.VITPOSE_BACKBONE', str(backbone_file)), \
                 mock.patch('densekp.core.densekp_trainer.create_backbone', side_effect=tiny_backbone), \
                 mock.patch('densekp.core.densekp_trainer.build_keypoints_head', side_effect=TinyHead):
                ds = DatasetKeypoints(cfg, 'test-fixture', use_augmentation=False)
                with mock.patch('densekp.core.datasets.crop_pipeline.sample_augmentation', side_effect=AssertionError('augmentation disabled')):
                    self.assertEqual(ds[0]['img'].shape, (3, 256, 256))
                dm = DataModule(cfg, dataset_config())
                dm.setup('fit')
                self.assertEqual([len(b['img']) for b in dm.val_dataloader()[0]], [2, 1])
                self.assertEqual(len(dm.train_dataloader()['img']), 2)
                metrics, objects = train(cfg)
                self.assertEqual(objects['trainer'].global_step, 2)
                self.assertEqual(float(metrics['val/scored_people_0']), 3.)
                self.assertEqual(float(metrics['val/scored_people_1']), 3.)
                self.assertTrue(torch.isfinite(metrics['val_loss']))
                checkpoint = objects['callbacks'][0].last_model_path
                self.assertTrue(Path(checkpoint).is_file())
                state = torch.load(checkpoint, map_location='cpu')
                self.assertEqual(state['global_step'], 2)
                self.assertTrue(state['optimizer_states'][0]['state'])
                callback_state = state['callbacks'][objects['callbacks'][0].state_key]
                current_score = metrics['val_loss']
                torch.testing.assert_close(callback_state['current_score'], current_score)
                self.assertEqual(callback_state['last_validation_save_step'], 2)
                old = objects['model'].head.linear.weight.detach().clone()
                cfg.ckpt_path = checkpoint
                cfg.paths.output_dir = str(folder / 'resumed')
                cfg.GENERAL.TOTAL_STEPS = 4
                cfg.test = True
                metrics, objects = train(cfg)
                self.assertEqual(objects['trainer'].global_step, 4)
                self.assertFalse(torch.equal(old, objects['model'].head.linear.weight))
                self.assertEqual(float(metrics['test/scored_people_0']), 3.)
                self.assertTrue(torch.isfinite(metrics['test_loss']))
