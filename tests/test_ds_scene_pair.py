"""Rig conventions for same-world, opposite-facing DS scene samples."""
import unittest
from dataclasses import replace
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from tools.fisheye import run_ds_scene_pair_pilot as pilot
from densekp_retrain.imaging import ds as processor
from densekp_retrain.imaging import source as source_buffers
from densekp_retrain.imaging.output_modes import (prepare_source_workspace,
    complete_output, record_failure, cleanup_owned_source)
from tools.fisheye.run_ds_scene_pair_pilot import rig_poses, point_in_view
from tools.fisheye.run_fisheye_multiperson_pilot import project_erp
from densekp_retrain.camera.double_sphere import DS_20260320, project_double_sphere


class ScenePairRigTests(unittest.TestCase):
    def test_output_mode_rejects_invalid_value(self):
        with self.assertRaises(ValueError):
            processor.validate_output_mode('small')

    def test_compact_and_full_compute_identical_deliverables(self):
        intrinsics = {
            name: replace(k, width=64, height=64, fx=k.fx/45, fy=k.fy/45,
                          cx=k.cx/45, cy=k.cy/45)
            for name, k in DS_20260320.items()
        }
        poses, _, _ = rig_poses([0, 0, 0], 0, .18)
        world = np.tile([[0., 2., 1.], [0., -2., 1.]], (69, 1))
        people = [dict(id=1, frame=0)]
        masks = {name: np.ones((32, 64), dtype=bool) for name in poses}
        lenses = {name: np.ones((64, 64), dtype=bool) for name in poses}
        with TemporaryDirectory() as temporary:
            roots = {mode: Path(temporary)/mode for mode in ('full','compact')}
            for root in roots.values():
                (root/'source').mkdir(parents=True)
                for name in poses:
                    Image.fromarray(np.full((32, 64, 3), 200, dtype=np.uint8)).save(
                        root/'source'/f'{name}_preview.png')
            with patch('densekp_retrain.camera.domain.DS_20260320', intrinsics), \
                 patch.object(processor, 'DS_20260320', intrinsics), \
                 patch.object(processor, 'world_points', return_value=world), \
                 patch.object(source_buffers, 'read_combined_depth_index', return_value=(
                     None, np.ones((32, 64)), np.ones((32, 64)), None)), \
                 patch.object(processor, 'erp_visibility', return_value=(None, np.ones(138, dtype=bool))):
                for mode, root in roots.items():
                    processor.process_scene(root, people, poses, masks,
                        np.zeros((1,1,3)), np.zeros((1,138,3)), lens_masks=lenses,
                        lens_metadata={name:{} for name in poses}, output_mode=mode)
            for camera in poses:
                for filename in ('normal_erp.png','valid_mask.png','person_1_mask.png'):
                    np.testing.assert_array_equal(
                        np.asarray(Image.open(roots['full']/camera/filename)),
                        np.asarray(Image.open(roots['compact']/camera/filename)))
                with np.load(roots['full']/camera/'annotations.npz') as full, \
                     np.load(roots['compact']/camera/'annotations.npz') as compact:
                    self.assertEqual(full.files, compact.files)
                    for key in full.files: np.testing.assert_array_equal(full[key],compact[key])
                self.assertTrue((roots['full']/camera/'overlay.png').is_file())
                self.assertFalse((roots['compact']/camera/'overlay.png').exists())
                self.assertFalse((roots['compact']/camera/'fisheye.png').exists())
                self.assertFalse((roots['compact']/camera/'reference_lens_mask.png').exists())
            with np.load(roots['full']/'pair_annotations.npz') as full, \
                 np.load(roots['compact']/'pair_annotations.npz') as compact:
                for key in full.files: np.testing.assert_array_equal(full[key],compact[key])
            self.assertTrue((roots['full']/'pair_review.jpg').is_file())
            self.assertFalse((roots['compact']/'pair_review.jpg').exists())

    def test_compact_source_lifecycle_cleans_only_owned_success(self):
        with TemporaryDirectory() as temporary:
            root=Path(temporary); output=root/'output'; output.mkdir()
            source=prepare_source_workspace(output,'compact')
            (source/'buffer.exr').write_bytes(b'x')
            with self.assertRaises(FileExistsError):
                prepare_source_workspace(output,'compact')
            complete_output(output,'compact',source,checks={'readback':True})
            self.assertFalse(source.parent.exists())
            status=json.loads((output/'generation_status.json').read_text())
            self.assertTrue(status['success']);self.assertEqual(status['cleanup'],'completed')
            failed=root/'failed';failed.mkdir()
            failed_source=prepare_source_workspace(failed,'compact')
            record_failure(failed,'compact',failed_source,'postprocess',RuntimeError('boom'))
            self.assertTrue(failed_source.exists())
            foreign=root/'foreign'/'source';foreign.mkdir(parents=True)
            (foreign/'keep').write_bytes(b'keep')
            with self.assertRaises(RuntimeError):cleanup_owned_source(output,foreign)
            self.assertTrue((foreign/'keep').is_file())

    def test_cleanup_failure_survives_outer_failure_recording(self):
        with TemporaryDirectory() as temporary:
            output=Path(temporary)/'output';output.mkdir()
            source=prepare_source_workspace(output,'compact')
            (source/'buffer.exr').write_bytes(b'x')
            checks={'readback':True,'packed_rows':6}
            cleanup_error=OSError('simulated cleanup denial')
            with patch('densekp_retrain.imaging.output_modes.shutil.rmtree',
                       side_effect=cleanup_error):
                with self.assertRaises(OSError):
                    complete_output(output,'compact',source,checks=checks)
            before=json.loads((output/'generation_status.json').read_text())
            self.assertFalse(before['success'])
            self.assertEqual(before['data_validation'],'passed')
            self.assertEqual(before['checks'],checks)
            self.assertEqual(before['cleanup'],'failed')
            self.assertEqual(before['cleanup_error'],
                             'OSError: simulated cleanup denial')
            self.assertEqual(Path(before['residual_path']),source.parent)
            self.assertTrue(before['residual_exists'])
            self.assertTrue(source.exists())
            returned=record_failure(output,'compact',source,'cleanup',cleanup_error)
            after=json.loads((output/'generation_status.json').read_text())
            self.assertEqual(after,before)
            self.assertEqual(returned,before)

    def test_rig_has_equal_height_opposite_axes_and_declared_baseline(self):
        erp, physical, _ = rig_poses([13.676, -4.25, 0], -17.2, .18)
        a, b = physical['cam2'], physical['cam4']
        self.assertAlmostEqual(a[2, 3], b[2, 3])
        self.assertAlmostEqual(np.linalg.norm(a[:3, 3] - b[:3, 3]), .18)
        np.testing.assert_allclose(a[:3, 2], -b[:3, 2], atol=1e-14)
        for pose in physical.values():
            np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-14)
            self.assertAlmostEqual(np.linalg.det(pose[:3, :3]), 1.)
        np.testing.assert_array_equal(erp['cam2'][:3, :3], erp['cam4'][:3, :3])

    def test_physical_camera_projection_agrees_with_normal_erp_axes(self):
        erp, physical, _ = rig_poses([0, 0, 0], 17, .18)
        points = np.array([[1., 0., 1.], [2., .1, 1.5], [-1., 0., .5]])
        mask = np.ones((288, 576), dtype=bool)
        for camera in ('cam2', 'cam4'):
            _, actual, _, domain, _ = point_in_view(points, erp[camera], camera, mask)
            pose = physical[camera]
            expected, expected_domain = project_double_sphere(
                (points - pose[:3, 3]) @ pose[:3, :3], DS_20260320[camera])
            np.testing.assert_allclose(actual, expected, atol=1e-10)
            np.testing.assert_array_equal(domain, expected_domain)

    def test_translation_changes_parallax_without_changing_world_points(self):
        poses, _, _ = rig_poses([0, 0, 0], 0, .18)
        points = np.array([[1., 0., 1.], [-1., 0., 1.]])
        first, _ = project_erp(points, poses['cam2'], 1536, 768)
        second, _ = project_erp(points, poses['cam4'], 1536, 768)
        self.assertGreater(float(np.abs(first - second).max()), 1.)
        poses, _, _ = rig_poses([0, 0, 0], 0, 0.)
        first, _ = project_erp(points, poses['cam2'], 1536, 768)
        second, _ = project_erp(points, poses['cam4'], 1536, 768)
        np.testing.assert_array_equal(first, second)

    def test_negative_or_nonfinite_baseline_is_rejected(self):
        for baseline in (-.1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                rig_poses([0, 0, 0], 0, baseline)

    def test_lens_mask_controls_rgb_instances_and_annotation_validity(self):
        # A fully blocked second lens must remove its labels/boxes as well as RGB.
        intrinsics = {
            name: replace(k, width=64, height=64, fx=k.fx/45, fy=k.fy/45,
                          cx=k.cx/45, cy=k.cy/45)
            for name, k in DS_20260320.items()
        }
        poses, _, _ = rig_poses([0, 0, 0], 0, .18)
        world = np.tile([[0., 2., 1.], [0., -2., 1.]], (69, 1))
        people = [dict(id=1, frame=0)]
        masks = {name: np.ones((32, 64), dtype=bool) for name in poses}
        lenses = {'cam2': np.ones((64, 64), dtype=bool),
                  'cam4': np.zeros((64, 64), dtype=bool)}
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output/'source').mkdir()
            for name in poses:
                Image.fromarray(np.full((32, 64, 3), 200, dtype=np.uint8)).save(
                    output/'source'/f'{name}_preview.png')
            with patch('densekp_retrain.camera.domain.DS_20260320', intrinsics), \
                 patch.object(processor, 'DS_20260320', intrinsics), \
                 patch.object(processor, 'world_points', return_value=world), \
                 patch.object(source_buffers, 'read_combined_depth_index', return_value=(
                     None, np.ones((32, 64)), np.ones((32, 64)), None)), \
                 patch.object(processor, 'erp_visibility', return_value=(None, np.ones(138, dtype=bool))):
                processor.process_scene(output, people, poses, masks, np.zeros((1, 1, 3)),
                                    np.zeros((1, 138, 3)), lens_masks=lenses,
                                    lens_metadata={name: {} for name in poses})
            with np.load(output/'cam2/annotations.npz') as a:
                self.assertTrue(a['visible'].any())
            with np.load(output/'cam4/annotations.npz') as a:
                self.assertTrue(a['surface_visible'].all())
                self.assertFalse(a['in_view'].any())
                self.assertFalse(a['visible'].any())
                self.assertTrue(a['out_of_view'].all())
                self.assertFalse(a['occluded_in_view'].any())
            for filename in ('fisheye.png', 'normal_erp.png', 'valid_mask.png', 'person_1_mask.png'):
                self.assertFalse(np.asarray(Image.open(output/'cam4'/filename)).any(), filename)
            self.assertIsNone(json.loads((output/'cam4/boxes.json').read_text())[0]['bbox_xyxy_exclusive'])

    def test_source_visibility_and_final_erp_use_independent_rasters(self):
        """Visibility stays on source pixels while delivery labels use final pixels."""
        intrinsics = {
            name: replace(k, width=64, height=64, fx=k.fx/45, fy=k.fy/45,
                          cx=k.cx/45, cy=k.cy/45)
            for name, k in DS_20260320.items()
        }
        poses, _, _ = rig_poses([0, 0, 0], 0, .18)
        world = np.tile([[0., 2., 1.], [0., -2., 1.]], (69, 1))
        people = [dict(id=1, frame=0)]
        masks = {name: np.ones((64, 128), dtype=bool) for name in poses}
        lenses = {name: np.ones((64, 64), dtype=bool) for name in poses}
        visibility_xy = []

        def record_visibility(xy, distance, depth, index, person_ids):
            visibility_xy.append(xy.copy())
            return np.ones(len(xy), dtype=bool), np.ones(len(xy), dtype=bool)

        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output/'source').mkdir()
            for name in poses:
                Image.fromarray(np.full((32, 64, 3), 200, dtype=np.uint8)).save(
                    output/'source'/f'{name}_preview.png')
            with patch('densekp_retrain.camera.domain.DS_20260320', intrinsics), \
                 patch.object(processor, 'DS_20260320', intrinsics), \
                 patch.object(processor, 'world_points', return_value=world), \
                 patch.object(source_buffers, 'read_combined_depth_index', return_value=(
                     None, np.ones((32, 64)), np.ones((32, 64)), None)), \
                 patch.object(processor, 'erp_visibility', side_effect=record_visibility):
                processor.process_scene(
                    output, people, poses, masks, np.zeros((1, 1, 3)),
                    np.zeros((1, 138, 3)), lens_masks=lenses,
                    lens_metadata={name: {} for name in poses}, output_size=(128, 64))
            self.assertEqual(np.asarray(Image.open(output/'cam2'/'normal_erp.png')).shape,
                             (64, 128, 3))
            with np.load(output/'cam2'/'annotations.npz') as data:
                final_xy = data['xy'].reshape(-1, 2)
            self.assertEqual(len(visibility_xy), 2)
            np.testing.assert_allclose(final_xy, visibility_xy[0] * 2, atol=1e-5)
