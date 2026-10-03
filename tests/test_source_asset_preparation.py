"""Small source-preparation contracts with analytic inputs; no production assets."""

import ast
import builtins
import copy
import json
from pathlib import Path
import pickle
import random
import sys
import tempfile
import unittest

import numpy as np
import torch

from densekp_retrain.body.parameters import parameters_from_pose
from densekp_retrain.clothing.preparation import prepare_simple_clothing
from densekp_retrain.clothing.groom_transfer import transfer_groom
from tools.experiments.adult_candidate_recipe import STYLES


def rng_state():
    return pickle.dumps(
        (random.getstate(), np.random.get_state(), torch.get_rng_state().numpy()),
        protocol=4,
    )


def _write_groom_input(root, dtype='float64', variant='ok'):
    root.mkdir()
    reference = root / 'reference'
    geometry = root / 'geometry'
    source = root / 'source'
    binding = root / 'binding'
    for p in (reference, geometry, source, binding):
        p.mkdir()
    rest = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], np.float64)
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
    body = rest @ rotation.T + np.array([0.125, 0.25, 0.5])
    np.save(reference / 'vertices.npy', rest[None])
    np.save(geometry / 'vertices.npy', body[None])
    np.save(geometry / 'faces.npy', np.array([[0, 1, 2], [1, 3, 2]], np.int32))
    offsets = np.array([0, 51, 102], np.int64)
    local = np.tile([0.125, 0.125, 0.0625], (102, 1)).astype(np.float32)
    local[:, 1] += np.arange(102) / 4096
    arrays = dict(body_faces=np.array([[0, 1, 2], [0, 1, 2]], np.int32), weights=np.array([[1, 0, 0], [0.5, 0.25, 0.25]], np.float32), curve_offsets=offsets, target_rest_frames=np.repeat(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=dtype)[None], 2, axis=0), local_points=local)
    if variant == 'bad_offsets':
        arrays['curve_offsets'] = np.array([0, 51, 50], np.int64)
    if variant == 'bad_weights':
        arrays['weights'] = np.ones((2, 2), np.float32)
    if variant != 'missing_binding':
        np.savez(binding / 'binding.npz', **arrays)
    np.save(source / 'collision_correction.npy', np.tile([0, 1 / 1024, 1 / 512], (102, 1)).astype(np.float32))
    if variant != 'missing_radius':
        np.save(source / 'radius.npy', np.linspace(0.001, 0.002, 102, dtype=np.float32))
    np.save(source / 'active_curve_mask.npy', np.array([variant != 'inactive', False], bool))
    info = dict(binding=str(binding), status='frozen', split='integration_only', source_record='synthetic-independent', head_shape='unchanged')
    (source / 'groom_manifest.json').write_text(json.dumps(info))
    return (dict(reference=dict(geometry=str(reference), groom_geometry=str(source))), geometry)


class SourceAssetPreparationTests(unittest.TestCase):
    def test_pose_blocks_preserve_dtype_alias_and_errors(self):
        blocks = [
            ('global_orient', 0, 3), ('body_pose', 3, 66), ('jaw_pose', 66, 69),
            ('leye_pose', 69, 72), ('reye_pose', 72, 75),
            ('left_hand_pose', 75, 120), ('right_hand_pose', 120, 165),
        ]
        for dtype in (np.float32, np.float64):
            with self.subTest(dtype=dtype):
                pose = np.arange(330, dtype=dtype)[::2]
                beta = np.arange(60, dtype=dtype)[::2]
                pelvis = np.array([1, 2, 3], dtype=dtype)
                before = [x.copy() for x in (pose, beta, pelvis)]
                rng = rng_state()
                result = parameters_from_pose(pose, beta, pelvis)
                self.assertEqual(rng_state(), rng)
                self.assertEqual(list(result), [x[0] for x in blocks] + ['betas', 'transl'])
                self.assertIs(result['betas'], beta)
                for name, start, end in blocks:
                    value = result[name]
                    self.assertEqual(value.shape, (1, end - start))
                    self.assertEqual(value.dtype, np.dtype('float32'))
                    np.testing.assert_array_equal(value[0], 2 * np.arange(start, end))
                    self.assertEqual(np.shares_memory(value, pose), dtype == np.float32)
                np.testing.assert_array_equal(result['transl'], [[-1, -2, -3]])
                self.assertEqual(result['transl'].dtype, pelvis.dtype)
                for value, old in zip((pose, beta, pelvis), before):
                    np.testing.assert_array_equal(value, old)
                result['betas'][0] = 7
                self.assertEqual(beta[0], 7)
                if dtype == np.float32:
                    result['body_pose'][0, 0] = 99
                    self.assertEqual(pose[3], 99)
        beta = [1., 2.]
        result = parameters_from_pose(np.arange(73), beta, np.zeros(3))
        self.assertIs(result['betas'], beta)
        self.assertEqual(result['reye_pose'].shape, (1, 1))
        self.assertEqual(result['left_hand_pose'].shape, (1, 0))
        self.assertEqual(result['right_hand_pose'].shape, (1, 0))
        with self.assertRaises(TypeError):
            parameters_from_pose(np.arange(165), beta, [1, 2, 3])
        with self.assertRaises(TypeError):
            parameters_from_pose(np.arange(165), beta, None)
        # Pose conversion fails before the invalid pelvis is indexed.
        with self.assertRaises(ValueError):
            parameters_from_pose(['invalid'] * 165, beta, None)

    def test_clothing_artifacts_and_failure_order(self):
        rest = np.array([[0, .2, 0], [.3, .2, 0], [0, .5, 0]], np.float32)
        joints = np.zeros((22, 3), np.float32)
        joints[12, 1] = 1
        joints[[16, 20], 0] = 1
        joints[[17, 21], 0] = -1
        style = {'sample': (.34, .36, 'sport', [.1, .2, .3], [.4, .5, .6], [.7, .8, .9])}
        original_style = copy.deepcopy(style)
        original_rest = rest.copy()
        original_joints = joints.copy()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            geometry = root / 'geometry'
            geometry.mkdir()
            destination = root / 'out'

            def save(key='sample'):
                return prepare_simple_clothing(
                    destination, geometry, rest, joints, key, 'controlled', style_table=style
                )

            with self.assertRaises(FileNotFoundError) as caught:
                save('unknown')
            self.assertTrue(caught.exception.filename.endswith('vertices.npy'))
            self.assertFalse(destination.exists())
            np.save(geometry / 'vertices.npy', rest[None])
            with self.assertRaises(FileNotFoundError) as caught:
                save('unknown')
            self.assertTrue(caught.exception.filename.endswith('faces.npy'))
            np.save(geometry / 'faces.npy', np.array([[0, 1, 2]], np.int32))
            with self.assertRaises(KeyError):
                save('unknown')
            self.assertFalse(destination.exists())
            rng = rng_state()
            self.assertEqual(save(), str(destination))
            self.assertEqual(rng_state(), rng)
            expected = rest.copy()
            expected[:, 2] = .004
            expected_arrays = {
                'clothing_vertices': expected[None],
                'clothing_faces': np.array([[0, 1, 2]], np.int32),
                'body_barycentric_indices': np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2]], np.int32),
                'body_barycentric_weights': np.tile(np.array([1, 0, 0], np.float32), (3, 1)),
                'reference_body_vertices': rest[None],
                'clothing_rest_vertices': rest,
                'face_material_ids': np.zeros(1, np.int32),
            }
            self.assertEqual(
                {p.name for p in destination.iterdir()},
                {k + '.npy' for k in expected_arrays} |
                {'materials.json', 'garment_material.json', 'clothing_manifest.json'},
            )
            for name, expected in expected_arrays.items():
                actual = np.load(destination / (name + '.npy'), allow_pickle=False)
                self.assertEqual(actual.dtype, expected.dtype)
                self.assertEqual(actual.shape, expected.shape)
                np.testing.assert_array_equal(actual, expected)
            materials = json.loads((destination / 'materials.json').read_text())
            self.assertEqual(materials, [
                {'name': 'top', 'color': [.1, .2, .3, 1]},
                {'name': 'pants', 'color': [.4, .5, .6, 1]},
                {'name': 'panel', 'color': [.7, .8, .9, 1]},
            ])
            self.assertEqual([list(x) for x in materials], [['name', 'color']] * 3)
            garment = json.loads((destination / 'garment_material.json').read_text())
            self.assertEqual(garment['profile'], dict(sleeve=.34, leg=.36, collar=.045, style='sport'))
            manifest = json.loads((destination / 'clothing_manifest.json').read_text())
            extra = dict(
                body_geometry=str(geometry), character_id='controlled', split='integration_only',
                method='existing clipped 4mm fitted shell with anatomical sleeve/leg lengths',
                limitation='fitted simple clothes; not straight trousers, polo or physically simulated cloth',
                appearance_status='diagnostic_pending', legacy_clearance_gate_m=.0004,
                legacy_offset_gate_m=.01,
            )
            self.assertEqual(manifest, dict(garment, **extra))
            self.assertEqual(list(manifest), list(garment) + list(extra))
            frozen = {p.name: p.read_bytes() for p in destination.iterdir()}
            with self.assertRaises(FileExistsError):
                save()
            self.assertEqual(frozen, {p.name: p.read_bytes() for p in destination.iterdir()})
            partial = root / 'partial'
            invalid = {'sample': (*style['sample'][:3], (.1, .2, .3), *style['sample'][4:])}
            with self.assertRaises(TypeError):
                prepare_simple_clothing(partial, geometry, rest, joints, 'sample', 'controlled', style_table=invalid)
            self.assertEqual({p.name for p in partial.iterdir()}, {k + '.npy' for k in expected_arrays})
        self.assertEqual(style, original_style)
        np.testing.assert_array_equal(rest, original_rest)
        np.testing.assert_array_equal(joints, original_joints)

    def test_recipe_import_is_data_only(self):
        import tools.experiments.adult_candidate_recipe as recipe
        source = Path(recipe.__file__).read_text()
        tree = ast.parse(source)
        assignments = [n for n in tree.body if isinstance(n, ast.Assign)]
        self.assertEqual(len(assignments), 1)
        # Only a docstring and a literal mapping: no implicit executable recipe.
        self.assertEqual(len(tree.body), 2)
        self.assertEqual(ast.literal_eval(assignments[0].value), STYLES)
        self.assertEqual(list(STYLES), ['F01','F02','F03','F04','F05','M01','M02','M03','M04','M05'])
        for value in STYLES.values():
            self.assertIsInstance(value, tuple)
            self.assertTrue(all(isinstance(c, list) for c in value[3:]))

        def reject(*args, **kwargs):
            raise AssertionError('recipe must not import or perform IO')

        safe_builtins = dict(vars(builtins), __import__=reject, open=reject)
        namespace = {'__builtins__': safe_builtins}
        before = rng_state(), list(sys.path)
        exec(compile(source, recipe.__file__, 'exec'), namespace)
        self.assertEqual((rng_state(), list(sys.path)), before)
        self.assertEqual(namespace['STYLES'], STYLES)


    def test_groom_rigid_transfer_and_binding_contract(self):
        # Two bound curves on a planar square, posed by an exact quarter turn.
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], np.float64)
        translation = np.array([.125, .25, .5])
        for dtype in ('float32', 'float64'):
            with self.subTest(frame_dtype=dtype), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest, geometry = _write_groom_input(root / 'input', dtype)
                original = copy.deepcopy(manifest)
                source = Path(manifest['reference']['groom_geometry'])
                info = json.loads((source / 'groom_manifest.json').read_text())
                with np.load(Path(info['binding']) / 'binding.npz') as binding:
                    local = binding['local_points']
                    offsets = binding['curve_offsets']
                    self.assertEqual(binding['target_rest_frames'].dtype, np.dtype(dtype))
                output = root / 'out'
                before = rng_state()
                self.assertEqual(transfer_groom(manifest, geometry, output), str(output))
                self.assertEqual(rng_state(), before)
                self.assertEqual(manifest, original)
                roots = np.array([[0, 0, .0004], [.25, .25, .0004]])
                posed_roots = roots @ rotation.T + translation
                correction = np.load(source / 'collision_correction.npy').astype(np.float64) @ rotation.T
                expected = (np.repeat(roots, [51, 51], axis=0) + local) @ rotation.T + translation + correction
                arrays = {
                    'points': expected.astype(np.float32),
                    'target_roots': posed_roots,
                    'collision_correction': correction,
                    'curve_offsets': offsets,
                    'radius': np.load(source / 'radius.npy'),
                    'active_curve_mask': np.array([True, False]),
                }
                for name, expected in arrays.items():
                    actual = np.load(output / (name + '.npy'), allow_pickle=False)
                    self.assertEqual(actual.dtype, expected.dtype)
                    self.assertEqual(actual.shape, expected.shape)
                    np.testing.assert_array_equal(actual, expected)
                report = json.loads((output / 'groom_manifest.json').read_text())
                self.assertEqual(report['sample_count'], 2)  # points 0, 50; 100 is inactive.
                self.assertEqual(report['sample_below_minus_1mm'], 0)
                # The plane normal is +Z; each sampled point has the same height.
                self.assertEqual(report['sample_min_signed_m'], float(arrays['points'][0, 2] - .5))
                for key in ['binding', 'source_record', 'head_shape']:
                    self.assertEqual(report[key], info[key])
                self.assertEqual(report['body_geometry'], str(geometry))
                self.assertEqual(report['parent_groom'], str(source))
                self.assertEqual(report['status'], 'diagnostic_pending')
                self.assertEqual(report['split'], 'integration_only')
                self.assertEqual(list(report), list(info) + [
                    'body_geometry', 'parent_groom', 'transfer', 'sample_min_signed_m',
                    'sample_below_minus_1mm', 'sample_count',
                ])
                self.assertEqual({p.name for p in output.iterdir()},
                                 {name + '.npy' for name in arrays} | {'groom_manifest.json'})

    def test_groom_failure_order_and_partial_outputs(self):
        cases = [
            ('missing_binding', FileNotFoundError, 'binding.npz', 0, False),
            ('bad_offsets', ValueError, 'negative', 0, False),
            ('bad_weights', ValueError, 'broadcast', 0, False),
            ('inactive', ValueError, 'concatenate', 6, True),
            ('existing', FileExistsError, 'exists', 0, True),
            ('missing_radius', FileNotFoundError, 'radius.npy', 0, True),
        ]
        for variant, error, message, count, exists in cases:
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                manifest, geometry = _write_groom_input(root / 'input', variant=variant)
                output = root / 'out'
                if variant == 'existing':
                    output.mkdir()
                before = rng_state()
                with self.assertRaisesRegex(error, message):
                    transfer_groom(manifest, geometry, output)
                self.assertEqual(rng_state(), before)
                self.assertEqual(output.exists(), exists)
                files = {p.name for p in output.glob('*')}
                self.assertEqual(len(files), count)
                if variant == 'inactive':
                    self.assertEqual(files, {name + '.npy' for name in (
                        'points', 'target_roots', 'collision_correction',
                        'radius', 'curve_offsets', 'active_curve_mask',
                    )})
                    np.testing.assert_array_equal(np.load(output / 'active_curve_mask.npy'), [False, False])
