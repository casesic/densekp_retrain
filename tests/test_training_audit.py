"""CPU lifecycle outcomes frozen before moving the historical audit callback."""
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from densekp.core.training_audit import CheckUpdates, probes
from tests.training_audit_fixture import quiet_contract


class TrainingAuditTests(unittest.TestCase):
    def test_probe_bounds_dtype_independence_and_complete_rng_state(self):
        model = torch.nn.Module()
        model.backbone = torch.nn.Linear(70, 70, dtype=torch.float64)
        model.head = torch.nn.Linear(3, 2, dtype=torch.float16)
        with torch.no_grad():
            model.backbone.weight.copy_(torch.arange(4900).reshape(70, 70))
            model.head.weight.copy_(torch.arange(6).reshape(2, 3))
        for training in (True, False):
            model.train(training)
            torch_rng = torch.get_rng_state().clone()
            numpy_rng = np.random.get_state()
            python_rng = random.getstate()
            values = probes(model)
            self.assertEqual(list(values), ['backbone', 'head'])
            self.assertEqual(model.training, training)
            self.assertTrue(torch.equal(torch.get_rng_state(), torch_rng))
            current_numpy_rng = np.random.get_state()
            self.assertEqual(current_numpy_rng[0], numpy_rng[0])
            np.testing.assert_array_equal(current_numpy_rng[1], numpy_rng[1])
            self.assertEqual(current_numpy_rng[2:], numpy_rng[2:])
            self.assertEqual(random.getstate(), python_rng)
            for name, size in [('backbone', 4096), ('head', 6)]:
                self.assertEqual(values[name].dtype, torch.float32)
                self.assertEqual(values[name].device.type, 'cpu')
                self.assertFalse(values[name].requires_grad)
                torch.testing.assert_close(values[name], torch.arange(size, dtype=torch.float32), rtol=0, atol=0)
            values['backbone'].fill_(-1)
            self.assertEqual(float(model.backbone.weight[0, 0]), 0.)

    def test_independent_lifecycle_and_gradient_failure_contract(self):
        fixtures = Path(__file__).parent / 'fixtures'
        with TemporaryDirectory() as directory:
            metadata, arrays = quiet_contract(CheckUpdates, probes, Path(directory))
        self.assertEqual(metadata, json.loads((fixtures / 'training_audit_v1.json').read_text()))
        frozen = json.loads((fixtures / 'training_audit_arrays_v1.json').read_text())
        self.assertEqual(set(arrays), set(frozen))
        for name, array in arrays.items():
            with self.subTest(array=name):
                record = frozen[name]
                previous = np.asarray(record['values'], dtype=record['dtype']).reshape(record['shape'])
                self.assertEqual(array.dtype, previous.dtype)
                self.assertEqual(array.shape, previous.shape)
                np.testing.assert_array_equal(array, previous)

    def test_active_subclass_uses_public_audit(self):
        from tools.experiments import train_source_isolated_trial

        # Numerical proof is the independent fixture above; this checks imports.
        self.assertIs(train_source_isolated_trial.CheckUpdates, CheckUpdates)
        self.assertIs(train_source_isolated_trial.probes, probes)
        self.assertIs(train_source_isolated_trial.RunAudit.__bases__[0], CheckUpdates)


if __name__ == '__main__':
    unittest.main()
