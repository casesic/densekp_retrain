"""Independent pre-migration outcomes; no models or local datasets required."""
import json
from pathlib import Path
import unittest

import numpy as np

from densekp.core import evaluation
from densekp.core.datasets.configuration import audit_cfg
from densekp_retrain.evaluation import candidates


FIXTURES = Path(__file__).parent / 'fixtures'


def decode(value):
    if isinstance(value, dict):
        if set(value) == {'array', 'dtype', 'shape'}:
            return np.asarray(value['array'], dtype=value['dtype']).reshape(value['shape'])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


class EvaluationContractTests(unittest.TestCase):
    def assert_exact(self, actual, expected):
        if isinstance(expected, np.ndarray):
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertEqual(actual.shape, expected.shape)
            np.testing.assert_array_equal(actual, expected)
        elif isinstance(expected, dict):
            self.assertEqual(list(actual), list(expected))
            for key in expected:
                self.assert_exact(actual[key], expected[key])
        elif isinstance(expected, list):
            self.assertEqual(len(actual), len(expected))
            for item, previous in zip(actual, expected):
                self.assert_exact(item, previous)
        else:
            self.assertEqual(actual, expected)

    def test_pre_migration_metric_outcomes(self):
        records = json.loads((FIXTURES / 'evaluation_contract_v1.json').read_text())
        self.assertEqual(len(records), 19)
        for record in records:
            with self.subTest(case=record['id']):
                module = candidates if record['function'] in ('summary', 'aggregate') else evaluation
                function = getattr(module, record['function'])
                arguments = decode(record['arguments'])
                outcome = decode(record['outcome'])
                if 'error_type' in outcome:
                    with self.assertRaises(ValueError) as raised:
                        function(*arguments)
                    self.assertEqual(type(raised.exception).__name__, outcome['error_type'])
                    self.assertEqual(str(raised.exception), outcome['message'])
                else:
                    self.assert_exact(function(*arguments), outcome['value'])

    def test_pre_migration_configuration_dumps(self):
        for semantics, filename in [(None, 'audit_undeclared_v1.yaml'),
                                    ('supervision_weight', 'audit_supervision_weight_v1.yaml')]:
            with self.subTest(semantics=semantics):
                self.assertEqual(audit_cfg(semantics).dump(), (FIXTURES / filename).read_text())
        self.assertEqual(audit_cfg().dump(), audit_cfg(None).dump())

    def test_configuration_instances_are_independent_and_extensible(self):
        first = audit_cfg('supervision_weight')
        first.MODEL.IMAGE_MEAN[0] = 123.
        first.DATASETS.CONFIG.SCALE_FACTOR = 1.
        second = audit_cfg('supervision_weight')
        self.assertEqual(second.dump(), (FIXTURES / 'audit_supervision_weight_v1.yaml').read_text())
        for node in [second, second.MODEL, second.DATASETS, second.DATASETS.CONFIG]:
            self.assertTrue(node.is_new_allowed())
            self.assertFalse(node.is_frozen())

    def test_historical_imports_remain_available(self):
        # Identity only verifies compatibility; numerical evidence is frozen above.
        from tools.validate import occlusion_metrics, evaluate_candidate_baseline
        from tools.validate import validate_bedlam_densekp_tiny

        for name in ['summarize', 'point_errors', 'evaluation_masks', 'prediction_retention', 'pair_masks']:
            self.assertIs(getattr(occlusion_metrics, name), getattr(evaluation, name))
        self.assertIs(evaluate_candidate_baseline.summary, candidates.summary)
        self.assertIs(evaluate_candidate_baseline.aggregate, candidates.aggregate)
        self.assertIs(validate_bedlam_densekp_tiny.audit_cfg, audit_cfg)


if __name__ == '__main__':
    unittest.main()
