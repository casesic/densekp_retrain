"""CPU interface checks only; real CUDA acceptance is verify_fixed_validation.py."""

from pathlib import Path
import pickle
import tempfile
from types import SimpleNamespace
import unittest

from densekp_retrain.training.trial_callbacks import FixedValidation


class TrialCallbackInterfaceTests(unittest.TestCase):
    def test_historical_state_key_and_class_resolution(self):
        from tools.experiments.train_source_isolated_trial import FixedValidation as historical

        callback = FixedValidation([], [], Path("/unused"), 100)
        self.assertIs(historical, FixedValidation)
        self.assertEqual(callback.state_key, "FixedValidation")
        self.assertEqual(callback.state_dict(), {})
        callback.load_state_dict({})
        self.assertEqual(callback.reports, [])
        # A protocol-0 GLOBAL with the original class path; no object execution.
        self.assertIs(
            pickle.loads(b"ctools.experiments.train_source_isolated_trial\nFixedValidation\n."),
            FixedValidation,
        )

    def test_zero_and_unscheduled_steps_do_not_touch_model_or_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            callback = FixedValidation(None, None, output, 100)
            for step in (0, 1, 99, 101):
                callback.on_train_batch_end(SimpleNamespace(global_step=step), None, None, None, 0)
            self.assertEqual(list(output.iterdir()), [])
            self.assertEqual(callback.reports, [])


if __name__ == "__main__":
    unittest.main()
