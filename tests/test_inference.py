"""CPU policy tests complement, but do not replace, real GPU equivalence evidence."""
import unittest
from unittest.mock import patch

import numpy as np
import torch

from densekp.core.inference import (
    load_checkpoint_state_dict,
    load_lightning_checkpoint,
    predict_batch,
    predict_item,
)


class RecordingModel(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output
        self.seen = None

    def forward(self, batch):
        self.seen = dict(training=self.training, inference=torch.is_inference_mode_enabled(),
                         grad=torch.is_grad_enabled(), image=batch['img'].clone())
        return {'pred_keypoints': self.output}


class InferenceTests(unittest.TestCase):
    def test_batch_preserves_mode_dtype_rng_and_delegates_rejection(self):
        image = np.zeros((1, 3, 256, 256), dtype=np.float32)
        image[:, :, :, :32] = 7
        image[:, :, :, -32:] = 9
        # Invalid/nonfinite output must reach the evaluator's own rejection policy.
        for shape in [(1, 138, 3), (1, 137, 3)]:
            raw = torch.full(shape, float('nan'), dtype=torch.float64)
            model = RecordingModel(raw)
            for training in (True, False):
                model.train(training)
                rng = torch.get_rng_state().clone()
                result = predict_batch(model, image, 'cpu')
                self.assertIs(result, raw)
                self.assertEqual(model.training, training)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertEqual(model.seen['training'], training)
                self.assertTrue(model.seen['inference'])
                self.assertFalse(model.seen['grad'])
                np.testing.assert_array_equal(model.seen['image'].numpy(), image)

    def test_item_keeps_historical_eval_side_effect_and_strict_policy(self):
        item = {'img': np.zeros((3, 256, 256), dtype=np.float32)}
        model = RecordingModel(torch.arange(414, dtype=torch.float64).reshape(1, 138, 3))
        result = predict_item(model, item, 'cpu')
        self.assertFalse(model.training)
        self.assertFalse(model.seen['training'])
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, np.arange(414, dtype=np.float32).reshape(1, 138, 3))
        for raw in [torch.zeros(1, 137, 3), torch.full((1, 138, 3), float('nan')),
                    torch.full((1, 138, 3), float('inf'))]:
            model = RecordingModel(raw)
            with self.assertRaisesRegex(ValueError, '^Expected finite 138x3 output$'):
                predict_item(model, item, 'cpu')
            self.assertFalse(model.training)

    def test_loading_paths_preserve_order_strictness_and_default_mode(self):
        events = []
        state = {'weight': torch.zeros(1)}
        payload = dict(state_dict=state, epoch=2, global_step=400)
        cfg = object()

        class FakeModel:
            def __init__(self, cfg):
                events.append(('construct', cfg))
                self.training = True

            def load_state_dict(self, value, strict):
                from types import SimpleNamespace
                events.append(('load_state_dict', value is state, strict))
                return SimpleNamespace(missing_keys=[], unexpected_keys=[])

            @classmethod
            def load_from_checkpoint(cls, checkpoint, **kwargs):
                events.append(('lightning', checkpoint, kwargs))
                return cls(kwargs['cfg'])

        def read(checkpoint, map_location):
            events.append(('read', checkpoint, map_location))
            return payload

        with patch('densekp.core.inference.DenseKP', FakeModel), \
             patch('densekp.core.inference.torch.load', side_effect=read):
            model, details = load_checkpoint_state_dict('weights.ckpt', cfg)
            self.assertEqual(events, [('read', 'weights.ckpt', 'cpu'), ('construct', cfg),
                                      ('load_state_dict', True, True)])
            self.assertTrue(model.training)
            self.assertEqual(details, dict(epoch=2, global_step=400, state_dict_keys=1,
                                          missing_keys=[], unexpected_keys=[], strict=True))
            events.clear()
            model = load_lightning_checkpoint('weights.ckpt', cfg)
            self.assertEqual(events[0], ('lightning', 'weights.ckpt',
                                        dict(cfg=cfg, map_location='cpu', strict=True)))
            self.assertTrue(model.training)

    def test_active_caller_is_direct(self):
        from tools.experiments import train_source_isolated_trial
        self.assertIs(train_source_isolated_trial.predict, predict_item)


if __name__ == '__main__':
    unittest.main()
