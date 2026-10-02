"""Small CPU network exercising the real DenseKP optimizer and Lightning hooks.

The audit implementation is injected so capture can run the independent original
before any migration. Network replacement avoids all external model assets.
"""
from contextlib import redirect_stdout
import io
import random
from unittest.mock import patch

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset

from densekp.core.densekp_trainer import DenseKP
from tests.test_densekp_boundaries import TinyHead, config


def backbone():
    return torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(),
                               torch.nn.Linear(3, 3), torch.nn.Dropout(.2))


class Rows(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        target = torch.zeros(138, 3)
        target[:, :2] = .05 * (index + 1)
        target[:100, 2] = 1
        return dict(img=torch.full((3, 256, 256), .2 * (index + 1)),
                    proj_verts_cropped=target, row=index)


class Recorder(pl.Callback):
    def __init__(self, prefix, metadata, arrays):
        self.prefix, self.metadata, self.arrays = prefix, metadata, arrays
        self.sequence = 0

    def record(self, event, trainer, model, row=None):
        key = f'{self.prefix}_{self.sequence:02d}_{event}'
        self.sequence += 1
        names = {id(value): name for name, value in model.named_parameters()}
        groups = [[names[id(value)] for value in group['params']]
                  for group in trainer.optimizers[0].param_groups]
        self.metadata['events'].append(dict(key=key, step=trainer.global_step,
                                           training=model.training, row=row,
                                           optimizer_parameter_order=groups))
        self.arrays[key + '_torch_rng'] = torch.get_rng_state().numpy().copy()
        self.arrays[key + '_numpy_rng'] = np.random.get_state()[1].copy()
        self.arrays[key + '_python_rng'] = np.asarray(random.getstate()[1], dtype=np.uint64)
        for name, value in model.named_parameters():
            self.arrays[key + '_parameter_' + name] = value.detach().cpu().numpy().copy()
            if event == 'before_step' and value.grad is not None:
                self.arrays[key + '_gradient_' + name] = value.grad.detach().cpu().numpy().copy()

    def on_train_batch_start(self, trainer, model, batch, batch_idx):
        self.record('batch_start', trainer, model, int(batch['img']['row'].item()))

    def on_before_optimizer_step(self, trainer, model, optimizer):
        self.record('before_step', trainer, model)

    def on_train_batch_end(self, trainer, model, outputs, batch, batch_idx):
        self.record('batch_end', trainer, model)

    def on_validation_epoch_start(self, trainer, model):
        self.record('validation_start', trainer, model)

    def on_validation_epoch_end(self, trainer, model):
        self.record('validation_end', trainer, model)


def run_contract(checker_class, probes, directory):
    """Return deterministic metadata/arrays after two updates and two resumed ones."""
    directory.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(131)
    np.random.seed(131)
    random.seed(131)
    metadata = dict(events=[], failures=[], runs=[])
    arrays = {}
    initial = directory / 'tiny_backbone.pt'
    torch.save({'state_dict': backbone().state_dict()}, initial)
    cfg = config(directory / 'run')
    cfg.TRAIN.LR = 1e-4
    checkpoint = directory / 'resume.ckpt'
    with patch('densekp.core.densekp_trainer.VITPOSE_BACKBONE', str(initial)), \
         patch('densekp.core.densekp_trainer.create_backbone', side_effect=backbone), \
         patch('densekp.core.densekp_trainer.build_keypoints_head', side_effect=TinyHead):
        for prefix, max_steps, resume in [('initial', 2, None), ('resumed', 4, str(checkpoint))]:
            model = DenseKP(cfg)
            checker = checker_class()
            recorder = Recorder(prefix, metadata, arrays)
            # Probe must be an independent snapshot and must preserve mode/RNG.
            mode = model.training
            rng = torch.get_rng_state().clone()
            start = probes(model)
            assert model.training == mode and torch.equal(rng, torch.get_rng_state())
            for name, value in start.items():
                arrays[prefix + '_probe_before_' + name] = value.numpy().copy()
            trainer = pl.Trainer(accelerator='cpu', devices=1, precision='32-true',
                                 max_steps=max_steps, max_epochs=-1, num_sanity_val_steps=0,
                                 logger=False, enable_checkpointing=False, enable_progress_bar=False,
                                 enable_model_summary=False, callbacks=[checker, recorder],
                                 default_root_dir=str(directory / prefix))
            train = DataLoader(Rows(), batch_size=1, shuffle=True, num_workers=0)
            val = DataLoader(Rows(), batch_size=1, shuffle=False, num_workers=0)
            trainer.fit(model, train_dataloaders={'img': train}, val_dataloaders=val, ckpt_path=resume)
            for name, value in probes(model).items():
                arrays[prefix + '_probe_after_' + name] = value.numpy().copy()
                assert torch.equal(start[name], torch.from_numpy(arrays[prefix + '_probe_before_' + name]))
                assert not torch.equal(start[name], value)
            optimizer = trainer.optimizers[0]
            named = dict(model.named_parameters())
            for name, value in named.items():
                for field, state in optimizer.state[value].items():
                    arrays[f'{prefix}_optimizer_{name}_{field}'] = state.detach().numpy().copy()
            metadata['runs'].append(dict(prefix=prefix, steps=trainer.global_step,
                                         attempts=checker.attempts, mode_after_fit=model.training,
                                         state_key=checker.state_key,
                                         callback_state=checker.state_dict(),
                                         parameter_order=list(named),
                                         optimizer_type=type(optimizer).__name__))
            if resume is None:
                trainer.save_checkpoint(str(checkpoint))
                saved = torch.load(checkpoint, map_location='cpu')
                metadata['checkpoint'] = dict(step=saved['global_step'],
                    schema=[dict(key=key, shape=list(value.shape), dtype=str(value.dtype))
                            for key, value in saved['state_dict'].items()],
                    optimizer_parameter_ids=saved['optimizer_states'][0]['param_groups'][0]['params'],
                    callback_keys=list(saved['callbacks']))
        # The checker must reject before any optimizer update and preserve the log.
        for name in ('backbone', 'head'):
            for problem in ('missing', 'nan', 'inf'):
                checker = checker_class()
                for value in model.parameters():
                    value.grad = torch.ones_like(value)
                module = getattr(model, name)
                if problem == 'missing':
                    for value in module.parameters():
                        value.grad = None
                else:
                    next(module.parameters()).grad.flatten()[0] = float(problem)
                previous = {key: value.detach().clone() for key, value in model.named_parameters()}
                rng = torch.get_rng_state().clone()
                try:
                    checker.on_before_optimizer_step(trainer, model, optimizer)
                except ValueError as error:
                    metadata['failures'].append(dict(module=name, problem=problem,
                                                     type=type(error).__name__, message=str(error),
                                                     attempts=checker.attempts))
                else:
                    raise AssertionError('Invalid gradients were accepted')
                assert torch.equal(rng, torch.get_rng_state())
                assert all(torch.equal(value, previous[key]) for key, value in model.named_parameters())
    return metadata, arrays


def quiet_contract(checker_class, probes, directory):
    stream = io.StringIO()
    with redirect_stdout(stream):
        metadata, arrays = run_contract(checker_class, probes, directory)
    metadata['stdout'] = stream.getvalue()
    return metadata, arrays
