#!/usr/bin/env python3
"""Bounded real-model GPU acceptance, in separate fresh/resume processes.

This checks numerical/runtime compatibility on existing tiny fixtures, not
accuracy or readiness of the eventual training corpus. It never replaces input
checkpoints. Outputs include a large full-state checkpoint in --output-dir.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from densekp.core.checkpointing import ValidationCheckpoint
from densekp.core.configs import dataset_config
from densekp.core.crop_geometry import restore_crop_xy
from densekp.core.datasets.datamodule_keypoints import DataModule
from densekp.core.densekp_trainer import DenseKP
from densekp.core.training_config import validate_training_config


class Acceptance(pl.Callback):
    def __init__(self, args, sample):
        self.args, self.sample = args, sample
        self.updates = []
        self.start_probes = {}
        self.resume_error = None

    def predict(self, model):
        was_training = model.training
        model.eval()
        with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
            result = model({'img': self.sample['img'].unsqueeze(0).to(model.device)})['pred_keypoints']
        model.train(was_training)
        if result.shape != (1, 138, 3) or not torch.isfinite(result).all():
            raise RuntimeError('Expected finite real-model [1,138,3] output')
        return result.float().cpu().numpy()

    @staticmethod
    def probes(model):
        return {name: next(module.parameters()).detach().flatten()[:4096].float().cpu().clone()
                for name, module in [('backbone', model.backbone), ('head', model.head)]}

    def on_train_start(self, trainer, model):
        self.start_probes = self.probes(model)
        prediction = self.predict(model)
        np.save(self.args.output_dir / f'{self.args.phase}_start_prediction.npy', prediction)
        self.initial_step = trainer.global_step
        self.initial_optimizer_steps = [float(state['step']) for state in trainer.optimizers[0].state.values() if 'step' in state]
        if self.args.phase == 'resume':
            expected = np.load(self.args.reference_dir / 'fresh_end_prediction.npy')
            self.resume_error = float(np.max(np.abs(prediction - expected)))
            if self.resume_error > 1e-5:
                raise RuntimeError(f'Restored prediction mismatch: {self.resume_error}')
            reference = json.loads((self.args.reference_dir / 'report.json').read_text())
            if self.initial_step != reference['final_step'] or not trainer.optimizers[0].state:
                raise RuntimeError('Full-state resume did not restore global step and AdamW state')

    def on_before_optimizer_step(self, trainer, model, optimizer):
        # Lightning calls this after AMP unscaling, before the actual update.
        groups = {}
        all_finite = True
        for name, module in [('backbone', model.backbone), ('head', model.head)]:
            gradients = [p.grad for p in module.parameters() if p.grad is not None]
            if not gradients:
                raise RuntimeError(f'No {name} gradients')
            finite = torch.stack([torch.isfinite(g).all() for g in gradients]).all().item()
            all_finite = all_finite and finite
            groups[name] = {'gradient_tensors': len(gradients), 'finite': finite}
        scaler = trainer.precision_plugin.scaler
        if not all_finite and scaler is None:
            raise RuntimeError('Nonfinite gradients without an AMP scaler')
        attempt = {'step_before_update': trainer.global_step, 'gradients': groups,
                   'finite': all_finite, 'amp_scale': float(scaler.get_scale()) if scaler else None}
        self.updates.append(attempt)
        print('Optimizer attempt: ' + json.dumps(attempt), flush=True)

    def on_train_end(self, trainer, model):
        prediction = self.predict(model)
        np.save(self.args.output_dir / f'{self.args.phase}_end_prediction.npy', prediction)
        self.changes = {name: float((value - self.start_probes[name]).abs().max())
                        for name, value in self.probes(model).items()}
        if not all(value > 0 for value in self.changes.values()):
            raise RuntimeError(f'Backbone/head probes did not both update: {self.changes}')
        xy = restore_crop_xy(prediction[0, :, :2], self.sample['_trans'].numpy(), (256, 256))
        if not np.isfinite(xy).all():
            raise RuntimeError('Restored model xy is nonfinite')
        np.save(self.args.output_dir / f'{self.args.phase}_restored_xy.npy', xy)
        self.final_optimizer_steps = [float(state['step']) for state in trainer.optimizers[0].state.values() if 'step' in state]
        self.successful_updates = sum(attempt['finite'] for attempt in self.updates)
        minimum_updates = 2 if self.args.phase == 'fresh' else 1
        if self.successful_updates < minimum_updates:
            raise RuntimeError('Insufficient finite optimizer updates after AMP warmup')
        before = max(self.initial_optimizer_steps, default=0)
        if not self.final_optimizer_steps or max(self.final_optimizer_steps) - before != self.successful_updates:
            raise RuntimeError('AdamW step count disagrees with finite-gradient updates')
        if len(self.updates) != trainer.global_step - self.initial_step:
            raise RuntimeError('Optimizer hook count does not match completed steps')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase', required=True, choices=['fresh', 'resume'])
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--reference-dir', type=Path)
    parser.add_argument('--gpu', type=int, default=0)
    args = parser.parse_args()
    if args.phase == 'resume' and args.reference_dir is None:
        parser.error('--reference-dir is required for resume')
    if not torch.cuda.is_available() or not 0 <= args.gpu < torch.cuda.device_count():
        parser.error('Requested CUDA device unavailable')
    if not args.checkpoint.is_file():
        parser.error('Input checkpoint does not exist')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    pl.seed_everything(0, workers=True)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('medium')
    with initialize_config_dir(version_base='1.2', config_dir=str(ROOT / 'configs/hydra')):
        cfg = compose(config_name='train', overrides=['data=p0_dressed_tiny', 'trainer=gpu'])
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.GENERAL.NUM_WORKERS = 0
    cfg.GENERAL.TOTAL_STEPS = (16 if args.phase == 'fresh' else
                               json.loads((args.reference_dir / 'report.json').read_text())['final_step'] + 1)
    cfg.GENERAL.VAL_STEPS = 1
    cfg.GENERAL.CHECKPOINT_STEPS = cfg.GENERAL.TOTAL_STEPS if args.phase == 'fresh' else 1
    cfg.paths.root_dir = str(ROOT)
    cfg.paths.work_dir = str(ROOT)
    cfg.paths.output_dir = str(args.output_dir)
    for name in ('SCALE_FACTOR', 'COLOR_SCALE', 'ROT_AUG_RATE', 'TRANS_AUG_RATE', 'EXTREME_CROP_AUG_RATE'):
        cfg.DATASETS.CONFIG[name] = 0.
    cfg.DATASETS.CONFIG.DO_FLIP = False
    cfg.DATASETS.CONFIG.USE_ALB = False
    cfg.trainer.devices = [args.gpu]
    cfg.trainer.log_every_n_steps = 1
    cfg.init_checkpoint = str(args.checkpoint) if args.phase == 'fresh' else None
    cfg.ckpt_path = str(args.checkpoint) if args.phase == 'resume' else None
    validate_training_config(cfg)
    datamodule = DataModule(cfg, dataset_config())
    datamodule.setup('fit')
    sample = next(iter(datamodule.val_dataloader()[0]))
    sample = {key: value[0] if isinstance(value, torch.Tensor) else value for key, value in sample.items()}
    checker = Acceptance(args, sample)
    checkpoint = ValidationCheckpoint(every_n_steps=cfg.GENERAL.CHECKPOINT_STEPS,
                                     dirpath=str(args.output_dir / 'checkpoints'),
                                     save_top_k=0, save_last=True, monitor='val_loss')
    print('Loading real DenseKP model', flush=True)
    model = (DenseKP.load_from_checkpoint(str(args.checkpoint), cfg=cfg, map_location='cpu')
             if args.phase == 'fresh' else DenseKP(cfg))
    trainer_options = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_options.pop('_target_')
    trainer = pl.Trainer(**trainer_options, callbacks=[checker, checkpoint], logger=False,
                         enable_progress_bar=False, enable_model_summary=False)
    OmegaConf.save(cfg, args.output_dir / 'config.yaml', resolve=True)
    try:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
    except Exception as error:
        failure = dict(status='failed', phase=args.phase, error=repr(error), attempts=checker.updates)
        (args.output_dir / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
        raise
    if trainer.global_step != cfg.GENERAL.TOTAL_STEPS:
        raise RuntimeError('Incorrect final global_step')
    if not Path(checkpoint.last_model_path).is_file():
        raise RuntimeError('Full-state checkpoint was not saved')
    metrics = {name: float(value) for name, value in trainer.callback_metrics.items()}
    if not all(np.isfinite(value) for value in metrics.values()):
        raise RuntimeError('Nonfinite metric')
    if metrics['val/scored_people_0'] != len(datamodule.val_dataset[0]):
        raise RuntimeError('Validation did not cover the complete tiny validation split')
    report = dict(status='passed', phase=args.phase, git_commit=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        torch=torch.__version__, lightning=pl.__version__, gpu=torch.cuda.get_device_name(args.gpu),
        precision=cfg.trainer.precision, with_var=cfg.MODEL.with_var,
        input_checkpoint=str(args.checkpoint.resolve()), checkpoint=checkpoint.last_model_path,
        dataset='p0-dressed-tiny (smoke only)', train_samples=len(datamodule.train_dataset),
        validation_samples=len(datamodule.val_dataset[0]), initial_step=checker.initial_step,
        final_step=trainer.global_step, optimizer_attempts=checker.updates,
        successful_updates=checker.successful_updates,
        amp_skipped_updates=len(checker.updates)-checker.successful_updates,
        final_amp_scale=float(trainer.precision_plugin.scaler.get_scale()),
        parameter_probe_changes=checker.changes,
        resume_prediction_max_error=checker.resume_error, metrics=metrics,
        max_cuda_memory_bytes=torch.cuda.max_memory_allocated(args.gpu),
        scope='real-model runtime compatibility; not an accuracy or dataset-independence evaluation')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
