"Part of the code has been taken from "
"4DHumans: https://github.com/shubham-goel/4D-Humans"

from typing import Optional, Tuple
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger

from yacs.config import CfgNode
from densekp.core.configs import dataset_config
from densekp.core.datasets.datamodule_keypoints import DataModule
from densekp.core.densekp_trainer import DenseKP
from densekp.core.utils.pylogger import get_pylogger
from densekp.core.utils.misc import task_wrapper, log_hyperparameters
from densekp.core.training_config import validate_training_config
from densekp.core.checkpointing import ValidationCheckpoint
import signal
signal.signal(signal.SIGUSR1, signal.SIG_DFL)
log = get_pylogger(__name__)
import torch
torch.set_float32_matmul_precision('medium')

@pl.utilities.rank_zero.rank_zero_only
def save_configs(model_cfg: CfgNode, dataset_cfg: CfgNode, rootdir: str):
    Path(rootdir).mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=model_cfg, f=os.path.join(rootdir, 'model_config.yaml'), resolve=True)
    with open(os.path.join(rootdir, 'dataset_config.yaml'), 'w') as f:
        f.write(dataset_cfg.dump())


@task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:

    validate_training_config(cfg)
    if cfg.get('seed') is not None:
        pl.seed_everything(int(cfg.seed), workers=True)
    dataset_cfg = dataset_config()
    save_configs(cfg, dataset_cfg, cfg.paths.output_dir)
    datamodule = DataModule(cfg, dataset_cfg)
    # Initializing from a trained DenseKP model starts a fresh optimizer/run.
    # ``ckpt_path`` below retains Lightning's full-state resume semantics.
    init_checkpoint = cfg.get('init_checkpoint')
    model = (DenseKP.load_from_checkpoint(init_checkpoint, cfg=cfg, map_location='cpu')
             if init_checkpoint else DenseKP(cfg))

    logger = TensorBoardLogger(os.path.join(cfg.paths.output_dir, 'tensorboard'), name='', version='', default_hp_metric=False)
    loggers = [logger]

    # Setup checkpoint saving
    checkpoint_callback = ValidationCheckpoint(
        dirpath=os.path.join(cfg.paths.output_dir, 'checkpoints'), 
        every_n_steps=cfg.GENERAL.CHECKPOINT_STEPS,
        save_last=True,
        save_top_k=cfg.GENERAL.CHECKPOINT_SAVE_TOP_K,
        monitor='val_loss'
    )

    rich_callback = pl.callbacks.TQDMProgressBar(refresh_rate=100)
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='step')
    callbacks = [
        checkpoint_callback, 
        lr_monitor,
        rich_callback
    ]

    trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks)
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        log_hyperparameters(object_dict)


    # Upstream hard-coded ``ckpt_path='last'``.  Keep fresh runs deterministic;
    # resume only when the caller supplies a checkpoint explicitly.
    if cfg.get('train', True):
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.get('ckpt_path'))
    if cfg.get('test', False):
        test_checkpoint = (checkpoint_callback.best_model_path or None) if cfg.get('train', True) else cfg.get('ckpt_path')
        trainer.test(model, datamodule=datamodule, ckpt_path=test_checkpoint)

    log.info("Run finished")
    return dict(trainer.callback_metrics), object_dict


@hydra.main(version_base="1.2", config_path=str(root/"configs/hydra"), config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    # train the model
    train(cfg)


if __name__ == "__main__":
    main()
