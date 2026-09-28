"""Validate controls that the fixed DenseKP/manual-optimization path supports."""
from .datasets.datamodule_keypoints import DataModule


def validate_training_config(cfg):
    if cfg.get('init_checkpoint') and cfg.get('ckpt_path'):
        raise ValueError('Choose weights-only init_checkpoint OR full-state ckpt_path')
    if cfg.MODEL.IMAGE_SIZE != 256:
        raise ValueError('The fixed DenseKP backbone requires MODEL.IMAGE_SIZE=256')
    if cfg.DATASETS.get('TARGET_VALUE_SEMANTICS') != 'supervision_weight':
        raise ValueError('Explicitly declare verified DATASETS.TARGET_VALUE_SEMANTICS=supervision_weight')
    if cfg.get('train', True):
        DataModule._names(cfg.DATASETS.get('DATASETS_AND_RATIOS'))
    if cfg.get('train', True) or cfg.get('test', False):
        DataModule._names(cfg.DATASETS.get('VAL_DATASETS'))
    if cfg.trainer.get('accumulate_grad_batches', 1) != 1:
        raise ValueError('DenseKP manual optimization does not implement gradient accumulation')
    if cfg.trainer.get('gradient_clip_val', 0) or cfg.TRAIN.get('CLIP_GRAD', False):
        raise ValueError('DenseKP manual optimization does not implement gradient clipping')
    if cfg.trainer.get('max_epochs') is None and cfg.trainer.get('max_steps', -1) < 0:
        raise ValueError('Set an explicit training duration')
