"""Dataset configuration factories shared by audits and experiment entry points."""
from yacs.config import CfgNode as CN


def audit_cfg(value_semantics=None) -> CN:
    """Build a fresh, mutable YACS config for deterministic 256-square audits.

    Augmentation is disabled and normalization matches the historical audit
    entry point. ``None`` leaves target-third-value semantics undeclared; callers
    using supervision weights must explicitly pass ``'supervision_weight'``.
    This factory does not register data sources or construct a Dataset.
    """
    cfg = CN(new_allowed=True)
    cfg.MODEL = CN(new_allowed=True)
    cfg.MODEL.IMAGE_SIZE = 256
    cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]
    cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
    cfg.DATASETS = CN(new_allowed=True)
    cfg.DATASETS.TARGET_VALUE_SEMANTICS = value_semantics
    cfg.DATASETS.USE_SKIMAGE_ANTIALIAS = False
    cfg.DATASETS.BORDER_MODE = "constant"
    cfg.DATASETS.CONFIG = CN(new_allowed=True)
    cfg.DATASETS.CONFIG.SCALE_FACTOR = 0.0
    cfg.DATASETS.CONFIG.ROT_FACTOR = 0.0
    cfg.DATASETS.CONFIG.TRANS_FACTOR = 0.0
    cfg.DATASETS.CONFIG.COLOR_SCALE = 0.0
    cfg.DATASETS.CONFIG.ROT_AUG_RATE = 0.0
    cfg.DATASETS.CONFIG.TRANS_AUG_RATE = 0.0
    cfg.DATASETS.CONFIG.DO_FLIP = False
    cfg.DATASETS.CONFIG.FLIP_AUG_RATE = 0.0
    cfg.DATASETS.CONFIG.EXTREME_CROP_AUG_RATE = 0.0
    cfg.DATASETS.CONFIG.EXTREME_CROP_AUG_LEVEL = 0
    cfg.DATASETS.CONFIG.USE_ALB = False
    cfg.DATASETS.CONFIG.ALB_PROB = 0.0
    return cfg
