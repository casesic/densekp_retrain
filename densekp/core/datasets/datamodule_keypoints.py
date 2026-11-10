"""DenseKP datasets and loaders; no model or image geometry ownership."""
from typing import Optional

import torch
import pytorch_lightning as pl
from yacs.config import CfgNode
from .dataset_keypoints import DatasetKeypoints


class DataModule(pl.LightningDataModule):
    def __init__(self, cfg: CfgNode, dataset_cfg: CfgNode) -> None:
        super().__init__()
        self.cfg = cfg
        self.dataset_cfg = dataset_cfg
        self.train_dataset = None
        self.val_dataset = None

    def setup(self, stage: Optional[str] = None):
        if stage in (None, 'fit'):
            self.train_dataset = self.train_dataset_prepare()
        if stage in (None, 'fit', 'validate', 'test'):
            self.val_dataset = self.val_dataset_prepare()

    @staticmethod
    def _names(value):
        # Legacy registry IDs contain hyphens; '_' separates datasets, not ratios.
        names = value.split('_') if isinstance(value, str) else list(value or [])
        if not names or any(not name for name in names):
            raise ValueError('Select nonempty training/validation dataset names explicitly')
        return names

    def train_dataset_prepare(self):
        datasets = [DatasetKeypoints(self.cfg, name) for name in
                    self._names(self.cfg.DATASETS.DATASETS_AND_RATIOS)]
        return torch.utils.data.ConcatDataset(datasets)

    def set_box_augmentation_epoch(self, epoch):
        """Call before the next loader iterator; no worker-order RNG state."""
        if self.train_dataset is None:
            raise ValueError('train_dataset_not_prepared')
        for dataset in self.train_dataset.datasets:
            dataset.set_box_augmentation_epoch(epoch)

    def val_dataset_prepare(self):
        return [DatasetKeypoints(self.cfg, name, is_train=False) for name in
                self._names(self.cfg.DATASETS.VAL_DATASETS)]

    def _loader(self, dataset, train=False):
        workers = self.cfg.GENERAL.NUM_WORKERS
        kwargs = dict(batch_size=self.cfg.TRAIN.BATCH_SIZE, shuffle=train,
                      drop_last=train, num_workers=workers)
        if train and len(dataset) < kwargs['batch_size']:
            raise ValueError('Training set is smaller than BATCH_SIZE with drop_last=True; '
                             'reduce BATCH_SIZE for a smoke run')
        if workers > 0:
            kwargs['prefetch_factor'] = self.cfg.GENERAL.PREFETCH_FACTOR
        return torch.utils.data.DataLoader(dataset, **kwargs)

    def train_dataloader(self):
        return {'img': self._loader(self.train_dataset, train=True)}

    def val_dataloader(self):
        return [self._loader(dataset) for dataset in self.val_dataset]

    def test_dataloader(self):
        return self.val_dataloader()
