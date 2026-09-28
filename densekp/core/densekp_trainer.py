import os
import torch
import numpy as np
import pytorch_lightning as pl
from typing import Dict, Tuple
from yacs.config import CfgNode

from .backbones import create_backbone
from .heads.smpl_head_keypoints import build_keypoints_head
from .utils.train_utils import denormalize_images
from .keypoint_objective import keypoint_loss, keypoint_error_totals
from .utils.pylogger import get_pylogger
from .constants import  VITPOSE_BACKBONE
log = get_pylogger(__name__)


class DenseKP(pl.LightningModule):

    def __init__(self, cfg: CfgNode, init_renderer: bool = True):
        super().__init__()

        self.save_hyperparameters(logger=False, ignore=['init_renderer'])
        self.cfg = cfg

        # Backbone feature extractor
        self.backbone = create_backbone()
        self.backbone.load_state_dict(torch.load(VITPOSE_BACKBONE, map_location='cpu')['state_dict'])

        self.head = build_keypoints_head()
        self._evaluation_totals = {}
        self.automatic_optimization = False

    def visualize(self, input_batch, output):
        images = input_batch['img']
        img = denormalize_images(images)[0]
        save_dir = os.path.join('.', 'output_images2')
        os.makedirs(save_dir, exist_ok=True)
        from matplotlib import pyplot as plt

        img = img.detach().cpu().numpy().transpose(1,2,0)*255
        img = np.clip(img, 0, 255).astype(np.uint8)
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(111)
        ax.imshow(img)

        pred_vertices = output['pred_keypoints'][0].detach().cpu().numpy().copy()
        pred_vertices[:, :2] = (pred_vertices[:, :2]+0.5)*self.cfg.MODEL.IMAGE_SIZE
 
        ax.scatter(pred_vertices[:, 0], pred_vertices[:, 1], s=1.0)# s=0.5)
        save_filename = os.path.join(save_dir, f'result_{self.current_epoch:04d}')
        plt.savefig(save_filename)
        plt.close(fig)
        
        
    def get_parameters(self):
        all_params = list(self.head.parameters())
        all_params += list(self.backbone.parameters())
        return all_params

    def configure_optimizers(self):

        return torch.optim.AdamW(
                self.get_parameters(),
                lr=self.cfg.TRAIN.LR,
                weight_decay=self.cfg.TRAIN.WEIGHT_DECAY)


    def forward_step(self, batch: Dict, train: bool = False) -> Dict:

        x = batch['img']
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError('DenseKP backbone requires [B,3,256,256] full-crop input')
        conditioning_feats = self.backbone(x[:,:,:,32:-32])
        output = self.head(conditioning_feats)
        # self.visualize(batch, output)
        return output


    def compute_loss(self, batch: Dict, output: Dict, train: bool = True) -> Tuple[torch.Tensor, Dict]:
        
        return keypoint_loss(output['pred_keypoints'], batch['proj_verts_cropped'],
                             self.cfg.MODEL.with_var)

    def forward(self, batch: Dict) -> Dict:
        return self.forward_step(batch, train=False)


    def training_step(self, joint_batch: Dict, batch_idx: int) -> Dict:

        batch = joint_batch['img']
        optimizer = self.optimizers(use_pl_optimizer=True)
  
        batch_size = batch['img'].shape[0]
        output = self.forward_step(batch, train=True)
       
        loss, _ = self.compute_loss(batch, output, train=True)


        optimizer.zero_grad()
        self.manual_backward(loss)
        optimizer.step()
 
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True, batch_size=batch_size)
        return output

   

    def _evaluation_step(self, batch, dataloader_idx, stage):
        output = self.forward_step(batch, train=False)
        # The dataset supplies the same transformed label contract for train/val.
        totals = keypoint_error_totals(output['pred_keypoints'], batch['proj_verts_cropped'])
        key = (stage, dataloader_idx)
        self._evaluation_totals[key] = self._evaluation_totals.get(key, 0) + totals

    def validation_step(self, batch: Dict, batch_idx: int, dataloader_idx=0):
        self._evaluation_step(batch, dataloader_idx, 'val')

    def _evaluation_epoch_end(self, stage):
        for (split, index), local_totals in sorted(self._evaluation_totals.items()):
            if split != stage:
                continue
            totals = local_totals.clone()
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(totals)
            error, count, empty = totals
            self.log(f'{stage}/scored_people_{index}', count, sync_dist=False)
            self.log(f'{stage}/empty_people_{index}', empty, sync_dist=False)
            if count > 0:
                mean = error / count
                self.log(f'{stage}/crop_l2_{index}', mean, sync_dist=False)
                # Preserve loader-0 checkpoint selection; all other loaders reported.
                if index == 0:
                    self.log(f'{stage}_loss', mean, prog_bar=True, sync_dist=False)
            elif index == 0:
                raise ValueError(f'{stage} loader 0 has no supervised people; cannot rank checkpoints')
        self._evaluation_totals = {key: value for key, value in self._evaluation_totals.items()
                                   if key[0] != stage}

    def on_validation_epoch_start(self):
        self._evaluation_totals = {}

    def on_validation_epoch_end(self):
        self._evaluation_epoch_end('val')

    def test_step(self, batch: Dict, batch_idx: int, dataloader_idx=0):
        self._evaluation_step(batch, dataloader_idx, 'test')

    def on_test_epoch_start(self):
        self._evaluation_totals = {}

    def on_test_epoch_end(self):
        self._evaluation_epoch_end('test')
