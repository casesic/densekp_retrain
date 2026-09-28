
import os
import cv2
import copy
import numpy as np
from torch.utils.data import Dataset

from .utils import expand_to_aspect_ratio, get_example_projverts
from ..configs import DATASET_FOLDERS, DATASET_FILES
from ..keypoint_contract import validate_annotations
from ..constants import FLIP_KEYPOINT_PERMUTATION, NUM_JOINTS
from ..utils.pylogger import get_pylogger

log = get_pylogger(__name__)

class DatasetKeypoints(Dataset):
    def __init__(self, cfg, dataset, use_augmentation=True, is_train=True):
        super(DatasetKeypoints, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.cfg = cfg
        self.use_augmentation = bool(use_augmentation)
        if is_train and self.use_augmentation and cfg.DATASETS.CONFIG.DO_FLIP and cfg.DATASETS.CONFIG.FLIP_AUG_RATE > 0:
            raise ValueError("DenseKP flipping requires a verified 138-point semantic permutation")

        self.IMG_SIZE = cfg.MODEL.IMAGE_SIZE
        self.BBOX_SHAPE = cfg.MODEL.get('BBOX_SHAPE', None)
        self.MEAN = 255. * np.array(cfg.MODEL.IMAGE_MEAN)
        self.STD = 255. * np.array(cfg.MODEL.IMAGE_STD)
        self.use_skimage_antialias = cfg.DATASETS.get('USE_SKIMAGE_ANTIALIAS', False)
        self.border_mode = {
            'constant': cv2.BORDER_CONSTANT,
            'replicate': cv2.BORDER_REPLICATE,
        }[cfg.DATASETS.get('BORDER_MODE', 'constant')]

        self.img_dir = DATASET_FOLDERS[dataset] 
        with np.load(DATASET_FILES[is_train][dataset], allow_pickle=True) as archive:
            self.data = {key: archive[key] for key in ('imgname', 'scale', 'center', 'proj_verts', 'part', 'gtkps') if key in archive}
        validate_annotations(self.data, cfg.DATASETS.get('TARGET_VALUE_SEMANTICS'))
        self.imgname = self.data['imgname']
        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']

        if 'part' in self.data:
            self.keypoints = self.data['part']
        elif 'gtkps' in self.data:
            self.keypoints = self.data['gtkps'][:,:NUM_JOINTS]
        else:
            self.keypoints = np.zeros((len(self.imgname), NUM_JOINTS, 3))    
        if self.keypoints.shape != (len(self.imgname), NUM_JOINTS, 3):
            raise ValueError(f'Auxiliary keypoints must have shape [N,{NUM_JOINTS},3]')
        self.keypoints = self.keypoints.astype(np.float32)
        self.proj_verts = self.data['proj_verts'].astype(np.float32)
        self.length = self.scale.shape[0]

        log.info(f'Loaded {self.dataset} dataset, num samples {self.length}')

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()
        keypoints_2d = self.keypoints[index].copy()
        center_x = center[0]
        center_y = center[1]
        bbox_size = expand_to_aspect_ratio(scale*200, target_aspect_ratio=self.BBOX_SHAPE).max()
        if bbox_size < 1:
            raise ValueError(f"{self.dataset}[{index}]: bbox size must be at least one pixel")

        imgname = os.path.join(self.img_dir, self.imgname[index])
        augm_config = copy.deepcopy(self.cfg.DATASETS.CONFIG)

        img_patch_rgba=None
        img_patch_cv = None
        img_patch_rgba, \
        img_patch_cv,\
        img_size, cx, cy, bbox_w, bbox_h, keypoints_2d, proj_verts_cropped, trans = get_example_projverts(imgname,
                                      center_x, center_y,
                                      bbox_size, bbox_size,
                                      keypoints_2d,
                                      FLIP_KEYPOINT_PERMUTATION,
                                      self.proj_verts[index].copy(),
                                      self.IMG_SIZE, self.IMG_SIZE,
                                      self.MEAN, self.STD, self.is_train and self.use_augmentation, augm_config,
                                      is_bgr=True, 
                                      use_skimage_antialias=self.use_skimage_antialias,
                                      border_mode=self.border_mode,
                                      dataset=self.dataset
                                      )
        new_center = np.array([cx, cy])
        img_patch = img_patch_rgba[:3,:,:]
        item['proj_verts'] = self.proj_verts[index].copy()
        item['proj_verts_cropped'] = proj_verts_cropped
        item['img'] = img_patch
        item['img_disp'] = img_patch_cv      
        item['box_center'] = new_center
        item['box_size'] = bbox_w
        item['img_size'] = 1.0 * img_size.copy()
        item['_scale'] = scale
        item['_trans'] = trans
        item['imgname'] = imgname
        item['dataset'] = self.dataset
      
        return item

    def __len__(self):
        return int(len(self.imgname))
        
