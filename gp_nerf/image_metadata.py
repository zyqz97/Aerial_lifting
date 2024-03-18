from pathlib import Path
from typing import Optional
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os
import cv2
class ImageMetadata:
    def __init__(self, image_path: Path, c2w: torch.Tensor, W: int, H: int, intrinsics: torch.Tensor, image_index: int,
                 mask_path: Optional[Path], is_val: bool, label_path: Optional[Path], sam_feature_path=None, normal_path=None, 
                 depth_path=None, depth_dji_path=None, left_or_right=None, hparams=None, instance_path=None):
        self.image_path = image_path
        self.c2w = c2w
        self.W = W
        self.H = H
        self.intrinsics = intrinsics
        self.image_index = image_index
        self._mask_path = mask_path
        self.is_val = is_val
        self.label_path = label_path
        self.sam_feature_path = sam_feature_path
        self.normal_path = normal_path
        self.depth_path = depth_path
        self.depth_dji_path = depth_dji_path
        self.left_or_right = left_or_right
        self.hparams = hparams
        self.instance_path = instance_path


    def load_image(self) -> torch.Tensor:
        rgbs = Image.open(self.image_path).convert('RGB')
        size = rgbs.size

        if size[0] != self.W or size[1] != self.H:
            # rgbs = rgbs.resize((self.W, self.H), Image.LANCZOS) 
            # rgbs = rgbs.resize((self.W, self.H), Image.NEAREST) 
            rgbs = rgbs.resize((self.W, self.H), Image.BILINEAR)

        return torch.ByteTensor(np.asarray(rgbs))

    def load_mask(self) -> Optional[torch.Tensor]:
        if self._mask_path is None:
            return None

        with ZipFile(self._mask_path) as zf:
            with zf.open(self._mask_path.name) as f:
                keep_mask = torch.load(f, map_location='cpu')

        if keep_mask.shape[0] != self.H or keep_mask.shape[1] != self.W:
            keep_mask = F.interpolate(keep_mask.unsqueeze(0).unsqueeze(0).float(),
                                      size=(self.H, self.W)).bool().squeeze()

        return keep_mask
    
    def load_label(self) -> torch.Tensor:
        if self.label_path is None:
            return None
        labels = Image.open(self.label_path)    #.convert('RGB')
        size = labels.size

        if size[0] != self.W or size[1] != self.H:
            labels = labels.resize((self.W, self.H), Image.NEAREST)
       
        return torch.ByteTensor(np.asarray(labels))
    
    
    def load_gt(self) -> torch.Tensor:
        gt_path = self.image_path.parent.parent / 'labels_gt' / f'{self.image_path.stem}.png'
        labels = Image.open(gt_path)    #.convert('RGB')
        size = labels.size

        if size[0] != self.W or size[1] != self.H:
            labels = labels.resize((self.W, self.H), Image.NEAREST)
       
        return torch.ByteTensor(np.asarray(labels))
    
    def load_sam_feature(self) -> torch.Tensor:
        sam_feature_path = self.sam_feature_path
        feature = np.load(sam_feature_path)
       
        return torch.from_numpy(feature)
        
    def load_normal(self) -> torch.Tensor:
        normal = np.load(self.normal_path)
        if 'sci-art' in self.normal_path:
            normal = np.moveaxis(normal, [0, 1, 2], [1, 2, 0]) # bug: from H*3*W to H*W*3
        else:
            normal = np.moveaxis(normal, [0, 1, 2], [2, 0, 1]) # from 3*H*W to H*W*3
        normal = 2.0 * normal - 1.0 # from [0,1] to [-1,1]
        normal[:, :, 1:] = normal[:, :, 1:] * -1 # from right-down-forward to right-up-back
        return torch.FloatTensor(np.asarray(normal)).contiguous()

    def load_depth(self):
        depth = np.load(self.depth_path)
        return torch.HalfTensor(depth)
    
    def load_depth_dji(self):
        if self.hparams != None and (self.hparams.render_zyq == False):
            if self.depth_dji_path != None:
                depth = np.load(self.depth_dji_path)
                depth = torch.HalfTensor(depth)
                # depth = torch.Tensor(depth)
                
                assert depth.shape[0] == self.H and depth.shape[1] == self.W
                return depth
            else:
                return None
        else:
            return None

    
    
    def load_instance(self) -> torch.Tensor:
        if self.instance_path is None:
            return None
        if self.instance_path.suffix == '.npy':
            labels = np.load(str(self.instance_path))
            return torch.tensor(labels.astype(np.int32),dtype=torch.int32)

        else:
            labels = Image.open(self.instance_path)    #.convert('RGB')
            # labels = cv2.imread(str(self.instance_path), cv2.IMREAD_UNCHANGED)
            size = labels.size
            if size[0] != self.W or size[1] != self.H:
                labels = labels.resize((self.W, self.H), Image.NEAREST)
       
            # return torch.ByteTensor(np.asarray(labels))
            return torch.tensor(np.asarray(labels),dtype=torch.int32)

    def load_instance_crossview(self) -> torch.Tensor:
        if self.instance_path is None:
            return None
        crossview_path = str(self.instance_path).replace(self.hparams.instance_name, self.hparams.instance_name+'_crossview_process')
        # print(crossview_path)
        if not os.path.exists(crossview_path):
            if Path(crossview_path).suffix == '.npy':
                crossview_path = crossview_path.replace('.npy', '.png')
                labels = Image.open(crossview_path)    #.convert('RGB')
                size = labels.size
                if size[0] != self.W or size[1] != self.H:
                    labels = labels.resize((self.W, self.H), Image.NEAREST)
                return torch.tensor(np.asarray(labels),dtype=torch.int32)
            else:
                crossview_path = crossview_path.replace('.png', '.npy')
                labels = np.load(crossview_path)
                return torch.tensor(labels.astype(np.int32),dtype=torch.int32)
        else:
            if self.instance_path.suffix == '.npy':
                labels = np.load(crossview_path)
                return torch.tensor(labels.astype(np.int32),dtype=torch.int32)
            else:
                labels = Image.open(crossview_path)    #.convert('RGB')
                size = labels.size
                if size[0] != self.W or size[1] != self.H:
                    labels = labels.resize((self.W, self.H), Image.NEAREST)
                return torch.tensor(np.asarray(labels),dtype=torch.int32)
        
    def load_instance_64(self) -> torch.Tensor:

        crossview_path = str(self.instance_path).replace(self.hparams.instance_name, 'instances_mask_64')[:-4]+'.npy'
        if os.path.exists(crossview_path):
            labels = np.load(crossview_path)
            labels = labels.astype(np.int32)
            labels = torch.tensor(labels,dtype=torch.int32)
            labels[labels != 0] += 200000
            return labels
        else:
            return None

    
    def load_instance_gt(self) -> torch.Tensor:

        gt_path = self.image_path.parent.parent / 'instances_gt' / f'{self.image_path.stem}.png'

        labels = Image.open(gt_path)    #.convert('RGB')
        size = labels.size

        if size[0] != self.W or size[1] != self.H:
            labels = labels.resize((self.W, self.H), Image.NEAREST)
        #  gt instance label will not be too huge
        return torch.tensor(np.asarray(labels),dtype=torch.int32)
        # return torch.tensor(np.asarray(labels))


