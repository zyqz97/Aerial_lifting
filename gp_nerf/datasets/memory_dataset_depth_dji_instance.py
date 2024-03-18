from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_depth_dji_instance
from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_rays, get_ray_directions

import numpy as np
import glob
import os
from pathlib import Path
import random
from PIL import Image
from tools.unetformer.uavid2rgb import remapping
import cv2

class MemoryDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams=None):
        super(MemoryDataset, self).__init__()
        self.hparams = hparams
        rgbs = []
        rays = []
        indices = []
        instances = []
        depth_djis = []
        depth_scales = []
        metadata_item = metadata_items[0]

        self.W = metadata_item.W
        self.H = metadata_item.H
        
        self._directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        center_pixels,
                                        device)
        self._depth_scale = torch.abs(self._directions[:, :, 2]).view(-1).cpu()
        
        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[::20]
        load_subset = 0
        for metadata_item in main_tqdm(metadata_items):
        # for metadata_item in main_tqdm(metadata_items[:40]):
            if hparams.enable_semantic and metadata_item.is_val:  # 训练语义的时候要去掉val图像
                continue
            if hparams.use_subset:
                used_files = []
                for ext in ('*.png', '*.jpg'):
                    used_files.extend(glob.glob(os.path.join(f'{hparams.dataset_path}/subset/rgbs', ext)))
                used_files.sort()
                file_names = [os.path.splitext(os.path.basename(file_path))[0] for file_path in used_files]
                if (metadata_item.label_path == None): 
                    continue
                else:
                    if (Path(metadata_item.label_path).stem not in file_names):
                        continue
                    else:
                        load_subset = load_subset+1

            image_data = get_rgb_index_mask_depth_dji_instance(metadata_item)

            if image_data is None:
                continue
            #改成读取instance label
            image_rgbs, image_indices, image_keep_mask, label, depth_dji, instance = image_data
            
            image_rays = get_rays(self._directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1, 8).cpu()
            
            depth_scale = self._depth_scale
            if image_keep_mask is not None:
                label[image_keep_mask] = 0
                # image_rays = image_rays[image_keep_mask == True]
                # depth_scale = depth_scale[image_keep_mask == True]

            label = remapping(label)
            building_mask = label==1
            instance[~building_mask] = 0

            rgbs.append(image_rgbs)
            rays.append(image_rays)
            indices.append(image_indices)
            instances.append(torch.tensor(instance, dtype=torch.int))
            depth_djis.append(depth_dji / depth_scale)

        print(f"load_subset: {load_subset}")
        main_print('Finished loading data')

        self._rgbs = rgbs
        self._rays = rays
        self._img_indices = indices  #  这个代码只存了一个
        self._instances = instances
        self._depth_djis = depth_djis 

    def __len__(self) -> int:
        return len(self._rgbs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        
            
        # 找到非零值的索引
        nonzero_indices = torch.nonzero(self._instances[idx]).squeeze()

        
        #### 2. 点不够， 则换张图采样
        if nonzero_indices.size(0) == 0 or nonzero_indices.size(0) < self.hparams.batch_size:

            index_shuffle= list(range(len(self._rgbs)))
            index_shuffle.remove(idx) 

            # 从剩下的数字中随机选择一个数
            next_idx = random.choice(index_shuffle)
            item = self.__getitem__(next_idx)
            return item

        sampling_idx = nonzero_indices[torch.randperm(nonzero_indices.size(0))[:self.hparams.batch_size]]



        item = {
            'rgbs': self._rgbs[idx][sampling_idx].float() / 255.,
            'rays': self._rays[idx][sampling_idx],
            'img_indices': self._img_indices[idx] * torch.ones(sampling_idx.shape[0], dtype=torch.int32),
            'labels': self._instances[idx][sampling_idx].int(),
        }
        if self._depth_djis[idx] is not None:
            item['depth_dji'] = self._depth_djis[idx][sampling_idx]
        
        return item
    
