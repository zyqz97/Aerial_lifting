from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_depth_dji
from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_rays, get_ray_directions

import numpy as np
import glob
import os
from pathlib import Path
import random
from PIL import Image

class MemoryDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams=None):
        super(MemoryDataset, self).__init__()
        self.hparams = hparams
        rgbs = []
        rays = []
        indices = []
        labels = []
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
        
        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[::20]
        load_subset = 0
        for metadata_item in main_tqdm(metadata_items):
        # for metadata_item in main_tqdm(metadata_items[:40]):
            if hparams.enable_semantic and metadata_item.is_val:  # 训练语义的时候要去掉val图像
                continue
            # if hparams.use_subset and ('Yingrenshi' in hparams.dataset_path):
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

            image_data = get_rgb_index_mask_depth_dji(metadata_item)

            if image_data is None:
                continue
            
            #zyq : add labels
            image_rgbs, image_indices, image_keep_mask, label, depth_dji = image_data

            # print("image index: {}, fx: {}, fy: {}".format(metadata_item.image_index, metadata_item.intrinsics[0], metadata_item.intrinsics[1]))
            
            depth_scale = torch.abs(self._directions[:, :, 2]).view(-1).cpu()
            image_rays = get_rays(self._directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1, 8).cpu()
            
            
            if image_keep_mask is not None:
                image_rays = image_rays[image_keep_mask == True]
                depth_scale = depth_scale[image_keep_mask == True]

            rgbs.append(image_rgbs)
            rays.append(image_rays)
            indices.append(image_indices)
            if label is not None:
                labels.append(torch.tensor(label, dtype=torch.int))
            if depth_dji != None:
                depth_djis.append(depth_dji / depth_scale)
            # depth_scales.append(depth_scale)

        print(f"load_subset: {load_subset}")
        main_print('Finished loading data')

        self._rgbs = torch.cat(rgbs)
        self._rays = torch.cat(rays)
        self._img_indices = torch.cat(indices)
        if labels != []:
            self._labels = torch.cat(labels)
        else:
            self._labels = []
        if depth_djis != []:
            self._depth_djis = torch.cat(depth_djis)
        else:
            self._depth_djis = []

        # self._depth_scales = torch.cat(depth_scales)

    def __len__(self) -> int:
        return self._rgbs.shape[0]

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        item = {
            'rgbs': self._rgbs[idx].float() / 255.,
            'rays': self._rays[idx],
            'img_indices': self._img_indices[idx],
            
        }
        if self._labels != []:
            item['labels'] = self._labels[idx].int()
        
        if self._depth_djis != []:
            item['depth_dji'] = self._depth_djis[idx]
            # item['depth_scale'] = self._depth_scales[idx]

        return item
    
