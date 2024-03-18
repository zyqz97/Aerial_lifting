from typing import List, Dict

import torch
from torch.utils.data import Dataset

from gp_nerf.datasets.dataset_utils import get_rgb_index_mask_depth_dji_instance_crossview
from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.misc_utils import main_tqdm, main_print
from mega_nerf.ray_utils import get_rays, get_ray_directions

import numpy as np
import glob
import os
from pathlib import Path
import random
from PIL import Image
import cv2
from tools.unetformer.uavid2rgb import remapping


class MemoryDataset(Dataset):

    def __init__(self, metadata_items: List[ImageMetadata], near: float, far: float, ray_altitude_range: List[float],
                 center_pixels: bool, device: torch.device, hparams=None):
        super(MemoryDataset, self).__init__()
        if hparams.debug:
            self.visualization=True
        else:
            self.visualization=False

        self.hparams = hparams
        rgbs = []
        rays = []
        indices = []
        instances = []
        instances_64 = []
        instances_crossview = []
        labels = []
        sample_dicts = []
        self.metadata_items = metadata_items

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
            metadata_items = metadata_items[155:156]
            pass

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

            image_data = get_rgb_index_mask_depth_dji_instance_crossview(metadata_item)

            if image_data is None:
                continue
            #改成读取instance label
            image_rgbs, image_indices, image_keep_mask, label, depth_dji, instance, instance_crossview, instance_64 = image_data
            
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
            labels.append(label)
            depth_djis.append(depth_dji / depth_scale)

            ### 下面提取crossview的dict

            sample_dict = {}


            unique_labels,counts = torch.unique(instance, return_counts=True)
            non_zero_indices = unique_labels != 0
            unique_labels = unique_labels[non_zero_indices]
            counts = counts[non_zero_indices]
            
            for uni in unique_labels:
                uni_mask = instance==uni
                label_in_crossview = instance_crossview[uni_mask]
                unique_label_in_crossview,counts_label_in_crossview = torch.unique(label_in_crossview, return_counts=True)
                non_zero_indices = (unique_label_in_crossview != 0)
                unique_label_in_crossview = unique_label_in_crossview[non_zero_indices]
                counts_label_in_crossview = counts_label_in_crossview[non_zero_indices]
                if counts_label_in_crossview.shape[0] == 0:
                    continue
                max_count_index = torch.argmax(counts_label_in_crossview)
                most_frequent_label = unique_label_in_crossview[max_count_index]
                if uni == 7027:
                    a = 1
                if uni_mask.nonzero().shape[0] > 0.0005 * self.H * self.W:
                    
                    if counts_label_in_crossview[max_count_index]/label_in_crossview.shape[0] < 0.5:
                        continue
                    if f'{most_frequent_label}' not in sample_dict:
                        sample_dict[f'{most_frequent_label}'] = []
                    each_mask = torch.zeros_like(instance)
                    each_mask[uni_mask] = instance[uni_mask]
                    sample_dict[f'{most_frequent_label}'].append(each_mask)
            
            sample_dicts.append(sample_dict)

        print(f"load_subset: {load_subset}")
        main_print('Finished loading data')

        self._rgbs = rgbs
        self._rays = rays
        self._img_indices = indices  #  这个代码只存了一个
        self._labels = labels
        self._depth_djis = depth_djis 
        self._sample_dicts = sample_dicts



    def __len__(self) -> int:
        return len(self._rgbs)

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        metadata_current = self.metadata_items[self._img_indices[idx]]

        

        selected_tensors = {}
        if self.hparams.crossview_all: # turn on when using longhua-b1 
            for key, tensor_list in self._sample_dicts[idx].items():
                if tensor_list:
                    modified_masks = [mask.masked_fill(mask != 0, int(key)) for mask in tensor_list]
                    union_modified_masks = torch.stack(modified_masks).sum(0)
                    selected_tensors[key] = union_modified_masks
        else:
            for key, tensor_list in self._sample_dicts[idx].items():
                if tensor_list:
                    random_index = np.random.randint(0, len(tensor_list))
                    selected_tensors[key] = tensor_list[random_index]

        tensor_list = list(selected_tensors.values())
        if tensor_list == []:
            return None
        stacked_tensors = torch.stack(tensor_list)
        instance_new = torch.sum(stacked_tensors, dim=0)



        nonzero_indices = torch.nonzero(instance_new).squeeze()

        
        
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
            'labels': instance_new[sampling_idx].int(),
        }
        if self._depth_djis[idx] is not None:
            item['depth_dji'] = self._depth_djis[idx][sampling_idx]
        
        return item
    
