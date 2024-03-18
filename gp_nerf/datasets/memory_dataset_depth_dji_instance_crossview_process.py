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
import cv2
from functools import reduce

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
        self.metadata_items =metadata_items

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
        depth_scale_full = torch.abs(self._directions[:, :, 2]).view(-1).cpu()
        
        main_print('Loading data')
        if hparams.debug:
            metadata_items = metadata_items[205:210]
            pass

        load_subset = 0
        for metadata_item in main_tqdm(metadata_items):
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
            image_rgbs, image_indices, image_keep_mask, labels, depth_dji, instance = image_data
            
            image_rays = get_rays(self._directions, metadata_item.c2w.to(device), near, far, ray_altitude_range).view(-1, 8).cpu()
            
            depth_scale = depth_scale_full.clone()
            if image_keep_mask is not None:
                ###  左右部分不做投影
                instance[image_keep_mask]=0
                # image_rays = image_rays[image_keep_mask == True]
                # depth_scale = depth_scale[image_keep_mask == True]
                depth_dji[image_keep_mask]=torch.inf

            rgbs.append(image_rgbs)
            rays.append(image_rays)
            indices.append(image_indices)
            instances.append(torch.tensor(instance, dtype=torch.int))
            depth_djis.append(depth_dji / depth_scale)
            depth_scales.append(depth_scale)
        if load_subset ==0:
            print('there is something wrong, please check if there have $dataset_path/subset/rgbs or train/labels_fusion')
        else:
            print(f"load_subset: {load_subset}")
        main_print('Finished loading data')

        self._rgbs = rgbs
        self._rays = rays
        self._img_indices = indices  #  这个代码只存了一个
        self._labels = instances
        self._depth_djis = depth_djis 
        self._depth_scales = depth_scales 


    def __len__(self) -> int:
        return len(self._rgbs)
    

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        device='cpu'
        overlap_threshold=0.5
        ###NOTE shuffle=False, process in order

        

        # 拿到当前图像的数据
        img_current = self._rgbs[idx].clone().view(self.H, self.W, 3).to(device)
        instances_current = self._labels[idx].clone().view(self.H, self.W).to(device)
        depth_current = (self._depth_djis[idx] * self._depth_scales[idx]).view(self.H, self.W).to(device)
        metadata_current = self.metadata_items[self._img_indices[idx]]
        
        if self.hparams.start != -1:
            if int(Path(metadata_current.image_path).stem) < self.hparams.start or int(Path(metadata_current.image_path).stem) > (self.hparams.start +100):
                return None
        
        visualization = False
        if visualization:
            color_current = torch.zeros_like(img_current)
            unique_label = torch.unique(instances_current)
            for uni in unique_label:
                if uni ==0:
                    continue
                random_color = torch.randint(0, 256, (3,), dtype=torch.uint8).to(device)
                if (instances_current==uni).sum() != 0:
                    color_current[instances_current==uni,:] = random_color
            vis_img1 = 0.7 * color_current + 0.3 * img_current




        index_list= list(range(len(self._rgbs)))
        index_list.remove(idx)  
        ##### 新图像，用于存储 cross view 并集
        new_instance = torch.zeros_like(instances_current)

        unique_labels,counts = torch.unique(instances_current, return_counts=True)
        sorted_indices = torch.argsort(counts, descending=True)
        sorted_labels = unique_labels[sorted_indices]


        project_instances = [] 
        instances_nexts = []
        for idx_next in index_list:
            img_next = self._rgbs[idx_next].clone().view(self.H, self.W, 3).to(device)
            instances_next = self._labels[idx_next].clone().view(self.H, self.W).to(device)
            depth_next = (self._depth_djis[idx_next] * self._depth_scales[idx_next]).to(device)
            inf_mask = torch.isinf(depth_next)
            depth_next[inf_mask] = depth_next[~inf_mask].max()
            metadata_next = self.metadata_items[self._img_indices[idx_next]]

            ###### 投影
            
            x_grid, y_grid = torch.meshgrid(torch.arange(self.W), torch.arange(self.H))
            x_grid, y_grid = x_grid.T.flatten().to(device), y_grid.T.flatten().to(device)

            pixel_coordinates = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
            K1 = metadata_next.intrinsics
            K1 = torch.tensor([[K1[0], 0, K1[2]], [0, K1[1], K1[3]], [0, 0, 1]]).to(device)
            pt_3d = depth_next[:, None] * (torch.linalg.inv(K1) @ pixel_coordinates[:, :, None].float()).squeeze()
            arr2 = torch.ones((pt_3d.shape[0], 1)).to(device)
            pt_3d = torch.cat([pt_3d, arr2], dim=-1)
            # pt_3d = pt_3d[valid_depth_mask]
            pt_3d = pt_3d.view(-1, 4)
            E1 = metadata_next.c2w.clone().detach()
            E1 = torch.stack([E1[:, 0], -E1[:, 1], -E1[:, 2], E1[:, 3]], dim=1).to(device)
            world_point = torch.mm(torch.cat([E1, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0), pt_3d.t()).t()
            world_point = world_point[:, :3] / world_point[:, 3:4]

            ### 投影回第一张图
            E2 = metadata_current.c2w.clone().detach()
            E2 = torch.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1).to(device)
            w2c = torch.inverse(torch.cat((E2, torch.tensor([[0, 0, 0, 1]], device=device)), dim=0))
            points_homogeneous = torch.cat((world_point, torch.ones((world_point.shape[0], 1), dtype=torch.float32, device=device)), dim=1)
            pt_3d_trans = torch.mm(w2c, points_homogeneous.t())
            pt_2d_trans = torch.mm(K1, pt_3d_trans[:3])
            pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
            projected_points = pt_2d_trans[:2].t()

            threshold= 0.02
            image_width, image_height = self.W, self.H
            mask_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width)
            mask_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
            mask = mask_x & mask_y
            x = projected_points[:, 0].long()
            y = projected_points[:, 1].long()
            x[~mask] = 0
            y[~mask] = 0
            depth_map_current = depth_current[y, x]
            depth_map_current[~mask] = -1e6
            depth_z = pt_3d_trans[2]
            mask_z = depth_z < (depth_map_current + threshold)
            mask_xyz = (mask & mask_z)

            x[~mask_xyz] = 0
            y[~mask_xyz] = 0
            
            project_instance = torch.zeros_like(instances_current)
            project_instance[y[mask_xyz], x[mask_xyz]] = instances_next[y_grid[mask_xyz], x_grid[mask_xyz]]


            project_instances.append(project_instance)
            instances_nexts.append(instances_next)
            
        ## 每一个mask进行操作
        union_masks = []
        union_masks_score_list = []
        union_mask_label_list = []
        for unique_label in unique_labels:
            ####  为每一个mask创建一个list， 存储 需要合并的mask和 overlap分数
            if unique_label==0:
                continue
            merge_unique_label_list = []
            score_list = []

            mask_idx = instances_current == unique_label
            mask_idx_area = mask_idx.sum()
            if mask_idx_area < 0.001 * self.H * self.W:
                continue

            
                
            ######接下来对每个mask进行cross view 操作
            ## 以上投影结束后， 进行overlap计算
            for project_instance, instances_next in zip(project_instances, instances_nexts):
                label_in_mask = project_instance[mask_idx]
                uni_label_in_mask, count_label_in_mask = torch.unique(label_in_mask, return_counts=True)
                for uni_2 in uni_label_in_mask:
                    if uni_2 == 0:
                        continue
                    mask_2 = project_instance == uni_2
                    mask_area_2 = mask_2.sum()
                    mask_area_overlap = (mask_idx * mask_2).sum()
                    # 存储符合条件的并集mask 和 对应要融合区域大小的score
                    if (mask_area_overlap / mask_area_2) > overlap_threshold or (mask_area_overlap / mask_idx_area) > overlap_threshold:
                        if mask_area_2 < 0.001 * self.H * self.W or (instances_next==uni_2).sum() < 0.001 * self.H * self.W:
                            continue
                        merge_unique_label_list.append(mask_2)
                        score_list.append(mask_area_2)
  

            if merge_unique_label_list != []:
                sorted_data = sorted(zip(merge_unique_label_list, score_list), key=lambda x: x[1], reverse=False)

                merge_unique_label_list, score_list = zip(*sorted_data)
                merge_unique_label_list = list(merge_unique_label_list)
                merge_unique_label_list.append(mask_idx)

                ## 从小到大进行覆盖
                union_mask = reduce(torch.logical_or, merge_unique_label_list)
                union_masks.append(union_mask)
                union_masks_score = union_mask.sum()
                union_masks_score_list.append(union_masks_score)
                union_mask_label_list.append(unique_label)


        #### reverse=False 
        sorted_data = sorted(zip(union_masks, union_masks_score_list, union_mask_label_list), key=lambda x: x[1], reverse=False)

        union_masks, union_masks_score_list, union_mask_label_list = zip(*sorted_data)
        union_masks, union_mask_label_list = list(union_masks), list(union_mask_label_list)

        color_result = torch.zeros_like(img_current)
        for union_mask, unique_label in zip(union_masks, union_mask_label_list):
            new_instance[union_mask] = unique_label
            if visualization:
                color_current = torch.zeros_like(img_current)
                random_color = torch.randint(0, 256, (3,), dtype=torch.uint8).to(device)
                color_current[union_mask] = random_color
                color_result[union_mask] = random_color
                vis_img1 = 0.7 * color_current + 0.3 * img_current
                Path(f"{self.hparams.crossview_process_path}/test_{overlap_threshold}/union_mask_each").mkdir(exist_ok=True, parents=True)
                cv2.imwrite(f"{self.hparams.crossview_process_path}/test_{overlap_threshold}/union_mask_each/%06d_results_%06d.jpg" % (int(Path(metadata_current.label_path).stem), unique_label), vis_img1.cpu().numpy())






        if new_instance.sum() != 0:
            color_result = torch.zeros_like(img_current)

            color_current = torch.zeros_like(img_current)
            unique_label_results = torch.unique(instances_current)
            for uni in unique_label_results:
                if uni ==0:
                    continue
                random_color = torch.randint(0, 256, (3,), dtype=torch.uint8).to(device)
                color_current[instances_current==uni,:] = random_color
                color_result[new_instance==uni,:] = random_color
            


            vis_img1 = 0.7 * color_current + 0.3 * img_current
            vis_img4 = 0.7 * color_result + 0.3 * img_current

            
            vis_img5 = np.concatenate([vis_img1.cpu().numpy(), vis_img4.cpu().numpy()], axis=1)
            
            

            Path(f"{self.hparams.crossview_process_path}/test_{overlap_threshold}/results").mkdir(exist_ok=True, parents=True)
            cv2.imwrite(f"{self.hparams.crossview_process_path}/test_{overlap_threshold}/results/%06d_results_%06d.jpg" % (int(Path(metadata_current.label_path).stem), unique_label), vis_img5)
            



            Path(f"{self.hparams.dataset_path}/train/{self.hparams.instance_name}_crossview_process").mkdir(exist_ok=True, parents=True)
            np.save(f"{self.hparams.dataset_path}/train/{self.hparams.instance_name}_crossview_process/{Path(metadata_current.label_path).stem}.npy", new_instance.cpu().numpy().astype(np.uint32))

            

        if idx == len(self._rgbs) - 1:
            return 'end'
        else:
            print(f"process idx : {int(Path(metadata_current.label_path).stem)}")
            return None
    
