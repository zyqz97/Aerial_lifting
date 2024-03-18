

import click
import os
import numpy as np
import cv2 as cv
from os.path import join as pjoin
from glob import glob
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import torch
from gp_nerf.runner_gpnerf import Runner
from gp_nerf.opts import get_opts_base
from argparse import Namespace
from tqdm import tqdm
import cv2
from tools.unetformer.uavid2rgb import rgb2custom, custom2rgb,remapping

from PIL import Image
from pathlib import Path
import open3d as o3d
import pickle
import math
import xml.etree.ElementTree as ET
from collections import Counter
from pyntcloud import PyntCloud
import pandas as pd
from torch.nn.functional import interpolate
from mega_nerf.ray_utils import get_ray_directions


# torch.cuda.set_device(6)


def calculate_entropy(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    total_labels = len(labels)

    probabilities = label_counts / total_labels
    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy


def custom2rgb_1(mask):
    N= mask.shape[0]
    mask_rgb = np.zeros(shape=(N, 3), dtype=np.uint8)
    mask_convert = mask[np.newaxis, :]
    mask_rgb[np.all(mask_convert == 0, axis=0)] = [0, 0, 0]             # cluster       black
    
    mask_rgb[np.all(mask_convert == 1, axis=0)] = [128, 0, 0]           # building      red
    mask_rgb[np.all(mask_convert == 2, axis=0)] = [192, 192, 192]       # road        grey  
    mask_rgb[np.all(mask_convert == 3, axis=0)] = [192, 0, 192]         # car           light violet
    mask_rgb[np.all(mask_convert == 4, axis=0)] = [0, 128, 0]           # tree          green
    mask_rgb[np.all(mask_convert == 5, axis=0)] = [128, 128, 0]         # vegetation    dark green
    mask_rgb[np.all(mask_convert == 6, axis=0)] = [255, 255, 0]         # human         yellow
    mask_rgb[np.all(mask_convert == 7, axis=0)] = [135, 206, 250]       # sky           light blue
    mask_rgb[np.all(mask_convert == 8, axis=0)] = [0, 0, 128]           # water         blue

    mask_rgb[np.all(mask_convert == 9, axis=0)] = [252,230,201]          # ground       egg
    mask_rgb[np.all(mask_convert == 10, axis=0)] = [128, 64, 128]        # mountain     dark violet

    # mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_RGB2BGR)
    return mask_rgb



def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs/test',required=False, help='')
    parser.add_argument('--far_paths', type=str, default='',required=False, help='')
    parser.add_argument('--output_path', type=str, default='',required=False, help='')
    parser.add_argument('--render_type', type=str, default='render_far0.3',required=False, help='')
    parser.add_argument('--eval', default=False, type=eval, choices=[True, False], help='')
    

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:


    far_paths = hparams.far_paths
    output_path = hparams.output_path
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'project_far_to_ori')):
        Path(os.path.join(output_path, 'project_far_to_ori')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'vis')):
        Path(os.path.join(output_path, 'vis')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'alpha')):
        Path(os.path.join(output_path, 'alpha')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'project_before_after')):
        Path(os.path.join(output_path, 'project_before_after')).mkdir(parents=True)


    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    device = 'cpu'
    hparams.label_name = 'm2f' # ['m2f', 'merge', 'gt']
    if 'Longhua' in hparams.dataset_path:
        hparams.train_scale_factor =1
        hparams.val_scale_factor =1

    runner = Runner(hparams)

    if not hparams.eval:
        train_items = runner.train_items
    else:
        train_items = runner.val_items



    if not hparams.eval:
        used_files = []
        for ext in ('*.png', '*.jpg'):
            used_files.extend(glob(os.path.join(hparams.dataset_path, 'subset', 'rgbs', ext)))
        used_files.sort()
        process_item = [Path(far_p).stem for far_p in used_files]
    
    for metadata_item in tqdm(train_items, desc="project to ori"):
        file_name = Path(metadata_item.image_path).stem
        
        if not hparams.eval:
            if file_name not in process_item or metadata_item.is_val: # or int(file_name) != 182:
                continue
        
        directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        False,
                                        'cpu')
        depth_scale = torch.abs(directions[:, :, 2]) # z-axis's values
        
        # 读取far的标签文件
        far_m2f = Image.open(os.path.join(far_paths, file_name+'.png'))    #.convert('RGB')
        far_m2f = torch.ByteTensor(np.asarray(far_m2f))


        # 读取ori的标签文件
        ori_m2f = metadata_item.load_label()

        # 读取深度
        depth_map = metadata_item.load_depth_dji().squeeze(-1).float()

        H, W = depth_map.shape

        # nan_mask = torch.isnan(depth_map)
        inf_mask = torch.isinf(depth_map)

        depth_map[inf_mask] = depth_map[~inf_mask].max()
        # depth_map[nan_mask] = interpolate(depth_map[None, None, ...], size=(H,W),mode='bilinear', align_corners=False)[0, 0, nan_mask]

        ## 1. 用2d和depth转换成点云
        x_grid, y_grid = torch.meshgrid(torch.arange(W), torch.arange(H))
        x_grid, y_grid = x_grid.T, y_grid.T

        pixel_coordinates = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
        K1 = metadata_item.intrinsics
        K1 = torch.tensor([[K1[0], 0, K1[2]], [0, K1[1], K1[3]], [0, 0, 1]])
        pt_3d = depth_map[:, :, None] * (torch.linalg.inv(K1) @ pixel_coordinates[:, :, :, None].float()).squeeze()
        arr2 = torch.ones((pt_3d.shape[0], pt_3d.shape[1], 1))
        pt_3d = torch.cat([pt_3d, arr2], dim=-1)
        # pt_3d = pt_3d[valid_depth_mask]
        pt_3d = pt_3d.view(-1, 4)
        E1 = torch.tensor(metadata_item.c2w)
        E1 = torch.stack([E1[:, 0], -E1[:, 1], -E1[:, 2], E1[:, 3]], dim=1)
        world_point = torch.mm(torch.cat([E1, torch.tensor([[0, 0, 0, 1]])], dim=0), pt_3d.t()).t()
        world_point = world_point[:, :3] / world_point[:, 3:4]



        # 点云投影到far
        metadata_far = torch.load(os.path.join(hparams.dataset_path, hparams.render_type, 'metadata', file_name+'.pt'), map_location='cpu')


        camera_rotation = metadata_far['c2w'][:3,:3].to(device)
        camera_position = metadata_far['c2w'][:3, 3].to(device)

        # camera_rotation = metadata_item.c2w[:3,:3].to(device)
        # camera_position = metadata_item.c2w[:3, 3].to(device)

        camera_matrix = torch.tensor([[metadata_item.intrinsics[0], 0, metadata_item.intrinsics[2]],
                            [0, metadata_item.intrinsics[1], metadata_item.intrinsics[3]],
                            [0, 0, 1]]).to(device)

        
        # NOTE: 
        E2 = torch.hstack((camera_rotation, camera_position.unsqueeze(-1)))
        E2 = torch.stack([E2[:, 0], E2[:, 1]*-1, E2[:, 2]*-1, E2[:, 3]], 1).to(device)
        w2c = torch.inverse(torch.cat((E2, torch.tensor([[0, 0, 0, 1]]).to(device)), dim=0))
        points_homogeneous = torch.cat((world_point, torch.ones((world_point.shape[0], 1), dtype=torch.float32).to(device)), dim=1)
        pt_3d_trans = torch.mm(w2c, points_homogeneous.t())
        
        pt_2d_trans = torch.mm(camera_matrix, pt_3d_trans[:3])
        pt_2d_trans = pt_2d_trans / pt_2d_trans[2]
        projected_points = pt_2d_trans[:2].t()
        


        ########### 考虑遮挡
        threshold= 0.02
        # threshold= 0.005

        large_int = 1e6
        image_width, image_height = int(W), int(H)
        project_m2f = torch.zeros((image_height, image_width)).long().to(device)

        mask_x = (projected_points[:, 0] >= 0) & (projected_points[:, 0] < image_width)
        mask_y = (projected_points[:, 1] >= 0) & (projected_points[:, 1] < image_height)
        mask = mask_x & mask_y
        # meshMask = metadata_item.load_depth_dji().float().to(device)

        ### NOTE:  NeRF render  depth
        farMask = torch.from_numpy(np.load(os.path.join(str(Path(far_paths).parent.parent), "pred_depth_save", f"{file_name}.npy")))
        farMask = (farMask) * depth_scale



        x = projected_points[:, 0].long()
        y = projected_points[:, 1].long()
        x[~mask] = 0
        y[~mask] = 0
        far_depths = farMask[y, x]
        far_depths[~mask] = -1e6

        depth_z = pt_3d_trans[2]
        # mask_z = depth_z < (far_depths + threshold)
        mask_z = depth_z < (far_depths + threshold)
        mask_xyz = mask & mask_z

        project_m2f=far_m2f[y, x].view(H,W)
        project_m2f[~(mask_xyz.view(H,W))]= 0
        ##########


        #  不做遮挡的考虑
        # x = projected_points[:, 0]
        # y = projected_points[:, 1]
        # x = x.long()
        # y = y.long()
        # # far_m2f = ori_m2f
        # project_m2f = far_m2f[y, x].view(H,W)

        Image.fromarray(project_m2f.numpy().astype(np.uint16)).save(os.path.join(output_path, 'project_far_to_ori', f"{file_name}.png"))
        

        color_label = custom2rgb(project_m2f.numpy())
        color_label = color_label.reshape(H, W, 3)
        Image.fromarray(color_label.astype(np.uint8)).save(os.path.join(output_path, 'vis',f"{file_name}.jpg"))


        img = metadata_item.load_image()
        merge = 0.7 * img.numpy() + 0.3 * color_label
        Image.fromarray(merge.astype(np.uint8)).save(os.path.join(output_path, 'alpha',f"{file_name}.jpg"))
        

        
        cat = torch.hstack([project_m2f, far_m2f])
        cat = custom2rgb(cat.numpy())
        cat = cat.reshape(H, 2*W, 3)
        Image.fromarray(cat.astype(np.uint8)).save(os.path.join(output_path, 'project_before_after', f"{file_name}.png"))
        a=1
           
    print('done')


if __name__ == '__main__':

    hello(_get_train_opts())
