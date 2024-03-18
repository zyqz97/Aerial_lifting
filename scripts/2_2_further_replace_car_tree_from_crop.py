####选择一个图像，往相机视角方向飞


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


def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs/test',required=False, help='')
    
    parser.add_argument('--only_sam_m2f_far_project_path', type=str, default='',required=False, help='')
    parser.add_argument('--crop_m2f', type=str, default='',required=False, help='')
    
    parser.add_argument('--output_path', type=str, default='',required=False, help='')
    parser.add_argument('--eval', default=False, type=eval, choices=[True, False], help='')
    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:

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

    only_sam_m2f_far_project_path = hparams.only_sam_m2f_far_project_path

    output_path = hparams.output_path
    if not os.path.exists(output_path):
        Path(output_path).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'replace_cartree_label')):
        Path(os.path.join(output_path, 'replace_cartree_label')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'vis')):
        Path(os.path.join(output_path, 'vis')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'alpha')):
        Path(os.path.join(output_path, 'alpha')).mkdir(parents=True)
    if not os.path.exists(os.path.join(output_path, 'project_before_after')):
        Path(os.path.join(output_path, 'project_before_after')).mkdir(parents=True)
        

    # used_files = []
    # for ext in ('*.png', '*.jpg'):
    #     used_files.extend(glob(os.path.join(only_sam_m2f_far_project_path, ext)))
    # used_files.sort()
    # process_item = [Path(far_p).stem for far_p in used_files]
    if not hparams.eval:

        used_files = []
        for ext in ('*.png', '*.jpg'):
            used_files.extend(glob(os.path.join(hparams.dataset_path, 'subset', 'rgbs', ext)))
        used_files.sort()
        process_item = [Path(far_p).stem for far_p in used_files]

    # process_item=[]
    # for metadata_item in tqdm(train_items):
    #     gt_label = metadata_item.load_gt()
    #     has_nonzero = (gt_label != 0).any()
    #     non_zero_ratio = torch.sum(gt_label != 0).item() / gt_label.numel()
    #     if has_nonzero and non_zero_ratio>0.1:
    #         process_item.append(f"{metadata_item.image_path.stem}")
    # print(len(process_item))
    
    for metadata_item in tqdm(train_items, desc="replace car tree"):
        file_name = Path(metadata_item.image_path).stem
        if not hparams.eval:
                    
            if file_name not in process_item or metadata_item.is_val: # or int(file_name) != 182:
                continue
        
        # 读取 replace building 的标签文件
        far_m2f = Image.open(os.path.join(only_sam_m2f_far_project_path, file_name+'.png'))    #.convert('RGB')
        far_m2f = torch.ByteTensor(np.asarray(far_m2f))


        # 读取 m2f downscale=2 的 crop 4块处理后的文件
        crop_m2f = Image.open(os.path.join(hparams.crop_m2f, file_name+'.png'))    #.convert('RGB')
        crop_m2f = torch.ByteTensor(np.asarray(crop_m2f))

        
        H, W = far_m2f.shape
        image_width, image_height = int(W), int(H)


        # 进一步覆盖掉car  tree 部分的标签
        car_mask = (crop_m2f==3)
        tree_mask = (crop_m2f==4)

        replace_label = far_m2f.clone()
        replace_label[replace_label==3]=0
        replace_label[replace_label==4]=0

        replace_label[car_mask] = 3
        replace_label[tree_mask] = 4


        Image.fromarray(replace_label.numpy().astype(np.uint16)).save(os.path.join(output_path, 'replace_cartree_label', f"{file_name}.png"))
        

        color_label = custom2rgb(replace_label.numpy())
        color_label = color_label.reshape(H, W, 3)
        Image.fromarray(color_label.astype(np.uint8)).save(os.path.join(output_path, 'vis',f"{file_name}.jpg"))


        img = metadata_item.load_image()
        merge = 0.7 * img.numpy() + 0.3 * color_label
        Image.fromarray(merge.astype(np.uint8)).save(os.path.join(output_path, 'alpha',f"{file_name}.jpg"))
        

        
        cat = torch.hstack([far_m2f, replace_label])
        cat = custom2rgb(cat.numpy())
        cat = cat.reshape(H, 2*W, 3)
        Image.fromarray(cat.astype(np.uint8)).save(os.path.join(output_path, 'project_before_after', f"{file_name}.png"))
        a=1

    print('done')


if __name__ == '__main__':

    hello(_get_train_opts())
