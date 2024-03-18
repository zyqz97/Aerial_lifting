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
from tools.unetformer.uavid2rgb import rgb2custom, custom2rgb
from PIL import Image
from pathlib import Path
import open3d as o3d


def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--rgbs_path', type=str, default='',required=False, help='')
    parser.add_argument('--save_path', type=str, default='',required=False, help='experiment name')

    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    rgbs_path = hparams.rgbs_path
    save_path = hparams.save_path

    Path(save_path).mkdir(exist_ok=True)

    used_files = []
    for ext in ('*.png', '*.jpg'):
        used_files.extend(glob(os.path.join(rgbs_path, ext)))
    used_files.sort()
    H, W = 1024, 1536

    
    for image_path in tqdm(used_files):
        rgbs = Image.open(image_path).convert('RGB')
        # W, H = rgbs.size
        # W, H = int(W/4), int(H/4) 
        rgbs = rgbs.resize((W, H), Image.LANCZOS)
        # rgbs = rgbs.resize((W, H), Image.BILINEAR)
        # rgbs = rgbs.resize((W, H), Image.INTER_AREA)

        
        rgbs.save(os.path.join(save_path, f'{Path(image_path).name}'))
        
    
    print("done")




if __name__ == '__main__':
    hello(_get_train_opts())
