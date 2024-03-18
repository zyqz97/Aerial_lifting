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

def inter_two_poses(pose_a, pose_b, alpha):
    ret = np.zeros([3, 4], dtype=np.float64)
    rot_a = R.from_matrix(pose_a[:3, :3])
    rot_b = R.from_matrix(pose_b[:3, :3])
    key_rots = R.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], 0))
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)
    rot = slerp(1. - alpha)
    ret[:3, :3] = rot.as_matrix()
    ret[:3, 3] = (pose_a * alpha + pose_b * (1. - alpha))[:3, 3]
    return ret


def inter_poses(key_poses, n_out_poses, sigma=1.):
    n_key_poses = len(key_poses)
    out_poses = []
    for i in range(n_out_poses):
        w = np.linspace(0, n_key_poses - 1, n_key_poses)
        w = np.exp(-(np.abs(i / n_out_poses * n_key_poses - w) / sigma)**2)
        w = w + 1e-6
        w /= np.sum(w)
        cur_pose = key_poses[0]
        cur_w = w[0]
        for j in range(0, n_key_poses - 1):
            cur_pose = inter_two_poses(cur_pose, key_poses[j + 1], cur_w / (cur_w + w[j + 1]))
            cur_w += w[j + 1]

        out_poses.append(cur_pose)

    return np.stack(out_poses)

from pathlib import Path
from mega_nerf.ray_utils import get_rays, get_ray_directions



def _get_train_opts() -> Namespace:
    parser = get_opts_base()
    parser.add_argument('--dataset_path', type=str, default='',required=False, help='')
    parser.add_argument('--exp_name', type=str, default='logs/test',required=False, help='experiment name')
    
    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    hparams.ray_altitude_range = [-95, 54]
    hparams.dataset_type='memory_depth_dji'
    
    runner = Runner(hparams)
    train_items = runner.train_items



    # for metadata_item in tqdm(train_items[200:240]):
    for metadata_item in tqdm(train_items):
        file_name = Path(metadata_item.image_path).stem
        
        pose = metadata_item.c2w
        

        directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        True,
                                        torch.device('cpu'))
        image_rays = get_rays(directions, metadata_item.c2w, 0, 1e5, [30, 118])
        ray_d = image_rays[int(metadata_item.H/2), int(metadata_item.W/2), 3:6]
        ray_o = image_rays[int(metadata_item.H/2), int(metadata_item.W/2), :3]
        

        z_vals_inbound = 0.3
        new_o = ray_o - ray_d * z_vals_inbound

        new_pose = pose
        new_pose[:,3]= new_o

        metadata_path = os.path.join(hparams.dataset_path, 'train', 'metadata', f"{metadata_item.image_path.stem}.pt")

        metadata_old = torch.load(metadata_path)
        metadata_old['c2w'] = new_pose

        if not os.path.exists(os.path.join(hparams.dataset_path, f'render_far{z_vals_inbound}', 'metadata')):
            Path(os.path.join(hparams.dataset_path, f'render_far{z_vals_inbound}')).mkdir(parents=True, exist_ok=True)
            Path(os.path.join(hparams.dataset_path, f'render_far{z_vals_inbound}', 'metadata')).mkdir(parents=True, exist_ok=True)

        torch.save(metadata_old, os.path.join(hparams.dataset_path, f'render_far{z_vals_inbound}', 'metadata', f"{metadata_item.image_path.stem}.pt"))



if __name__ == '__main__':
    hello(_get_train_opts())
