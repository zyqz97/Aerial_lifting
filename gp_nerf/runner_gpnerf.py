import datetime
import faulthandler
import math
import os
import random
import shutil
import signal
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.cuda.amp import GradScaler
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch.utils.data import DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
from tqdm import tqdm
import gc


from gp_nerf.image_metadata import ImageMetadata
from mega_nerf.metrics import psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
from mega_nerf.ray_utils import get_rays, get_ray_directions
from gp_nerf.models.model_utils import get_nerf, get_bg_nerf

import wandb
from torchvision.utils import make_grid

#semantic
from tools.unetformer.uavid2rgb import uavid2rgb, custom2rgb, remapping
from tools.unetformer.uavid2rgb import custom2rgb_point
from tools.unetformer.metric import Evaluator

import pandas as pd

from gp_nerf.eval_utils import get_depth_vis, get_semantic_gt_pred, get_sdf_normal_map, get_semantic_gt_pred_render_zyq, get_instance_pred, calculate_panoptic_quality_folders
from tools.contrastive_lift.utils import cluster, visualize_panoptic_outputs, assign_clusters
from gp_nerf.eval_utils import calculate_metric_rendering, write_metric_to_folder_logger, save_semantic_metric
from gp_nerf.eval_utils import prepare_depth_normal_visual


from tools.segment_anything import sam_model_registry, SamPredictor

# nr3d_sdf
from typing import Literal, Union
from nr3d_lib.logger import Logger
from nr3d_lib.models.loss.safe import safe_mse_loss
from gp_nerf.sample_bg import contract_to_unisphere_new
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import scipy
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree
import pickle

from tools.contrastive_lift.utils import create_instances_from_semantics

import xml.etree.ElementTree as ET
from pyntcloud import PyntCloud


to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            # print(s)
            nn = nn*s
        pp += nn
    return pp

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None

    return torch.utils.data.dataloader.default_collate(batch)

def init_predictor(device):
    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    
    model_type = "vit_h"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    print("SAM initializd.")
    predictor = SamPredictor(sam)
    return predictor

def contrastive_loss(features, instance_labels, temperature):
    bsize = features.size(0)
    masks = instance_labels.view(-1, 1).repeat(1, bsize).eq_(instance_labels.clone())
    masks = masks.fill_diagonal_(0, wrap=False)

    # compute similarity matrix based on Euclidean distance
    distance_sq = torch.pow(features.unsqueeze(1) - features.unsqueeze(0), 2).sum(dim=-1)
    # temperature = 1 for positive pairs and temperature for negative pairs
    temperature = torch.ones_like(distance_sq) * temperature
    temperature = torch.where(masks==1, temperature, torch.ones_like(temperature))

    similarity_kernel = torch.exp(-distance_sq/temperature)
    logits = torch.exp(similarity_kernel)

    p = torch.mul(logits, masks).sum(dim=-1)
    Z = logits.sum(dim=-1)

    prob = torch.div(p, Z)
    prob_masked = torch.masked_select(prob, prob.ne(0))
    loss = -prob_masked.log().sum()/bsize
    return loss


def rad(x):
    return math.radians(x)

def zhitu_2_nerf(dataset_path, metaXml_path, points_xyz):
    # to process points_rgb
    coordinate_info = torch.load(dataset_path + '/coordinates.pt')
    origin_drb = coordinate_info['origin_drb'].numpy()
    pose_scale_factor = coordinate_info['pose_scale_factor']

    root = ET.parse(metaXml_path).getroot()
    translation = np.array(root.find('SRSOrigin').text.split(',')).astype(np.float)    

    #######################################
    ZYQ = torch.DoubleTensor([[0, 0, -1],
                            [0, 1, 0],
                            [1, 0, 0]])
    ZYQ_1 = torch.DoubleTensor([[1, 0, 0],
                            [0, math.cos(rad(135)), math.sin(rad(135))],
                            [0, -math.sin(rad(135)), math.cos(rad(135))]])      
    # points_nerf = np.array(xyz)
    points_nerf = points_xyz
    points_nerf += translation
    points_nerf = ZYQ.numpy() @ points_nerf.T
    points_nerf = (ZYQ_1.numpy() @ points_nerf).T
    points_nerf = (points_nerf - origin_drb) / pose_scale_factor
    return points_nerf

class Runner:
    def __init__(self, hparams: Namespace, set_experiment_path: bool = True):
        faulthandler.register(signal.SIGUSR1)
        print(f"ignore_index: {hparams.ignore_index}")
        self.temperature = 100
        self.thing_classes=[1]
        # use when instance_loss_mode == 'linear_assignment'
        self.loss_instances_cluster = torch.nn.CrossEntropyLoss(reduction='none')

        if hparams.balance_weight:
            balance_weight = torch.FloatTensor([1, 1, 1, 2, 1]).cuda()

            CrossEntropyLoss = nn.CrossEntropyLoss(weight=balance_weight, ignore_index=hparams.ignore_index)
        else:
            CrossEntropyLoss = nn.CrossEntropyLoss(ignore_index=hparams.ignore_index)

        self.crossentropy_loss = lambda logit, label: CrossEntropyLoss(logit, label)
        self.logits_2_label = lambda x: torch.argmax(torch.nn.functional.softmax(x, dim=-1),dim=-1)
        

        self.color_list = torch.randint(0, 255,(100, 3)).to(torch.float32)

        if hparams.depth_loss:
            from gp_nerf.loss_monosdf import ScaleAndShiftInvariantLoss
            self.depth_loss = ScaleAndShiftInvariantLoss(alpha=0.5, scales=1)

        if hparams.normal_loss:
            from gp_nerf.loss_monosdf import get_l1_normal_loss
            self.normal_loss = get_l1_normal_loss

        if hparams.ckpt_path is not None:
            checkpoint = torch.load(hparams.ckpt_path, map_location='cpu')
            np.random.set_state(checkpoint['np_random_state'])     
            torch.set_rng_state(checkpoint['torch_random_state'])  
            random.setstate(checkpoint['random_state'])
        else:
            np.random.seed(hparams.random_seed)
            torch.manual_seed(hparams.random_seed)
            random.seed(hparams.random_seed)

        self.hparams = hparams
        
 
        
        if 'RANK' in os.environ:
            dist.init_process_group(backend='nccl', timeout=datetime.timedelta(0, hours=24))
            torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
            self.is_master = (int(os.environ['RANK']) == 0)
        else:
            self.is_master = True

        self.is_local_master = ('RANK' not in os.environ) or int(os.environ['LOCAL_RANK']) == 0
        main_print(hparams)

        if set_experiment_path:
            self.experiment_path = self._get_experiment_path() if self.is_master else None
            self.model_path = self.experiment_path / 'models' if self.is_master else None

        
        self.wandb = None
        self.writer = None

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu')
        self.origin_drb = coordinate_info['origin_drb']
        self.pose_scale_factor = coordinate_info['pose_scale_factor']
        main_print('Origin: {}, scale factor: {}'.format(self.origin_drb, self.pose_scale_factor))

        self.near = (hparams.near / self.pose_scale_factor)

        if self.hparams.far is not None:
            self.far = hparams.far / self.pose_scale_factor
        elif hparams.bg_nerf:
            self.far = 1e5
        elif hparams.gpnerf:
            self.far = 1e5
        else:
            self.far = 2
        main_print('Ray bounds: {}, {}'.format(self.near, self.far))

            
        self.ray_altitude_range = [(x - self.origin_drb[0]) / self.pose_scale_factor for x in
                                hparams.ray_altitude_range] if hparams.ray_altitude_range is not None else None
        main_print('Ray altitude range in [-1, 1] space: {}'.format(self.ray_altitude_range))
        main_print('Ray altitude range in metric space: {}'.format(hparams.ray_altitude_range))

        if self.ray_altitude_range is not None:
            assert self.ray_altitude_range[0] < self.ray_altitude_range[1]


        if self.hparams.cluster_mask_path is not None:
            cluster_params = torch.load(Path(self.hparams.cluster_mask_path).parent / 'params.pt', map_location='cpu')
            assert cluster_params['near'] == self.near
            assert (torch.allclose(cluster_params['origin_drb'], self.origin_drb))
            assert cluster_params['pose_scale_factor'] == self.pose_scale_factor

            if self.ray_altitude_range is not None:
                assert (torch.allclose(torch.FloatTensor(cluster_params['ray_altitude_range']),
                                    torch.FloatTensor(self.ray_altitude_range))), \
                    '{} {}'.format(self.ray_altitude_range, cluster_params['ray_altitude_range'])

        self.train_items, self.val_items = self._get_image_metadata()



        main_print('Using {} train images and {} val images'.format(len(self.train_items), len(self.val_items)))

        camera_positions = torch.cat([x.c2w[:3, 3].unsqueeze(0) for x in self.train_items + self.val_items])
        min_position = camera_positions.min(dim=0)[0]
        max_position = camera_positions.max(dim=0)[0]

        main_print('Camera range in metric space: {} {}'.format(min_position * self.pose_scale_factor + self.origin_drb,
                                                                max_position * self.pose_scale_factor + self.origin_drb))

        main_print('Camera range in [-1, 1] space: {} {}'.format(min_position, max_position))

        if hparams.ellipse_bounds or (hparams.ellipse_bounds and hparams.gpnerf):
            assert hparams.ray_altitude_range is not None

            if self.ray_altitude_range is not None:
                ground_poses = camera_positions.clone()
                ground_poses[:, 0] = self.ray_altitude_range[1]
                air_poses = camera_positions.clone()
                air_poses[:, 0] = self.ray_altitude_range[0]
                used_positions = torch.cat([camera_positions, air_poses, ground_poses])
            else:
                used_positions = camera_positions

            max_position[0] = self.ray_altitude_range[1]
            main_print('Camera range in [-1, 1] space with ray altitude range: {} {}'.format(min_position,
                                                                                            max_position))

            self.sphere_center = ((max_position + min_position) * 0.5).to(self.device)
            self.sphere_radius = ((max_position - min_position) * 0.5).to(self.device)
            scale_factor = ((used_positions.to(self.device) - self.sphere_center) / self.sphere_radius).norm(
                dim=-1).max()
            self.sphere_radius *= (scale_factor * hparams.ellipse_scale_factor)
            main_print('Sphere center: {}, radius: {}'.format(self.sphere_center, self.sphere_radius))
            hparams.z_range = self.ray_altitude_range
            hparams.sphere_center=self.sphere_center
            hparams.sphere_radius=self.sphere_radius
            hparams.aabb_bound = max(self.sphere_radius)

            #aabb for nr3d 
            z_range = torch.tensor(hparams.z_range, dtype=torch.float32)
            hparams.stretch = torch.tensor([[z_range[0], hparams.sphere_center[1] - hparams.sphere_radius[1], hparams.sphere_center[2] - hparams.sphere_radius[2]], 
                                            [z_range[1], hparams.sphere_center[1] + hparams.sphere_radius[1], hparams.sphere_center[2] + hparams.sphere_radius[2]]]).to(self.device)
            hparams.pose_scale_factor = self.pose_scale_factor

        else:
            self.sphere_center = None
            self.sphere_radius = None
        
        self.nerf = get_nerf(hparams, len(self.train_items)).to(self.device)

        if 'RANK' in os.environ:
            self.nerf = torch.nn.parallel.DistributedDataParallel(self.nerf, device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                output_device=int(os.environ['LOCAL_RANK']))

        if hparams.bg_nerf:
            self.bg_nerf = get_bg_nerf(hparams, len(self.train_items)).to(self.device)
            if 'RANK' in os.environ:
                self.bg_nerf = torch.nn.parallel.DistributedDataParallel(self.bg_nerf,
                                                                        device_ids=[int(os.environ['LOCAL_RANK'])],
                                                                        output_device=int(os.environ['LOCAL_RANK']))
        else:
            self.bg_nerf = None


        if hparams.bg_nerf:
            bg_parameters = get_n_params(self.bg_nerf)
        else:
            bg_parameters = 0
        fg_parameters = get_n_params(self.nerf)
        print("the parameters of whole model:\t total: {}, fg: {}, bg: {}".format(fg_parameters+bg_parameters,fg_parameters,bg_parameters))
        if self.wandb is not None:
            self.wandb.log({"parameters/fg": fg_parameters})
            self.wandb.log({"parameters/bg": bg_parameters})

    def train(self):

        self._setup_experiment_dir()
        scaler = torch.cuda.amp.GradScaler(enabled=self.hparams.amp)

        if (self.hparams.enable_semantic or self.hparams.enable_instance) and self.hparams.freeze_geo and self.hparams.ckpt_path is not None:
            
            for p_base in self.nerf.encoder_bg.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.plane_encoder.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.sigma_net.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.sigma_net_bg.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.color_net.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.color_net_bg.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.embedding_a.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.embedding_xyz.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.encoder_dir.parameters():
                p_base.requires_grad = False
            for p_base in self.nerf.encoder_dir_bg.parameters():
                p_base.requires_grad = False
            if 'nr3d' in self.hparams.network_type:
                for p_base in self.nerf.encoding.parameters():
                    p_base.requires_grad = False
                for p_base in self.nerf.decoder.parameters():
                    p_base.requires_grad = False
            else:
                for p_base in self.nerf.encoder.parameters():
                    p_base.requires_grad = False
            #  train instance， freeze  semantic
            if self.hparams.freeze_semantic:
                for p_base in self.nerf.semantic_linear.parameters():
                    p_base.requires_grad = False
                for p_base in self.nerf.semantic_linear_bg.parameters():
                    p_base.requires_grad = False
                
                
            non_frozen_parameters = [p for p in self.nerf.parameters() if p.requires_grad]
            optimizers = {}
            # optimizers['nerf'] = Adam(non_frozen_parameters, lr=self.hparams.lr)
            optimizers['nerf'] = torch.optim.SGD(non_frozen_parameters, lr=self.hparams.lr)
            
        else:
            optimizers = {}
            optimizers['nerf'] = Adam(self.nerf.parameters(), lr=self.hparams.lr)
        
        if self.bg_nerf is not None:
            optimizers['bg_nerf'] = Adam(self.bg_nerf.parameters(), lr=self.hparams.lr)

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            # # add by zyq : load the pretrain-gpnerf to train the semantic
            # if self.hparams.resume_ckpt_state:
            # if not (self.hparams.enable_semantic or self.hparams.enable_instance):
            if not (self.hparams.enable_semantic or self.hparams.enable_instance) or self.hparams.continue_train:
                train_iterations = checkpoint['iteration']
                for key, optimizer in optimizers.items():
                    optimizer_dict = optimizer.state_dict()
                    optimizer_dict.update(checkpoint['optimizers'][key])
                    optimizer.load_state_dict(optimizer_dict)
            else:
                print(f'load weights from {self.hparams.ckpt_path}, strat training from 0')
                train_iterations = 0


            scaler_dict = scaler.state_dict()
            scaler_dict.update(checkpoint['scaler'])
            scaler.load_state_dict(scaler_dict)
            
            discard_index = checkpoint['dataset_index'] if (self.hparams.resume_ckpt_state and not self.hparams.debug) and (not self.hparams.freeze_geo) else -1
            print(f"dicard_index:{discard_index}")
        else:
            train_iterations = 0
            discard_index = -1

        schedulers = {}
        for key, optimizer in optimizers.items():
            schedulers[key] = StepLR(optimizer,
                                     step_size=self.hparams.train_iterations / 4,
                                     gamma=self.hparams.lr_decay_factor,
                                     last_epoch=train_iterations - 1)
            

        # load data
        if self.hparams.dataset_type == 'memory':
            from gp_nerf.datasets.memory_dataset import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
        elif self.hparams.dataset_type == 'memory_depth_dji':
            from gp_nerf.datasets.memory_dataset_depth_dji import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
        elif self.hparams.dataset_type == 'memory_depth_dji_instance':
            assert self.hparams.enable_semantic == True
            from gp_nerf.datasets.memory_dataset_depth_dji_instance import MemoryDataset                
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
            self.H = dataset.H
            self.W = dataset.W
        elif self.hparams.dataset_type == 'memory_depth_dji_instance_crossview':
            assert self.hparams.enable_semantic == True
            from gp_nerf.datasets.memory_dataset_depth_dji_instance_crossview import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
            self.H = dataset.H
            self.W = dataset.W
        elif self.hparams.dataset_type == 'memory_depth_dji_instance_crossview_process':
            assert self.hparams.enable_semantic == True
            from gp_nerf.datasets.memory_dataset_depth_dji_instance_crossview_process import MemoryDataset
            dataset = MemoryDataset(self.train_items, self.near, self.far, self.ray_altitude_range,
                                    self.hparams.center_pixels, self.device, self.hparams)
            self.H = dataset.H
            self.W = dataset.W
        else:
            raise Exception('Unrecognized dataset type: {}'.format(self.hparams.dataset_type))

        if self.is_master:
            pbar = tqdm(total=self.hparams.train_iterations)
            pbar.update(train_iterations)
        else:
            pbar = None
        
        # start training
        chunk_id = 0
        while train_iterations < self.hparams.train_iterations:


            if 'RANK' in os.environ:
                world_size = int(os.environ['WORLD_SIZE'])
                sampler = DistributedSampler(dataset, world_size, int(os.environ['RANK']))
                assert self.hparams.batch_size % world_size == 0
                data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size // world_size, sampler=sampler,
                                         num_workers=0, pin_memory=True)
            else:
                if self.hparams.dataset_type == 'memory_depth':
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=0,
                                                pin_memory=False)
                
                elif self.hparams.dataset_type == 'memory_depth_dji_instance':
                    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                elif self.hparams.dataset_type =='memory_depth_dji_instance_crossview':
                    data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                elif self.hparams.dataset_type == 'memory_depth_dji_instance_crossview_process':
                    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0,
                                                pin_memory=False, collate_fn=custom_collate)
                    
                else:
                    data_loader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=16,
                                                pin_memory=False)

            for dataset_index, item in enumerate(data_loader):
                if self.hparams.dataset_type == 'memory_depth_dji_instance_crossview_process':
                    if item == ['end']:
                        print('done')
                        raise TypeError
                    continue
                
                if item == None:
                    continue
                elif item == ['end']:
                    self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)
                    
                    val_metrics, _ = self._run_validation(train_iterations)
                    self._write_final_metrics(val_metrics, train_iterations)
                    # raise TypeError

                


                if dataset_index <= discard_index:
                    continue
                discard_index = -1

                # amp: Automatic mixed precision
                with torch.cuda.amp.autocast(enabled=self.hparams.amp):
                    if self.hparams.enable_semantic or self.hparams.enable_instance:
                        for key in item.keys():
                            if self.hparams.enable_instance:
                                if item[key].dim() == 2:
                                    item[key] = item[key].reshape(-1)
                                elif item[key].dim() == 3:
                                    item[key] = item[key].reshape(-1, *item[key].shape[2:])

                            if item[key].shape[0]==1:
                                item[key] = item[key].squeeze(0)
                            if item[key].dim() != 1:
                                if item[key].shape[-1] == 1:
                                    item[key] = item[key].reshape(-1)
                                else:
                                    item[key] = item[key].reshape(-1, item[key].shape[-1])
                                


                        for key in item.keys():
                            if 'random' in key:
                                continue
                            elif 'random_'+key in item.keys():
                                item[key] = torch.cat((item[key], item['random_'+key]))


                    if (self.hparams.enable_semantic or self.hparams.enable_instance) and 'labels' in item.keys():
                        labels = item['labels'].to(self.device, non_blocking=True)
                        if not self.hparams.enable_instance:
                            from tools.unetformer.uavid2rgb import remapping
                            labels = remapping(labels)
                    else:
                        labels = None

                    rgbs = item['rgbs'].to(self.device, non_blocking=True)
                    groups = None

                    # training_step
                    metrics, bg_nerf_rays_present = self._training_step(
                        rgbs,
                        item['rays'].to(self.device, non_blocking=True),
                        item['img_indices'].to(self.device, non_blocking=True), 
                        labels, groups, train_iterations, item)
                    

                    with torch.no_grad():
                        for key, val in metrics.items():
                            if key == 'psnr' and math.isinf(val):  # a perfect reproduction will give PSNR = infinity
                                continue

                            if not math.isfinite(val):
                                np.save(f"{train_iterations}.npy", item)
                                raise Exception('Train metrics not finite: {}'.format(metrics))
                            if math.isnan(val):
                                np.save(f"{train_iterations}.npy", item)
                                raise Exception('Train metrics is nan: {}'.format(metrics))

                for optimizer in optimizers.values():
                    # optimizer.zero_grad(set_to_none=True)
                    optimizer.zero_grad()

                scaler.scale(metrics['loss']).backward()   # 在这之后用torch.cuda.empty_cache() 有效



                if self.hparams.clip_grad_max != 0:
                    torch.nn.utils.clip_grad_norm_(self.nerf.parameters(), self.hparams.clip_grad_max)

                for key, optimizer in optimizers.items():
                    if key == 'bg_nerf' and (not bg_nerf_rays_present):
                        continue
                    else:
                        lr_temp = optimizer.param_groups[0]['lr']
                        if self.wandb is not None and train_iterations % self.hparams.logger_interval == 0:
                            self.wandb.log({"train/optimizer_{}_lr".format(key): lr_temp, 'epoch':train_iterations})
                        if self.writer is not None and train_iterations % self.hparams.logger_interval == 0:
                            self.writer.add_scalar('1_train/optimizer_{}_lr'.format(key), lr_temp, train_iterations)
                        scaler.step(optimizer)
                

                scaler.update()
                for scheduler in schedulers.values():
                    scheduler.step()

                train_iterations += 1
                if self.is_master:
                    pbar.update(1)
                    if train_iterations % self.hparams.logger_interval == 0:
                        for key, value in metrics.items():
                            if self.writer is not None:
                                self.writer.add_scalar('1_train/{}'.format(key), value, train_iterations)
                            if self.wandb is not None:
                                self.wandb.log({"train/{}".format(key): value, 'epoch':train_iterations})

                    if train_iterations > 0 and train_iterations % self.hparams.ckpt_interval == 0:
                        self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                            dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)
            
                if (train_iterations > 0 and train_iterations % self.hparams.val_interval == 0) or train_iterations == self.hparams.train_iterations:
                    
                    val_metrics, all_centroids = self._run_validation(train_iterations)
                        
                    
                    self._write_final_metrics(val_metrics, train_iterations)
                    
                    if self.hparams.enable_instance and train_iterations == self.hparams.train_iterations:
                        self.hparams.render_zyq = True 
                        if self.hparams.instance_loss_mode == 'linear_assignment':
                            all_centroids=None
                        self.hparams.fushi=True
                        _ = self._run_validation_render_zyq(train_iterations, all_centroids)
                
                if train_iterations >= self.hparams.train_iterations:
                    break
        

        if 'RANK' in os.environ:
            dist.barrier()

        if self.is_master:
            pbar.close()
            self._save_checkpoint(optimizers, scaler, train_iterations, dataset_index,
                                  dataset.get_state() if self.hparams.dataset_type == 'filesystem' else None)

    def eval(self):

        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']
        self._setup_experiment_dir()

        if self.hparams.enable_instance:
            all_centroids=None
            if self.hparams.cached_centroids_path is not None:
                with open(self.hparams.cached_centroids_path, 'rb') as f:
                    all_centroids = pickle.load(f)
            val_metrics, all_centroids = self._run_validation(train_iterations, all_centroids)
        else:
            val_metrics, _ = self._run_validation(train_iterations)
        
        self._write_final_metrics(val_metrics, train_iterations=train_iterations)

        # 这里是渲染俯视图
        if self.hparams.enable_instance:
            self.hparams.render_zyq = True
            if self.hparams.instance_loss_mode == 'linear_assignment':
                all_centroids=None
            self.hparams.fushi=True
            val_metrics = self._run_validation_render_zyq(train_iterations, all_centroids)
    
    def _run_validation(self, train_index=-1, all_centroids=None) -> Dict[str, float]:
        from tools.unetformer.uavid2rgb import remapping
        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            val_type = self.hparams.val_type  # train  val
            print('val_type: ', val_type)
            try:
                if val_type == 'val':
                    if 'residence'in self.hparams.dataset_path:
                        self.val_items=self.val_items[:19]
                    # elif 'building'in self.hparams.dataset_path or 'campus'in self.hparams.dataset_path:
                        # self.val_items=self.val_items[:10]
                    indices_to_eval = np.arange(len(self.val_items))
                elif 'train' in val_type:
                    indices_to_eval = np.arange(len(self.train_items))
                    print(len(self.train_items))
                    # indices_to_eval = np.arange(370,490)  
                    
                if self.hparams.enable_instance:
                    all_instance_features, all_thing_features = [], []
                    all_thing_features_building = []
                    all_points_rgb, all_points_semantics = [], []
                    gt_points_rgb, gt_points_semantic, gt_points_instance = [], [], []
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
                if self.hparams.cached_centroids_path is None and self.hparams.enable_instance:
                    Path(str(experiment_path_current)).mkdir(exist_ok=True)
                    Path(str(experiment_path_current / 'val_rgbs')).mkdir(exist_ok=True)
                else:
                    Path(str(experiment_path_current)).mkdir()
                    Path(str(experiment_path_current / 'val_rgbs')).mkdir()
                with (experiment_path_current / 'psnr.txt').open('w') as f:
                    
                    samantic_each_value = {}
                    for class_name in CLASSES:
                        samantic_each_value[f'{class_name}_iou'] = []
                    samantic_each_value['mIoU'] = []
                    samantic_each_value['FW_IoU'] = []
                    samantic_each_value['F1'] = []
                    # samantic_each_value['OA'] = []

                    
                    if self.hparams.debug:
                        indices_to_eval = indices_to_eval[:2]
                        # indices_to_eval = indices_to_eval[:2]
                    
                    for i in main_tqdm(indices_to_eval):
                        self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                        # if i != 0:
                        #     break
                        if val_type == 'val':
                            metadata_item = self.val_items[i]
                        elif 'train' in val_type:
                            metadata_item = self.train_items[i]
                            if metadata_item.is_val:
                                continue

                        if self.hparams.enable_instance:
                            gt_instance_label = metadata_item.load_instance_gt()
                            gt_points_instance.append(gt_instance_label.view(-1))
                        

                        results, _ = self.render_image(metadata_item, train_index)


                        typ = 'fine' if 'rgb_fine' in results else 'coarse'
                        
                        viz_rgbs = metadata_item.load_image().float() / 255.
                        self.H, self.W = viz_rgbs.shape[0], viz_rgbs.shape[1]
                        if self.hparams.save_depth:
                            save_depth_dir = os.path.join(str(self.experiment_path), "depth_{}".format(train_index))
                            if not os.path.exists(save_depth_dir):
                                os.makedirs(save_depth_dir)
                            depth_map = results[f'depth_{typ}'].view(viz_rgbs.shape[0], viz_rgbs.shape[1]).numpy().astype(np.float16)
                            np.save(os.path.join(save_depth_dir, metadata_item.image_path.stem + '.npy'), depth_map)
                            continue
                        
                        # get rendering rgbs and depth
                        viz_result_rgbs = results[f'rgb_{typ}'].view(*viz_rgbs.shape).cpu()

                        


                        viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                        if val_type == 'val':   # calculate psnr  ssim  lpips when val (not train)
                            val_metrics = calculate_metric_rendering(viz_rgbs, viz_result_rgbs, train_index, self.wandb, self.writer, val_metrics, i, f, self.hparams, metadata_item, typ, results, self.device, self.pose_scale_factor)
                        
                        viz_result_rgbs = viz_result_rgbs.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()
                        
                        # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                        img_list = [viz_rgbs * 255, viz_result_rgbs * 255]

                        image_diff = np.abs(viz_rgbs.numpy() - viz_result_rgbs.numpy()).mean(2) # .clip(0.2) / 0.2
                        image_diff_color = cv2.applyColorMap((image_diff*255).astype(np.uint8), cv2.COLORMAP_JET)
                        image_diff_color = cv2.cvtColor(image_diff_color, cv2.COLOR_RGB2BGR)
                        img_list.append(torch.from_numpy(image_diff_color))
                        
                        
                        prepare_depth_normal_visual(img_list, self.hparams, metadata_item, typ, results, Runner.visualize_scalars, experiment_path_current, i)
                        
                        get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, img_list, experiment_path_current, 
                                            i, self.writer, self.hparams, viz_result_rgbs * 255, self.metrics_val, self.metrics_val_each)
                        
                        
                        if self.hparams.enable_instance:
                            instances, p_instances, all_points_rgb, all_points_semantics, gt_points_semantic, p_instances_building = get_instance_pred(
                                results, val_type, metadata_item, viz_rgbs, self.logits_2_label, typ, remapping, 
                                experiment_path_current, i, self.writer, self.hparams, viz_result_rgbs, self.thing_classes,
                                all_points_rgb, all_points_semantics, gt_points_semantic)
                            
                            all_instance_features.append(instances)
                            all_thing_features.append(p_instances)
                            

                            gt_points_rgb.append(viz_rgbs.view(-1,3))



                        # NOTE: 对需要可视化的list进行处理
                        # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                        # 将None元素转换为zeros矩阵
                        img_list = [torch.zeros_like(viz_rgbs) if element is None else element for element in img_list]
                        img_list = torch.stack(img_list).permute(0,3,1,2)
                        img = make_grid(img_list, nrow=3)
                        img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                        Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / ("%06d_all.jpg" % i)))

                        if self.writer is not None and (train_index % 50000 == 0):
                            self.writer.add_image('5_val_images/{}'.format(i), img.byte(), train_index)
                        if self.wandb is not None and (train_index % 50000 == 0):
                            Img = wandb.Image(img, caption="ckpt {}: {} th".format(train_index, i))
                            self.wandb.log({"images_all/{}".format(train_index): Img, 'epoch': i})
                        

                        if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')):
                            Path(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')).mkdir()
                        Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                            str(experiment_path_current / 'val_rgbs' / 'pred_rgb' / ("%06d_pred_rgb.jpg" % i)))
                        
                        
                        if val_type == 'val':
                            #save  [pred_label, pred_rgb, fg_bg] to the folder 

                            if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'gt_rgb')) and self.hparams.save_individual:
                                Path(str(experiment_path_current / 'val_rgbs' / 'gt_rgb')).mkdir()
                            if self.hparams.save_individual:
                                Image.fromarray((viz_rgbs.numpy() * 255).astype(np.uint8)).save(
                                    str(experiment_path_current / 'val_rgbs' / 'gt_rgb' / ("%06d_gt_rgb.jpg" % i)))


                            if self.hparams.bg_nerf or f'bg_rgb_{typ}' in results:
                                img = Runner._create_fg_bg_image(results[f'fg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu(),
                                                                results[f'bg_rgb_{typ}'].view(viz_rgbs.shape[0],viz_rgbs.shape[1], 3).cpu())
                                
                                if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'fg_bg')):
                                    Path(str(experiment_path_current / 'val_rgbs' / 'fg_bg')).mkdir()
                                
                                img.save(str(experiment_path_current / 'val_rgbs' / 'fg_bg' / ("%06d_fg_bg.jpg" % i)))
                            
                            # logger
                            samantic_each_value = save_semantic_metric(self.metrics_val_each, CLASSES, samantic_each_value, self.wandb, self.writer, train_index, i)
                            self.metrics_val_each.reset()
                        del results
                
                if self.hparams.enable_instance:
                    

                    # 'linear_assignment' 是直接得到一个伪标签
                    if self.hparams.instance_loss_mode == 'linear_assignment':
                        all_points_instances = torch.stack(all_thing_features, dim=0) # N x d
                        output_dir = str(experiment_path_current / 'panoptic')
                        if not os.path.exists(output_dir):
                            Path(output_dir).mkdir()
                    else:
                            
                        # instance clustering
                        all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
                        all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
                        
                        output_dir = str(experiment_path_current / 'panoptic')
                        if not os.path.exists(output_dir):
                            Path(output_dir).mkdir()
                        if train_index == self.hparams.train_iterations:
                            np.save(os.path.join(output_dir, "all_instance_features.npy"), all_instance_features)
                            np.save(os.path.join(output_dir, "all_points_semantics.npy"), torch.stack(all_points_semantics).cpu().numpy())
                            np.save(os.path.join(output_dir, "all_points_rgb.npy"), torch.stack(all_points_rgb).cpu().numpy())
                            np.save(os.path.join(output_dir, "gt_points_rgb.npy"), torch.stack(gt_points_rgb).cpu().numpy())
                            np.save(os.path.join(output_dir, "gt_points_semantic.npy"), torch.stack(gt_points_semantic).cpu().numpy())
                            np.save(os.path.join(output_dir, "gt_points_instance.npy"), torch.stack(gt_points_instance).cpu().numpy())

                            
                        if all_centroids is not None:
                            
                            all_points_instances, _ = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                                                        num_images=len(indices_to_eval), use_dbscan=self.hparams.use_dbscan, all_centroids=all_centroids)
        
                        else:
                            
                            all_points_instances, all_centroids = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                                                        num_images=len(indices_to_eval), use_dbscan=self.hparams.use_dbscan)
                            output_dir = str(experiment_path_current)
                            if not os.path.exists(output_dir):
                                Path(output_dir).mkdir(parents=True)
                                
                            all_centroids_path = os.path.join(output_dir, f"test_centroids.npy")
                            with open(all_centroids_path, "wb") as file:
                                pickle.dump(all_centroids, file)
                            print(f"save all_centroids_cache to : {all_centroids_path}")
                            
                            
                                
                    if not os.path.exists(str(experiment_path_current / 'pred_semantics')):
                        Path(str(experiment_path_current / 'pred_semantics')).mkdir()
                    if not os.path.exists(str(experiment_path_current / 'pred_surrogateid')):
                        Path(str(experiment_path_current / 'pred_surrogateid')).mkdir()




                    for save_i in range(len(indices_to_eval)):
                        p_rgb = all_points_rgb[save_i]
                        p_semantics = all_points_semantics[save_i]
                        # p_semantics = gt_points_semantic[save_i]
                        p_instances = all_points_instances[save_i]

                        gt_rgb = gt_points_rgb[save_i]
                        gt_semantics = gt_points_semantic[save_i]
                        gt_instances = gt_points_instance[save_i]

                        
                        output_semantics_with_invalid = p_semantics.detach()
                        Image.fromarray(output_semantics_with_invalid.reshape(self.H, self.W).cpu().numpy().astype(np.uint8)).save(
                                str(experiment_path_current / 'pred_semantics'/ ("%06d.png" % self.val_items[save_i].image_index)))

                        Image.fromarray(p_instances.argmax(dim=1).reshape(self.H, self.W).cpu().numpy().astype(np.uint16)).save(
                                str(experiment_path_current / 'pred_surrogateid'/ ("%06d.png" % self.val_items[save_i].image_index)))
                        
                    
                    # calculate the panoptic quality
                    
                    path_target_sem = os.path.join(self.hparams.dataset_path, 'val', 'labels_gt')
                    path_target_inst = os.path.join(self.hparams.dataset_path, 'val', 'instances_gt')
                    path_pred_sem = str(experiment_path_current / 'pred_semantics')
                    path_pred_inst = str(experiment_path_current / 'pred_surrogateid')
                    if Path(path_target_inst).exists():
                        pq, sq, rq, metrics_each, pred_areas, target_areas, zyq_TP, zyq_FP, zyq_FN, matching = calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, 
                                        path_target_sem, path_target_inst, image_size=[self.W, self.H])
                        with (experiment_path_current / 'instance.txt').open('w') as f:
                            f.write(f'\n\npred_areas\n')  

                            for key, value in pred_areas.items():
                                f.write(f"    {key}: {value}\n")
                            f.write(f'\n\ntarget_areas\n')  
                            for key, value in target_areas.items():
                                f.write(f"    {key}: {value}\n")
                            
                            f.write(f'\n\nTP\n')  
                            for item in zyq_TP:
                                f.write(    f"{item}\n")
                            
                            f.write(f'\n\nFP\n')  
                            for item in zyq_FP:
                                f.write(    f"{item}\n")
                            
                            f.write(f'\n\nFN\n')  
                            for item in zyq_FN:
                                f.write(f"    {item}\n")

                        val_metrics['pq'] = pq
                        val_metrics['sq'] = sq
                        val_metrics['rq'] = rq
                        val_metrics['metrics_each']=metrics_each
                        if all_centroids is not None:
                            val_metrics['all_centroids_shape'] = all_centroids.shape
                            print(f"all_centroids: {all_centroids.shape}")

                        # 对TP进行处理
                        TP = torch.tensor([value[1] for value in zyq_TP if value[0] == 1])
                        FP = torch.tensor([value[1] for value in zyq_FP if value[0] == 1])
                        FN = torch.tensor([value[1] for value in zyq_FN if value[0] == 1])

                        for save_i in range(len(indices_to_eval)):
                            p_rgb = all_points_rgb[save_i]
                            p_semantics = all_points_semantics[save_i]
                            p_instances = all_points_instances[save_i]

                            gt_rgb = gt_points_rgb[save_i]
                            gt_semantics = gt_points_semantic[save_i]
                            gt_instances = gt_points_instance[save_i]
                            stack = visualize_panoptic_outputs(
                                p_rgb, p_semantics, p_instances, None, gt_rgb, gt_semantics, gt_instances,
                                self.H, self.W, thing_classes=self.thing_classes, visualize_entropy=False,
                                TP=TP, FP=FP, FN=FN, matching=matching
                            )
                            
                            grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=4).permute((1, 2, 0)).contiguous()
                            grid = (grid * 255).cpu().numpy().astype(np.uint8)
                            
                            Image.fromarray(grid).save(str(experiment_path_current / 'panoptic' / ("%06d.jpg" % save_i)))
                        

                    ## 对分类结果进行可视化

                    

                # logger
                write_metric_to_folder_logger(self.metrics_val, CLASSES, experiment_path_current, samantic_each_value, self.wandb, self.writer, train_index, self.hparams)
                self.metrics_val.reset()
                
                self.writer.flush()
                self.writer.close()
                self.nerf.train()
            finally:
                if self.is_master and base_tmp_path is not None:
                    shutil.rmtree(base_tmp_path)
            
            return val_metrics, all_centroids
               
    def render_zyq(self):
        if self.hparams.ckpt_path is not None:
            checkpoint = torch.load(self.hparams.ckpt_path, map_location='cpu')
            train_iterations = checkpoint['iteration']
        self._setup_experiment_dir()

        if self.hparams.enable_instance:
            if self.hparams.instance_loss_mode != 'linear_assignment':
                with open(self.hparams.cached_centroids_path, 'rb') as f:
                    all_centroids = pickle.load(f)
            else:
                all_centroids=None
            val_metrics = self._run_validation_render_zyq(train_iterations, all_centroids)
        else:
            val_metrics = self._run_validation_render_zyq(train_iterations)

        self._write_final_metrics(val_metrics, train_iterations=train_iterations)
               
    def _run_validation_render_zyq(self, train_index=-1, all_centroids=None) -> Dict[str, float]:

        with torch.inference_mode():
            #semantic 
            self.metrics_val = Evaluator(num_class=self.hparams.num_semantic_classes)
            CLASSES = ('Cluster', 'Building', 'Road', 'Car', 'Tree', 'Vegetation', 'Human', 'Sky', 'Water', 'Ground', 'Mountain')
            self.nerf.eval()
            val_metrics = defaultdict(float)
            base_tmp_path = None
            
            dataset_path = Path(self.hparams.dataset_path)

            if self.hparams.fushi:
                    
                if 'Yingrenshi' in self.hparams.dataset_path:
                    val_paths = sorted(list((dataset_path / 'render_far0.3' / 'metadata').iterdir()))
                else:
                    val_paths = sorted(list((dataset_path / 'render_far0.3_val' / 'metadata').iterdir()))
            else:
                val_paths = sorted(list((dataset_path / self.hparams.render_zyq_far_view / 'metadata').iterdir()))



            train_paths = val_paths
            train_paths.sort(key=lambda x: x.name)
            val_paths_set = set(val_paths)
            image_indices = {}
            for i, train_path in enumerate(train_paths):
                image_indices[train_path.name] = i
            render_items = [
                self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, x in val_paths_set) for x
                in train_paths]

            H = render_items[0].H
            W = render_items[0].W

            indices_to_eval = np.arange(len(render_items))
            

            if self.hparams.enable_instance and self.hparams.render_zyq and self.hparams.fushi:
                experiment_path_current = self.experiment_path / "eval_fushi"
            else:
                experiment_path_current = self.experiment_path / "eval_{}".format(train_index)
            Path(str(experiment_path_current)).mkdir()
            Path(str(experiment_path_current / 'val_rgbs')).mkdir()
            Path(str(experiment_path_current / 'val_rgbs'/'pred_all')).mkdir()
            with (experiment_path_current / 'psnr.txt').open('w') as f:
                
                samantic_each_value = {}
                for class_name in CLASSES:
                    samantic_each_value[f'{class_name}_iou'] = []
                samantic_each_value['mIoU'] = []
                samantic_each_value['FW_IoU'] = []
                samantic_each_value['F1'] = []
                
                if self.hparams.enable_instance:
                    all_instance_features, all_thing_features = [], []
                    all_points_rgb, all_points_semantics = [], []
                    gt_points_rgb, gt_points_semantic, gt_points_instance = [], [], []
                    if self.hparams.fushi:
                        if 'Yingrenshi' in self.hparams.dataset_path:
                            indices_to_eval = indices_to_eval[:1]
                        elif 'Longhua_block2' in self.hparams.dataset_path:
                            indices_to_eval = indices_to_eval[4:5]
                        elif 'Longhua_block1' in self.hparams.dataset_path:
                            indices_to_eval = indices_to_eval[5:6]
                        elif 'Campus_new' in self.hparams.dataset_path:
                            # indices_to_eval = indices_to_eval[3:4]
                            indices_to_eval = indices_to_eval[7:8]
                            # indices_to_eval = indices_to_eval[7:8]

                
                for i in main_tqdm(indices_to_eval):
                    self.metrics_val_each = Evaluator(num_class=self.hparams.num_semantic_classes)
                    metadata_item = render_items[i]

                    # file_name = Path(metadata_item.image_path).stem
                    # if file_name not in process_item:
                    #     continue
                    i = int(Path(metadata_item.depth_dji_path).stem)
                    # i = metadata_item.image_index
                    self.hparams.sampling_mesh_guidance = False
                    results, _ = self.render_image(metadata_item, train_index)
                    typ = 'fine' if 'rgb_fine' in results else 'coarse'
                    
                    # get rendering rgbs and depth
                    viz_result_rgbs = results[f'rgb_{typ}'].view(H, W, 3).cpu()
                    viz_result_rgbs = viz_result_rgbs.clamp(0,1)
                    self.H, self.W = viz_result_rgbs.shape[0],viz_result_rgbs.shape[1]

                    save_depth_dir = os.path.join(str(experiment_path_current), 'val_rgbs', "pred_depth_save")
                    if not os.path.exists(save_depth_dir):
                        os.makedirs(save_depth_dir)
                    depth_map = results[f'depth_{typ}'].view(viz_result_rgbs.shape[0], viz_result_rgbs.shape[1]).numpy().astype(np.float16)
                    np.save(os.path.join(save_depth_dir, ("%06d.npy" % i)), depth_map)

                    if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')):
                        Path(str(experiment_path_current / 'val_rgbs' / 'pred_rgb')).mkdir()
                    Image.fromarray((viz_result_rgbs.numpy() * 255).astype(np.uint8)).save(
                        str(experiment_path_current / 'val_rgbs' / 'pred_rgb' / ("%06d.jpg" % i)))
                    
                    
                     
                    # NOTE: 这里初始化了一个list，需要可视化的东西可以后续加上去
                    img_list = [viz_result_rgbs * 255]

                    
                    prepare_depth_normal_visual(img_list, self.hparams, metadata_item, typ, results, Runner.visualize_scalars, experiment_path_current, i)

                    get_semantic_gt_pred_render_zyq(results, 'val', metadata_item, viz_result_rgbs, self.logits_2_label, typ, remapping,
                                        self.metrics_val, self.metrics_val_each, img_list, experiment_path_current, i, self.writer, self.hparams)
                    
                    if self.hparams.enable_instance:
                        instances, p_instances, all_points_rgb, all_points_semantics, gt_points_semantic, p_instances_building = get_instance_pred(
                                        results, 'val', metadata_item, viz_result_rgbs, self.logits_2_label, typ, remapping, 
                                        experiment_path_current, i, self.writer, self.hparams, viz_result_rgbs, self.thing_classes,
                                        all_points_rgb, all_points_semantics, gt_points_semantic)
            
                        all_instance_features.append(instances)
                        all_thing_features.append(p_instances)


                    # NOTE: 对需要可视化的list进行处理
                    # save images: list：  N * (H, W, 3),  -> tensor(N, 3, H, W)
                    # 将None元素转换为zeros矩阵
                    img_list = [torch.zeros_like(viz_result_rgbs) if element is None else element for element in img_list]
                    img_list = torch.stack(img_list).permute(0,3,1,2)
                    img = make_grid(img_list, nrow=3)
                    img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                    Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / 'pred_all'/ ("%06d_all.jpg" % i)))

                    del results

            if self.hparams.enable_instance:
                if self.hparams.instance_loss_mode == 'linear_assignment':
                    all_points_instances = torch.stack(all_thing_features, dim=0) # N x d
                    output_dir = str(experiment_path_current / 'panoptic')
                    if not os.path.exists(output_dir):
                        Path(output_dir).mkdir()
                else:
                    # instance clustering
                    all_instance_features = torch.cat(all_instance_features, dim=0).cpu().numpy()
                    all_thing_features = torch.cat(all_thing_features, dim=0).cpu().numpy() # N x d
                    output_dir = str(experiment_path_current / 'panoptic')
                    if not os.path.exists(output_dir):
                        Path(output_dir).mkdir()
                    
                    all_points_instances, all_centroids = cluster(all_thing_features, bandwidth=0.2, device=self.device, 
                                                num_images=len(indices_to_eval), use_dbscan=self.hparams.use_dbscan, all_centroids=all_centroids)

                if not os.path.exists(str(experiment_path_current / 'pred_semantics')):
                    Path(str(experiment_path_current / 'pred_semantics')).mkdir()
                if not os.path.exists(str(experiment_path_current / 'pred_surrogateid')):
                    Path(str(experiment_path_current / 'pred_surrogateid')).mkdir()

                for save_i in range(len(indices_to_eval)):
                    p_rgb = all_points_rgb[save_i]
                    p_semantics = all_points_semantics[save_i]
                    p_instances = all_points_instances[save_i]

                    
                    output_semantics_with_invalid = p_semantics.detach()
                    Image.fromarray(output_semantics_with_invalid.reshape(self.H, self.W).cpu().numpy().astype(np.uint8)).save(
                            str(experiment_path_current / 'pred_semantics'/ ("%06d.png" % self.val_items[save_i].image_index)))
                    
                    Image.fromarray(p_instances.argmax(dim=1).reshape(self.H, self.W).cpu().numpy().astype(np.uint16)).save(
                            str(experiment_path_current / 'pred_surrogateid'/ ("%06d.png" % self.val_items[save_i].image_index)))
                    
                    stack = visualize_panoptic_outputs(
                        p_rgb, p_semantics, p_instances, None, None, None, None,
                        self.H, self.W, thing_classes=self.thing_classes, visualize_entropy=False
                    )
                    grid = make_grid(stack, value_range=(0, 1), normalize=True, nrow=3).permute((1, 2, 0)).contiguous()
                    grid = (grid * 255).cpu().numpy().astype(np.uint8)
                    
                    Image.fromarray(grid).save(str(experiment_path_current / 'panoptic' / ("%06d.jpg" % save_i)))

            return val_metrics

    def _write_final_metrics(self, val_metrics: Dict[str, float], train_iterations) -> None:
        if self.is_master:
            
            experiment_path_current = self.experiment_path / "eval_{}".format(train_iterations)
            with (experiment_path_current /'metrics.txt').open('a') as f:
                if 'pq' in val_metrics:
                    pq, sq, rq = val_metrics['pq'],val_metrics['sq'],val_metrics['rq']
                    print(f'pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}')

                    f.write(f'\n pq, sq, rq: {pq:.5f} {sq:.5f} {rq:.5f}\n')  
                    self.writer.add_scalar('2_val_metric_average/pq', pq, train_iterations)
                    self.writer.add_scalar('2_val_metric_average/sq', sq, train_iterations)
                    self.writer.add_scalar('2_val_metric_average/rq', rq, train_iterations)
                    
                    if 'all_centroids_shape' in val_metrics:
                        all_centroids_shape = val_metrics['all_centroids_shape']
                        self.writer.add_scalar('2_val_metric_average/all_centroids_shape', all_centroids_shape[0], train_iterations)
                        f.write('all_centroids_shape: {}\n'.format(all_centroids_shape))
                        del val_metrics['all_centroids_shape']

                    
                    metrics_each = val_metrics['metrics_each']
                    # f.write(f'panoptic metrics_each: {metrics_each} \n')  
                    
                    for key in metrics_each['all']:
                        avg_val = metrics_each['all'][key]
                        message = '      {}: {}'.format(key, avg_val)
                        f.write('{}\n'.format(message))
                        print(message)
                    f.write('{}\n')
                    f.write(f"pq, rq, sq, mIoU, TP, FP, FN: {metrics_each['all']['pq'][0].item()},{metrics_each['all']['rq'][0].item()},{metrics_each['all']['sq'][0].item()},{metrics_each['all']['iou_sum'][0].item()},{metrics_each['all']['true_positives'][0].item()},{metrics_each['all']['false_positives'][0].item()},{metrics_each['all']['false_negatives'][0].item()}\n")
                    del val_metrics['pq'],val_metrics['sq'],val_metrics['rq'], val_metrics['metrics_each']
                
                for key in val_metrics:
                    avg_val = val_metrics[key] / len(self.val_items)
                    if key== 'val/psnr':
                        if self.wandb is not None:
                            self.wandb.log({'val/psnr_avg': avg_val, 'epoch':train_iterations})
                        if self.writer is not None:
                            self.writer.add_scalar('2_val_metric_average/psnr_avg', avg_val, train_iterations)
                    if key== 'val/ssim':
                        if self.wandb is not None:
                            self.wandb.log({'val/ssim_avg': avg_val, 'epoch':train_iterations})
                        if self.writer is not None:
                            self.writer.add_scalar('2_val_metric_average/ssim_avg', avg_val, train_iterations)

                    message = 'Average {}: {}'.format(key, avg_val)
                    main_print(message)
                    f.write('{}\n'.format(message))
                psnr = val_metrics['val/psnr'] / len(self.val_items)
                ssim = val_metrics['val/ssim'] / len(self.val_items)
                abs_rel = val_metrics['val/abs_rel'] / len(self.val_items)
                rmse_actual = val_metrics['val/rmse_actual'] / len(self.val_items)
                if self.writer is not None:
                    self.writer.add_scalar('2_val_metric_average/abs_rel', abs_rel, train_iterations)
                    self.writer.add_scalar('2_val_metric_average/rmse_actual', rmse_actual, train_iterations)

                # f.write('arg_psnr, arg_ssim: {arg_psnr:.5f}, {arg_ssim:.5f}\n')  
                f.write(f'\n psnr, ssim, rmse_actual, abs_rel: {psnr:.5f}, {ssim:.5f}, {rmse_actual:.5f} ,{abs_rel:.5f}\n')  
                print(f'psnr, ssim, rmse_actual, abs_rel: {psnr:.5f}, {ssim:.5f}, {rmse_actual:.5f} ,{abs_rel:.5f}')

            self.writer.flush()
            self.writer.close()


    def _setup_experiment_dir(self) -> None:
        if self.is_master:
            self.experiment_path.mkdir()
            with (self.experiment_path / 'hparams.txt').open('w') as f:
                for key in vars(self.hparams):
                    f.write('{}: {}\n'.format(key, vars(self.hparams)[key]))
                if 'WORLD_SIZE' in os.environ:
                    f.write('WORLD_SIZE: {}\n'.format(os.environ['WORLD_SIZE']))

            with (self.experiment_path / 'command.txt').open('w') as f:
                f.write(' '.join(sys.argv))
                f.write('\n')

            self.model_path.mkdir(parents=True)

            with (self.experiment_path / 'image_indices.txt').open('w') as f:
                for i, metadata_item in enumerate(self.train_items):
                    f.write('{},{}\n'.format(metadata_item.image_index, metadata_item.image_path.name))
        if self.hparams.writer_log:
            self.writer = SummaryWriter(str(self.experiment_path / 'tb')) if self.is_master else None
        if 'RANK' in os.environ:
            dist.barrier()
        
        if self.hparams.wandb_id =='None':
            self.wandb = None
            print('no using wandb')
        else:
            self.wandb = wandb.init(project=self.hparams.wandb_id, entity="mega-ingp", name=self.hparams.wandb_run_name, dir=self.experiment_path)
            
    def ema_update_slownet(self, slownet, fastnet, momentum):
        # EMA update for the teacher
        with torch.no_grad():
            for param_q, param_k in zip(fastnet.parameters(), slownet.parameters()):
                param_k.data.mul_(momentum).add_((1 - momentum) * param_q.detach().data)

    def _training_step(self, rgbs: torch.Tensor, rays: torch.Tensor, image_indices: Optional[torch.Tensor], labels: Optional[torch.Tensor], groups: Optional[torch.Tensor], train_iterations = -1, item=None) \
            -> Tuple[Dict[str, Union[torch.Tensor, float]], bool]:
        
        from gp_nerf.rendering_gpnerf import render_rays
        
        if 'depth_dji' in item:
            gt_depths = item['depth_dji'].to(self.device, non_blocking=True)
        else:
            gt_depths = None

        results, bg_nerf_rays_present = render_rays(nerf=self.nerf,
                                                    bg_nerf=self.bg_nerf,
                                                    rays=rays,
                                                    image_indices=image_indices,
                                                    hparams=self.hparams,
                                                    sphere_center=self.sphere_center,
                                                    sphere_radius=self.sphere_radius,
                                                    get_depth=False,
                                                    get_depth_variance=True,
                                                    get_bg_fg_rgb=False,
                                                    train_iterations=train_iterations,
                                                    gt_depths=gt_depths,
                                                    pose_scale_factor = self.pose_scale_factor
                                                    )
        
        typ = 'fine' if 'rgb_fine' in results else 'coarse'
        if not self.hparams.freeze_geo and not ('sam' in self.hparams.dataset_type):
            
            with torch.no_grad():
                psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
                depth_variance = results[f'depth_variance_{typ}'].mean()
            metrics = {
                'psnr': psnr_,
                'depth_variance': depth_variance,
                }
            metrics['loss'] = 0

            photo_loss = F.mse_loss(results[f'rgb_{typ}'], rgbs, reduction='mean')
            metrics['photo_loss'] = photo_loss
            metrics['loss'] += photo_loss
        else:
            metrics = {}
            metrics['loss'] = 0
            photo_loss = torch.zeros(1, device='cuda')
            metrics['photo_loss'] = photo_loss
            metrics['loss'] += photo_loss

        if 'air_sigma_loss' in results:
            air_sigma_loss = results['air_sigma_loss'] * self.hparams.wgt_air_sigma_loss
            metrics['air_sigma_loss'] = air_sigma_loss
            metrics['loss'] +=  air_sigma_loss


        # depth_dji loss
        if self.hparams.depth_dji_loss:
            valid_depth_mask = ~torch.isinf(gt_depths)
            pred_depths = results['depth_fine']
            pred_depths_valid = pred_depths[valid_depth_mask]
            gt_depths_valid = gt_depths[valid_depth_mask]

            if self.hparams.wgt_depth_mse_loss != 0:
                depth_mse_loss = torch.mean((pred_depths_valid - gt_depths_valid)**2)
                metrics['depth_mse_loss'] = depth_mse_loss
                metrics['loss'] += self.hparams.wgt_depth_mse_loss * depth_mse_loss

            if self.hparams.wgt_sigma_loss != 0:
                ### sigma_loss from dsnerf (Ray distribution loss): add additional depth supervision
                # z_vals = results['zvals_fine'][valid_depth_mask]
                # deltas = results['deltas_fine'][valid_depth_mask]
                # weights = results[f'weights_fine'][valid_depth_mask]
                # rays_d = rays[valid_depth_mask, 3:6]
                # err = 1
                # dists = deltas * torch.norm(rays_d[...,None,:], dim=-1)
                # sigma_loss = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - gt_depths_valid[:,None]) ** 2 / (2 * err)) * dists
                # sigma_loss = torch.sum(sigma_loss, dim=1).mean()
                # metrics['sigma_loss'] = sigma_loss

                metrics['sigma_loss'] = results['sigma_loss']
                metrics['loss'] += self.hparams.wgt_sigma_loss * sigma_loss
                
        # instance loss
        if self.hparams.enable_instance:
            instance_features = results[f'instance_map_{typ}']
            labels_gt = labels.type(torch.long)
            sem_logits = results[f'sem_map_{typ}']
            sem_label = self.logits_2_label(sem_logits)
            sem_label = remapping(sem_label)


            # contrastive loss or slow-fast loss
            instance_loss, concentration_loss = self.calculate_instance_clustering_loss(instance_features, labels_gt)

            # Concentration loss from contrastive lift

            metrics['instance_loss'] = instance_loss

            metrics['concentration_loss'] = concentration_loss

            metrics['loss'] += self.hparams.wgt_instance_loss * instance_loss + self.hparams.wgt_concentration_loss * concentration_loss

        #semantic loss
        if self.hparams.enable_semantic and (not self.hparams.freeze_semantic):
            sem_logits = results[f'sem_map_{typ}']
            semantic_loss = self.crossentropy_loss(sem_logits, labels.type(torch.long))
            metrics['semantic_loss'] = semantic_loss
            metrics['loss'] += self.hparams.wgt_sem_loss * semantic_loss
            
            with torch.no_grad():
                if train_iterations % 1000 == 0:
                    sem_label = self.logits_2_label(sem_logits)
                    if self.writer is not None:
                        self.writer.add_scalar('1_train/accuracy', sum(labels == sem_label) / labels.shape[0], train_iterations)
                    if self.wandb is not None:
                        self.wandb.log({'train/accuracy': sum(labels == sem_label) / labels.shape[0], 'epoch': train_iterations})



        return metrics, bg_nerf_rays_present
    
    def create_virtual_gt_with_linear_assignment(self, labels_gt, predicted_scores):
        labels = sorted(torch.unique(labels_gt).cpu().tolist())[:predicted_scores.shape[-1]]
        predicted_probabilities = torch.softmax(predicted_scores, dim=-1).detach()
        cost_matrix = np.zeros([len(labels), predicted_probabilities.shape[-1]])
        for lidx, label in enumerate(labels):
            cost_matrix[lidx, :] = -(predicted_probabilities[labels_gt == label, :].sum(dim=0) / ((labels_gt == label).sum() + 1e-4)).cpu().numpy()
        assignment = scipy.optimize.linear_sum_assignment(np.nan_to_num(cost_matrix))
        new_labels = torch.zeros_like(labels_gt)
        for aidx, lidx in enumerate(assignment[0]):
            new_labels[labels_gt == labels[lidx]] = assignment[1][aidx]
        return new_labels
    
    def calculate_instance_clustering_loss(self, instance_features, labels_gt):
        instance_loss = 0    #torch.tensor(0., device=instance_features.device, requires_grad=True)
        concentration_loss = 0 ##torch.tensor(0., device=instance_features.device, requires_grad=True)
        if instance_features == []:
            return torch.tensor(0., device=instance_features.device), torch.tensor(0., device=instance_features.device)
        if self.hparams.instance_loss_mode == "linear_assignment":
            virtual_gt_labels = self.create_virtual_gt_with_linear_assignment(labels_gt, instance_features)
            predicted_labels = instance_features.argmax(dim=-1)
            if torch.any(virtual_gt_labels != predicted_labels):  # should never reinforce correct labels
                # return (self.loss_instances_cluster(instance_features, virtual_gt_labels) * confidences).mean()
                # 我们这里先不考虑confidences
                return (self.loss_instances_cluster(instance_features, virtual_gt_labels)).mean(), concentration_loss
            return torch.tensor(0., device=instance_features.device, requires_grad=True), concentration_loss
        
        elif self.hparams.instance_loss_mode == "contrastive": # vanilla contrastive loss
            instance_loss = contrastive_loss(instance_features, labels_gt, self.temperature)
        
        elif self.hparams.instance_loss_mode == "slow_fast":    
            # EMA update of slow network; done before everything else
            ema_momentum = 0.9 # CONSTANT MOMENTUM
            
            self.ema_update_slownet(self.nerf.instance_linear_slow, self.nerf.instance_linear, ema_momentum)
            self.ema_update_slownet(self.nerf.instance_linear_slow_bg, self.nerf.instance_linear_bg, ema_momentum)

            fast_features, slow_features = instance_features.split(
                [self.hparams.num_instance_classes, self.hparams.num_instance_classes], dim=-1)
            
            fast_projections, slow_projections = fast_features, slow_features # no projection layer
            slow_projections = slow_projections.detach() # no gradient for slow projections

            # sample two random batches from the current batch
            fast_mask = torch.zeros_like(labels_gt).bool()
            random_indices = torch.randperm(len(labels_gt))[:len(labels_gt) // 2]
            fast_mask[random_indices] = True
            slow_mask = ~fast_mask # non-overlapping masks for slow and fast models
            ## compute centroids
            slow_centroids = []
            fast_labels, slow_labels = torch.unique(labels_gt[fast_mask]), torch.unique(labels_gt[slow_mask])
            for l in slow_labels:
                mask_ = torch.logical_and(slow_mask, labels_gt==l) #.unsqueeze(-1)
                slow_centroids.append(slow_projections[mask_].mean(dim=0))
            slow_centroids = torch.stack(slow_centroids)
            # DEBUG edge case:
            if len(fast_labels) == 0 or len(slow_labels) == 0:
                print("Length of fast labels", len(fast_labels), "Length of slow labels", len(slow_labels))
                # This happens when labels_gt of shape 1
                return torch.tensor(0.0, device=instance_features.device),torch.tensor(0.0, device=instance_features.device)
            
            ### Concentration loss
            intersecting_labels = fast_labels[torch.where(torch.isin(fast_labels, slow_labels))] # [num_centroids]
            for l in intersecting_labels:
                mask_ = torch.logical_and(fast_mask, labels_gt==l)
                centroid_ = slow_centroids[slow_labels==l] # [1, d]
                # distance between fast features and slow centroid
                dist_sq = torch.pow(fast_projections[mask_] - centroid_, 2).sum(dim=-1) # [num_points]
                concentration_loss += -1.0 * (torch.exp(-dist_sq / 1.0)).mean()  # 暂时不考虑confidence

            if intersecting_labels.shape[0] > 0: 
                concentration_loss /= intersecting_labels.shape[0]
            
            ### Contrastive loss
            label_matrix = labels_gt[fast_mask].unsqueeze(1) == labels_gt[slow_mask].unsqueeze(0) # [num_points1, num_points2]
            similarity_matrix = torch.exp(-torch.cdist(fast_projections[fast_mask], slow_projections[slow_mask], p=2) / 1.0) # [num_points1, num_points2]
            logits = torch.exp(similarity_matrix)
            # compute loss
            prob = torch.mul(logits, label_matrix).sum(dim=-1) / logits.sum(dim=-1)
            prob_masked = torch.masked_select(prob, prob.ne(0))
            if prob_masked.shape[0] == 0:
                return torch.tensor(0.0, device=instance_features.device, requires_grad=True),torch.tensor(0.0, device=instance_features.device, requires_grad=True)
            instance_loss += -torch.log(prob_masked).mean()

        return instance_loss, concentration_loss

    def _save_checkpoint(self, optimizers: Dict[str, any], scaler: GradScaler, train_index: int, dataset_index: int,
                         dataset_state: Optional[str]) -> None:
        dict = {
            'model_state_dict': self.nerf.state_dict(),
            'scaler': scaler.state_dict(),
            'optimizers': {k: v.state_dict() for k, v in optimizers.items()},
            'iteration': train_index,
            'torch_random_state': torch.get_rng_state(),
            'np_random_state': np.random.get_state(),
            'random_state': random.getstate(),
            'dataset_index': dataset_index
        }

        if dataset_state is not None:
            dict['dataset_state'] = dataset_state

        if self.bg_nerf is not None:
            dict['bg_model_state_dict'] = self.bg_nerf.state_dict()

        torch.save(dict, self.model_path / '{}.pt'.format(train_index))

    def render_image(self, metadata: ImageMetadata, train_index=-1) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        
        from gp_nerf.rendering_gpnerf import render_rays
        directions = get_ray_directions(metadata.W,
                                        metadata.H,
                                        metadata.intrinsics[0],
                                        metadata.intrinsics[1],
                                        metadata.intrinsics[2],
                                        metadata.intrinsics[3],
                                        self.hparams.center_pixels,
                                        self.device)
        depth_scale = torch.abs(directions[:, :, 2]).view(-1)

        with torch.cuda.amp.autocast(enabled=self.hparams.amp):
            ###############3 . 俯视图，  render0.3视角下第一张图片
            if self.hparams.render_zyq and self.hparams.enable_instance and self.hparams.fushi:
                if 'Yingrenshi' in self.hparams.dataset_path:
                    image_rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
                    ray_d = image_rays[int(metadata.H/2), int(metadata.W/2), 3:6]
                    ray_o = image_rays[int(metadata.H/2), int(metadata.W/2), :3]

                    z_vals_inbound = 1
                    new_o = ray_o - ray_d * z_vals_inbound
                    metadata.c2w[:,3]= new_o
                    metadata.c2w[1:3,3]=self.sphere_center[1:3]
                    def rad(x):
                        return math.radians(x)
                    angle=30
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_x = torch.tensor([[1, 0, 0],
                                    [0, cosine, sine],
                                    [0, -sine, cosine]])
                    angle=-40
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_y = torch.tensor([[cosine, 0, sine],
                                    [0, 1, 0],
                                    [-sine, 0, cosine]])
                    metadata.c2w[:3,:3]=rotation_matrix_y @ (rotation_matrix_x @ metadata.c2w[:3,:3])
                    metadata.c2w[1,3]=metadata.c2w[1,3]-0.4
                    metadata.c2w[2,3]=metadata.c2w[2,3]+0.05
                elif 'Longhua_block' in self.hparams.dataset_path:

                    image_rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
                    ray_d = image_rays[int(metadata.H/2), int(metadata.W/2), 3:6]
                    ray_o = image_rays[int(metadata.H/2), int(metadata.W/2), :3]

                    z_vals_inbound = 1
                    new_o = ray_o - ray_d * z_vals_inbound
                    metadata.c2w[:,3]= new_o
                    metadata.c2w[1:3,3]=self.sphere_center[1:3]
                elif 'Campus_new' in self.hparams.dataset_path:

                    image_rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)
                    ray_d = image_rays[int(metadata.H/2), int(metadata.W/2), 3:6]
                    ray_o = image_rays[int(metadata.H/2), int(metadata.W/2), :3]

                    z_vals_inbound = 0.6
                    new_o = ray_o - ray_d * z_vals_inbound
                    metadata.c2w[:,3]= new_o
                    # metadata.c2w[1:3,3]=self.sphere_center[1:3]
                    def rad(x):
                        return math.radians(x)
                    angle=0 #-10
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_x = torch.tensor([[1, 0, 0],
                                    [0, cosine, sine],
                                    [0, -sine, cosine]])
                    angle=0
                    cosine = math.cos(rad(angle))
                    sine = math.sin(rad(angle))
                    rotation_matrix_y = torch.tensor([[cosine, 0, sine],
                                    [0, 1, 0],
                                    [-sine, 0, cosine]])
                    metadata.c2w[:3,:3]=rotation_matrix_y @ (rotation_matrix_x @ metadata.c2w[:3,:3])
                    # metadata.c2w[1,3]=metadata.c2w[1,3]-0.4
                    # metadata.c2w[2,3]=metadata.c2w[2,3]+0.05

                    
                    
            #########################################################


            rays = get_rays(directions, metadata.c2w.to(self.device), self.near, self.far, self.ray_altitude_range)

            rays = rays.view(-1, 8).to(self.device, non_blocking=True).cuda()  # (H*W, 8)
            if self.hparams.render_zyq:
                image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device)
                # image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device)
            elif 'train' in self.hparams.val_type:
                if 'b1' in self.hparams.dataset_path:
                    image_indices = 200 * torch.ones(rays.shape[0], device=rays.device)
                else:    
                    image_indices = 300 * torch.ones(rays.shape[0], device=rays.device)

                # image_indices = 0 * torch.ones(rays.shape[0], device=rays.device)
            else:
                image_indices = metadata.image_index * torch.ones(rays.shape[0], device=rays.device) \
                    if self.hparams.appearance_dim > 0 else None
            results = {}



            if 'RANK' in os.environ:
                nerf = self.nerf.module
            else:
                nerf = self.nerf

            if self.bg_nerf is not None and 'RANK' in os.environ:
                bg_nerf = self.bg_nerf.module
            else:
                bg_nerf = self.bg_nerf

            for i in range(0, rays.shape[0], self.hparams.image_pixel_batch_size):
                if self.hparams.depth_dji_type == "mesh" and self.hparams.sampling_mesh_guidance:
                    if 'train' in self.hparams.val_type:
                        # # print('load depth')
                        # gt_depths = metadata.load_depth_dji().view(-1).to(self.device)
                        # gt_depths = gt_depths / depth_scale    # 这里读的是depth mesh， 存储的是z分量
                        gt_depths = None
                        
                    else:
                            
                        gt_depths = metadata.load_depth_dji().view(-1).to(self.device)
                        gt_depths = gt_depths / depth_scale    # 这里读的是depth mesh， 存储的是z分量
                else: 
                    gt_depths = None
                result_batch, _ = render_rays(nerf=nerf, bg_nerf=bg_nerf,
                                              rays=rays[i:i + self.hparams.image_pixel_batch_size],
                                              image_indices=image_indices[
                                                            i:i + self.hparams.image_pixel_batch_size] if self.hparams.appearance_dim > 0 else None,
                                              hparams=self.hparams,
                                              sphere_center=self.sphere_center,
                                              sphere_radius=self.sphere_radius,
                                              get_depth=True,
                                              get_depth_variance=False,
                                              get_bg_fg_rgb=True,
                                              train_iterations=train_index,
                                              gt_depths= gt_depths[i:i + self.hparams.image_pixel_batch_size] if gt_depths is not None else None,
                                              pose_scale_factor = self.pose_scale_factor)
                if 'air_sigma_loss' in result_batch:
                    del result_batch['air_sigma_loss']
                for key, value in result_batch.items():
                    if key not in results:
                        results[key] = []
                    results[key].append(value.cpu())

            for key, value in results.items():
                results[key] = torch.cat(value)
            return results, rays

    @staticmethod
    def _create_result_image(rgbs: torch.Tensor, result_rgbs: torch.Tensor, result_depths: torch.Tensor) -> Image:
        if result_depths is not None:
            depth_vis = Runner.visualize_scalars(torch.log(result_depths + 1e-8).view(rgbs.shape[0], rgbs.shape[1]).cpu())
            images = (rgbs * 255, result_rgbs * 255, depth_vis)
            return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))
        else:
            images = (rgbs * 255, result_rgbs * 255) #, depth_vis)
            return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))
    
    def _create_fg_bg_image(fgs: torch.Tensor, bgs: torch.Tensor) -> Image:
        images = (fgs * 255, bgs * 255) #, depth_vis)
        return Image.fromarray(np.concatenate(images, 1).astype(np.uint8))
    
    def _create_rendering_semantic(rgbs: torch.Tensor, gt_semantic: torch.Tensor, pseudo_semantic: torch.Tensor,
                                   pred_rgb: torch.Tensor, pred_semantic: torch.Tensor, pred_depth_or_normal: torch.Tensor) -> Image:
        if gt_semantic is None:
            gt_semantic = torch.zeros_like(rgbs)
        if pred_depth_or_normal is None:
            pred_depth_or_normal = torch.zeros_like(rgbs)
        else:
            pred_depth_or_normal = (pred_depth_or_normal+1)*0.5*255
        image_1 = (rgbs * 255, gt_semantic, pseudo_semantic)
        image_1 = Image.fromarray(np.concatenate(image_1, 1).astype(np.uint8))
        image_2 = (pred_rgb * 255, pred_semantic, pred_depth_or_normal)
        image_2 = Image.fromarray(np.concatenate(image_2, 1).astype(np.uint8))
        
        return Image.fromarray(np.concatenate((image_1, image_2), 0).astype(np.uint8))

    def _create_res_list(rgbs, gt_semantic, pseudo_semantic, pred_rgb, pred_semantic, pred_normal) -> Image:
        if gt_semantic is None:
            gt_semantic = torch.zeros_like(rgbs)
        pred_normal = (pred_normal+1)*0.5*255

        res_list = [rgbs * 255, gt_semantic, pseudo_semantic, pred_rgb * 255, pred_semantic, pred_normal]
        
        return res_list

    def visualize_scalars(scalar_tensor: torch.Tensor, ma=None, mi=None, invalid_mask=None) -> np.ndarray:
        
        if ma is not None and mi is not None:
            pass
        else:
            if invalid_mask is not None:
                w, h, _ = invalid_mask.shape
                to_use = scalar_tensor[~invalid_mask].view(-1)
            else:
                to_use = scalar_tensor.view(-1)
            while to_use.shape[0] > 2 ** 24:
                to_use = to_use[::2]

            mi = torch.quantile(to_use, 0.05)
            ma = torch.quantile(to_use, 0.95)
        # print(mi)
        # print(ma)
        scalar_tensor = (scalar_tensor - mi) / max(ma - mi, 1e-8)  # normalize to 0~1
        scalar_tensor = scalar_tensor.clamp_(0, 1)

        scalar_tensor = ((1 - scalar_tensor) * 255).byte().numpy()  # inverse heatmap
        return cv2.cvtColor(cv2.applyColorMap(scalar_tensor, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)

    def _get_image_metadata(self) -> Tuple[List[ImageMetadata], List[ImageMetadata]]:
        dataset_path = Path(self.hparams.dataset_path)

        train_path_candidates = sorted(list((dataset_path / 'train' / 'metadata').iterdir()))
        train_paths = [train_path_candidates[i] for i in
                       range(0, len(train_path_candidates), self.hparams.train_every)]
        

        val_paths = sorted(list((dataset_path / 'val' / 'metadata').iterdir()))

        # train_paths=train_paths[:10]
        # val_paths = val_paths[:4]

        train_paths += val_paths
        train_paths.sort(key=lambda x: x.name)
        val_paths_set = set(val_paths)
        image_indices = {}
        for i, train_path in enumerate(train_paths):
            image_indices[train_path.name] = i

        train_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.train_scale_factor, x in val_paths_set) for x
            in train_paths]
        val_items = [
            self._get_metadata_item(x, image_indices[x.name], self.hparams.val_scale_factor, True) for x in val_paths]

        return train_items, val_items

    def _get_metadata_item(self, metadata_path: Path, image_index: int, scale_factor: int,
                           is_val: bool) -> ImageMetadata:
        image_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            candidate = metadata_path.parent.parent / 'rgbs' / '{}{}'.format(metadata_path.stem, extension)
            if candidate.exists():
                image_path = candidate
                break

        # assert image_path.exists()

        metadata = torch.load(metadata_path, map_location='cpu')
        intrinsics = metadata['intrinsics'] / scale_factor
        # print(f"{metadata['W']} {metadata['H']} {scale_factor}")
        assert metadata['W'] % scale_factor == 0
        assert metadata['H'] % scale_factor == 0

        dataset_mask = metadata_path.parent.parent.parent / 'masks' / metadata_path.name
        if self.hparams.cluster_mask_path is not None:
            if image_index == 0:
                main_print('Using cluster mask path: {}'.format(self.hparams.cluster_mask_path))
            mask_path = Path(self.hparams.cluster_mask_path) / metadata_path.name
        elif dataset_mask.exists():
            if image_index == 0:
                main_print('Using dataset mask path: {}'.format(dataset_mask.parent))
            mask_path = dataset_mask
        else:
            mask_path = None

        
        label_path = None
        for extension in ['.jpg', '.JPG', '.png', '.PNG']:
            
            candidate = metadata_path.parent.parent / f'labels_{self.hparams.label_name}' / '{}{}'.format(metadata_path.stem, extension)

            if candidate.exists():
                label_path = candidate
                break
        instance_path = None
        if self.hparams.enable_instance:
            for extension in ['.jpg', '.JPG', '.png', '.PNG', '.npy']:
                
                candidate = metadata_path.parent.parent / f'{self.hparams.instance_name}' / '{}{}'.format(metadata_path.stem, extension)

                if candidate.exists():
                    instance_path = candidate
                    break
        if 'memory_depth_dji' in self.hparams.dataset_type:
            if self.hparams.depth_dji_type=='las':
                depth_dji_path = os.path.join(metadata_path.parent.parent, 'depth_dji', '%s.npy' % metadata_path.stem) 
            elif self.hparams.depth_dji_type=='mesh':
                depth_dji_path = os.path.join(metadata_path.parent.parent, 'depth_mesh', '%s.npy' % metadata_path.stem) 
            
            if not Path(depth_dji_path).exists() and not self.hparams.render_zyq:
                depth_dji_path=None
            if 'left_or_right' in metadata:
                left_or_right = metadata['left_or_right']
            else:
                left_or_right = None
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                 intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, 
                                 depth_dji_path=depth_dji_path, left_or_right=left_or_right, hparams=self.hparams, instance_path=instance_path)

        else:
            return ImageMetadata(image_path, metadata['c2w'], metadata['W'] // scale_factor, metadata['H'] // scale_factor,
                                intrinsics, image_index, None if (is_val and self.hparams.all_val) else mask_path, is_val, label_path, instance_path=instance_path)

    def _get_experiment_path(self) -> Path:
        exp_dir = Path(self.hparams.exp_name)
        exp_dir.mkdir(parents=True, exist_ok=True)
        existing_versions = [int(x.name) for x in exp_dir.iterdir()]
        version = 0 if len(existing_versions) == 0 else max(existing_versions) + 1
        experiment_path = exp_dir / str(version)
        return experiment_path
