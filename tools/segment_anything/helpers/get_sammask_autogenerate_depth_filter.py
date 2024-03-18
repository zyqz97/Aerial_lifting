import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
import sys
import time
from tools.segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import glob
import os
import tqdm
from pathlib import Path

from argparse import Namespace
import configargparse
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
from gp_nerf.runner_gpnerf import Runner
from gp_nerf.opts import get_opts_base
from mega_nerf.ray_utils import get_rays, get_ray_directions


# torch.cuda.set_device(7)
device= 'cuda'

def get_depth_range(depth_map, mask):

    to_use = depth_map[mask].view(-1)
    mi = torch.quantile(to_use, 0.05)
    ma = torch.quantile(to_use, 0.95)
    depth_range = ma - mi

    return depth_range


def save_mask_anns_torch(anns, img_name, hparams, id, output_path=None):
    if len(anns) == 0:
        return
    
    #  reverse=True， 从大到小排序
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # 创建一个初始的图像张量
    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1])
    img = torch.zeros(img_shape, dtype=torch.int32, device=device)

    # 设置透明度通道
    # img[:, :, 3] = 0

    # id = 1
    for ann in sorted_anns:
        if ann['area'] < hparams.threshold * img_shape[0] * img_shape[1]:
            continue
        m = ann['segmentation']
        img[m] = id
        id = id + 1



    return img, id

def save_mask_anns_torch_depth(anns, img_name, hparams, id, output_path, depth):
    if len(anns) == 0:
        return
    
    #  reverse=True， 从大到小排序
    sorted_anns = sorted(anns, key=lambda x: x['area'], reverse=True)

    # 创建一个初始的图像张量
    img_shape = (sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1])
    img = torch.zeros(img_shape, dtype=torch.int32, device=device)

    # 设置透明度通道
    # img[:, :, 3] = 0

    # id = 1
    for ann in sorted_anns:
        if ann['area'] < hparams.threshold * img_shape[0] * img_shape[1]:
            continue
        m = ann['segmentation']

        # 加一个极大值抑制
        label_in_exsit = img[m]
        unique_label_in_exsit,counts_label_in_exsit = torch.unique(label_in_exsit, return_counts=True)
        non_zero_indices = (unique_label_in_exsit != 0)
        unique_label_in_exsit = unique_label_in_exsit[non_zero_indices]
        counts_label_in_exsit = counts_label_in_exsit[non_zero_indices]
        if counts_label_in_exsit.shape[0] == 0:
            img[m] = id
            id = id + 1

        else:
            max_count_index = torch.argmax(counts_label_in_exsit)
            most_frequent_label = unique_label_in_exsit[max_count_index]

            exsit_max_label_mask = img==most_frequent_label

            #####################################################
            # # 1. 根据IOU
            # intersection = torch.logical_and(exsit_max_label_mask, torch.from_numpy(m).to(exsit_max_label_mask.device)).sum()
            # union = torch.logical_or(exsit_max_label_mask, torch.from_numpy(m).to(exsit_max_label_mask.device)).sum()
            # iou = intersection.float() / union.float()
            # if iou > 0.9:
            #     # img[exsit_max_label_mask]=id
            #     img[m] = most_frequent_label
            # else:
            #     img[m]=id
            # id = id + 1
            #####################################################


            # #####################################################
            # ## 2. 根据交集与小的比例
            # intersection = torch.logical_and(exsit_max_label_mask, torch.from_numpy(m).to(exsit_max_label_mask.device)).sum()
            # if (intersection / torch.from_numpy(m).to(exsit_max_label_mask.device).sum()) > 0.9:  # 小的在大的上面，就不要了
            #     # img[exsit_max_label_mask]=id
            #     continue
            # img[m] = id
            # id = id + 1
            # #####################################################


            #####################################################
            ## 3. 根据深度进行过滤
            intersection = torch.logical_and(exsit_max_label_mask, torch.from_numpy(m).to(exsit_max_label_mask.device)).sum()
            union = torch.logical_or(exsit_max_label_mask, torch.from_numpy(m).to(exsit_max_label_mask.device)).sum()
            iou = intersection.float() / union.float()

            coordinate_info = torch.load(Path(hparams.dataset_path) / 'coordinates.pt', map_location='cpu')
            pose_scale_factor = coordinate_info['pose_scale_factor']
            depth_range = get_depth_range(depth, m)
            if iou > 0.9 or (depth_range < (hparams.geo_filter_long / pose_scale_factor) and (intersection / torch.from_numpy(m).to(exsit_max_label_mask.device).sum()) > 0.9):
                continue
            img[m] = id
            id = id + 1
            #####################################################


        # viz_img = visualize_labels(img)
        # cv2.imwrite(os.path.join(output_path, 'instances_mask_vis_each', f"{img_name}_%06d.jpg" % id), viz_img)


    return img, id

def visualize_labels(labels_tensor):
    non_zero_labels = labels_tensor[labels_tensor != 0]
    unique_labels = torch.unique(non_zero_labels)
    num_labels = len(unique_labels)

    # 创建颜色映射
    label_colors = torch.rand((num_labels, 3))  # 生成随机颜色
    colored_image = torch.zeros((labels_tensor.size(0), labels_tensor.size(1), 3))  # 创建空的彩色图像

    for i, label in enumerate(unique_labels):
        label_mask = (labels_tensor == label)
        colored_image[label_mask] = label_colors[i]

    # 将 PyTorch 张量转换为 NumPy 数组
    colored_image_np = (colored_image.numpy() * 255).astype(np.uint8)

    return colored_image_np

def _get_train_opts() -> Namespace:
    # parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    parser = get_opts_base()

    parser.add_argument('--output_path', type=str, default='',required=True, help='')
    parser.add_argument('--exp_name', type=str, default='logs/test',required=False, help='experiment name')
    parser.add_argument('--threshold', type=float, default=0.001,required=False, help='')

    parser.add_argument('--dataset_path', type=str, default='',required=False, help='')
    parser.add_argument('--eval', default=False, type=eval, choices=[True, False], help='')
    
    parser.add_argument('--geo_filter_long', default=10, type=float, help='')
    

    return parser.parse_args()


def hello(hparams: Namespace) -> None:
    if 'Longhua' in hparams.dataset_path:
        hparams.train_scale_factor = 1
        hparams.val_scale_factor = 1

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


    points_per_side=32
    if points_per_side ==32:
        output_path = hparams.output_path
    else:
        output_path = hparams.output_path + f'_{points_per_side}'

    sam_checkpoint = "tools/segment_anything/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"
    

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=points_per_side)

    
    
    Path(output_path).mkdir(exist_ok=True, parents=True)
    Path(os.path.join(output_path,f'instances_mask_{hparams.threshold}_depth')).mkdir(exist_ok=True)
    Path(os.path.join(output_path,'instances_mask_vis')).mkdir(exist_ok=True)
    Path(os.path.join(output_path,'image_cat')).mkdir(exist_ok=True)
    Path(os.path.join(output_path,'instances_mask_vis_each')).mkdir(exist_ok=True)

    
    if not hparams.eval:
        
        used_files = []
        for ext in ('*.png', '*.jpg'):
            used_files.extend(glob.glob(os.path.join(hparams.dataset_path, 'subset', 'rgbs', ext)))
        used_files.sort()
        process_item = [Path(far_p).stem for far_p in used_files]


    # process_item = process_item[350:]
    visualize_ori = True
    if visualize_ori:
        id = 1 
    id_depth=1
    print(f"geo-filter:{hparams.geo_filter_long}")           
    
    for metadata_item in tqdm.tqdm(train_items):
        img_name = Path(metadata_item.image_path).stem
        if not hparams.eval:
            if img_name not in process_item or metadata_item.is_val: # or int(file_name) != 182:
                continue

        image = metadata_item.load_image()
        


        feature = torch.from_numpy(np.load(str(metadata_item.depth_dji_path).replace('depth_mesh', 'sam_features')))

        depth = metadata_item.load_depth_dji()
        depth = depth.float().view(-1).to(device)



        ### NOTE: 有时候会报维度不匹配的错误，修改下面的代码
        masks = mask_generator.generate(image, feature[0])
        # masks = mask_generator.generate(image, feature)
        

        ## 把单张depth投影到三维

        directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        False,
                                        device)
        depth_scale_full = torch.abs(directions[:, :, 2]).view(-1).to(device)

        inf_mask = torch.isinf(depth)
        depth[inf_mask] = depth[~inf_mask].max()
        x_grid, y_grid = torch.meshgrid(torch.arange(metadata_item.W), torch.arange(metadata_item.H))
        x_grid, y_grid = x_grid.T.flatten().to(device), y_grid.T.flatten().to(device)
        ## 第二张图先得到点云
        pixel_coordinates = torch.stack([x_grid, y_grid, torch.ones_like(x_grid)], dim=-1)
        K1 = metadata_item.intrinsics
        K1 = torch.tensor([[K1[0], 0, K1[2]], [0, K1[1], K1[3]], [0, 0, 1]]).to(device)
        pt_3d = depth[:, None] * (torch.linalg.inv(K1) @ pixel_coordinates[:, :, None].float()).squeeze()
        arr2 = torch.ones((pt_3d.shape[0], 1)).to(device)
        pt_3d = torch.cat([pt_3d, arr2], dim=-1)
        # pt_3d = pt_3d[valid_depth_mask]
        pt_3d = pt_3d.view(-1, 4)
        E1 = metadata_item.c2w.clone().detach()
        E1 = torch.stack([E1[:, 0], -E1[:, 1], -E1[:, 2], E1[:, 3]], dim=1).to(device)
        world_point = torch.mm(torch.cat([E1, torch.tensor([[0, 0, 0, 1]], device=device)], dim=0), pt_3d.t()).t()
        world_point = world_point[:, :3] / world_point[:, 3:4]

        # 第一维表示高度
        world_point = world_point.view(metadata_item.H, metadata_item.W, 3)


        mask, id_depth = save_mask_anns_torch_depth(masks, img_name, hparams, id_depth, output_path, world_point[:,:,0])
        mask_vis = visualize_labels(mask)
        np.save(os.path.join(output_path, f'instances_mask_{hparams.threshold}_depth', f"{img_name}.npy"), mask.cpu().numpy().astype(np.uint32))
        # print(np.array(id_depth, dtype=np.uint32))
        
        cv2.imwrite(os.path.join(output_path, 'instances_mask_vis', f"{img_name}.png"), mask_vis)
        image_cat = image.cpu().numpy()*0.6+ mask_vis*0.4


        if visualize_ori:
            mask, id = save_mask_anns_torch(masks, img_name, hparams, id, output_path)
            mask_vis = visualize_labels(mask)
            vis_ori = image.cpu().numpy()*0.6+ mask_vis*0.4
            img_compare = np.concatenate([vis_ori, image_cat], axis=1)
            cv2.imwrite(os.path.join(output_path,'image_cat', f"{img_name}.png"), img_compare)

        else:
            cv2.imwrite(os.path.join(output_path,'image_cat', f"{img_name}.png"), image_cat)



    print('done')
    print(id)


if __name__ == '__main__':
    hello(_get_train_opts())