#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5


dataset_path=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi
config_file=configs/yingrenshi.yaml
dataset_type=memory_depth_dji_instance_crossview_process

### denotes the semantic ckpt
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_dji/1018_yingrenshi_density_depth_hash22_far0.3_car2/0/continue150k/0/models/180000.pt 

instance_name=instances_mask_0.001_depth
crossview_process_path=logs_instance_mask/Yingrenshi_depth_0.001_process



### this will genarate the cross-view guidance map under the dataset root, namely $instance_name_crossview_process
### then you can train the instance field

### if you suffer from long processing time, you can turn on parallel processing by manully setting --start={0,100,200 ...}
### --start is used to process 100 images, please refer to the Line 127 in gp_nerf/datasets/memory_dataset_depth_dji_instance_crossview_process.py

python gp_nerf/train.py  \
    --exp_name=./logs/test  \
    --dataset_path  $dataset_path  \
    --config_file  $config_file   \
    --dataset_type $dataset_type    \
    --ckpt_path=$ckpt_path  \
    --instance_name=$instance_name    \
    --crossview_process_path=$crossview_process_path    
    # --start=100
