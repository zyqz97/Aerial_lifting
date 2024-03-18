#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2


dataset_path=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi
config_file=configs/yingrenshi.yaml


batch_size=10240
train_iterations=200000
val_interval=50000
ckpt_interval=50000

exp_name=logs/yingrenshi_geo
dataset_type=memory_depth_dji

enable_semantic=False







python gp_nerf/train.py  \
    --dataset_path  $dataset_path  --config_file  $config_file   \
    --batch_size  $batch_size  --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type    \
    --enable_semantic  $enable_semantic  \
    --exp_name  $exp_name




######## if you do not have depth when using custom dataset
######## set  
# --sampling_mesh_guidance=False
# --depth_dji_loss=False
# --wgt_depth_mse_loss=0


# python gp_nerf/train.py  \
#     --dataset_path  $dataset_path  --config_file  $config_file   \
#     --batch_size  $batch_size  --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
#     --dataset_type $dataset_type    \
#     --enable_semantic  $enable_semantic  \
#     --exp_name  $exp_name   \
#     --sampling_mesh_guidance=False  --depth_dji_loss=False  --wgt_depth_mse_loss=0

