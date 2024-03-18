#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=7


dataset_path=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi
config_file=configs/yingrenshi.yaml


batch_size=16384
train_iterations=100000
val_interval=50000
ckpt_interval=50000

exp_name=logs/yingrenshi_instance_la
dataset_type=memory_depth_dji_instance_crossview

enable_semantic=True
ckpt_path=logs/yingrenshi_samantic_old/0/models/160000.pt

enable_instance=True
instance_loss_mode=linear_assignment
instance_name=instances_mask_0.001_depth

python gp_nerf/train.py  \
    --dataset_path  $dataset_path  --config_file  $config_file   \
    --batch_size  $batch_size  --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type    \
    --enable_semantic  $enable_semantic  \
    --exp_name  $exp_name   \
    --enable_semantic=$enable_semantic    \
    --instance_loss_mode=$instance_loss_mode   --instance_name=$instance_name --enable_instance=$enable_instance    \
    --ckpt_path=$ckpt_path 

