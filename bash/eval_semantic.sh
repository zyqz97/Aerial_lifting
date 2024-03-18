#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2


dataset_path=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi
config_file=configs/yingrenshi.yaml


batch_size=40960
train_iterations=100000
val_interval=50000
ckpt_interval=50000

exp_name=
dataset_type=memory_depth_dji

enable_semantic=True
ckpt_path=





python gp_nerf/eval.py  \
    --dataset_path  $dataset_path  --config_file  $config_file   \
    --batch_size  $batch_size  --train_iterations   $train_iterations   --val_interval  $val_interval   --ckpt_interval   $ckpt_interval  \
    --dataset_type $dataset_type    \
    --enable_semantic  $enable_semantic  \
    --exp_name  $exp_name   \
    --enable_semantic=$enable_semantic    \
    --ckpt_path=$ckpt_path 
