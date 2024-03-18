#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5


config_file=configs/yingrenshi.yaml


exp_name=
dataset_type=memory_depth_dji_instance_crossview

enable_semantic=True
ckpt_path=

enable_instance=True
instance_loss_mode=linear_assignment
instance_name=instances_mask_0.001_depth

python gp_nerf/eval.py  \
    --config_file  $config_file   \
    --dataset_type $dataset_type    \
    --enable_semantic  $enable_semantic  \
    --exp_name  $exp_name   \
    --instance_loss_mode=$instance_loss_mode   --instance_name=$instance_name --enable_instance=$enable_instance    \
    --ckpt_path=$ckpt_path 

