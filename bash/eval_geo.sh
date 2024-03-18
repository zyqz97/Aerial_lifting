#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=2


config_file=configs/yingrenshi.yaml



exp_name=
dataset_type=memory_depth_dji

enable_semantic=False

ckpt_path=

python gp_nerf/eval.py  \
    --config_file  $config_file   \
    --dataset_type $dataset_type    \
    --enable_semantic  $enable_semantic  \
    --exp_name  $exp_name   \
    --ckpt_path=$ckpt_path 

######## if you do not have depth when using custom dataset
######## set  --save_depth=True   after   training the geo model

# python gp_nerf/eval.py  \
#     --dataset_path  $dataset_path  --config_file  $config_file   \
#     --dataset_type $dataset_type    \
#     --enable_semantic  $enable_semantic  \
#     --exp_name  $exp_name   \
#     --ckpt_path=$ckpt_path \
#     --save_depth=True 

