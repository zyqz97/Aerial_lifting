#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=3

exp_name=/data/yuqi/code/GP-NeRF-semantic/logs_313/0425_4_merge_ignore-1
ckpt_path=/data/yuqi/code/GP-NeRF-semantic/logs_313/0425_4_merge_ignore-1/0/models/100000.pt  # give the checkpoint path  
dataset1='UrbanScene3D'  #  "Mill19"  "Quad6k"   "UrbanScene3D"
dataset2='sci-art' #  "building"  "rubble"  "quad"  "residence"  "sci-art"  "campus"
python gp_nerf/eval.py     --config_file  configs/$dataset2.yaml   --dataset_path  /data/yuqi/Datasets/MegaNeRF/$dataset1/$dataset2/$dataset2-pixsfm    --exp_name  $exp_name    --ckpt_path  $ckpt_path













