#!/bin/bash
export OMP_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=5




dataset_path=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi
## the same as 2_2
exp_name=logs/yingrenshi_geo_farview
M2F_path=/data/yuqi/code/private/Mask2Former_aeriallift_private




crop_m2f=$dataset_path/train/mask2former_output_ds2/labels_m2f_final_sam/labels_merge
output_path=$exp_name/0
far_paths=$output_path/1_labels_m2f_only_sam/labels_merge
render_type=render_far0.3
eval=False


###  extract the SAM feature embedding of original images 
python tools/segment_anything/helpers/extract_embeddings.py   \
    --rgbs_path=$dataset_path/train/rgbs


###  extract the SAM feature embedding of far-view images 
python tools/segment_anything/helpers/extract_embeddings.py   \
    --rgbs_path=$output_path/pred_rgb



###    fusion far-view    Building category  of SAM and   M2F
python tools/segment_anything/helpers/combine_sam_m2f3_only_sam.py  \
    --sam_features_path=$output_path/sam_features  \
    --rgbs_path=$output_path/pred_rgb   \
    --labels_m2f_path=$output_path/mask2former_output/labels_m2f  \
    --output_path=$output_path/1_labels_m2f_only_sam






### project the far-view results to the original images 
python   scripts/2_0_project_far_to_ori.py \
    --dataset_path=$dataset_path  \
    --exp_name=logs/test  \
    --far_paths=$far_paths   \
    --output_path=$output_path/2_project_to_ori_gt_only_sam \
    --render_type=$render_type  \
    --eval=$eval




### replace building
python    scripts/2_1_only_sam_building_label_replace_nocar.py  \
    --dataset_path=$dataset_path    \
    --exp_name=logs/test    \
    --only_sam_m2f_far_project_path=$output_path/2_project_to_ori_gt_only_sam/project_far_to_ori    \
    --output_path=$output_path/4_replace_building  \
    --eval=$eval






### get crop images  M2F results  M2F_path
original_dir=$(pwd)
cd $M2F_path
/data/yuqi/anaconda3/envs/mask2former/bin/python $M2F_path/demo/demo_zyq_augment_ds2.py  \
    --config-file  $M2F_path/configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml \
    --input  $dataset_path/train/rgbs  \
    --output  $dataset_path/train/mask2former_output_ds2 \
    --zyq_code  True  \
    --zyq_mapping True \
    --opts MODEL.WEIGHTS  pretrained_ckpt/ade20k/model_final_e0c58e_panoptic_swinL.pkl
cd $original_dir



###  fusion SAM and crop M2F
python tools/segment_anything/helpers/combine_sam_m2f2.py  \
    --sam_features_path=$dataset_path/train/sam_features  \
    --rgbs_path=$dataset_path/train//rgbs    \
    --labels_m2f_path=$dataset_path/train/mask2former_output_ds2/labels_m2f_final  \
    --output_path=$dataset_path/train/mask2former_output_ds2/labels_m2f_final_sam




### replace car and tree
python     scripts/2_2_further_replace_car_tree_from_crop.py  \
    --dataset_path=$dataset_path    \
    --exp_name=logs/test    \
    --only_sam_m2f_far_project_path=$output_path/4_replace_building/replace_building_label  \
    --crop_m2f=$crop_m2f  \
    --output_path=$output_path/5_replace_building_cartree  \
    --eval=$eval

###  move to the dataset root
mv  $output_path/5_replace_building_cartree/replace_cartree_label    $dataset_path/train/labels_fusion
