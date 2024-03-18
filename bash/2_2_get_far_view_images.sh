
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=1


dataset_path=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi
config_file=configs/yingrenshi.yaml

M2F_path=/data/yuqi/code/private/Mask2Former_aeriallift_private
exp_name=logs/yingrenshi_geo_farview
ckpt_path=logs/yingrenshi_geo/0/models/200000.pt


###  get far-view pose, it will save pose file at $dataset_path/render_far0.3
python /data/yuqi/code/private/Aerial_lifting_private/scripts/2_augment_set_far_fix_z_yingrenshi.py \
    --dataset_path=$dataset_path


###  get far-view RGB images from trained ckpt point
python gp_nerf/eval.py  \
    --dataset_path  $dataset_path  --config_file  $config_file   \
    --exp_name  $exp_name   \
    --ckpt_path=$ckpt_path \
    --render_zyq
mv $exp_name/0/eval_200000/val_rgbs/pred_rgb   $exp_name/0
mv $exp_name/0/eval_200000/val_rgbs/pred_depth_save   $exp_name/0


### ger far-view mask2former semantic labels
cd $M2F_path
/data/yuqi/anaconda3/envs/mask2former/bin/python $M2F_path/demo/demo_zyq_augment.py  \
    --config-file  $M2F_path/configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml \
    --input  $exp_name/0/pred_rgb  \
    --output  $exp_name/0/mask2former_output \
    --zyq_code  True  \
    --zyq_mapping True \
    --opts MODEL.WEIGHTS  pretrained_ckpt/ade20k/model_final_e0c58e_panoptic_swinL.pkl

