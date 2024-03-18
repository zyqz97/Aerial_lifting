
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5

### this scripts is use to genarate the original Mask2former semantic label

dataset_path_train=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi/train

cd /data/yuqi/code/private/Mask2Former_aeriallift_private
/data/yuqi/anaconda3/envs/mask2former/bin/python demo/demo_zyq_augment.py  \
    --config-file  configs/ade20k/panoptic-segmentation/swin/maskformer2_swin_large_IN21k_384_bs16_160k.yaml \
    --input  $dataset_path_train/rgbs  \
    --output  $dataset_path_train/mask2former_output \
    --zyq_code  True  \
    --zyq_mapping True \
    --resize True   \
    --opts MODEL.WEIGHTS  pretrained_ckpt/ade20k/model_final_e0c58e_panoptic_swinL.pkl


### move the output labels_m2f to the folder
mv $dataset_path_train/mask2former_output/labels_m2f  $dataset_path_train/
