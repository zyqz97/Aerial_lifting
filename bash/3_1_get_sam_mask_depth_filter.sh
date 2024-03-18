
export OMP_NUM_THREADS=16
export CUDA_VISIBLE_DEVICES=5

dataset_path=/data/yuqi/Datasets/Aerial_lifting_data/Yingrenshi
output_path=logs_instance_mask/Yingrenshi_depth_filter

python tools/segment_anything/helpers/get_sammask_autogenerate_depth_filter.py  \
    --dataset_path=$dataset_path   \
    --output_path=$output_path 

mv $output_path/instances_mask_0.001_depth   $dataset_path/train/


