import configargparse


def get_opts_base():

    parser = configargparse.ArgParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
        
    parser.add_argument('--gpnerf', default=True, type=eval, choices=[True, False], help='if true use gp-nerf, else mega-nerf') 

    parser.add_argument('--debug', type=eval, default=False, help='shuffle=False and ignore the trained data')
    parser.add_argument('--val_type', type=str, default='val', choices=['val', 'train', 'train_instance', 'train_save_depth'], help='')
    parser.add_argument('--logger_interval', type=int, default=10, help='training iterations')
    
    parser.add_argument('--separate_semantic', type=eval, default=True, choices=[True, False], help='')
    parser.add_argument('--freeze_geo', default=False, type=eval, choices=[True, False], help='if true use gp-nerf, else mega-nerf')
    parser.add_argument('--dataset_type', type=str, default='', choices=['memory_depth_dji_instance_crossview_process',
                                                                         'memory_depth_dji_instance_crossview',
                                                                         'memory_depth_dji_instance',
                                                                         'memory_depth_dji',
                                                                         'memory', ],
                        help="""specifies whether to hold all images in CPU memory during training, or whether to write randomized
                        batches or pixels/rays to disk""")
   

    parser.add_argument('--balance_weight', type=eval, default=True, help='')
    
    parser.add_argument('--remove_cluster', type=eval, default=True, help='')

    parser.add_argument('--use_subset', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--label_name_3d_to_2d', type=str, default='label_pc',help='')
    parser.add_argument('--start', type=int,default=-1, help='')
    parser.add_argument('--end', type=int, default=-1,help='')

    parser.add_argument('--check_depth', type=eval, default=False, help='shuffle=False and ignore the trained data')

    # nr3d 
    parser.add_argument('--contract_new', default=True, type=eval, choices=[True, False])
    parser.add_argument('--use_plane', default=True, type=eval, choices=[True, False])
    parser.add_argument('--geo_init_method', default='idr', type=str, choices=['idr', 'road_surface'], help='')
    parser.add_argument('--save_individual', default=False, type=eval, choices=[True, False])
    parser.add_argument('--continue_train', default=False, type=eval, choices=[True, False])


    # depth_dji_loss
    parser.add_argument('--depth_dji_loss', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--depth_dji_type', default='mesh', type=str, choices=['mesh', 'las'], help='')
    parser.add_argument('--sampling_mesh_guidance', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--wgt_air_sigma_loss', default=0, type=float, help='')
    parser.add_argument('--around_mesh_meter', type=int, default=5, help='')
    parser.add_argument('--wgt_depth_mse_loss', default=0, type=float, help='')
    parser.add_argument('--wgt_sigma_loss', default=0, type=float, help='')




    # normal and depth
    parser.add_argument('--sample_ray_num', default=1024, type=int, help='')
    parser.add_argument('--visual_normal', default=True, type=eval, choices=[True, False], help='')
    parser.add_argument('--normal_loss', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--wgt_nl1_loss', default=1e-4, type=float, help='')
    parser.add_argument('--wgt_ncos_loss', default=1e-4, type=float, help='')
    parser.add_argument('--depth_loss', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--wgt_depth_loss', default=0.000, type=float, help='')
    parser.add_argument('--auto_grad', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--decay_min', default=0.1, type=float, help='')
    parser.add_argument('--save_depth', default=False, type=eval, choices=[True, False], help='')


    # instance
    parser.add_argument('--fushi', default=False, type=eval, choices=[True, False], help='brid eyes view')
    parser.add_argument('--enable_instance', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--num_instance_classes', type=int, default=25, help='is different when using LA or CL')
    parser.add_argument('--wgt_instance_loss', default=1, type=float, help='')
    parser.add_argument('--freeze_semantic', default=False, type=eval, choices=[True, False], help='free semantic when training the instance field')
    parser.add_argument('--instance_name', type=str, default='instances_mask_0.001',choices=['instances_mask_0.001', 'instances_mask_0.001_depth'], help='')
    parser.add_argument('--instance_loss_mode', type=str, default='linear_assignment', choices=['contrastive', 'linear_assignment', 'slow_fast'], help='')
    parser.add_argument('--cached_centroids_path', type=str, default=None, help='')
    parser.add_argument('--use_dbscan', default=True, type=eval, choices=[True, False], help='')
    parser.add_argument('--wgt_concentration_loss', default=1, type=float, help='')
    
    
    parser.add_argument('--crossview_process_path', type=str, default='zyq/test', help='')
    parser.add_argument('--crossview_all', default=False, type=eval, choices=[True, False], help='The SAM of Longhua-b1 dataset is too small, need to turn on this switch')
    

    # semantic
    parser.add_argument('--stop_semantic_grad', default=True, type=eval, choices=[True, False], help='stop the semantic grad toward the geo')
    parser.add_argument('--ignore_index', type=int, default=0, help='ignore cluster category when training semantic')
    # parser.add_argument('--ignore_index', type=int, nargs='+', default=-1, help='List of indices to ignore')
    parser.add_argument('--label_name', type=str, default='fusion', choices=['m2f', 'fusion', 'gt'], help='')

    parser.add_argument('--enable_semantic', default=False, type=eval, choices=[True, False], help='')
    parser.add_argument('--num_semantic_classes', type=int, default=5, help='')
    parser.add_argument('--num_layers_semantic_hidden', type=int, default=3, choices=[1, 3], help='')
    parser.add_argument('--semantic_layer_dim', type=int, default=128, help='')
    parser.add_argument('--wgt_sem_loss', default=1, type=float, help='')
    parser.add_argument('--network_type', type=str, default='gpnerf_nr3d', choices=['gpnerf', 'gpnerf_nr3d', 'mlp'], help='')
    
    parser.add_argument('--clip_grad_max', type=float, default=0, help='use clip_grad_norm and set the max_value')

    # network setting
    parser.add_argument('--num_layers', type=int, default=2, help='change our sigma layer')
    parser.add_argument('--num_layers_color', type=int, default=3, help='change our color layer')
    parser.add_argument('--layer_dim', type=int, default=64, help='number of channels in foreground MLP')
    parser.add_argument('--appearance_dim', type=int, default=48, help='dimension of appearance embedding vector (set to 0 to disable)')
    parser.add_argument('--geo_feat_dim', type=int, default=15, help='')

    parser.add_argument('--num_levels', type=int, default=16, help='')
    parser.add_argument('--base_resolution', type=int, default=16, help='')
    parser.add_argument('--desired_resolution', type=int, default=2048, help='')
    parser.add_argument('--log2_hashmap_size', type=int, default=19, help='')
    parser.add_argument('--hash_feat_dim', type=int, default=2, help='')

    

    # logger
    parser.add_argument('--writer_log', default=True, type=eval, choices=[True, False], help='')
    parser.add_argument('--wandb_id', default='None', type=str, help='')
    parser.add_argument('--wandb_run_name', default='test', type=str, help='')

    parser.add_argument('--use_scaling', default=False, type=eval, choices=[True, False], help='scale plane feature')
    parser.add_argument('--contract_norm', type=str, default='l2', choices=['l2', 'inf'], help='')
    parser.add_argument('--contract_bg_len', default=1, type=float, help='set 0.4 of 1:1')
    parser.add_argument('--aabb_bound', default=1.6, type=float, help='work only when not use ellipsoid')


    parser.add_argument('--train_iterations', type=int, default=100000, help='training iterations')
    parser.add_argument('--val_interval', type=int, default=100000, help='validation interval')
    parser.add_argument('--ckpt_interval', type=int, default=100000, help='checkpoint interval')
    parser.add_argument('--model_chunk_size', type=int, default=10*1024*1024, help='chunk size to split the input to avoid OOM')
    parser.add_argument('--ray_chunk_size', type=int, default=20*1024, help='chunk size to split the input to avoid OOM')

    # parser.add_argument('--model_chunk_size', type=int, default=32 * 1024, help='chunk size to split the input to avoid OOM')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--coarse_samples', type=int, default=128, help='number of coarse samples')
    parser.add_argument('--fine_samples', type=int, default=128, help='number of additional fine samples')

    parser.add_argument('--ckpt_path', type=str, default=None, help='path towards serialized model checkpoint')
    parser.add_argument('--config_file', is_config_file=True)
    
    parser.add_argument('--chunk_paths', type=str, nargs='+', default=None,
                        help="""scratch directory to write shuffled batches to when training using the filesystem dataset. 
    Should be set to a non-existent path when first created, and can then be reused by subsequent training runs once all chunks are written""")
    parser.add_argument('--desired_chunks', type=int, default=20,
                        help='due to the long time and hugh space consumption,we only keep part of chunk')
    parser.add_argument('--num_chunks', type=int, default=20,
                        help='number of shuffled chunk files to write to disk. Each chunk should be small enough to fit into CPU memory')
    

    parser.add_argument('--disk_flush_size', type=int, default=10000000)
    parser.add_argument('--train_every', type=int, default=1,
                        help='if set to larger than 1, subsamples each n training images')
    parser.add_argument('--cluster_mask_path', type=str, default=None,
                        help='directory containing pixel masks for all training images (generated by create_cluster_masks.py)')
    parser.add_argument('--container_path', type=str, default=None,
                        help='path towards merged Mega-NeRF model generated by merged_submodules.py')
    parser.add_argument('--bg_layer_dim', type=int, default=256, help='number of channels in background MLP, NO use in gpnerf')

    parser.add_argument('--near', type=float, default=1, help='ray near bounds')
    parser.add_argument('--far', type=float, default=None,
                        help='ray far bounds. Will be automatically set if not explicitly set')
    parser.add_argument('--ray_altitude_range', nargs='+', type=float, default=None,
                        help='constrains ray sampling to the given altitude')
    parser.add_argument('--train_scale_factor', type=int, default=4,
                        help='downsamples training images if greater than 1')
    parser.add_argument('--val_scale_factor', type=int, default=4,
                        help='downsamples validation images if greater than 1')

    parser.add_argument('--pos_xyz_dim', type=int, default=10,
                        help='frequency encoding dimension applied to xyz position')
    parser.add_argument('--pos_dir_dim', type=int, default=4,
                        help='frequency encoding dimension applied to view direction (set to 0 to disable)')
    parser.add_argument('--layers', type=int, default=8, help='number of layers in MLP')
    parser.add_argument('--skip_layers', type=int, nargs='+', default=[4], help='indices of the skip connections')
    parser.add_argument('--affine_appearance', default=False, action='store_true',
                        help='set to true to use affine transformation for appearance instead of latent embedding')

    parser.add_argument('--use_cascade', default=False, action='store_true',
                        help='use separate MLPs to query coarse and fine samples')
    parser.add_argument('--train_mega_nerf', type=str, default=None,
                        help='directory train a Mega-NeRF architecture (point this towards the params.pt file generated by create_cluster_masks.py)')
    parser.add_argument('--boundary_margin', type=float, default=1.15,
                        help='overlap factor between different spatial cells')
    parser.add_argument('--all_val', default=False, action='store_true',
                        help='use all pixels for validation images instead of those specified in cluster masks')
    parser.add_argument('--cluster_2d', default=False, action='store_true', help='cluster without altitude dimension')

    parser.add_argument('--no_center_pixels', dest='center_pixels', default=True, action='store_false', help='do not shift pixels by +0.5 when computing ray directions')
    parser.add_argument('--no_shifted_softplus', dest='shifted_softplus', default=True, action='store_false', help='use ReLU instead of shifted softplus activation')
    parser.add_argument('--image_pixel_batch_size', type=int, default=8 * 1024, help='number of pixels to evaluate per split when rendering validation images')
    parser.add_argument('--perturb', type=float, default=1.0, help='factor to perturb depth sampling points')
    parser.add_argument('--noise_std', type=float, default=1.0, help='std dev of noise added to regularize sigma')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--lr_decay_factor', type=float, default=1, help='learning rate decay factor')
    parser.add_argument('--no_bg_nerf', dest='bg_nerf', default=True, action='store_false',help='do not use background MLP')
    parser.add_argument('--ellipse_scale_factor', type=float, default=1.1, help='Factor to scale foreground bounds')
    parser.add_argument('--no_ellipse_bounds', dest='ellipse_bounds', default=True, action='store_false', help='use spherical foreground bounds instead of ellipse')
    parser.add_argument('--no_resume_ckpt_state', dest='resume_ckpt_state', default=True, action='store_false')
    parser.add_argument('--no_amp', dest='amp', default=True, action='store_false')
    parser.add_argument('--detect_anomalies', default=False, action='store_true')
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--render_zyq', default=False, action='store_true')
    parser.add_argument('--render_zyq_far_view', type=str, default='render_far0.3')



    return parser
