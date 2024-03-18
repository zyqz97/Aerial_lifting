
from mega_nerf.metrics import psnr, ssim, lpips
from mega_nerf.misc_utils import main_print, main_tqdm
import torch
from tools.unetformer.uavid2rgb import custom2rgb
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchvision.utils import make_grid

from tools.contrastive_lift.utils import create_instances_from_semantics
from tqdm import tqdm
from tools.contrastive_lift.util.panoptic_quality import panoptic_quality

def read_and_resize_labels(path, size):
    image = Image.open(path)
    return np.array(image.resize(size, Image.NEAREST))

def calculate_panoptic_quality_folders(path_pred_sem, path_pred_inst, path_target_sem, path_target_inst, image_size):
    path_pred_sem = Path(path_pred_sem)
    path_pred_inst = Path(path_pred_inst)
    path_target_sem = Path(path_target_sem)
    path_target_inst = Path(path_target_inst)


    faulty_gt = [0]
    things = set([1])
    stuff = set([0,2,3,4])
    val_paths = [y for y in sorted(list(path_pred_sem.iterdir()), key=lambda x: int(x.stem))]

    pred, target = [], []
    for p in tqdm(val_paths):
        img_target_sem = read_and_resize_labels((path_target_sem / p.name), image_size)
        valid_mask = ~np.isin(img_target_sem, faulty_gt)
        img_pred_sem = torch.from_numpy(read_and_resize_labels(p, image_size)[valid_mask]).unsqueeze(-1)
        img_target_sem = torch.from_numpy(img_target_sem[valid_mask]).unsqueeze(-1)
        img_pred_inst = torch.from_numpy(read_and_resize_labels((path_pred_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        img_target_inst = torch.from_numpy(read_and_resize_labels((path_target_inst / p.name), image_size)[valid_mask]).unsqueeze(-1)
        pred_ = torch.cat([img_pred_sem, img_pred_inst], dim=1).reshape(-1, 2)
        target_ = torch.cat([img_target_sem, img_target_inst], dim=1).reshape(-1, 2)
        pred.append(pred_)
        target.append(target_)
    pq, sq, rq, metrics_each, pred_areas, target_areas, zyq_TP, zyq_FP, zyq_FN, matching  = panoptic_quality(torch.cat(pred, dim=0).cuda(), torch.cat(target, dim=0).cuda(), things, stuff, allow_unknown_preds_category=True)
    return pq.item(), sq.item(), rq.item(), metrics_each, pred_areas, target_areas, zyq_TP, zyq_FP, zyq_FN, matching




### https://github.com/nianticlabs/monodepth2/blob/b676244e5a1ca55564eb5d16ab521a48f823af31/evaluate_depth.py#L214
def compute_errors(gt, pred):
    """Computation of error metrics between predicted and ground truth depths
    """
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25     ).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)

    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3




def calculate_metric_rendering(viz_rgbs, viz_result_rgbs, train_index, wandb, writer, val_metrics, i, f, hparams, metadata_item, typ, results, device, pose_scale_factor):                            
    eval_rgbs = viz_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
    eval_result_rgbs = viz_result_rgbs[:, viz_rgbs.shape[1] // 2:].contiguous()
    
    val_psnr = psnr(eval_result_rgbs.view(-1, 3), eval_rgbs.view(-1, 3))
    metric_key = 'val/psnr/{}'.format(train_index)
    
    if wandb is not None:
        wandb.log({'val/psnr/{}'.format(train_index): val_psnr, 'epoch': i})
    if writer is not None:
        writer.add_scalar('3_val_each_image/psnr/{}'.format(train_index), val_psnr, i)
    val_metrics['val/psnr'] += val_psnr
    main_print('The psnr of the {} image is: {}'.format(i, val_psnr))
    f.write('The psnr of the {} image is: {}\n'.format(i, val_psnr))

    val_ssim = ssim(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs, 1)

    metric_key = 'val/ssim/{}'.format(train_index)
    # TODO: 暂时不放ssim
    if wandb is not None:
        wandb.log({'val/ssim/{}'.format(train_index): val_ssim, 'epoch':i})
    if writer is not None:
        writer.add_scalar('3_val_each_image/ssim/{}'.format(train_index), val_ssim, i)
    val_metrics['val/ssim'] += val_ssim

    val_lpips_metrics = lpips(eval_result_rgbs.view(*eval_rgbs.shape), eval_rgbs)
    for network in val_lpips_metrics:
       agg_key = 'val/lpips/{}'.format(network)
       metric_key = '{}/{}'.format(agg_key, train_index)
       # TODO: 暂时不放lpips
       # if self.wandb is not None:
       #     self.wandb.log({'val/lpips/{}/{}'.format(network, train_index): val_lpips_metrics[network], 'epoch':i})
       # if self.writer is not None:
       #     self.writer.add_scalar('3_val_each_image/lpips/{}'.format(network), val_lpips_metrics[network], i)
       val_metrics[agg_key] += val_lpips_metrics[network]


    # Depth metric
    if hparams.depth_dji_loss:
        gt_depths = metadata_item.load_depth_dji()
        valid_depth_mask = ~torch.isinf(gt_depths)
        # if hparams.depth_dji_type == 'mesh':
        #     valid_depth_mask[:,:]=False
        #     valid_depth_mask[::3]=True
        #     valid_depth_mask[gt_depths==-1] = False 


        gt_depths_valid = gt_depths[valid_depth_mask]
        
        from mega_nerf.ray_utils import get_ray_directions
        directions = get_ray_directions(metadata_item.W,
                                        metadata_item.H,
                                        metadata_item.intrinsics[0],
                                        metadata_item.intrinsics[1],
                                        metadata_item.intrinsics[2],
                                        metadata_item.intrinsics[3],
                                        hparams.center_pixels,
                                        torch.device('cpu'))
        depth_scale = torch.abs(directions[:, :, 2])
        pred_depths = (results[f'depth_{typ}'].view(gt_depths.shape[0],gt_depths.shape[1],1)) * (depth_scale.unsqueeze(-1))
        pred_depths_valid = pred_depths[valid_depth_mask]
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_errors(gt_depths_valid.view(-1).numpy(), pred_depths_valid.view(-1).numpy())
        rmse_actual = rmse * pose_scale_factor
        
        if wandb is not None:
            wandb.log({'val/depth_abs_rel/{}'.format(train_index): abs_rel, 'epoch':i})
        if writer is not None:
            writer.add_scalar('3_val_each_image/abs_rel/{}'.format(train_index), abs_rel, i)
        val_metrics['val/abs_rel'] += abs_rel

        if wandb is not None:
            wandb.log({'val/depth_rmse_actual/{}'.format(train_index): rmse_actual, 'epoch':i})
        if writer is not None:
            writer.add_scalar('3_val_each_image/rmse_actual/{}'.format(train_index), rmse_actual, i)
        val_metrics['val/rmse_actual'] += rmse_actual
        


    return val_metrics


def get_semantic_gt_pred_render_zyq(results, val_type, metadata_item, viz_rgbs, logits_2_label, typ, remapping, 
                         metrics_val, metrics_val_each, img_list, experiment_path_current, i, writer, hparams):
    if f'sem_map_{typ}' in results:
        sem_logits = results[f'sem_map_{typ}']
        sem_label = logits_2_label(sem_logits)

        sem_label = remapping(sem_label)

        visualize_sem = custom2rgb(sem_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
        img_list.append(torch.from_numpy(visualize_sem))

        if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_label')):
            Path(str(experiment_path_current / 'val_rgbs' / 'pred_label')).mkdir()
        Image.fromarray((visualize_sem).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'pred_label' / ("%06d_pred_label.jpg" % i)))
        # Image.fromarray((visualize_sem).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / ("%06d_pred_label.jpg" % i)))
    return

def get_semantic_gt_pred(results, val_type, metadata_item, viz_rgbs, logits_2_label, typ, remapping, img_list, 
                        experiment_path_current, i, writer, hparams, viz_result_rgbs, metrics_val, metrics_val_each, save_left_or_right=None):
    if f'sem_map_{typ}' in results:
        sem_logits = results[f'sem_map_{typ}']
        sem_label = logits_2_label(sem_logits)


        sem_label = remapping(sem_label)

        # invalid_mask = gt_label==0
        # sem_label[invalid_mask.view(sem_label.shape)] = 0

        visualize_sem = custom2rgb(sem_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())
        
        if val_type == 'val':
            gt_label = metadata_item.load_gt()
            gt_label = remapping(gt_label)
            gt_label_rgb = custom2rgb(gt_label.view(*viz_rgbs.shape[:-1]).cpu().numpy())

            if hparams.remove_cluster:

                # ignore_cluster_index = gt_label.view(-1)
                # gt_label_ig = gt_label.view(-1)[ignore_cluster_index.nonzero()].view(-1)
                # sem_label_ig = sem_label[ignore_cluster_index.nonzero()].view(-1)
                # metrics_val.add_batch(gt_label_ig.cpu().numpy(), sem_label_ig.cpu().numpy())
                # metrics_val_each.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())
                gt_label = gt_label.view(-1)
                sem_label = sem_label.view(-1)
                gt_no_zero_mask = (gt_label != 0)
                gt_label_ig = gt_label[gt_no_zero_mask]
                sem_label_ig = sem_label[gt_no_zero_mask]
                metrics_val.add_batch(gt_label_ig.cpu().numpy(), sem_label_ig.cpu().numpy())
                metrics_val_each.add_batch(gt_label_ig.view(-1).cpu().numpy(), sem_label_ig.cpu().numpy())
            else:
                metrics_val.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())
                metrics_val_each.add_batch(gt_label.view(-1).cpu().numpy(), sem_label.cpu().numpy())

            gt_label_rgb = torch.from_numpy(gt_label_rgb)
            if hparams.label_name not in ['m2f', 'merge', 'gt']:
                pseudo_gt_label_rgb = None
            else:
                pseudo_gt_label_rgb = metadata_item.load_label()
                pseudo_gt_label_rgb = remapping(pseudo_gt_label_rgb)
                pseudo_gt_label_rgb = custom2rgb(pseudo_gt_label_rgb.view(*viz_rgbs.shape[:-1]).cpu().numpy())
                pseudo_gt_label_rgb = torch.from_numpy(pseudo_gt_label_rgb)
            

            img_list.append(pseudo_gt_label_rgb)
            img_list.append(gt_label_rgb)
            img_list.append(torch.from_numpy(visualize_sem))
            
            if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'pred_label')):
                Path(str(experiment_path_current / 'val_rgbs' / 'pred_label')).mkdir()
            Image.fromarray((visualize_sem).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'pred_label' / ("%06d_pred_label.jpg" % i)))

            if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'gt_label')) and hparams.save_individual:
                Path(str(experiment_path_current / 'val_rgbs' / 'gt_label')).mkdir()
            if hparams.save_individual and gt_label_rgb is not None:
                Image.fromarray((gt_label_rgb.cpu().numpy()).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'gt_label' / ("%06d_gt_label.jpg" % i)))


            alpha = 0.35
            label_list = [viz_result_rgbs]
            if pseudo_gt_label_rgb is not None:
                if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'm2f_label')):
                    Path(str(experiment_path_current / 'val_rgbs' / 'm2f_label')).mkdir()
                Image.fromarray((pseudo_gt_label_rgb.cpu().numpy()).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'm2f_label' / ("%06d_m2f_label.jpg" % i)))
                
                if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'alpha_m2f_label')):
                    Path(str(experiment_path_current / 'val_rgbs' / 'alpha_m2f_label')).mkdir()
                Image.fromarray((pseudo_gt_label_rgb.cpu().numpy() * (1-alpha) + viz_result_rgbs.cpu().numpy() * alpha).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'alpha_m2f_label' / ("%06d_m2f_label.jpg" % i)))
                
                label_list.append(pseudo_gt_label_rgb * (1-alpha) + viz_result_rgbs * alpha)
            
            label_list.append(gt_label_rgb * (1-alpha) + viz_result_rgbs * alpha)
            label_list.append(torch.from_numpy(visualize_sem) * (1-alpha) + viz_result_rgbs * alpha)
            label_list = [torch.zeros_like(viz_rgbs) if element is None else element for element in label_list]
            label_list = torch.stack(label_list).permute(0,3,1,2)
            img = make_grid(label_list, nrow=2)
            img_grid = img.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'all_label')):
                Path(str(experiment_path_current / 'val_rgbs' / 'all_label')).mkdir()
            Image.fromarray(img_grid).save(str(experiment_path_current / 'val_rgbs' / 'all_label' / ("%06d_all_label.jpg" % i)))
            
            if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'alpha_pred_label')):
                Path(str(experiment_path_current / 'val_rgbs' / 'alpha_pred_label')).mkdir()
            Image.fromarray((visualize_sem * (1-alpha) + viz_result_rgbs.cpu().numpy() * alpha).astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'alpha_pred_label' / ("%06d_pred_label.jpg" % i)))

            
            if not os.path.exists(str(experiment_path_current / 'val_rgbs' / 'alpha_gt_label')):
                Path(str(experiment_path_current / 'val_rgbs' / 'alpha_gt_label')).mkdir()
            # if hparams.save_individual:
            valid_gt_label = (gt_label != 0).view(*viz_rgbs.shape[:-1]).cpu().numpy() # no supervision
            gt_label_rgb_1 = viz_result_rgbs.cpu().numpy()
            gt_label_rgb_1[valid_gt_label] = (gt_label_rgb.cpu().numpy())[valid_gt_label] * (1-alpha) + viz_result_rgbs.cpu().numpy()[valid_gt_label] * alpha
            # gt_label_rgb_1 = (gt_label_rgb.cpu().numpy() * (1-alpha) + viz_result_rgbs.cpu().numpy() * alpha).astype(np.uint8)
            Image.fromarray(gt_label_rgb_1.astype(np.uint8)).save(str(experiment_path_current / 'val_rgbs' / 'alpha_gt_label' / ("%06d_gt_label.jpg" % i)))

            if writer is not None:
                writer.add_image('5_val_images_semantic/{}'.format(i), torch.from_numpy(visualize_sem).permute(2, 0, 1), i)

        
        elif 'train' in val_type:  
        ### 这里存三个东西： 
        # 1. 原始的semantic 结果
        # 2. 上色的semantic 结果
        # 3. alpha 叠加的semantic 结果
            pseudo_gt_label_rgb = None
            gt_label_rgb = None

            if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_label_png')):
                Path(str(experiment_path_current / save_left_or_right / 'pred_label_png')).mkdir()
            Image.fromarray((sem_label.view(*viz_rgbs.shape[:-1]).cpu().numpy()).astype(np.uint8)).save(str(experiment_path_current / save_left_or_right / 'pred_label_png' / ("%06d_pred_label.png" % i)))

            img_list.append(torch.from_numpy(visualize_sem))
            
            if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_label')):
                Path(str(experiment_path_current / save_left_or_right / 'pred_label')).mkdir()
            Image.fromarray((visualize_sem).astype(np.uint8)).save(str(experiment_path_current / save_left_or_right / 'pred_label' / ("%06d_pred_label.jpg" % i)))
            
            alpha = 0.35
            if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_label_alpha')):
                Path(str(experiment_path_current / save_left_or_right / 'pred_label_alpha')).mkdir()
            Image.fromarray((visualize_sem * (1-alpha) + viz_result_rgbs.cpu().numpy() * alpha).astype(np.uint8)).save(str(experiment_path_current / save_left_or_right / 'pred_label_alpha' / ("%06d_pred_label.jpg" % i)))
    return

def get_instance_pred(results, val_type, metadata_item, viz_rgbs, logits_2_label, typ, remapping, 
                        experiment_path_current, i, writer, hparams, viz_result_rgbs, thing_classes,
                        all_points_rgb, all_points_semantics, gt_points_semantic=None):
    if f'instance_map_{typ}' in results:
        instances = results[f'instance_map_{typ}']
        device = instances.device
        # gt_semantic
        # if not hparams.render_zyq and hparams.val_type !='train_instance':
        if not hparams.render_zyq and 'train' not in  hparams.val_type:
            gt_label = metadata_item.load_gt()
            gt_label = remapping(gt_label.view(-1))
            if hparams.val_type == 'val':
                gt_points_semantic.append(gt_label)

        # 如果pred semantic存在，则使用
        # 若不存在， 则创建一个全是things的semantic 
        if f'sem_map_{typ}' in results:
            sem_logits = results[f'sem_map_{typ}']
            sem_label = logits_2_label(sem_logits)
            sem_label = remapping(sem_label)
            # if not hparams.render_zyq:
                # invalid_mask = gt_label==0
                # sem_label[invalid_mask.view(sem_label.shape)] = 0
        
        # else:

            # sem_label = torch.ones_like(instances)

        
        if hparams.instance_loss_mode == 'slow_fast':
            slow_features = instances[...,hparams.num_instance_classes:] 
            # all_slow_features.append(slow_features)
            instances = instances[...,0:hparams.num_instance_classes] # keep fast features only
        p_instances_building = None
        # if not hparams.render_zyq and hparams.val_type !='train_instance':
        
        p_instances = create_instances_from_semantics(instances, sem_label, thing_classes,device=device)
        
        # pred_instances = cluster(padded_instances, device)
        all_points_rgb.append(viz_result_rgbs.view(-1,3))
        all_points_semantics.append(sem_label.view(-1))
        return instances, p_instances, all_points_rgb, all_points_semantics, gt_points_semantic, p_instances_building
    else:
        return None, None, None, None, None, None

def get_sdf_normal_map(metadata_item, results, typ, viz_rgbs):
    #  NSR  SDF ------------------------------------  save the normal_map
    # world -> camera 
    w2c = torch.linalg.inv(torch.cat((metadata_item.c2w,torch.tensor([[0,0,0,1]])),0))
    viz_result_normal_map = results[f'normal_map_{typ}']
    viz_result_normal_map = torch.mm(w2c[:3,:3],viz_result_normal_map.T).T
    # normalize 
    viz_result_normal_map = viz_result_normal_map / (1e-5 + torch.linalg.norm(viz_result_normal_map, ord = 2, dim=-1, keepdim=True))
    viz_result_normal_map = viz_result_normal_map.view(viz_rgbs.shape[0], viz_rgbs.shape[1], 3).cpu()

    normal_viz = (viz_result_normal_map+1)*0.5*255

    return normal_viz

def save_semantic_metric(metrics_val_each, CLASSES, samantic_each_value, wandb, writer, train_index, i):
    mIoU = np.nanmean(metrics_val_each.Intersection_over_Union())
    F1 = np.nanmean(metrics_val_each.F1())
    # OA = np.nanmean(metrics_val_each.OA())
    FW_IoU = metrics_val_each.Frequency_Weighted_Intersection_over_Union()
    iou_per_class = metrics_val_each.Intersection_over_Union()

    samantic_each_value['mIoU'].append(mIoU)
    samantic_each_value['FW_IoU'].append(FW_IoU)
    samantic_each_value['F1'].append(F1)
    # samantic_each_value['OA'].append(OA)

    for class_name, iou in zip(CLASSES, iou_per_class):
        samantic_each_value[f'{class_name}_iou'].append(iou)
    

    for class_name, iou in zip(CLASSES, iou_per_class):
        if np.isnan(iou):
            continue
        if wandb is not None:
            wandb.log({f'val/mIoU_each_class/{train_index}_{class_name}': iou, 'epoch':i})
            wandb.log({'val/FW_IoU_each_images/{}'.format(train_index): FW_IoU, 'epoch':i})
        if writer is not None:
            writer.add_scalar(f'4_{class_name}/{i}', iou, train_index)
            writer.add_scalar('3_val_each_image_FW_IoU/{}'.format(train_index), FW_IoU, i)

    return samantic_each_value

def write_metric_to_folder_logger(metrics_val, CLASSES, experiment_path_current, samantic_each_value, wandb, writer, train_index, hparams):
    if hparams.remove_cluster:

        mIoU = np.nanmean(metrics_val.Intersection_over_Union()[1:])
        FW_IoU = metrics_val.Frequency_Weighted_Intersection_over_Union()
        F1 = np.nanmean(metrics_val.F1()[1:])
    else:
        mIoU = np.nanmean(metrics_val.Intersection_over_Union())
        FW_IoU = metrics_val.Frequency_Weighted_Intersection_over_Union()
        F1 = np.nanmean(metrics_val.F1())
    # OA = np.nanmean(metrics_val.OA())
    iou_per_class = metrics_val.Intersection_over_Union()

    eval_value = {'mIoU': mIoU,
                    'FW_IoU': FW_IoU,
                    'F1': F1,
                #   'OA': OA,
                    }
    print("eval_value")
    print('val:', eval_value)

    iou_value = {}
    for class_name, iou in zip(CLASSES, iou_per_class):
        iou_value[class_name] = iou
    print(iou_value)

    with(experiment_path_current / 'semantic_each.txt').open('w') as f2:
        for key in samantic_each_value:
            f2.write(f'{key}:\n')
            for k in range(len(samantic_each_value[key])):
                f2.write(f'\t\t{k:<3}: {samantic_each_value[key][k]}\n')

    with (experiment_path_current /'metrics.txt').open('a') as f:
        for key in eval_value:
            f.write(f'{eval_value[key]} ')
        for key in iou_value:
            f.write(f'{iou_value[key]} ')
        f.write(f'\n\n')
        f.write('eval_value:\n')
        for key in eval_value:
            f.write(f'\t\t{key:<12}: {eval_value[key]}\n')
        f.write('iou_value:\n')
        for key in iou_value:
            f.write(f'\t\t{key:<12}: {iou_value[key]}\n' )
    

    if wandb is not None:
        wandb.log({'val/mIoU': mIoU, 'epoch':train_index})
        wandb.log({'val/FW_IoU': FW_IoU, 'epoch':train_index})
        wandb.log({'val/F1': F1, 'epoch':train_index})
        # self.wandb.log({'val/OA': OA, 'epoch':train_index})
    if writer is not None:
        writer.add_scalar('2_val_metric_average/mIoU', mIoU, train_index)
        writer.add_scalar('2_val_metric_average/FW_IoU', FW_IoU, train_index)
        writer.add_scalar('2_val_metric_average/F1', F1, train_index)
        # self.writer.add_scalar('val/OA', OA, train_index)



def prepare_depth_normal_visual(img_list, hparams, metadata_item, typ, results, visualize_scalars, experiment_path_current, i, save_left_or_right='val_rgbs', ray_altitude_range=None):
    depth_map = None
    H, W = metadata_item.H, metadata_item.W

    from mega_nerf.ray_utils import get_ray_directions
    directions = get_ray_directions(metadata_item.W,
                                    metadata_item.H,
                                    metadata_item.intrinsics[0],
                                    metadata_item.intrinsics[1],
                                    metadata_item.intrinsics[2],
                                    metadata_item.intrinsics[3],
                                    hparams.center_pixels,
                                    torch.device('cpu'))
    depth_scale = torch.abs(directions[:, :, 2])
    
    if f'depth_{typ}' in results:
        ma, mi = None, None 
        if (hparams.depth_dji_loss and ('memory_depth_dji' in hparams.dataset_type)) and not hparams.render_zyq and 'train' not in hparams.val_type:  # DJI Gt depth
            depth_dji = metadata_item.load_depth_dji().float()
            invalid_mask = torch.isinf(depth_dji)
        
            to_use = depth_dji[~invalid_mask].view(-1)
            while to_use.shape[0] > 2 ** 24:
                to_use = to_use[::2]
            
            mi = torch.quantile(to_use, 0.05)
            ma = torch.quantile(to_use, 0.95)

            depth_dji = torch.from_numpy(visualize_scalars(depth_dji, ma, mi))
            img_list.append(depth_dji)

            if not os.path.exists(str(experiment_path_current / save_left_or_right / 'gt_dji_depth')) and hparams.save_individual:
                Path(str(experiment_path_current / save_left_or_right / 'gt_dji_depth')).mkdir()
            if hparams.save_individual:
                Image.fromarray((depth_dji.cpu().numpy()).astype(np.uint8)).save(str(experiment_path_current / save_left_or_right / 'gt_dji_depth' / ("%06d_gt_dji_depth.jpg" % i)))


        # gt_depth 是z， 网络得到的depth是z_val， 所以需要用scale进行处理
        depth_map = results[f'depth_{typ}'] * depth_scale.view(-1)
        
        if hparams.val_type =='train_instance':
            if 'Yingrenshi' in hparams.dataset_path:
                mi=0.17
                ma=0.6
            elif 'Campus' in hparams.dataset_path:
                mi=0.2
                ma=0.65

        
        depth_vis = torch.from_numpy(visualize_scalars(depth_map.view(H, W).cpu(), ma, mi))
        img_list.append(depth_vis)


        if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_depth')):
            Path(str(experiment_path_current / save_left_or_right / 'pred_depth')).mkdir()
        Image.fromarray((depth_vis.cpu().numpy()).astype(np.uint8)).save(str(experiment_path_current / save_left_or_right / 'pred_depth' / ("%06d_pred_depth.jpg" % i)))


    if hparams.depth_loss:  # GT depth
        depth_cue = metadata_item.load_depth().float()
        depth_cue = torch.from_numpy(visualize_scalars(depth_cue))
        img_list.append(depth_cue)
    
    
        

    if f'normal_map_{typ}' in results:
        # world -> camera 
        w2c = torch.linalg.inv(torch.cat((metadata_item.c2w,torch.tensor([[0,0,0,1]])), 0))
        normal_map = results[f'normal_map_{typ}']
        # normal_map = torch.mm(normal_map, w2c[:3,:3])# + w2c[:3,3]
        normal_map = torch.mm(w2c[:3,:3], normal_map.T).T
        # normalize 
        normal_map = normal_map / (1e-5 + torch.linalg.norm(normal_map, ord = 2, dim=-1, keepdim=True))

        if 'sdf' not in hparams.network_type:
            normal_map = normal_map * -1
        normal_map = normal_map.view(H, W, 3).cpu()
        
        normal_viz = (normal_map + 1)*0.5
        img_list.append(normal_viz*255)

        if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_normal')) and hparams.save_individual:
            Path(str(experiment_path_current / save_left_or_right / 'pred_normal')).mkdir()
        if hparams.save_individual:
            Image.fromarray(((normal_viz*255).cpu().numpy()).astype(np.uint8)).save(str(experiment_path_current / save_left_or_right / 'pred_normal' / ("%06d_pred_normal.jpg" % i)))

        

        # camera_world = origin_to_world(metadata_item)
        # light_source = camera_world[0,0] 

        light_source = torch.Tensor([0.05, -0.05, 0.05]).float()
        
        # camera_world = torch.concat((metadata_item.c2w, torch.tensor([[0,0,0,1]])), 0)
        # light_source = torch.inverse(camera_world)[:3,3].float()

        # light_source = metadata_item.c2w[:3,3]
        # light_source = torch.Tensor([0.05, light_source[1], light_source[2]]).float()

        light = (light_source / light_source.norm(2)).unsqueeze(1)

        diffuse_per = torch.Tensor([0.7,0.7,0.7]).float()
        ambiant = torch.Tensor([0.3,0.3,0.3]).float()
        
        diffuse = torch.mm(normal_viz.view(-1,3), light).clamp_min(0).repeat(1, 3) * diffuse_per.unsqueeze(0)

        geo_viz = (ambiant.unsqueeze(0) + diffuse).clamp_max(1.0)
        geo_viz = geo_viz.view(H, W, 3).cpu()
        img_list.append(geo_viz*255)

        if not os.path.exists(str(experiment_path_current / save_left_or_right / 'pred_shading')) and hparams.save_individual:
            Path(str(experiment_path_current / save_left_or_right / 'pred_shading')).mkdir()
        if hparams.save_individual:
            Image.fromarray(((geo_viz*255).cpu().numpy()).astype(np.uint8)).save(str(experiment_path_current / save_left_or_right / 'pred_shading' / ("%06d_pred_shading.jpg" % i)))



    if hparams.normal_loss:
        normal_cue = metadata_item.load_normal()
        normal_cue = (normal_cue + 1) * 0.5 * 255
        img_list.append(normal_cue)


    if 'bg_lambda_fine' in results:
        fg_mask = (results['bg_lambda_fine'] < 0.01).reshape(H, W, 1)
        fg_mask = fg_mask.repeat(1, 1, 3) * 255
        img_list.append(fg_mask)
    return


def origin_to_world(metadata_item, pose_scale_factor, invert=True):
    ''' Transforms origin (camera location) to world coordinates.

    Args:
        n_points (int): how often the transformed origin is repeated in the
            form (batch_size, n_points, 3)
        camera_mat (tensor): camera matrix
        world_mat (tensor): world matrix
        scale_mat (tensor): scale matrix
        invert (bool): whether to invert the matrices (default: true)
    '''

    # Create origin in homogen coordinates
    p = torch.zeros(1, 4, 1)
    p[:, -1] = 1.

    K1 = metadata_item.intrinsics
    K1 = np.array([[K1[0], 0, K1[2]],[0, K1[1], K1[3]],[0,0,1]])

    E1 = np.array(metadata_item.c2w)
    E1 = np.stack([E1[:, 0], E1[:, 1]*-1, E1[:, 2]*-1, E1[:, 3]], 1)

    camera_mat = np.concatenate((E1, [[0,0,0,1]]), 0)

    world_mat = K1

    # # Invert matrices
    # if invert:
    #     camera_mat = torch.inverse(torch.from_numpy(camera_mat)).float()
    #     world_mat = torch.inverse(torch.from_numpy(world_mat)).float()

    # Apply transformation
    p_world = world_mat @ camera_mat @ p[:,:3,:]

    # Transform points back to 3D coordinates
    p_world = p_world[:, :3].permute(0, 2, 1)
    return p_world

def get_depth_vis(results, typ):
    if f'depth_{typ}' in results:
        viz_depth = results[f'depth_{typ}']
        if f'fg_depth_{typ}' in results:
            to_use = results[f'fg_depth_{typ}'].view(-1)
            while to_use.shape[0] > 2 ** 24:
                to_use = to_use[::2]
            ma = torch.quantile(to_use, 0.95)

            viz_depth = viz_depth.clamp_max(ma)
    else: 
        viz_depth = None
