import os
from argparse import Namespace
from typing import Optional, Dict, Callable, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from mega_nerf.spherical_harmonics import eval_sh
from gp_nerf.sample_bg import bg_sample_inv, contract_to_unisphere, contract_to_unisphere_new

import gc
from torch_scatter import segment_coo
from scripts.visualize_points import visualize_points

# TO_COMPOSITE = {'rgb', 'depth', 'sem_map'}
TO_COMPOSITE = {'rgb', 'depth'}
INTERMEDIATE_KEYS = {'zvals_coarse', 'raw_rgb_coarse', 'raw_sigma_coarse', 'depth_real_coarse', 'raw_sem_logits_coarse', 'raw_sem_feature_coarse','raw_instance_logits_coarse'}

def render_rays(nerf: nn.Module,
                bg_nerf: Optional[nn.Module],
                rays: torch.Tensor,
                image_indices: Optional[torch.Tensor],
                hparams: Namespace,
                sphere_center: Optional[torch.Tensor],
                sphere_radius: Optional[torch.Tensor],
                get_depth: bool,
                get_depth_variance: bool,
                get_bg_fg_rgb: bool,
                train_iterations=-1,
                gt_depths=None,
                pose_scale_factor=None) -> Tuple[Dict[str, torch.Tensor], bool]:
    
    N_rays = rays.shape[0]
    device = rays.device
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (N_rays, 3)
    near, far = rays[:, 6:7], rays[:, 7:8]  # both (N_rays, 1)
    near = torch.clamp(near, max=1e4-1)
    if image_indices is not None:
        image_indices = image_indices.unsqueeze(-1).unsqueeze(-1)

    perturb = hparams.perturb if nerf.training else 0
    last_delta = 1e10 * torch.ones(N_rays, 1, device=device)


    fg_far = _intersect_sphere(rays_o, rays_d, sphere_center, sphere_radius, hparams.render_zyq)
    fg_far = torch.maximum(fg_far, near.squeeze())
    # 划分bg ray
    rays_with_bg = torch.arange(N_rays, device=device)[far.squeeze() > fg_far]
    rays_with_fg = torch.arange(N_rays, device=device)[far.squeeze() <= fg_far]

    if not hparams.render_zyq:
    # if False:
        assert rays_with_bg.shape[0] + rays_with_fg.shape[0] == far.shape[0]
    rays_o = rays_o.view(rays_o.shape[0], 1, rays_o.shape[1])
    rays_d = rays_d.view(rays_d.shape[0], 1, rays_d.shape[1])
    if rays_with_bg.shape[0] > 0:
        last_delta[rays_with_bg, 0] = fg_far[rays_with_bg]

    #  zyq:    初始化
    far_ellipsoid = torch.minimum(far.squeeze(), fg_far).unsqueeze(-1)
    z_vals_inbound = torch.zeros([rays_o.shape[0], hparams.coarse_samples], device=device)
    
    valid_depth_mask=None
    s_near = None
    if hparams.depth_dji_type == "mesh" and hparams.sampling_mesh_guidance:
        valid_depth_mask = ~torch.isinf(gt_depths)
        z_fg = torch.linspace(0, 1, hparams.coarse_samples, device=device)
        # mesh中没有深度的部分
        z_vals_inbound[~valid_depth_mask] = near[~valid_depth_mask] * (1 - z_fg) + far_ellipsoid[~valid_depth_mask] * z_fg
        # mesh中有深度的， 分为三部分： 相机->表面， 表面附近， 表面-> 椭圆界
        z_1 = torch.linspace(0, 1, int(hparams.coarse_samples * 0.25), device=device)
        z_2 = torch.linspace(0, 1, int(hparams.coarse_samples * 0.625), device=device)
        z_3 = torch.linspace(0, 1, int(hparams.coarse_samples * 0.125), device=device)
        surface_point = gt_depths[valid_depth_mask]  # 得到表面点的far
        # epsilon =  (far-near).max() / 100
        epsilon =  hparams.around_mesh_meter / pose_scale_factor
        s_near = (surface_point - epsilon).unsqueeze(-1)
        s_far = (surface_point + epsilon).unsqueeze(-1)

        larger_than_s_near = (near[valid_depth_mask] >= s_near).squeeze(-1)

        mesh_sample1 = near[valid_depth_mask] * (1 - z_1) + s_near * z_1
        mesh_sample2 = s_near * (1 - z_2) + s_far * z_2
        mesh_sample3 = s_far * (1 - z_3) + far_ellipsoid[valid_depth_mask] * z_3
        if not hparams.check_depth:
            z_vals_inbound[valid_depth_mask] = torch.cat([mesh_sample1, mesh_sample2, mesh_sample3], dim=1)
        else:
            z_vals_inbound[valid_depth_mask] = torch.cat([torch.zeros_like(mesh_sample1), mesh_sample2, torch.zeros_like(mesh_sample3)], dim=1)

        z_vals_inbound, _ = torch.sort(z_vals_inbound, -1)


    else:
        z_fg = torch.linspace(0, 1, hparams.coarse_samples, device=device)
        z_vals_inbound = near * (1 - z_fg) + far_ellipsoid * z_fg
    
    # 随机扰动，并生成采样点
    z_vals_inbound = _expand_and_perturb_z_vals(z_vals_inbound, hparams.coarse_samples, perturb, N_rays)
    xyz_coarse_fg = rays_o + rays_d * z_vals_inbound.unsqueeze(-1)
    if hparams.check_depth:
        visualize_points_list = [xyz_coarse_fg.view(-1, 3).cpu().numpy()]
        visualize_points(visualize_points_list)
        print('save the sampling points.')
        import sys
        sys.exit(0)
    

    if hparams.contract_new:
        xyz_coarse_fg = contract_to_unisphere_new(xyz_coarse_fg, hparams)
        
    else:
        xyz_coarse_fg = contract_to_unisphere(xyz_coarse_fg, hparams)

    


    results = _get_results(point_type='fg',
                           nerf=nerf,
                           rays_d=rays_d,
                           image_indices=image_indices,
                           hparams=hparams,
                           xyz_coarse=xyz_coarse_fg,
                           z_vals=z_vals_inbound,
                           last_delta=last_delta,
                           get_depth=get_depth,
                           get_depth_variance=get_depth_variance,
                           get_bg_lambda=True,
                           depth_real=None,
                           xyz_fine_fn=lambda fine_z_vals: (rays_o + rays_d * fine_z_vals.unsqueeze(-1), None),
                           train_iterations=train_iterations,
                           gt_depths=gt_depths,
                           valid_depth_mask=valid_depth_mask,
                           s_near=s_near)
    
    if rays_with_bg.shape[0] != 0:
        z_vals_outer = bg_sample_inv(far_ellipsoid[rays_with_bg], 1e4+1, hparams.coarse_samples // 2, device)
        z_vals_outer = _expand_and_perturb_z_vals(z_vals_outer, hparams.coarse_samples // 2, perturb, rays_with_bg.shape[0])

        xyz_coarse_bg = rays_o[rays_with_bg] + rays_d[rays_with_bg] * z_vals_outer.unsqueeze(-1)
        if hparams.contract_new:
            xyz_coarse_bg = contract_to_unisphere_new(xyz_coarse_bg, hparams)
        else:
            xyz_coarse_bg = contract_to_unisphere(xyz_coarse_bg, hparams)
        
        bg_results = _get_results(point_type='bg',
                                  nerf=nerf,
                                  rays_d=rays_d[rays_with_bg],
                                  image_indices=image_indices[rays_with_bg] if image_indices is not None else None,
                                  hparams=hparams,
                                  xyz_coarse=xyz_coarse_bg,
                                  z_vals=z_vals_outer,
                                  # bg_nerf的last_dalta为1e10
                                  last_delta=1e10 * torch.ones(rays_with_bg.shape[0], 1, device=device),
                                  get_depth=get_depth,
                                  get_depth_variance=get_depth_variance,
                                  get_bg_lambda=False,
                                  depth_real=None,
                                  xyz_fine_fn=lambda fine_z_vals: (rays_o[rays_with_bg] + rays_d[rays_with_bg] * fine_z_vals.unsqueeze(-1), None),
                                  train_iterations=train_iterations,
                                  gt_depths=gt_depths,
                                  valid_depth_mask=None,
                                  s_near=None)
        
    # merge the result of inner and outer
    types = ['fine' if hparams.fine_samples > 0 else 'coarse']
    if hparams.use_cascade and hparams.fine_samples > 0:
        types.append('coarse')
    for typ in types:
        if rays_with_bg.shape[0] > 0:
            bg_lambda = results[f'bg_lambda_{typ}'][rays_with_bg]

            for key in TO_COMPOSITE:
                if f'{key}_{typ}' not in results:
                    continue

                val = results[f'{key}_{typ}']

                if get_bg_fg_rgb:
                    results[f'fg_{key}_{typ}'] = val

                expanded_bg_val = torch.zeros_like(val)

                mult = bg_lambda
                if len(val.shape) > 1:
                    mult = mult.unsqueeze(-1)


                if hparams.stop_semantic_grad and key == 'sem_map':
                    mult = mult.detach()
                expanded_bg_val[rays_with_bg] = bg_results[f'{key}_{typ}'] * mult

                if get_bg_fg_rgb:
                    results[f'bg_{key}_{typ}'] = expanded_bg_val
                results[f'{key}_{typ}'] = val + expanded_bg_val

        elif get_bg_fg_rgb:
            for key in TO_COMPOSITE:
                if f'{key}_{typ}' not in results:
                    continue

                val = results[f'{key}_{typ}']
                results[f'fg_{key}_{typ}'] = val
                results[f'bg_{key}_{typ}'] = torch.zeros_like(val)

    bg_nerf_rays_present = False
    
    return results, bg_nerf_rays_present


def _get_results(point_type,
                 nerf: nn.Module,
                 rays_d: torch.Tensor,
                 image_indices: Optional[torch.Tensor],
                 hparams: Namespace,
                 xyz_coarse: torch.Tensor,
                 z_vals: torch.Tensor,
                 last_delta: torch.Tensor,
                 get_depth: bool,
                 get_depth_variance: bool,
                 get_bg_lambda: bool,
                 depth_real: Optional[torch.Tensor],
                 xyz_fine_fn: Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]],
                 train_iterations=-1,
                 gt_depths=None,
                 valid_depth_mask=None,
                 s_near=None)-> Dict[str, torch.Tensor]:
    results = {}

    last_delta_diff = torch.zeros_like(last_delta)
    last_delta_diff[last_delta.squeeze() < 1e10, 0] = z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]


    _inference(point_type=point_type,
               results=results,
               typ='coarse',
               nerf=nerf,
               rays_d=rays_d,
               image_indices=image_indices,
               hparams=hparams,
               xyz=xyz_coarse,
               z_vals=z_vals,
               last_delta=last_delta - last_delta_diff,
               composite_rgb=hparams.use_cascade,
               get_depth=hparams.fine_samples == 0 and get_depth,
               get_depth_variance=hparams.fine_samples == 0 and get_depth_variance,
               get_weights=hparams.fine_samples > 0,
               get_bg_lambda=get_bg_lambda and hparams.use_cascade,
               depth_real=depth_real,
               train_iterations=train_iterations,
               gt_depths=gt_depths,
               valid_depth_mask=valid_depth_mask,
               s_near=s_near)


    if hparams.fine_samples > 0:  # sample points for fine model
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        perturb = hparams.perturb if nerf.training else 0
        if point_type == 'fg':
            fine_z_vals = _sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),hparams.fine_samples, det=(perturb == 0))
        elif point_type == 'bg_same_as_fg':
            fine_z_vals = _sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),hparams.fine_samples // 2, det=(perturb == 0))
        else:
            fine_z_vals = _sample_pdf(z_vals_mid, results['weights_coarse'][:, 1:-1].detach(),hparams.fine_samples // 2, det=(perturb == 0))

        if hparams.use_cascade:
            fine_z_vals, _ = torch.sort(torch.cat([z_vals, fine_z_vals], -1), -1)

        del results['weights_coarse']


        xyz_fine, depth_real_fine = xyz_fine_fn(fine_z_vals)

        # visualize_points_list = [xyz_fine.view(-1, 3).cpu().numpy()]
        # visualize_points(visualize_points_list)
        if hparams.contract_new:
            xyz_fine = contract_to_unisphere_new(xyz_fine, hparams)
        else:
            xyz_fine = contract_to_unisphere(xyz_fine, hparams)
        last_delta_diff = torch.zeros_like(last_delta)
        last_delta_diff[last_delta.squeeze() < 1e10, 0] = fine_z_vals[last_delta.squeeze() < 1e10].max(dim=-1)[0]

        _inference(point_type=point_type,
                   results=results,
                   typ='fine',
                   nerf=nerf,
                   rays_d=rays_d,
                   image_indices=image_indices,
                   hparams=hparams,
                   xyz=xyz_fine,
                   z_vals=fine_z_vals,
                   last_delta=last_delta - last_delta_diff,
                   composite_rgb=True,
                   get_depth=get_depth,
                   get_depth_variance=get_depth_variance,
                   get_weights=False,
                   get_bg_lambda=get_bg_lambda,
                   depth_real=depth_real_fine,
                   train_iterations=train_iterations,
                   gt_depths=gt_depths,
                   valid_depth_mask=valid_depth_mask,
                   s_near=s_near)

        for key in INTERMEDIATE_KEYS:
            if key in results:
                del results[key]

    return results


def _inference(point_type,
               results: Dict[str, torch.Tensor],
               typ: str,
               nerf: nn.Module,
               rays_d: torch.Tensor,
               image_indices: Optional[torch.Tensor],
               hparams: Namespace,
               xyz: torch.Tensor,
               z_vals: torch.Tensor,
               last_delta: torch.Tensor,
               composite_rgb: bool,
               get_depth: bool,
               get_depth_variance: bool,
               get_weights: bool,
               get_bg_lambda: bool,
               depth_real: Optional[torch.Tensor],
               train_iterations=-1,
               gt_depths=None,
               valid_depth_mask=None,
               s_near=None):


    N_rays_ = xyz.shape[0]
    N_samples_ = xyz.shape[1]
    xyz_ = xyz.view(-1, xyz.shape[-1])

    # Perform model inference to get rgb and raw sigma
    B = xyz_.shape[0]
    out_chunks = []
    out_semantic_chunk = [] 
    out_semantic_feature_chunk = []
    out_instance_chunk = []
    rays_d_ = rays_d.repeat(1, N_samples_, 1).view(-1, rays_d.shape[-1])

    if image_indices is not None:
        image_indices_ = image_indices.repeat(1, N_samples_, 1).view(-1, 1)


    # (N_rays*N_samples_, embed_dir_channels)
    for i in range(0, B, hparams.model_chunk_size):
        xyz_chunk = xyz_[i:i + hparams.model_chunk_size]

        if image_indices is not None:
            xyz_chunk = torch.cat([xyz_chunk,
                                   rays_d_[i:i + hparams.model_chunk_size],
                                   image_indices_[i:i + hparams.model_chunk_size]], 1)
        else:
            xyz_chunk = torch.cat([xyz_chunk, rays_d_[i:i + hparams.model_chunk_size]], 1)

        # sigma_noise = torch.rand(len(xyz_chunk), 1, device=xyz_chunk.device) if nerf.training else None
        sigma_noise=None

        if hparams.enable_semantic:
            if hparams.dataset_type == 'sam':
                model_chunk, semantic_chunk, semantic_feature_chunk= nerf(point_type, xyz_chunk, sigma_noise=sigma_noise, train_iterations=train_iterations)
                out_chunks += [model_chunk]
                out_semantic_chunk += [semantic_chunk]
                out_semantic_feature_chunk += [semantic_feature_chunk]
            else:
                model_chunk, semantic_chunk= nerf(point_type, xyz_chunk, sigma_noise=sigma_noise, train_iterations=train_iterations)
                out_chunks += [model_chunk]
                out_semantic_chunk += [semantic_chunk]
        else:
            model_chunk= nerf(point_type, xyz_chunk, sigma_noise=sigma_noise, train_iterations=train_iterations)
            out_chunks += [model_chunk]
        if hparams.enable_instance:
            instance_chunk= nerf.forward_instance(point_type, xyz_chunk)
            out_instance_chunk += [instance_chunk]


    out = torch.cat(out_chunks, 0)
    out = out.view(N_rays_, N_samples_, out.shape[-1])

    rgbs = out[..., :3]  # (N_rays, N_samples_, 3)
    sigmas = out[..., 3]  # (N_rays, N_samples_)

    

    gradient, normals = None, None
    if point_type == 'fg' and (hparams.visual_normal or hparams.normal_loss) and not hparams.save_depth: # for surface normal extraction
        gradient, normals = extract_gradients(nerf, xyz_, train_iterations, hparams, N_rays_, N_samples_)

    if hparams.enable_semantic:
        out_semantic = torch.cat(out_semantic_chunk, 0)
        if len(out_semantic.shape)== 1:
            out_semantic = out_semantic.unsqueeze(-1)
        sem_logits = out_semantic.view(N_rays_, N_samples_, out_semantic.shape[-1])
        if hparams.dataset_type == 'sam':
            out_semantic_fea = torch.cat(out_semantic_feature_chunk, 0)
            sem_feature = out_semantic_fea.view(N_rays_, N_samples_, out_semantic_fea.shape[-1])  
    if hparams.enable_instance:
        out_instance = torch.cat(out_instance_chunk, 0)
        if len(out_instance.shape)== 1:
            out_instance = out_instance.unsqueeze(-1)
        instance_logits = out_instance.view(N_rays_, N_samples_, out_instance.shape[-1])

    # del out, out_chunks, out_semantic_chunk
    # gc.collect()
    # torch.cuda.empty_cache()

    if 'zvals_coarse' in results:
        # combine coarse and fine samples
        z_vals, ordering = torch.sort(torch.cat([z_vals, results['zvals_coarse']], -1), -1, descending=False)
        ordering_3c = ordering.view(ordering.shape[0], ordering.shape[1], 1).repeat(1,1,rgbs.shape[-1])
        rgbs = torch.gather(torch.cat((rgbs, results['raw_rgb_coarse']), 1), 1, ordering_3c)
        sigmas = torch.gather(torch.cat((sigmas, results['raw_sigma_coarse']), 1), 1, ordering)
        if hparams.enable_semantic:
            ordering_Kc = ordering.view(ordering.shape[0], ordering.shape[1], 1).repeat(1,1,sem_logits.shape[-1])
            sem_logits = torch.gather(torch.cat((sem_logits, results['raw_sem_logits_coarse']), 1), 1, ordering_Kc)
            if hparams.dataset_type == 'sam':
                ordering_Cc = ordering.view(ordering.shape[0], ordering.shape[1], 1).repeat(1,1,sem_feature.shape[-1])
                sem_feature = torch.gather(torch.cat((sem_feature, results['raw_sem_feature_coarse']), 1), 1, ordering_Cc)
        if hparams.enable_instance:
            ordering_Kc = ordering.view(ordering.shape[0], ordering.shape[1], 1).repeat(1,1,instance_logits.shape[-1])
            instance_logits = torch.gather(torch.cat((instance_logits, results['raw_instance_logits_coarse']), 1), 1, ordering_Kc)

        if point_type == 'fg' and normals is not None:
            normals = torch.gather(torch.cat((normals, results['raw_rgb_coarse']), 1), 1, ordering_3c)


        if depth_real is not None:
            depth_real = torch.gather(torch.cat((depth_real, results['depth_real_coarse']), 1), 1,
                                      ordering)

    # Convert these values using volume rendering (Section 4)

    deltas = z_vals[:, 1:] - z_vals[:, :-1]  # (N_rays, N_samples_-1)
    deltas = torch.cat([deltas, last_delta], -1)  # (N_rays, N_samples_)

    alphas = 1 - torch.exp(-deltas * sigmas)  # (N_rays, N_samples_)


    T = torch.cumprod(1 - alphas + 1e-8, -1)
    if get_bg_lambda: # only when foreground fine =True
        results[f'bg_lambda_{typ}'] = T[..., -1]

    T = torch.cat((torch.ones_like(T[..., 0:1]), T[..., :-1]), dim=-1)  # [..., N_samples]

    weights = alphas * T  # (N_rays, N_samples_)

    if get_weights: # coarse = True, fine = False
        results[f'weights_{typ}'] = weights

    

    if composite_rgb: # coarse = False, fine = True
        results[f'rgb_{typ}'] = (weights.unsqueeze(-1) * rgbs).sum(dim=1)  # n1 n2 c -> n1 c
        
        # if valid_depth_mask is not None and s_near is not None and hparams.wgt_air_sigma_loss != 0:
        #     if 'air_sigma_loss' not in results:
        #         results['air_sigma_loss'] = 0
        #     air_point = z_vals[valid_depth_mask]<s_near
        #     criterion = nn.MSELoss()
        #     zeros_target = torch.zeros(sigmas[valid_depth_mask][air_point].shape).to(weights.device)
        #     air_sigma_loss = criterion(sigmas[valid_depth_mask][air_point], zeros_target)
        #     results['air_sigma_loss'] += air_sigma_loss
            
        # if hparams.depth_dji_loss and hparams.wgt_sigma_loss !=0:
        #     valid_depth_mask = ~torch.isinf(gt_depths)
        #     err = 1
        #     dists = z_vals[:, 1:] - z_vals[:, :-1]
        #     dists = torch.cat([dists, last_delta], -1)  # (N_rays, N_samples_)

        #     dists = dists * torch.norm(rays_d, dim=-1)
        #     sigma_loss = -torch.log(weights + 1e-5) * torch.exp(-(z_vals - (gt_depths)[:,None]) ** 2 / (2 * err)) * dists
        #     sigma_loss = sigma_loss[valid_depth_mask]
        #     sigma_loss = torch.sum(sigma_loss, dim=1).mean()
        #     if 'sigma_loss' in results:
        #         results['sigma_loss'] += sigma_loss
        #     else:
        #         results['sigma_loss'] = sigma_loss
                
        if hparams.enable_semantic:
            if hparams.stop_semantic_grad:
                w = weights[..., None].detach()
                sem_map = torch.sum(w * sem_logits, -2)      
                if hparams.dataset_type == 'sam':
                    semantic_feature = torch.sum(w * sem_feature, -2)
                    results[f'semantic_feature_{typ}'] = semantic_feature
            else:
                sem_map = torch.sum(weights[..., None] * sem_logits, -2)
            results[f'sem_map_{typ}'] = sem_map
        if hparams.enable_instance:
            w = weights[..., None].detach()
            instance_map = torch.sum(w * instance_logits, -2)
            results[f'instance_map_{typ}'] = instance_map
            
        if point_type == 'fg' and normals is not None:
            normal_map = (weights.unsqueeze(-1) * normals).sum(dim=1)
            # normal_map[:, 1:] = normal_map[:, 1:] * -1 # flip normal map
            results[f'normal_map_{typ}'] = normal_map
    else:
        results[f'zvals_{typ}'] = z_vals
        results[f'raw_rgb_{typ}'] = rgbs
        results[f'raw_sigma_{typ}'] = sigmas
        if depth_real is not None:
            results[f'depth_real_{typ}'] = depth_real
        if hparams.enable_semantic:
            results[f'raw_sem_logits_{typ}'] = sem_logits
            if hparams.dataset_type == 'sam':
                results[f'raw_sem_feature_{typ}'] = sem_feature
        if hparams.enable_instance:
            results[f'raw_instance_logits_{typ}'] = instance_logits
        
        if point_type == 'fg' and normals is not None:
            results[f'raw_normal_{typ}'] = normals
        
    if hparams.depth_loss or hparams.depth_dji_loss:
        depth_map = (weights * z_vals).sum(dim=1)  # n1 n2 -> n1
        results[f'depth_{typ}'] = depth_map
        with torch.no_grad():
            if get_depth_variance:# coarse = False, fine = True
                results[f'depth_variance_{typ}'] = (weights * (z_vals - depth_map.unsqueeze(1)).square()).sum(
                    axis=-1)
        

    else:
        with torch.no_grad():
            if get_depth or get_depth_variance:
                if depth_real is not None:
                    depth_map = (weights * depth_real).sum(dim=1)  # n1 n2 -> n1
                else:
                    depth_map = (weights * z_vals).sum(dim=1)  # n1 n2 -> n1

            if get_depth: # always False
                results[f'depth_{typ}'] = depth_map

            if get_depth_variance:# coarse = False, fine = True
                results[f'depth_variance_{typ}'] = (weights * (z_vals - depth_map.unsqueeze(1)).square()).sum(
                    axis=-1)

def _intersect_sphere(rays_o: torch.Tensor, rays_d: torch.Tensor, sphere_center: torch.Tensor,
                      sphere_radius: torch.Tensor, render_zyq: bool) -> torch.Tensor:
    if sphere_radius is not None:
        rays_o = (rays_o - sphere_center) / sphere_radius
        rays_d = rays_d / sphere_radius

    '''
    rays_o, rays_d: [..., 3]
    compute the depth of the intersection point between this ray and unit sphere
    '''
    # note: d1 becomes negative if this mid point is behind camera
    d1 = -torch.sum(rays_d * rays_o, dim=-1) / torch.sum(rays_d * rays_d, dim=-1)
    p = rays_o + d1.unsqueeze(-1) * rays_d
    # consider the case where the ray does not intersect the sphere
    ray_d_cos = 1. / torch.norm(rays_d, dim=-1)
    p_norm_sq = torch.sum(p * p, dim=-1)
    if (p_norm_sq >= 1.).any() and render_zyq == False:
    # if False:
        raise Exception(
            'Not all your cameras are bounded by the unit sphere; please make sure the cameras are normalized properly!')
    d2 = torch.sqrt(1. - p_norm_sq) * ray_d_cos

    return d1 + d2

def _expand_and_perturb_z_vals(z_vals, samples, perturb, N_rays):
    z_vals = z_vals.expand(N_rays, samples)
    if perturb > 0:  # perturb sampling depths (z_vals)
        z_vals_mid = 0.5 * (z_vals[:, :-1] + z_vals[:, 1:])  # (N_rays, N_samples-1) interval mid points
        # get intervals between samples
        upper = torch.cat([z_vals_mid, z_vals[:, -1:]], -1)
        lower = torch.cat([z_vals[:, :1], z_vals_mid], -1)

        perturb_rand = perturb * torch.rand_like(z_vals)
        z_vals = lower + (upper - lower) * perturb_rand

    return z_vals


def _sample_pdf(bins: torch.Tensor, weights: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        fine_samples: the number of samples to draw from the distribution
        det: deterministic or not
    Outputs:
        samples: the sampled samples
    """
    weights = weights + 1e-8  # prevent division by zero (don't do inplace op!)

    pdf = weights / weights.sum(-1).unsqueeze(-1)  # (N_rays, N_samples_)

    cdf = torch.cumsum(pdf, -1)  # (N_rays, N_samples), cumulative distribution function
    return _sample_cdf(bins, cdf, fine_samples, det)


def _sample_cdf(bins: torch.Tensor, cdf: torch.Tensor, fine_samples: int, det: bool) -> torch.Tensor:
    N_rays, N_samples_ = cdf.shape

    cdf = torch.cat([torch.zeros_like(cdf[:, :1]), cdf], -1)  # (N_rays, N_samples_+1)
    # padded to 0~1 inclusive
    if det:
        u = torch.linspace(0, 1, fine_samples, device=bins.device)
        u = u.expand(N_rays, fine_samples)
    else:
        u = torch.rand(N_rays, fine_samples, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds - 1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = torch.stack([below, above], -1)
    inds_sampled = inds_sampled.view(inds_sampled.shape[0], -1)  # n1 n2 2 -> n1 (n2 2)

    cdf_g = torch.gather(cdf, 1, inds_sampled)
    cdf_g = cdf_g.view(cdf_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    bins_g = torch.gather(bins, 1, inds_sampled)
    bins_g = bins_g.view(bins_g.shape[0], -1, 2)  # n1 (n2 2) -> n1 n2 2

    denom = cdf_g[..., 1] - cdf_g[..., 0]
    denom[denom < 1e-8] = 1  # denom equals 0 means a bin has weight 0,
    # in which case it will not be sampled
    # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[..., 0] + (u - cdf_g[..., 0]) / denom * (bins_g[..., 1] - bins_g[..., 0])
    return samples

def extract_gradients(nerf, xyz_, train_iterations, hparams, N_rays_, N_samples_):
    normal_epsilon_ratio = min((train_iterations) / hparams.train_iterations, 0.50)
    if hparams.normal_loss:
        if hparams.auto_grad:
            gradient = nerf.auto_gradient(xyz_)
        else:
            gradient = nerf.gradient(xyz_, 0.005 * (1.0 - normal_epsilon_ratio)).squeeze()
    elif hparams.visual_normal:
        with torch.no_grad():
            if hparams.auto_grad:
                gradient = nerf.auto_gradient(xyz_)#.squeeze()
            else:
                gradient = nerf.gradient(xyz_, 0.005 * (1.0 - normal_epsilon_ratio)).squeeze()

    if gradient is not None:
        normals = gradient / (1e-5 + torch.linalg.norm(gradient, ord=2, dim=-1,  keepdim = True))
        #print('xyz, normal', xyz_.shape, normals.shape, results[f'rgb_{typ}'].shape)
        normals = normals.view(N_rays_, N_samples_, 3)
    return gradient, normals
