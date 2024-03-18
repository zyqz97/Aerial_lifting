import torch
import numpy as np

def  bg_sample_inv(near, far, point_num, device):
    z = torch.linspace(0, 1, point_num, device=device)
    z_vals = 1. / near * (1 - z) + 1. / far * (z) # linear combination in the inveres space
    z_vals = 1. / z_vals # inverse back
    return z_vals


#@torch.no_grad()
def contract_to_unisphere(x: torch.Tensor, hparams):
    aabb_bound = hparams.aabb_bound
    aabb = torch.tensor([-aabb_bound, -aabb_bound, -aabb_bound, aabb_bound, aabb_bound, aabb_bound]).to(x.device)
    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    if hparams.contract_norm == 'inf':
        mag = x.abs().amax(dim=-1, keepdim=True)
    elif hparams.contract_norm == 'l2':
        mag = x.norm(dim=-1, keepdim=True)
    else:
        print("the norm of contract is wrong!")
        raise NotImplementedError
    mask = mag.squeeze(-1) > 1
    x[mask] = (1 + hparams.contract_bg_len - hparams.contract_bg_len / mag[mask]) * (x[mask] / mag[mask])  # out of bound points trun to [-2, 2]
    return x


@torch.no_grad()
def contract_to_unisphere_new(x: torch.Tensor, hparams):
    aabb = hparams.stretch
    # aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    aabb_min, aabb_max = aabb[0], aabb[1]
    x = (x - aabb_min) / (aabb_max - aabb_min)   #[0, 1]
    x = x * 2 - 1  # aabb is at [-1, 1]
    if hparams.contract_norm == 'inf':
        mag = x.abs().amax(dim=-1, keepdim=True)
    elif hparams.contract_norm == 'l2':
        mag = x.norm(dim=-1, keepdim=True)
    else:
        print("the norm of contract is wrong!")
        raise NotImplementedError
    mask = mag.squeeze(-1) > 1
    x[mask] = (1 + hparams.contract_bg_len - hparams.contract_bg_len / mag[mask]) * (x[mask] / mag[mask])  # out of bound points to [-2, 2]
    return x







def contract_to_unisphere_box(x: torch.Tensor, hparams):
    aabb_bound = hparams.fg_box_bound
    aabb = torch.tensor([aabb_bound[0,0], aabb_bound[0,1], aabb_bound[0,2], aabb_bound[1,0], aabb_bound[1,1], aabb_bound[1,2]]).to(dtype=torch.float32, device=x.device)

    aabb_min, aabb_max = torch.split(aabb, 3, dim=-1)
    x = (x - aabb_min) / (aabb_max - aabb_min)
    x = x * 2 - 1  # aabb is at [-1, 1]
    if hparams.contract_norm == 'inf':
        mag = x.abs().amax(dim=-1, keepdim=True)
    elif hparams.contract_norm == 'l2':
        mag = x.norm(dim=-1, keepdim=True)
    else:
        print("the norm of contract is wrong!")
        raise NotImplementedError
    mask = mag.squeeze(-1) > 1
    x[mask] = (1 + hparams.contract_bg_len - hparams.contract_bg_len / mag[mask]) * (x[mask] / mag[mask])  # out of bound points trun to [-2, 2]
    return x


# ray box intersection from  https://github.com/zju3dv/neuralbody/blob/301ab711418dd118b59eed833e34aa4d39e1cc0b/lib/utils/if_nerf/if_nerf_data_utils.py#L54C5-L54C17
# def get_near_far(bounds, ray_o, ray_d):
def get_box_intersection(bounds, ray_o, ray_d):
    """calculate intersections with 3d bounding box"""
    norm_d = np.linalg.norm(ray_d, axis=-1, keepdims=True)
    viewdir = ray_d / norm_d
    viewdir[(viewdir < 1e-5) & (viewdir > -1e-10)] = 1e-5
    viewdir[(viewdir > -1e-5) & (viewdir < 1e-10)] = -1e-5
    tmin = (bounds[:1] - ray_o) / viewdir
    tmax = (bounds[1:2] - ray_o) / viewdir
    t1 = np.minimum(tmin, tmax)
    t2 = np.maximum(tmin, tmax)
    near = np.max(t1, axis=-1)
    far = np.min(t2, axis=-1)
    mask_at_box = near < far
    near = near[mask_at_box] / norm_d[mask_at_box, 0]
    far = far[mask_at_box] / norm_d[mask_at_box, 0]
    return near, far, mask_at_box

# def get_box_intersection(bounds, ray_o, ray_d):
#     tmin = (bounds[:1] - ray_o[:1]) / viewdir
    