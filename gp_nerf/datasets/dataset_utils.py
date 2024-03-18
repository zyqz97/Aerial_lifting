from typing import Tuple, Optional

import torch

from gp_nerf.image_metadata import ImageMetadata


def get_rgb_index_mask(metadata: ImageMetadata) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    rgbs = metadata.load_image().view(-1, 3)

    keep_mask = metadata.load_mask()


    if metadata.is_val:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        else:
            # Get how many pixels we're discarding that would otherwise be added
            discard_half = keep_mask[:, metadata.W // 2:]
            discard_pos_count = discard_half[discard_half == True].shape[0]

            candidates_to_add = torch.arange(metadata.H * metadata.W).view(metadata.H, metadata.W)[:, :metadata.W // 2]
            keep_half = keep_mask[:, :metadata.W // 2]
            candidates_to_add = candidates_to_add[keep_half == False].reshape(-1)
            to_add = candidates_to_add[torch.randperm(candidates_to_add.shape[0])[:discard_pos_count]]

            keep_mask.view(-1).scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))

        keep_mask[:, metadata.W // 2:] = False

    if keep_mask is not None:
        if keep_mask[keep_mask == True].shape[0] == 0:
            return None

        keep_mask = keep_mask.view(-1)
        rgbs = rgbs[keep_mask == True]

    labels = metadata.load_label()
    if labels is not None:
        labels = labels.view(-1)
        if keep_mask is not None :
            labels = labels[keep_mask == True]

    assert metadata.image_index <= torch.iinfo(torch.int32).max
    return rgbs, metadata.image_index * torch.ones(rgbs.shape[0], dtype=torch.int32), keep_mask, labels



def get_rgb_index_mask_depth_dji(metadata: ImageMetadata) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    rgbs = metadata.load_image().view(-1, 3)

    keep_mask = metadata.load_mask()


    if metadata.is_val:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        else:
            # Get how many pixels we're discarding that would otherwise be added
            discard_half = keep_mask[:, metadata.W // 2:]
            discard_pos_count = discard_half[discard_half == True].shape[0]

            candidates_to_add = torch.arange(metadata.H * metadata.W).view(metadata.H, metadata.W)[:, :metadata.W // 2]
            keep_half = keep_mask[:, :metadata.W // 2]
            candidates_to_add = candidates_to_add[keep_half == False].reshape(-1)
            to_add = candidates_to_add[torch.randperm(candidates_to_add.shape[0])[:discard_pos_count]]

            keep_mask.view(-1).scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))

        keep_mask[:, metadata.W // 2:] = False

    # # -----过滤黑影
    if (metadata.left_or_right) != None:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        if metadata.left_or_right == 'left':
            keep_mask[:, :int(metadata.W/5)] = False
        elif metadata.left_or_right == 'right':
            keep_mask[:, -int(metadata.W/5):] = False

    if keep_mask is not None:
        if keep_mask[keep_mask == True].shape[0] == 0:
            return None
        keep_mask = keep_mask.view(-1)
        rgbs = rgbs[keep_mask == True]

    labels = metadata.load_label()
    if labels is not None:
        labels = labels.view(-1)
        if keep_mask is not None :
            labels = labels[keep_mask == True]
    
    depth_dji = metadata.load_depth_dji()
    if depth_dji is not None:
        depth_dji = depth_dji.view(-1)
        if keep_mask is not None :
            depth_dji = depth_dji[keep_mask == True]


    assert metadata.image_index <= torch.iinfo(torch.int32).max
    return rgbs, metadata.image_index * torch.ones(rgbs.shape[0], dtype=torch.int32), keep_mask, labels, depth_dji


#这里把keep masks 筛点的操作直接放到dataloader里面，根据semantic进行过滤，这样保证都是（H,W）
def get_rgb_index_mask_depth_dji_instance(metadata: ImageMetadata) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    rgbs = metadata.load_image().view(-1, 3)

    keep_mask = metadata.load_mask()


    if metadata.is_val:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        else:
            # Get how many pixels we're discarding that would otherwise be added
            discard_half = keep_mask[:, metadata.W // 2:]
            discard_pos_count = discard_half[discard_half == True].shape[0]

            candidates_to_add = torch.arange(metadata.H * metadata.W).view(metadata.H, metadata.W)[:, :metadata.W // 2]
            keep_half = keep_mask[:, :metadata.W // 2]
            candidates_to_add = candidates_to_add[keep_half == False].reshape(-1)
            to_add = candidates_to_add[torch.randperm(candidates_to_add.shape[0])[:discard_pos_count]]

            keep_mask.view(-1).scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))

        keep_mask[:, metadata.W // 2:] = False

    # # -----过滤黑影
    if (metadata.left_or_right) != None:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        if metadata.left_or_right == 'left':
            keep_mask[:, :int(metadata.W/5)] = False
        elif metadata.left_or_right == 'right':
            keep_mask[:, -int(metadata.W/5):] = False


    if keep_mask is not None:
        if keep_mask[keep_mask == True].shape[0] == 0:
            return None
        keep_mask = keep_mask.view(-1)
        # rgbs = rgbs[keep_mask == True]

    labels = metadata.load_label()
    if labels is not None:
        labels = labels.view(-1)
        # if keep_mask is not None :
            # labels = labels[keep_mask == True]

    instance = metadata.load_instance()
    if instance is not None:
        instance = instance.view(-1)
        # if keep_mask is not None :
            # instance = instance[keep_mask == True]

    depth_dji = metadata.load_depth_dji()
    if depth_dji is not None:
        depth_dji = depth_dji.view(-1)
        # if keep_mask is not None :
            # depth_dji = depth_dji[keep_mask == True]

    
    assert metadata.image_index <= torch.iinfo(torch.int32).max
    # return rgbs, metadata.image_index * torch.ones(rgbs.shape[0], dtype=torch.int32), keep_mask, instance, depth_dji
    return rgbs, metadata.image_index, keep_mask, labels, depth_dji, instance


def get_rgb_index_mask_depth_dji_instance_crossview(metadata: ImageMetadata) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
    rgbs = metadata.load_image().view(-1, 3)

    keep_mask = metadata.load_mask()


    if metadata.is_val:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        else:
            # Get how many pixels we're discarding that would otherwise be added
            discard_half = keep_mask[:, metadata.W // 2:]
            discard_pos_count = discard_half[discard_half == True].shape[0]

            candidates_to_add = torch.arange(metadata.H * metadata.W).view(metadata.H, metadata.W)[:, :metadata.W // 2]
            keep_half = keep_mask[:, :metadata.W // 2]
            candidates_to_add = candidates_to_add[keep_half == False].reshape(-1)
            to_add = candidates_to_add[torch.randperm(candidates_to_add.shape[0])[:discard_pos_count]]

            keep_mask.view(-1).scatter_(0, to_add, torch.ones_like(to_add, dtype=torch.bool))

        keep_mask[:, metadata.W // 2:] = False

    # # -----过滤黑影
    if (metadata.left_or_right) != None:
        if keep_mask is None:
            keep_mask = torch.ones(metadata.H, metadata.W, dtype=torch.bool)
        if metadata.left_or_right == 'left':
            keep_mask[:, :int(metadata.W/5)] = False
        elif metadata.left_or_right == 'right':
            keep_mask[:, -int(metadata.W/5):] = False

    if keep_mask is not None:
        if keep_mask[keep_mask == True].shape[0] == 0:
            return None
        keep_mask = keep_mask.view(-1)
        # rgbs = rgbs[keep_mask == True]

    
    labels = metadata.load_label()
    if labels is not None:
        labels = labels.view(-1)
        # if keep_mask is not None :
            # labels = labels[keep_mask == True]

    instance = metadata.load_instance()
    if instance is not None:
        instance = instance.view(-1)
        # if keep_mask is not None :
            # instance = instance[keep_mask == True]

    instance_crossview = metadata.load_instance_crossview()
    if instance_crossview is not None:
        instance_crossview = instance_crossview.view(-1)
        # if keep_mask is not None :
            # instance_crossview = instance_crossview[keep_mask == True]
    
    instance_64 = metadata.load_instance_64()
    if instance_64 is not None:
        instance_64 = instance_64.view(-1)
        # if keep_mask is not None :
            # instance_64 = instance_64[keep_mask == True]

    depth_dji = metadata.load_depth_dji()
    if depth_dji is not None:
        depth_dji = depth_dji.view(-1)
        # if keep_mask is not None :
            # depth_dji = depth_dji[keep_mask == True]

    


    assert metadata.image_index <= torch.iinfo(torch.int32).max
    # return rgbs, metadata.image_index * torch.ones(rgbs.shape[0], dtype=torch.int32), keep_mask, instance, depth_dji
    # return rgbs, metadata.image_index, keep_mask, instance, depth_dji, instance_crossview
    return rgbs, metadata.image_index, keep_mask, labels, depth_dji, instance, instance_crossview, instance_64

