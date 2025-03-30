#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Data loader."""

import itertools
from functools import partial
from typing import Optional, List

import numpy as np
import torch

from core.dataset.multigrid_helper import ShortCycleBatchSampler
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, Sampler

from . import utils as utils
from .build import build_dataset


def multiple_samples_collate(batch, fold=False):
    """
    Collate function for repeated augmentation. Each instance in the batch has
    more than one sample.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)
    inputs = [item for sublist in inputs for item in sublist]
    labels = [item for sublist in labels for item in sublist]
    video_idx = [item for sublist in video_idx for item in sublist]
    time = [item for sublist in time for item in sublist]

    inputs, labels, video_idx, time, extra_data = (
        default_collate(inputs),
        default_collate(labels),
        default_collate(video_idx),
        default_collate(time),
        default_collate(extra_data),
    )
    if fold:
        return [inputs], labels, video_idx, time, extra_data
    else:
        return inputs, labels, video_idx, time, extra_data


def detection_collate(batch):
    """
    Collate function for detection task. Concatanate bboxes, labels and
    metadata from different samples in the first dimension instead of
    stacking them to have a batch-size dimension.
    Args:
        batch (tuple or list): data batch to collate.
    Returns:
        (tuple): collated detection data batch.
    """
    inputs, labels, video_idx, time, extra_data = zip(*batch)
    # labels = torch.tensor(np.concatenate(labels, axis=0)).float()
    inputs, video_idx = default_collate(inputs), default_collate(video_idx)
    batch_lengths = [array.shape[0] for array in labels]
    time = default_collate(time)
    action_labels = torch.tensor(np.concatenate(labels, axis=0)).float()
    collated_extra_data = {}
    for key in extra_data[0].keys():
        data = [d[key] for d in extra_data]
        if key == "boxes" or key == 'ori_boxes':
            # collated_extra_data["unstack_boxes"] = [data[i] for i in range(len(data))]
            # Append idx info to the bboxes before concatenating them.
            bboxes = [
                np.concatenate(
                    [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
                )
                for i in range(len(data))
            ]
            bboxes = np.concatenate(bboxes, axis=0)
            collated_extra_data[key] = torch.tensor(bboxes).float()
        elif key == "metadata":
            collated_extra_data[key] = torch.tensor(list(itertools.chain(*data))).view(
                -1, 2
            )
        # elif key == 'ori_boxes':
        #     collated_extra_data[key] = torch.tensor(data).float()
        else:
            collated_extra_data[key] = default_collate(data)
            # collated_extra_data[key] = [data[key] for data in extra_data]#todo: do not stack labels
    collated_extra_data['action_labels'] = action_labels
    for key, val in collated_extra_data.items():    #todo
        if isinstance(val, np.ndarray):
            collated_extra_data[key] = torch.tensor(val).float()
        elif isinstance(val, list):
            try:
                collated_extra_data[key] = torch.tensor(val).float()
            except Exception:
                pass

    return inputs, action_labels, video_idx, time, collated_extra_data, batch_lengths




# def detection_collate(batch):
#     """
#     Collate function for detection task. Concatanate bboxes, labels and
#     metadata from different samples in the first dimension instead of
#     stacking them to have a batch-size dimension.
#     Args:
#         batch (tuple or list): data batch to collate.
#     Returns:
#         (tuple): collated detection data batch.
#     """
#     inputs, labels, video_idx, time, extra_data = zip(*batch)
#     inputs, video_idx = default_collate(inputs), default_collate(video_idx)
#     time = default_collate(time)
#     labels = torch.tensor(np.concatenate(labels, axis=0)).float()
#
#     collated_extra_data = {}
#     for key in extra_data[0].keys():
#         data = [d[key] for d in extra_data]
#         if key == "boxes" or key == "ori_boxes":
#             # Append idx info to the bboxes before concatenating them.
#             bboxes = [
#                 np.concatenate(
#                     [np.full((data[i].shape[0], 1), float(i)), data[i]], axis=1
#                 )
#                 for i in range(len(data))
#             ]
#             bboxes = np.concatenate(bboxes, axis=0)
#             collated_extra_data[key] = torch.tensor(bboxes).float()
#         elif key == "metadata":
#             collated_extra_data[key] = torch.tensor(list(itertools.chain(*data))).view(
#                 -1, 2
#             )
#         else:
#             collated_extra_data[key] = default_collate(data)
#
#     return inputs, labels, video_idx, time, collated_extra_data

def nested_tensor_from_tensor_list(tensor_list: List[torch.Tensor]):
    """
    Converts a list of tensors (possibly with different sizes) into a NestedTensor.
    Pads tensors to the same size and creates a mask to indicate valid regions.
    Args:
        tensor_list (List[Tensor]): List of tensors to be converted.
    Returns:
        NestedTensor: A NestedTensor containing the padded tensors and a mask.
    """
    #slowfast 的 dataset 处理的input是包含了slow和fast的，不能直接这样处理
    if tensor_list[0].ndim == 3:  # Case for 3D tensors (e.g., images)
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size  # [batch_size, C, H, W]

        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        # Initialize padded tensor and mask
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)

        # Pad tensors and update mask
        for i, img in enumerate(tensor_list):
            tensor[i, : img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            mask[i, : img.shape[1], : img.shape[2]] = False

    elif tensor_list[0].ndim == 4:  # Case for 4D tensors (e.g., video clips)
        max_size = _max_by_axis([list(clip.shape) for clip in tensor_list])
        batch_shape = [len(tensor_list)] + max_size  # [batch_size, C, T, H, W]

        b, c, t, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device

        # Initialize padded tensor and mask
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, t, h, w), dtype=torch.bool, device=device)

        # Pad tensors and update mask
        for i, clip in enumerate(tensor_list):
            tensor[i, : clip.shape[0], : clip.shape[1], : clip.shape[2], : clip.shape[3]].copy_(clip)
            mask[i, : clip.shape[2], : clip.shape[3]] = False

    else:
        raise ValueError("Only 3D or 4D tensors are supported.")

    return NestedTensor(tensor, mask)

def _max_by_axis(sizes):
    """
    Find the maximum size along each axis in a list of sizes.
    Args:
        sizes (List[List[int]]): List of sizes for each tensor.
    Returns:
        List[int]: Maximum size along each axis.
    """
    maxes = sizes[0]
    for size in sizes[1:]:
        maxes = [max(max_val, cur_val) for max_val, cur_val in zip(maxes, size)]
    return maxes

class NestedTensor(object):
    """
    A data structure that holds a tensor and its corresponding mask.
    Useful for handling batches with variable-sized inputs.
    """
    def __init__(self, tensors, mask: Optional[torch.Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        """
        Move tensors and mask to the specified device.
        Args:
            device: Target device.
        Returns:
            NestedTensor: Moved NestedTensor.
        """
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        """
        Decompose the NestedTensor into its tensors and mask.
        Returns:
            (Tensor, Tensor): Tensors and mask.
        """
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def construct_loader(cfg, split, is_precise_bn=False):
    """
    Constructs the data loader for the given dataset.
    Args:
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        split (str): the split of the data loader. Options include `train`,
            `val`, and `test`.
    """
    assert split in ["train", "val", "test"]
    if split in ["train"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = True
        drop_last = True
    elif split in ["val"]:
        dataset_name = cfg.TRAIN.DATASET
        batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False
    elif split in ["test"]:
        dataset_name = cfg.TEST.DATASET
        batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
        shuffle = False
        drop_last = False

    # Construct the dataset
    dataset = build_dataset(dataset_name, cfg, split)

    if isinstance(dataset, torch.utils.data.IterableDataset):
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=cfg.DATA_LOADER.NUM_WORKERS,
            pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
            drop_last=drop_last,
            collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
            worker_init_fn=utils.loader_worker_init_fn(dataset),
        )
    else:
        if cfg.MULTIGRID.SHORT_CYCLE and split in ["train"] and not is_precise_bn:
            # Create a sampler for multi-process training
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            batch_sampler = ShortCycleBatchSampler(
                sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
            )
            # Create a loader
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                worker_init_fn=utils.loader_worker_init_fn(dataset),
            )
        else:
            # Create a sampler for multi-process training
            sampler = utils.create_sampler(dataset, shuffle, cfg)
            # Create a loader
            if cfg.DETECTION.ENABLE:
                collate_func = detection_collate
            elif (
                (
                    cfg.AUG.NUM_SAMPLE > 1
                    or cfg.DATA.TRAIN_CROP_NUM_TEMPORAL > 1
                    or cfg.DATA.TRAIN_CROP_NUM_SPATIAL > 1
                )
                and split in ["train"]
                and not cfg.MODEL.MODEL_NAME == "ContrastiveModel"
            ):
                collate_func = partial(
                    multiple_samples_collate, fold="imagenet" in dataset_name
                )
            else:
                collate_func = None
            loader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(False if sampler else shuffle),
                sampler=sampler,
                num_workers=cfg.DATA_LOADER.NUM_WORKERS,
                pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
                drop_last=drop_last,
                collate_fn=collate_func,
                worker_init_fn=utils.loader_worker_init_fn(dataset),
            )
    # for batch_idx, (inputs, labels, *others) in enumerate(loader):
    #     print(f"Batch {batch_idx}:")
    #     print(f"  Inputs shape: {[x.shape for x in inputs] if isinstance(inputs, list) else inputs.shape}")
    #     print(f"  Labels shape: {labels.shape if isinstance(labels, torch.Tensor) else type(labels)}")
    #     print(f"  Other data: {others}")
    #     break  # 查看第一个批次即可，避免打印过多数据
    return loader, dataset


# def construct_loader(cfg, split, is_precise_bn=False):
#     """
#     Constructs the data loader for the given dataset.
#     Args:
#         cfg (CfgNode): configs. Details can be found in
#             slowfast/config/defaults.py
#         split (str): the split of the data loader. Options include `train`,
#             `val`, and `test`.
#     """
#     assert split in ["train", "val", "test"]
#     if split in ["train"]:
#         dataset_name = cfg.TRAIN.DATASET
#         batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
#         shuffle = True
#         drop_last = True
#     elif split in ["val"]:
#         dataset_name = cfg.TRAIN.DATASET
#         batch_size = int(cfg.TRAIN.BATCH_SIZE / max(1, cfg.NUM_GPUS))
#         shuffle = False
#         drop_last = False
#     elif split in ["test"]:
#         dataset_name = cfg.TEST.DATASET
#         batch_size = int(cfg.TEST.BATCH_SIZE / max(1, cfg.NUM_GPUS))
#         shuffle = False
#         drop_last = False
#
#     # Construct the dataset
#     dataset = build_dataset(dataset_name, cfg, split)
#     if isinstance(dataset, torch.utils.data.IterableDataset):
#         loader = torch.utils.data.DataLoader(
#             dataset,
#             batch_size=batch_size,
#             num_workers=cfg.DATA_LOADER.NUM_WORKERS,
#             pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
#             drop_last=drop_last,
#             collate_fn=detection_collate if cfg.DETECTION.ENABLE else None,
#             worker_init_fn=utils.loader_worker_init_fn(dataset),
#         )
#     else:
#         if cfg.MULTIGRID.SHORT_CYCLE and split in ["train"] and not is_precise_bn:
#             # Create a sampler for multi-process training
#             sampler = utils.create_sampler(dataset, shuffle, cfg)
#             batch_sampler = ShortCycleBatchSampler(
#                 sampler, batch_size=batch_size, drop_last=drop_last, cfg=cfg
#             )
#             # Create a loader
#             loader = torch.utils.data.DataLoader(
#                 dataset,
#                 batch_sampler=batch_sampler,
#                 num_workers=cfg.DATA_LOADER.NUM_WORKERS,
#                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
#                 worker_init_fn=utils.loader_worker_init_fn(dataset),
#             )
#         else:   #--->
#             # Create a sampler for multi-process training
#             sampler = utils.create_sampler(dataset, shuffle, cfg)
#             # Create a loader
#             if cfg.DETECTION.ENABLE:
#                 collate_func = detection_collate
#             elif (
#                 (
#                     cfg.AUG.NUM_SAMPLE > 1
#                     or cfg.DATA.TRAIN_CROP_NUM_TEMPORAL > 1
#                     or cfg.DATA.TRAIN_CROP_NUM_SPATIAL > 1
#                 )
#                 and split in ["train"]
#                 and not cfg.MODEL.MODEL_NAME == "ContrastiveModel"
#             ):
#                 collate_func = partial(
#                     multiple_samples_collate, fold="imagenet" in dataset_name
#                 )
#             else:
#                 collate_func = None
#             loader = torch.utils.data.DataLoader(
#                 dataset,
#                 batch_size=batch_size,
#                 shuffle=(False if sampler else shuffle),
#                 sampler=sampler,
#                 num_workers=cfg.DATA_LOADER.NUM_WORKERS,
#                 pin_memory=cfg.DATA_LOADER.PIN_MEMORY,
#                 drop_last=drop_last,
#                 collate_fn=collate_func,
#                 worker_init_fn=utils.loader_worker_init_fn(dataset),
#             )
#
#     return loader, dataset


def shuffle_dataset(loader, cur_epoch):
    """ "
    Shuffles the data.
    Args:
        loader (loader): data loader to perform shuffle.
        cur_epoch (int): number of the current epoch.
    """
    if loader._dataset_kind == torch.utils.data.dataloader._DatasetKind.Iterable:
        if hasattr(loader.dataset, "sampler"):
            sampler = loader.dataset.sampler
        else:
            raise RuntimeError(
                "Unknown sampler for IterableDataset when shuffling dataset"
            )
    else:
        sampler = (
            loader.batch_sampler.sampler
            if isinstance(loader.batch_sampler, ShortCycleBatchSampler)
            else loader.sampler
        )
    assert isinstance(
        sampler, (RandomSampler, DistributedSampler)
    ), "Sampler type '{}' not supported".format(type(sampler))
    # RandomSampler handles shuffling automatically
    if isinstance(sampler, DistributedSampler):
        # DistributedSampler shuffles data based on epoch
        sampler.set_epoch(cur_epoch)

    if hasattr(loader.dataset, "prefetcher"):
        sampler = loader.dataset.prefetcher.sampler
        if isinstance(sampler, DistributedSampler):
            # DistributedSampler shuffles data based on epoch
            print("prefetcher sampler")
            sampler.set_epoch(cur_epoch)