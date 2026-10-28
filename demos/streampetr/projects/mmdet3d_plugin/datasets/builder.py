# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shihao Wang
# (StreamPETR: https://github.com/exiawsh/StreamPETR)
# ---------------------------------------------
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Added option to disable group-orderrandomizations from config by forwarding of `randomize` 
#          parameter from the config when creating the `InfiniteGroupEachSampleInBatchSampler` instance.
#          Select the sequence batch sampler by configuration for both training and load-only runners.
#          Allow per-GPU dataset replication for performance evaluations.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import platform
import random
from functools import partial

import numpy as np
from mmcv.parallel import collate
from mmcv.runner import get_dist_info
from mmcv.utils import Registry, build_from_cfg
from torch.utils.data import DataLoader

from mmdet.datasets.samplers import GroupSampler
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import DistributedGroupSampler
from projects.mmdet3d_plugin.datasets.samplers.distributed_sampler import DistributedSampler
from projects.mmdet3d_plugin.datasets.samplers.group_sampler import InfiniteGroupEachSampleInBatchSampler
from projects.mmdet3d_plugin.datasets.samplers.sampler import build_sampler

def build_dataloader(dataset,
                     samples_per_gpu,
                     workers_per_gpu,
                     num_gpus=1,
                     dist=True,
                     shuffle=True,
                     seed=None,
                     shuffler_sampler=None,
                     nonshuffler_sampler=None,
                     runner_type=dict(type='EpochBasedRunner'),
                     replicate_dataset_per_gpu=False,
                     **kwargs):
    """Build PyTorch DataLoader.
    In distributed training, each GPU/process has a dataloader.
    In non-distributed training, there is only one dataloader for all GPUs.
    Args:
        dataset (Dataset): A PyTorch dataset.
        samples_per_gpu (int): Number of training samples on each GPU, i.e.,
            batch size of each GPU.
        workers_per_gpu (int): How many subprocesses to use for data loading
            for each GPU.
        num_gpus (int): Number of GPUs. Only used in non-distributed training.
        dist (bool): Distributed training/test or not. Default: True.
        shuffle (bool): Whether to shuffle the data at every epoch.
            Default: True.
        replicate_dataset_per_gpu (bool): Whether each training sequence sampler
            uses the full dataset. Used for performance evaluations with NuScenes
            mini, which has too few sequence groups for the multi-GPU batch size.
            Worker seeding and validation still use the real distributed rank.
        kwargs: any keyword argument to be used to initialize DataLoader
    Returns:
        DataLoader: A PyTorch dataloader.
    """
    rank, world_size = get_dist_info()
    # NuScenes mini has too few sequence groups for 8 GPUs x 8 samples. Replication
    # uses an unsharded sampler on each GPU while keeping the real rank for worker
    # seeding. Both sampler constructions below must use these sampling values.
    replicate_training_data = (
        replicate_dataset_per_gpu and shuffle and shuffler_sampler is not None
        and shuffler_sampler.get('type') == 'InfiniteGroupEachSampleInBatchSampler'
    )
    sampling_rank = 0 if replicate_training_data else rank
    sampling_world_size = 1 if replicate_training_data else world_size

    if dist:
        # DistributedGroupSampler will definitely shuffle the data to satisfy
        # that images on each GPU are in the same group
        if shuffle:
            sampler = build_sampler(shuffler_sampler if shuffler_sampler is not None else dict(type='DistributedGroupSampler'),
                                     dict(
                                         dataset=dataset,
                                         samples_per_gpu=samples_per_gpu,
                                         num_replicas=sampling_world_size,
                                         rank=sampling_rank,
                                         seed=seed)
                                     )

        else:
            sampler = build_sampler(nonshuffler_sampler if nonshuffler_sampler is not None else dict(type='DistributedSampler'),
                                     dict(
                                         dataset=dataset,
                                         num_replicas=world_size,
                                         rank=rank,
                                         shuffle=shuffle,
                                         seed=seed)
                                     )

        batch_size = samples_per_gpu
        num_workers = workers_per_gpu
        batch_sampler = None


    else:
        # assert False, 'not support in bevformer'
        print('WARNING!!!!, Only can be used for obtain inference speed!!!!')
        sampler = GroupSampler(dataset, samples_per_gpu) if shuffle else None
        batch_size = num_gpus * samples_per_gpu
        num_workers = num_gpus * workers_per_gpu
        batch_sampler = None


    # This sampler yields batches of indices, not individual indices. Select it
    # from the sampler configuration rather than the runner's class name so that
    # custom runners (e.g. load-only runs without out training for loading time evaluation) 
    # use the same sequence sampling. Validation (shuffle=False) must keep its finite, 
    # non-shuffling sampler.
    if (shuffle and shuffler_sampler is not None
            and shuffler_sampler.get('type') == 'InfiniteGroupEachSampleInBatchSampler'):
        # TODO: original has more options, but I'm not using them 
        # https://github.com/open-mmlab/mmdetection/blob/3b72b12fe9b14de906d1363982b9fba05e7d47c1/mmdet/datasets/builder.py#L145-L157
        randomize = shuffler_sampler["randomize"] if "randomize" in shuffler_sampler else True
        batch_sampler = InfiniteGroupEachSampleInBatchSampler(
            dataset,
            samples_per_gpu,
            sampling_world_size,
            sampling_rank,
            seed=seed,
            randomize=randomize)
        batch_size = 1
        sampler = None


    init_fn = partial(
        worker_init_fn, num_workers=num_workers, rank=rank,
        seed=seed) if seed is not None else None

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=partial(collate, samples_per_gpu=samples_per_gpu),
        pin_memory=False,
        worker_init_fn=init_fn,
        **kwargs)

    return data_loader


def worker_init_fn(worker_id, num_workers, rank, seed):
    # The seed of each worker equals to
    # num_worker * rank + worker_id + user_seed
    worker_seed = num_workers * rank + worker_id + seed
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Copyright (c) OpenMMLab. All rights reserved.
import platform
from mmcv.utils import Registry, build_from_cfg

from mmdet.datasets import DATASETS
from mmdet.datasets.builder import _concat_dataset

if platform.system() != 'Windows':
    # https://github.com/pytorch/pytorch/issues/973
    import resource
    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    base_soft_limit = rlimit[0]
    hard_limit = rlimit[1]
    soft_limit = min(max(4096, base_soft_limit), hard_limit)
    resource.setrlimit(resource.RLIMIT_NOFILE, (soft_limit, hard_limit))

OBJECTSAMPLERS = Registry('Object sampler')


def custom_build_dataset(cfg, default_args=None):
    from mmdet3d.datasets.dataset_wrappers import CBGSDataset
    from mmdet.datasets.dataset_wrappers import (ClassBalancedDataset,
                                                 ConcatDataset, RepeatDataset)
    if isinstance(cfg, (list, tuple)):
        dataset = ConcatDataset([custom_build_dataset(c, default_args) for c in cfg])
    elif cfg['type'] == 'ConcatDataset':
        dataset = ConcatDataset(
            [custom_build_dataset(c, default_args) for c in cfg['datasets']],
            cfg.get('separate_eval', True))
    elif cfg['type'] == 'RepeatDataset':
        dataset = RepeatDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['times'])
    elif cfg['type'] == 'ClassBalancedDataset':
        dataset = ClassBalancedDataset(
            custom_build_dataset(cfg['dataset'], default_args), cfg['oversample_thr'])
    elif cfg['type'] == 'CBGSDataset':
        dataset = CBGSDataset(custom_build_dataset(cfg['dataset'], default_args))
    elif isinstance(cfg.get('ann_file'), (list, tuple)):
        dataset = _concat_dataset(cfg, default_args)
    else:
        dataset = build_from_cfg(cfg, DATASETS, default_args)

    return dataset
