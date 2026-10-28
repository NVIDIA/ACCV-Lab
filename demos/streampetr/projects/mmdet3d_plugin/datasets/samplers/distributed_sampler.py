# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# (BEVFormer: https://github.com/fundamentalvision/BEVFormer)
# ---------------------------------------------
#  Modified by Shihao Wang
# (StreamPETR: https://github.com/exiawsh/StreamPETR)
# ---------------------------------------------
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

import math
import torch
from torch.utils.data import DistributedSampler as _DistributedSampler
from .sampler import SAMPLER


@SAMPLER.register_module()
class DistributedSampler(_DistributedSampler):

    def __init__(self,
                 dataset=None,
                 num_replicas=None,
                 rank=None,
                 shuffle=True,
                 seed=0):
        super().__init__(
            dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)
        # for the compatibility from PyTorch 1.3+
        self.seed = seed if seed is not None else 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            assert False
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        # in case that indices is shorter than half of total_size
        indices = (indices *
                   math.ceil(self.total_size / len(indices)))[:self.total_size]
        assert len(indices) == self.total_size

        # subsample
        per_replicas = self.total_size//self.num_replicas
        # indices = indices[self.rank:self.total_size:self.num_replicas]
        indices = indices[self.rank*per_replicas:(self.rank+1)*per_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)
