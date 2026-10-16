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

from mmcv.utils.registry import Registry, build_from_cfg

SAMPLER = Registry('sampler')


def build_sampler(cfg, default_args):
    return build_from_cfg(cfg, SAMPLER, default_args)
