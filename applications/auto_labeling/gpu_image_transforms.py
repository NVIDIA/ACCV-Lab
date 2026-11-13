# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Deterministic GPU resize, crop, BGR normalization and padding for StreamPETR."""

import math

import torch
import torch.nn.functional as F


class GPUImageTransforms:
    """Process a batch of owned HWC float32 GPU images; calibration is updated separately."""

    def __init__(self, img_norm_cfg, ida_aug_conf, device='cuda:0'):
        self.mean = torch.tensor(img_norm_cfg['mean'], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(img_norm_cfg['std'], dtype=torch.float32, device=device).view(1, 3, 1, 1)
        self.to_rgb = img_norm_cfg.get('to_rgb', True)
        self.final_dim = tuple(ida_aug_conf['final_dim'])
        height, width = ida_aug_conf['H'], ida_aug_conf['W']
        self.resize = max(self.final_dim[0] / height, self.final_dim[1] / width)
        self.newH, self.newW = int(height * self.resize), int(width * self.resize)
        bottom = sum(ida_aug_conf.get('bot_pct_lim', (0.0, 0.0))) / 2
        self.crop_h = int((1 - bottom) * self.newH) - self.final_dim[0]
        self.crop_w = max(0, self.newW - self.final_dim[1]) // 2
        self.pad_h, self.pad_w = self.get_pad_shape()

    def __call__(self, images):
        images = torch.stack(images).permute(0, 3, 1, 2).contiguous()
        images = F.interpolate(images, size=(self.newH, self.newW), mode='bilinear', align_corners=False)
        height, width = self.final_dim
        images = images[
            :, :, self.crop_h : self.crop_h + height, self.crop_w : self.crop_w + width
        ].contiguous()
        if self.to_rgb:
            images = images[:, [2, 1, 0]]
        images = (images - self.mean) / self.std
        if (self.pad_h, self.pad_w) != self.final_dim:
            images = F.pad(images, (0, self.pad_w - width, 0, self.pad_h - height))
        return images

    def get_pad_shape(self):
        return tuple(math.ceil(size / 32) * 32 for size in self.final_dim)
