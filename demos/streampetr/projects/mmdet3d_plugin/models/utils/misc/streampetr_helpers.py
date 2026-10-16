# Extracted from `projects/mmdet3d_plugin/models/utils/misc.py` in the
# original StreamPETR repository.
#
# From StreamPETR (https://github.com/exiawsh/StreamPETR).
# No copyright holder is named upstream; this does not imply the absence of
# copyright.
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

import torch
import torch.nn as nn
from mmdet.core import bbox_xyxy_to_cxcywh
from mmdet.models.utils.transformer import inverse_sigmoid


def memory_refresh(memory, prev_exist):
    memory_shape = memory.shape
    view_shape = [1 for _ in range(len(memory_shape))]
    prev_exist = prev_exist.view(-1, *view_shape[1:]) 
    return memory * prev_exist
    
def topk_gather(feat, topk_indexes):
    if topk_indexes is not None:
        feat_shape = feat.shape
        topk_shape = topk_indexes.shape
        
        view_shape = [1 for _ in range(len(feat_shape))] 
        view_shape[:2] = topk_shape[:2]
        topk_indexes = topk_indexes.view(*view_shape)
        
        feat = torch.gather(feat, 1, topk_indexes.repeat(1, 1, *feat_shape[2:]))
    return feat


def apply_ltrb(locations, pred_ltrb): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4)
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])# x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])# y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])# x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])# y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes  = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes  = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)


        return pred_boxes    

def apply_center_offset(locations, center_offset): 
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4) 
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

@torch.no_grad()
def locations(features, stride, pad_h, pad_w):
        """
        Arguments:
            features:  (N, C, H, W)
        Return:
            locations:  (H, W, 2)
        """

        h, w = features.size()[-2:]
        device = features.device
        
        shifts_x = (torch.arange(
            0, stride*w, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2 ) / pad_w
        shifts_y = (torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        ) + stride // 2) / pad_h
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1)
        
        locations = locations.reshape(h, w, 2)
        
        return locations


class MLN(nn.Module):
    ''' 
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out


def transform_reference_points(reference_points, egopose, reverse=False, translation=True):
    reference_points = torch.cat([reference_points, torch.ones_like(reference_points[..., 0:1])], dim=-1)
    if reverse:
        matrix = egopose.inverse()
    else:
        matrix = egopose
    if not translation:
        matrix[..., :3, 3] = 0.0
    reference_points = (matrix.unsqueeze(1) @ reference_points.unsqueeze(-1)).squeeze(-1)[..., :3]
    return reference_points
