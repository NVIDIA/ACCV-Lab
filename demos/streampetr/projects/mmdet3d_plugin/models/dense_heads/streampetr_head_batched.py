# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
#  (StreamPETR: https://github.com/exiawsh/StreamPETR)
# ------------------------------------------------------------------------
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: This is based on `streampetr_head.py`. Changes include: 
#           - Added batched loss computation; adjusted the implementation and the interface accordingly.
#           - Added NVTX range-based profiling and runtime measurements.
#           - Added comments & notes describing the approach & implementation.
#
# ------------------------------------------------------------------------
#
# This file contains contributions made under each of the licenses below;
# both apply to it. Each license governs the contributions made under it.
#
# ........................................................................
#
# MIT License
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# ........................................................................
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

# Note that this docstring is included into the demo documentation from here
'''
The more fully batched loss computation is based on the same approach as used in ``focal_head_batched.py``.
Please, see the notes in ``focal_head_batched.py`` first for a detailed explanation of the approach.

The original StreamPETR head already batches the final classification and bounding-box loss-function calls
over samples within each decoder layer: it concatenates the per-sample targets and flattens the sample and
query dimensions before invoking the loss functions. The changes here instead batch the surrounding target
preparation and cost-matrix computation over samples and decoder layers, preserve the structured batch
dimensions until the loss-function boundary, and compute bounding-box loss only for matched
prediction/target pairs. The SciPy Hungarian solver itself still runs separately for each layer/sample
entry. The changes also add batching of the loss-function calls across decoder layers, including the DN
losses, which is not present in the original implementation.

Here, we extend the approach to the case of multiple decoder layers, where the loss is computed 
for each decoder layer (i.e. batched, but not reduced over the layers). The explanation here is focused 
on this extension. The extension is done as follows:

  1) The GT data is duplicated for each decoder layer to match the shape of the predictions.
     This is similar to the original implementation. However, lists are used there. Here,
     we use :class:`~accvlab.batching_helpers.RaggedBatch` instances. After converting the GT data 
     to :class:`~accvlab.batching_helpers.RaggedBatch` objects, an additional batch dimension is added, and 
     the GT data is duplicated for each decoder layer, so that the additional batch dimension iterates over 
     the decoder layers.
  2) The method ``_compute_per_layer_loss()`` is used as a wrapper around the actual loss 
     functions and allows for reduction of the loss over the samples while retaining the 
     per-decoder-layer separation. Note that this is done to work around the fact that the loss
     functions can only reduce over all objects in the batch at once, but cannot return per-sample 
     losses, i.e. they cannot reduce over the objects only, keeping the batch dimension(s) intact.

Also note that the loss computation includes the DN loss computation (``dn_loss_batched()``),
which is not present for the focal head. The inputs for the DN loss computation are obtained from
``prepare_for_loss()``, which returns the data with the first dimension being the decoder layers,
so that the inputs can be directly used for batched loss computation. 

The implementation of the loss computation is done in the ``loss()`` function. Please see the 
notes & doc-strings there and in the related functions called from there for more details.
'''

from typing import Union, Callable, List, Dict, Optional, Tuple

import torch
import torch.nn as nn
from mmcv.cnn import Linear, bias_init_with_prob

from mmcv.runner import force_fp32
from mmdet.core import build_assigner, build_sampler, reduce_mean
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmdet3d.core.bbox.structures import LiDARInstance3DBoxes
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

from mmdet.models.utils import NormedLinear
from projects.mmdet3d_plugin.models.utils.positional_encoding import pos2posemb3d, pos2posemb1d, nerf_positional_encoding
from projects.mmdet3d_plugin.models.utils.misc import MLN, topk_gather, transform_reference_points, memory_refresh, SELayer_Linear

import accvlab.batching_helpers as bhl

from accvlab.optim_test_tools import NVTXRangeWrapper, Stopwatch
nvtx_wrp = NVTXRangeWrapper()
stopwatch = Stopwatch()

@HEADS.register_module()
class StreamPETRHeadBatched(AnchorFreeHead):
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    _version = 2

    def __init__(self,
                 num_classes,
                 in_channels=256,
                 stride=16,
                 embed_dims=256,
                 num_query=100,
                 num_reg_fcs=2,
                 memory_len=1024,
                 topk_proposals=256,
                 num_propagated=256,
                 with_dn=True,
                 with_ego_pos=True,
                 match_with_velo=True,
                 match_costs=None,
                 transformer=None,
                 sync_cls_avg_factor=False,
                 code_weights=None,
                 bbox_coder=None,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                    assigner=dict(
                        type='HungarianAssigner3DBatched',
                        cls_cost=dict(type='FocalLossCostBatched', weight=2.0),
                        reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                        iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                        pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])),
                 test_cfg=dict(max_per_img=100),
                 depth_step=0.8,
                 depth_num=64,
                 LID=False,
                 depth_start = 1,
                 position_range=[-65, -65, -8.0, 65, 65, 8.0],
                 scalar = 5,
                 noise_scale = 0.4,
                 noise_trans = 0.0,
                 dn_weight = 1.0,
                 split = 0.5,
                 init_cfg=None,
                 normedlinear=False,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.code_weights = self.code_weights[:self.code_size]

        if match_costs is not None:
            self.match_costs = match_costs
        else:
            self.match_costs = self.code_weights

        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor
        class_weight = loss_cls.get('class_weight', None)
        if class_weight is not None and (self.__class__ is StreamPETRHeadBatched):
            assert isinstance(class_weight, float), 'Expected ' \
                'class_weight to have type float. Found ' \
                f'{type(class_weight)}.'
            # NOTE following the official DETR rep0, bg_cls_weight means
            # relative classification weight of the no-object class.
            bg_cls_weight = loss_cls.get('bg_cls_weight', class_weight)
            assert isinstance(bg_cls_weight, float), 'Expected ' \
                'bg_cls_weight to have type float. Found ' \
                f'{type(bg_cls_weight)}.'
            class_weight = torch.ones(num_classes + 1) * class_weight
            # set background class as the last indice
            class_weight[num_classes] = bg_cls_weight
            loss_cls.update({'class_weight': class_weight})
            if 'bg_cls_weight' in loss_cls:
                loss_cls.pop('bg_cls_weight')
            self.bg_cls_weight = bg_cls_weight

        if train_cfg:
            assert 'assigner' in train_cfg, 'assigner should be provided '\
                'when train_cfg is set.'
            assigner = train_cfg['assigner']


            self.assigner = build_assigner(assigner)
            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.memory_len = memory_len
        self.topk_proposals = topk_proposals
        self.num_propagated = num_propagated
        self.with_dn = with_dn
        self.with_ego_pos = with_ego_pos
        self.match_with_velo = match_with_velo
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.embed_dims = embed_dims
        self.depth_step = depth_step
        self.depth_num = depth_num
        self.position_dim = depth_num * 3
        self.LID = LID
        self.depth_start = depth_start
        self.stride=stride

        self.scalar = scalar
        self.bbox_noise_scale = noise_scale
        self.bbox_noise_trans = noise_trans
        self.dn_weight = dn_weight
        self.split = split 

        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.num_pred = 6
        self.normedlinear = normedlinear
        super(StreamPETRHeadBatched, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)

        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1

        self.transformer = build_transformer(transformer)

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights), requires_grad=False)

        self.match_costs = nn.Parameter(torch.tensor(
            self.match_costs), requires_grad=False)

        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.pc_range = nn.Parameter(torch.tensor(
            self.bbox_coder.pc_range), requires_grad=False)

        self.position_range = nn.Parameter(torch.tensor(
            position_range), requires_grad=False)
        
        if self.LID:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            index_1 = index + 1
            bin_size = (self.position_range[3] - self.depth_start) / (self.depth_num * (1 + self.depth_num))
            coords_d = self.depth_start + bin_size * index * index_1
        else:
            index  = torch.arange(start=0, end=self.depth_num, step=1).float()
            bin_size = (self.position_range[3] - self.depth_start) / self.depth_num
            coords_d = self.depth_start + bin_size * index

        self.coords_d = nn.Parameter(coords_d, requires_grad=False)

        self._init_layers()
        self.reset_memory()

    def _init_layers(self):
        """Initialize layers of the transformer head."""

        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        if self.normedlinear:
            cls_branch.append(NormedLinear(self.embed_dims, self.cls_out_channels))
        else:
            cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.position_encoder = nn.Sequential(
                nn.Linear(self.position_dim, self.embed_dims*4),
                nn.ReLU(),
                nn.Linear(self.embed_dims*4, self.embed_dims),
            )

        self.memory_embed = nn.Sequential(
                nn.Linear(self.in_channels, self.embed_dims),
                nn.ReLU(),
                nn.Linear(self.embed_dims, self.embed_dims),
            )
        
        # can be replaced with MLN
        self.featurized_pe = SELayer_Linear(self.embed_dims)

        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)


        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims*3//2, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(8)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

        # encoding ego pose
        if self.with_ego_pos:
            self.ego_pose_pe = MLN(180)
            self.ego_pose_memory = MLN(180)

    def init_weights(self):
        """Initialize weights of the transformer head."""
        # The initialization for transformer is important
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            for m in self.cls_branches:
                nn.init.constant_(m[-1].bias, bias_init)


    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None
        self.memory_timestamp = None
        self.memory_egopose = None
        self.memory_velo = None

    def pre_update_memory(self, data):
        x = data['prev_exists']
        B = x.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_timestamp = x.new_zeros(B, self.memory_len, 1)
            self.memory_egopose = x.new_zeros(B, self.memory_len, 4, 4)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
            self.memory_timestamp += data['timestamp'].unsqueeze(-1).unsqueeze(-1)
            self.memory_egopose = data['ego_pose_inv'].unsqueeze(1) @ self.memory_egopose
            self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose_inv'], reverse=False)
            self.memory_timestamp = memory_refresh(self.memory_timestamp[:, :self.memory_len], x)
            self.memory_reference_point = memory_refresh(self.memory_reference_point[:, :self.memory_len], x)
            self.memory_embedding = memory_refresh(self.memory_embedding[:, :self.memory_len], x)
            self.memory_egopose = memory_refresh(self.memory_egopose[:, :self.memory_len], x)
            self.memory_velo = memory_refresh(self.memory_velo[:, :self.memory_len], x)
        
        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = self.pseudo_reference_points.weight * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
            self.memory_reference_point[:, :self.num_propagated]  = self.memory_reference_point[:, :self.num_propagated] + (1 - x).view(B, 1, 1) * pseudo_reference_points
            self.memory_egopose[:, :self.num_propagated]  = self.memory_egopose[:, :self.num_propagated] + (1 - x).view(B, 1, 1, 1) * torch.eye(4, device=x.device)

    def post_update_memory(self, data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict):
        if self.training and mask_dict and mask_dict['pad_size'] > 0:
            rec_reference_points = all_bbox_preds[:, :, mask_dict['pad_size']:, :3][-1]
            rec_velo = all_bbox_preds[:, :, mask_dict['pad_size']:, -2:][-1]
            rec_memory = outs_dec[:, :, mask_dict['pad_size']:, :][-1]
            rec_score = all_cls_scores[:, :, mask_dict['pad_size']:, :][-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        else:
            rec_reference_points = all_bbox_preds[..., :3][-1]
            rec_velo = all_bbox_preds[..., -2:][-1]
            rec_memory = outs_dec[-1]
            rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]
            rec_timestamp = torch.zeros_like(rec_score, dtype=torch.float64)
        
        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.topk_proposals, dim=1)
        rec_timestamp = topk_gather(rec_timestamp, topk_indexes)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_ego_pose = topk_gather(rec_ego_pose, topk_indexes)
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_timestamp = torch.cat([rec_timestamp, self.memory_timestamp], dim=1)
        self.memory_egopose= torch.cat([rec_ego_pose, self.memory_egopose], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
        self.memory_reference_point = transform_reference_points(self.memory_reference_point, data['ego_pose'], reverse=False)
        self.memory_timestamp -= data['timestamp'].unsqueeze(-1).unsqueeze(-1)
        self.memory_egopose = data['ego_pose'].unsqueeze(1) @ self.memory_egopose

    def position_embeding(self, data, memory_centers, topk_indexes, img_metas):
        eps = 1e-5
        BN, H, W, _ = memory_centers.shape
        B = data['intrinsics'].size(0)

        intrinsic = torch.stack([data['intrinsics'][..., 0, 0], data['intrinsics'][..., 1, 1]], dim=-1)
        intrinsic = torch.abs(intrinsic) / 1e3
        intrinsic = intrinsic.repeat(1, H*W, 1).view(B, -1, 2)
        LEN = intrinsic.size(1)

        num_sample_tokens = topk_indexes.size(1) if topk_indexes is not None else LEN

        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        memory_centers[..., 0] = memory_centers[..., 0] * pad_w
        memory_centers[..., 1] = memory_centers[..., 1] * pad_h

        D = self.coords_d.shape[0]

        memory_centers = memory_centers.detach().view(B, LEN, 1, 2)
        topk_centers = topk_gather(memory_centers, topk_indexes).repeat(1, 1, D, 1)
        coords_d = self.coords_d.view(1, 1, D, 1).repeat(B, num_sample_tokens, 1 , 1)
        coords = torch.cat([topk_centers, coords_d], dim=-1)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)
        coords[..., :2] = coords[..., :2] * torch.maximum(coords[..., 2:3], torch.ones_like(coords[..., 2:3])*eps)

        coords = coords.unsqueeze(-1)

        # @MARK: The commented out line below replaced as it does not work with the used config
        #img2lidars = data['lidar2img'].inverse()
        img2lidars = torch.inverse(data['lidar2img'].cpu()).cuda()

        img2lidars = img2lidars.view(BN, 1, 1, 4, 4).repeat(1, H*W, D, 1, 1).view(B, LEN, D, 4, 4)
        img2lidars = topk_gather(img2lidars, topk_indexes)

        coords3d = torch.matmul(img2lidars, coords).squeeze(-1)[..., :3]
        coords3d[..., 0:3] = (coords3d[..., 0:3] - self.position_range[0:3]) / (self.position_range[3:6] - self.position_range[0:3])
        coords3d = coords3d.reshape(B, -1, D*3)
      
        pos_embed  = inverse_sigmoid(coords3d)
        coords_position_embeding = self.position_encoder(pos_embed)
        intrinsic = topk_gather(intrinsic, topk_indexes)

        # for spatial alignment in focal petr
        cone = torch.cat([intrinsic, coords3d[..., -3:], coords3d[..., -90:-87]], dim=-1)

        return coords_position_embeding, cone

    def temporal_alignment(self, query_pos, tgt, reference_points):
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (self.pc_range[3:6] - self.pc_range[0:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point)) 
        temp_memory = self.memory_embedding
        rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.size(1), 1, 1)
        
        if self.with_ego_pos:
            rec_ego_motion = torch.cat([torch.zeros_like(reference_points[...,:3]), rec_ego_pose[..., :3, :].flatten(-2)], dim=-1)
            rec_ego_motion = nerf_positional_encoding(rec_ego_motion)
            tgt = self.ego_pose_memory(tgt, rec_ego_motion)
            query_pos = self.ego_pose_pe(query_pos, rec_ego_motion)
            memory_ego_motion = torch.cat([self.memory_velo, self.memory_timestamp, self.memory_egopose[..., :3, :].flatten(-2)], dim=-1).float()
            memory_ego_motion = nerf_positional_encoding(memory_ego_motion)
            temp_pos = self.ego_pose_pe(temp_pos, memory_ego_motion)
            temp_memory = self.ego_pose_memory(temp_memory, memory_ego_motion)

        query_pos += self.time_embedding(pos2posemb1d(torch.zeros_like(reference_points[...,:1])))
        temp_pos += self.time_embedding(pos2posemb1d(self.memory_timestamp).float())

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            rec_ego_pose = torch.eye(4, device=query_pos.device).unsqueeze(0).unsqueeze(0).repeat(B, query_pos.shape[1]+self.num_propagated, 1, 1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]
            
        return tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose

    def prepare_for_dn(self, batch_size, reference_points, img_metas):
        if self.training and self.with_dn:
            targets = [torch.cat((img_meta['gt_bboxes_3d']._data.gravity_center, img_meta['gt_bboxes_3d']._data.tensor[:, 3:]),dim=1) for img_meta in img_metas ]
            labels = [img_meta['gt_labels_3d']._data for img_meta in img_metas ]
            known = [(torch.ones_like(t)).cuda() for t in labels]
            know_idx = known
            unmask_bbox = unmask_label = torch.cat(known)
            #gt_num
            known_num = [t.size(0) for t in targets]
        
            labels = torch.cat([t for t in labels])
            boxes = torch.cat([t for t in targets])
            batch_idx = torch.cat([torch.full((t.size(0), ), i) for i, t in enumerate(targets)])
        
            known_indice = torch.nonzero(unmask_label + unmask_bbox)
            known_indice = known_indice.view(-1)
            # add noise
            known_indice = known_indice.repeat(self.scalar, 1).view(-1)
            known_labels = labels.repeat(self.scalar, 1).view(-1).long().to(reference_points.device)
            known_bid = batch_idx.repeat(self.scalar, 1).view(-1)
            known_bboxs = boxes.repeat(self.scalar, 1).to(reference_points.device)
            known_bbox_center = known_bboxs[:, :3].clone()
            known_bbox_scale = known_bboxs[:, 3:6].clone()

            if self.bbox_noise_scale > 0:
                diff = known_bbox_scale / 2 + self.bbox_noise_trans
                rand_prob = torch.rand_like(known_bbox_center) * 2 - 1.0
                known_bbox_center += torch.mul(rand_prob,
                                            diff) * self.bbox_noise_scale
                known_bbox_center[..., 0:3] = (known_bbox_center[..., 0:3] - self.pc_range[0:3]) / (self.pc_range[3:6] - self.pc_range[0:3])

                known_bbox_center = known_bbox_center.clamp(min=0.0, max=1.0)
                mask = torch.norm(rand_prob, 2, 1) > self.split
                known_labels[mask] = self.num_classes
            
            single_pad = int(max(known_num))
            pad_size = int(single_pad * self.scalar)
            padding_bbox = torch.zeros(pad_size, 3).to(reference_points.device)
            padded_reference_points = torch.cat([padding_bbox, reference_points], dim=0).unsqueeze(0).repeat(batch_size, 1, 1)

            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(self.scalar)]).long()
            if len(known_bid):
                padded_reference_points[(known_bid.long(), map_known_indice)] = known_bbox_center.to(reference_points.device)

            tgt_size = pad_size + self.num_query
            attn_mask = torch.ones(tgt_size, tgt_size).to(reference_points.device) < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(self.scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == self.scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
             
            # update dn mask for temporal modeling
            query_size = pad_size + self.num_query + self.num_propagated
            tgt_size = pad_size + self.num_query + self.memory_len
            temporal_attn_mask = torch.ones(query_size, tgt_size).to(reference_points.device) < 0
            temporal_attn_mask[:attn_mask.size(0), :attn_mask.size(1)] = attn_mask 
            temporal_attn_mask[pad_size:, :pad_size] = True
            attn_mask = temporal_attn_mask

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_bboxs),
                'know_idx': know_idx,
                'pad_size': pad_size
            }
            
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        """load checkpoints."""
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since `AnchorFreeHead._load_from_state_dict` should not be
        # called here. Invoking the default `Module._load_from_state_dict`
        # is enough.

        # Names of some parameters in has been changed.
        version = local_metadata.get('version', None)
        if (version is None or version < 2) and self.__class__ is StreamPETRHeadBatched:
            convert_dict = {
                '.self_attn.': '.attentions.0.',
                # '.ffn.': '.ffns.0.',
                '.multihead_attn.': '.attentions.1.',
                '.decoder.norm.': '.decoder.post_norm.'
            }
            state_dict_keys = list(state_dict.keys())
            for k in state_dict_keys:
                for ori_key, convert_key in convert_dict.items():
                    if ori_key in k:
                        convert_key = k.replace(ori_key, convert_key)
                        state_dict[convert_key] = state_dict[k]
                        del state_dict[k]

        super(AnchorFreeHead,
              self)._load_from_state_dict(state_dict, prefix, local_metadata,
                                          strict, missing_keys,
                                          unexpected_keys, error_msgs)
    

    def forward(self, memory_center, img_metas, topk_indexes=None,  **data):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        # zero init the memory bank
        self.pre_update_memory(data)

        x = data['img_feats']
        B, N, C, H, W = x.shape
        num_tokens = N * H * W
        memory = x.permute(0, 1, 3, 4, 2).reshape(B, num_tokens, C)
        memory = topk_gather(memory, topk_indexes)

        pos_embed, cone = self.position_embeding(data, memory_center, topk_indexes, img_metas)

        memory = self.memory_embed(memory)

        # spatial_alignment in focal petr
        memory = self.spatial_alignment(memory, cone)
        pos_embed = self.featurized_pe(pos_embed, memory)

        reference_points = self.reference_points.weight
        reference_points, attn_mask, mask_dict = self.prepare_for_dn(B, reference_points, img_metas)
        query_pos = self.query_embedding(pos2posemb3d(reference_points))
        tgt = torch.zeros_like(query_pos)

        # prepare for the tgt and query_pos using mln.
        tgt, query_pos, reference_points, temp_memory, temp_pos, rec_ego_pose = self.temporal_alignment(query_pos, tgt, reference_points)

        # transformer here is a little different from PETR
        outs_dec, _ = self.transformer(memory, tgt, query_pos, pos_embed, attn_mask, temp_memory, temp_pos)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(reference_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
        
        # update the memory bank
        self.post_update_memory(data, rec_ego_pose, all_cls_scores, all_bbox_preds, outs_dec, mask_dict)

        if mask_dict and mask_dict['pad_size'] > 0:
            output_known_class = all_cls_scores[:, :, :mask_dict['pad_size'], :]
            output_known_coord = all_bbox_preds[:, :, :mask_dict['pad_size'], :]
            outputs_class = all_cls_scores[:, :, mask_dict['pad_size']:, :]
            outputs_coord = all_bbox_preds[:, :, mask_dict['pad_size']:, :]
            mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
            outs = {
                'all_cls_scores': outputs_class,
                'all_bbox_preds': outputs_coord,
                'dn_mask_dict':mask_dict,

            }
        else:
            outs = {
                'all_cls_scores': all_cls_scores,
                'all_bbox_preds': all_bbox_preds,
                'dn_mask_dict':None,
            }

        return outs
    
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice'].long()
        known_indice = mask_dict['known_indice'].long().cpu()
        batch_idx = mask_dict['batch_idx'].long()
        bid = batch_idx[known_indice]
        if len(output_known_class) > 0:
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        num_tgt = known_indice.numel()
        return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt
    

    def loss_batched(self,
                     cls_scores: torch.Tensor,
                     bbox_preds: torch.Tensor,
                     gt_bboxes: bhl.RaggedBatch,
                     gt_labels: bhl.RaggedBatch,
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        """"Loss function for outputs from all decoder layers

        The input has two batch dimensions: decoder layers followed by samples. The loss-function boundary
        temporarily flattens these dimensions together with the object dimension, after which
        ``_compute_per_layer_loss()`` restores the decoder-layer dimension for per-layer reduction.

        Args:
            cls_scores (Tensor): Classification score logits with shape
                [num_decoder_layers, num_samples, num_query, cls_out_channels].
            bbox_preds (Tensor): Bbox regression outputs with shape
                [num_decoder_layers, num_samples, num_query, 10].
            gt_bboxes (RaggedBatch): Ground truth bboxes with batch dimensions
                [num_decoder_layers, num_samples] and element shape [num_gts, 9].
            gt_labels (RaggedBatch): Ground truth class indexes with batch dimensions
                [num_decoder_layers, num_samples] and element shape [num_gts].
        Returns:
            loss_cls (Tensor): Classification loss (per-decoder-layer).
            loss_bbox (Tensor): Bounding box regression loss (per-decoder-layer).
        """

        #NOTE: Also see notes in `loss()` for overview of the batched loss implementation approach.

        # ===== Pre-processing (`loss_batched()`) =====
        nvtx_wrp.range_push("StreamPETRHeadBatched.loss_batched")
        num_dec_layers = cls_scores.size(0)
        num_samples_in_batch = cls_scores.size(1)
        num_preds = cls_scores.size(2)
        
        # ===== Perform the matching =====
        #@NOTE
        # The following is the same as the case of `FocalHeadBatched.loss_batched()`, i.e. we
        # call the assign() method directly and obtain the matched indices.
        # 
        # This is the same as in the focal head case. However, the assigner itself if a different class
        # (i.e. `HungarianAssigner3DBatched` instead of `HungarianAssigner2DBatched`). Also, the data
        # has multiple batch dimensions (which are handled automatically by the assigner implementation).
        matched_pred_indices, matched_gt_indices = self.assigner.assign(bbox_preds, cls_scores, gt_bboxes,
                                                                        gt_labels, None, self.match_costs, 
                                                                        self.match_with_velo)
        
        # ===== Per-layer counts and avg factors =====
        #@NOTE: `per_sample_nums_pos` will have the same shape as the batch dimensions.
        per_sample_nums_pos = matched_pred_indices.sample_sizes
        #@NOTE
        # Here, we sum over the samples (the second batch dimension, i.e. `dim==1`) to obtain the total 
        # number of positive samples for each decoder layer. Note that the first batch dimension, i.e. 
        # `dim==0`, is over the decoder layers, and we keep that dimension.
        #   
        # For comparison: In the original implementation, the number of positive samples is 
        # computed for each decoder layer separately (in the `get_targets()` function).
        per_layer_nums_pos = per_sample_nums_pos.sum(dim=1)
        per_layer_nums_neg = num_preds * num_samples_in_batch - per_layer_nums_pos

        #@NOTE
        # The following is the batched version of the per-layer normalization factor computation (for the 
        # classification loss). This is equivalent to the (per-sample) computation in the original 
        # implementation.
        per_layer_cls_avg_factors = per_layer_nums_pos * 1.0 + \
            per_layer_nums_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            per_layer_cls_avg_factors = reduce_mean(per_layer_cls_avg_factors)
        #@NOTE 
        # Make sure we don't have any division by zero (similar to original implementation).
        # This is equivalent to the original implementation, but done here for tensor elements
        # instead of for a scalar value.
        per_layer_cls_avg_factors[per_layer_cls_avg_factors < 1.0] = 1.0

        # ===== Align labels and align matched boxes =====
        nvtx_wrp.range_push("Indexing [StreamPETRHeadBatched.loss_batched]")
        #@NOTE:
        # Align/map back the GT labels to the predicted labels as is done for losses which 
        # are computed for all predictions, not just the matched ones. This is analogous to the focal head 
        # classification loss. Please have a look at the notes in `focal_head_batched.py` for a more detailed 
        # description. The main difference is that here, we have multiple batch dimensions. However, this is 
        # handled automatically by `bhl.batched_index_mapping()`.
        gt_labels_mapped_to_pred = torch.full(
            (num_dec_layers, num_samples_in_batch, num_preds), 
            fill_value=self.cls_out_channels, 
            dtype=gt_labels.dtype, device=gt_labels.device
        )
        gt_labels_mapped_to_pred = bhl.batched_index_mapping(
            gt_labels, matched_gt_indices, matched_pred_indices, gt_labels_mapped_to_pred
        )

        #@NOTE:
        # Get the aligned (i.e. indexed using indices obtained from the matching) versions of the 
        # bounding boxes. Again, this is analogous to the focal head case (except for the additional batch 
        # dimensions). Please have a look at the notes in `focal_head_batched.py` for a more detailed 
        # description.
        gt_bboxes_matched = bhl.batched_indexing_access(gt_bboxes, matched_gt_indices)
        pred_bboxes_matched = bhl.batched_indexing_access(bbox_preds, matched_pred_indices)
        nvtx_wrp.range_pop()

        label_weights = torch.ones(
            (num_dec_layers, num_samples_in_batch, num_preds), 
            dtype=torch.float32, device = gt_labels.device
        )

        # ===== Losses: Classification (per-layer wrapper) =====
        nvtx_wrp.range_push("loss_cls computation[StreamPETRHeadBatched.loss_batched]")
        #@NOTE
        # The method `_compute_per_layer_loss()` is a wrapper to the actual loss computation and enables
        # batching over both samples and decoder layers, with the possibility to compute the loss for each
        # layer separately, while combining (i.e. summing or computing the average) the loss over the samples.
        # This is necessary as the loss function does not differentiate between the entries in a batch. 
        # Using the loss function directly, it would be not possible to compute the per-layer loss separately.
        # Please refer to the documentation of `_compute_per_layer_loss()` and the comments
        # in that function for more details.
        # 
        # Furthermore, note that the classification loss is computed for all predictions, not just the matched 
        # ones (also see the discussion of the classification loss in `focal_head_batched.py`).
        loss_cls = self._compute_per_layer_loss(
            self.loss_cls, gt_labels_mapped_to_pred, cls_scores, label_weights, per_layer_cls_avg_factors, 
            num_batch_dims=2
        )

        nvtx_wrp.range_pop()
        loss_cls = torch.nan_to_num(loss_cls)

        # ===== Synchronize positives =====
        #@NOTE: reduce_mean is used to synchronize over all devices, as in the original implementation.
        per_layer_nums_pos = per_layer_nums_pos.type(loss_cls.dtype)
        per_layer_nums_pos = reduce_mean(per_layer_nums_pos)
        per_layer_nums_pos[per_layer_nums_pos < 1.0] = 1.0

        # ===== Prepare bbox targets & masks =====
        gt_bboxes_matched_normalized = gt_bboxes_matched.apply(
            lambda bboxs: normalize_bbox(bboxs, self.pc_range)
        )

        isnotnan = torch.isfinite(gt_bboxes_matched_normalized.tensor).all(dim=-1, keepdim=True)
        
        #@NOTE
        # Get the existence weights. In the original implementation, the weights are returned 
        # by the `get_targets()` function (also see notes in `FocalHeadBatched.loss_batched()`). Note that
        # the weights here do not exactly correspond to the weights in the original implementation, as the 
        # existence weights here are computed for the data tensor (i.e. `tensor`) of the `RaggedBatch` 
        # instance (containing filler values to ensure uniform shape), while in the original implementation, 
        # the weights are computed for a tensor with the same shape as the predictions.
        bbox_existence_weights_matched = gt_bboxes_matched_normalized.get_existence_weights()

        #@NOTE 
        # The code weights are per-element weights for the individual elements which describe a single
        # bounding box. Therefore, they can be directly multiplied with the existence weights without any
        # adaptations regarding the batch shape, as long as the size of the inner-most dimension of the 
        # existence weights corresponds to the bounding box descriptors.
        bbox_weights_matched = bbox_existence_weights_matched * self.code_weights

        #@NOTE
        # The handling of non-finite values differs from the original implementation. In the original code,
        # boolean indexing with `isnotnan` as a mask is used to filter out non-finite values. However, 
        # this would lead to flattening of the batch dimensions and the dimension iterating over individual 
        # objects inside a sample. Here, we use `apply_mask_to_tensor()` to zero out non-finite values while 
        # preserving the current shape of the data.
        # This approach:
        #   1. Maintains the batch dimensions for efficient batched processing
        #   2. Properly handles `NaN` values (unlike simple multiplication, where `NaN * 0 = NaN`)
        # 
        # Note that our approach changes elements containing `NaN` values to filler (0) values, while the 
        # original implementation removes the `NaN` elements instead (using boolean indexing).
        gt_bboxes_matched_normalized = gt_bboxes_matched_normalized.apply(
            lambda tensor: bhl.apply_mask_to_tensor(tensor, isnotnan)
        )

        #@NOTE 
        # Here, we can multiply instead of using `apply_mask_to_tensor()` as the predictions and weights do 
        # not contain `NaN` values.
        pred_bboxes_matched = pred_bboxes_matched.apply(lambda bboxs: bboxs * isnotnan)
        bbox_weights_matched = bbox_weights_matched * isnotnan

        # ===== Losses: BBox (per-layer wrapper) =====
        nvtx_wrp.range_push("loss_bbox computation[StreamPETRHeadBatched.loss_batched]")
        #@NOTE
        # The wrapper method `_compute_per_layer_loss()` is used again here.
        # 
        # In contrast to the classification loss, the bounding box loss is computed only for matched 
        # predictions. See notes in `focal_head_batched.py` for a discussion of how the input data differs 
        # between the two cases.
        loss_bbox = self._compute_per_layer_loss(
            self.loss_bbox, gt_bboxes_matched_normalized.tensor, pred_bboxes_matched.tensor, 
            bbox_weights_matched, per_layer_nums_pos, num_batch_dims=2
        )

        loss_bbox = torch.nan_to_num(loss_bbox)
        nvtx_wrp.range_pop()

        # ===== Finalize (`loss_batched()`) =====
        nvtx_wrp.range_pop("StreamPETRHeadBatched.loss_batched")
        return loss_cls, loss_bbox


    def dn_loss_batched(self,
                        cls_scores: torch.Tensor,
                        bbox_preds: torch.Tensor,
                        known_bboxs: torch.Tensor,
                        known_labels: torch.Tensor,
                        nums_pos: torch.Tensor
                        ) -> Tuple[torch.Tensor, torch.Tensor]:
        ''' DN loss computation batched over the decoder layers.

        Args:
            cls_scores (Tensor): Classification score logits from all decoder layer
                for all images. Shape [num_dec_layers, num_entries, cls_out_channels].
            bbox_preds (Tensor): Bbox regression outputs from all decoder layer
                for all images.
                shape [num_dec_layers, num_entries, 10].
            known_bboxs (Tensor): Ground truth bboxes for each decoder layer and image
                with shape (num_dec_layers, num_entries, 9).
            known_labels (Tensor): Ground truth class indexes for each decoder layer and image
                with shape (num_dec_layers, num_entries).
            nums_pos (torch.Tensor): Number of positive samples for each decoder layer
                with shape (num_dec_layers,).
        Returns:
            loss_cls (Tensor): Classification loss (per-decoder-layer).
            loss_bbox (Tensor): Bounding box regression loss (per-decoder-layer).
        '''
        
        #@NOTE
        #   This method computes the DN loss for the individual decoder layers in a batched manner.
        # Please also see the doc-string above for more details on the input & output. Note that
        # compared to the original implementation, the inputs have an additional dimension (with 
        # size `num_dec_layers`) which is used to batch over the decoder layers. Also, the number of
        # positive samples is given per decoder layer instead of being a scalar (as in the original 
        # implementation, where `dn_loss_single()` computes the loss for a single decoder layer at a time).
        #
        #   The loss computation is batched in a similar manner as the loss computation in `loss_batched()`.
        # Please refer to the notes in `loss_batched()` for more details on the approach. Note that
        # this function is simpler than `loss_batched()` as it does not require to compute & apply
        # the matching between the predictions and the GT, the GT data is of fixed size, and the 
        # predictions and GT data do not need to be aligned.

        # ===== DN: pre-processing =====
        nvtx_wrp.range_push("StreamPETRHeadBatched.dn_loss_batched")
 
        cls_avg_factors = nums_pos * 3.14159 / 6 * self.split * self.split * self.split
        if self.sync_cls_avg_factor:
            cls_avg_factors = reduce_mean(cls_avg_factors)
        
        bbox_weights = torch.ones_like(bbox_preds)
        label_weights = torch.ones_like(known_labels)        

        cls_avg_factors[cls_avg_factors < 1.0] = 1.0

        # ===== DN: cls loss =====
        nvtx_wrp.range_push("loss_cls computation[StreamPETRHeadBatched.dn_loss_batched]")
        loss_cls = self._compute_per_layer_loss(
            self.loss_cls, known_labels, cls_scores, label_weights, cls_avg_factors, num_batch_dims=1
        )
        nvtx_wrp.range_pop()

        nums_pos = nums_pos.type(loss_cls.dtype)
        nums_pos = reduce_mean(nums_pos)
        nums_pos[nums_pos < 1.0] = 1.0

        # ===== DN: bbox loss =====
        # regression L1 loss
        normalized_bbox_targets = normalize_bbox(known_bboxs, self.pc_range)
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1, keepdim=True)

        normalized_bbox_targets = bhl.apply_mask_to_tensor(normalized_bbox_targets, isnotnan)
        bbox_preds = bbox_preds * isnotnan
        bbox_weights = bbox_weights * isnotnan

        bbox_weights = bbox_weights * self.code_weights

        nvtx_wrp.range_push("loss_bbox computation[StreamPETRHeadBatched.dn_loss_batched]")
        loss_bbox = self._compute_per_layer_loss(self.loss_bbox, normalized_bbox_targets, bbox_preds, 
                                                     bbox_weights, nums_pos, num_batch_dims=1)
        loss_bbox = torch.nan_to_num(loss_bbox)
        nvtx_wrp.range_pop()

        # ===== DN: finalize =====
        res = self.dn_weight * loss_cls, self.dn_weight * loss_bbox

        nvtx_wrp.range_pop("StreamPETRHeadBatched.dn_loss_batched")
        return res


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list: List[LiDARInstance3DBoxes],
             gt_labels_list: List[torch.Tensor],
             preds_dicts: Dict[str, Union[torch.Tensor, dict]],
             gt_bboxes_ignore: Optional[List[torch.Tensor]] = None):
        '''Optimized loss computation.

        Args:
            gt_bboxes_list: Ground truth bboxes for each sample.
            gt_labels_list: Ground truth class indexes for each sample.
               Each entry is a tensor with shape (num_gts_i, ).
               Note that `num_gts_i` is the number of ground truth objects for the `i`-th sample.
            preds_dicts:
                all_cls_scores (Tensor): Classification score from all decoder layers.
                    Shape: [num_dec_layers, num_samples, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Bbox regression outputs from all decoder layers.
                    Shape: [num_dec_layers, num_samples, num_query, 10].
                dn_mask_dict (dict): DN mask dictionary
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
                Not supported and needs to be set to `None`. This is consistent with the original 
                implementation, where the HungarianAssigner3D does not support ignore lists.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        '''

        # @NOTE 
        #   For an overview of the batching strategy used here, please see the note at the 
        # beginning of the file.

        # ===== Pre-processing (`loss()`) =====
        nvtx_wrp.range_push("StreamPETRHeadBatched.loss")
        stopwatch.start_meas("StreamPETRHeadBatched.loss")
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'
        
        loss_dict = dict()

        all_cls_scores = preds_dicts['all_cls_scores']
        all_bbox_preds = preds_dicts['all_bbox_preds']

        num_dec_layers = len(all_cls_scores)
        device = gt_labels_list[0].device
        #@NOTE
        # Note that `gt_bboxes_list` is on the CPU, so we need to transfer it to the GPU.
        # However, this is done when calling `combine_data()` below. Typically, it is more efficient to
        # do the combining of samples onto batched on the CPU, so we do not not transfer the data to the GPU
        # before calling `combine_data()`.
        gt_bboxes_list = [
            torch.cat((gt_bboxes.gravity_center, gt_bboxes.tensor[:, 3:]), dim=1)
            for gt_bboxes in gt_bboxes_list
        ]
        gt_bboxes = bhl.combine_data(gt_bboxes_list, device=device, flatten_batch_dims=True)

        #@NOTE
        # Using `other_with_same_sample_sizes=gt_bboxes` to share the memory for the 
        # `sample_sizes` and `mask` tensors of the two RaggedBatch instances.
        gt_labels = bhl.combine_data(gt_labels_list, other_with_same_sample_sizes=gt_bboxes, flatten_batch_dims=True)

        # ===== Prepare per-layer GT =====
        #@NOTE
        # Duplicating the GT for each layer in order to match the shape of the predictions, which
        # are also per-layer (but in contrast to the GT, the predictions differ for each layer). 
        # This is done by introducing a new batch dimension (iterating over decoder layers) and repeating the GT data 
        # for each decoder layer along that dimension.
        gt_bboxes = gt_bboxes.unsqueeze_batch_dim(0)
        gt_bboxes = gt_bboxes.repeat_samples(num_dec_layers, batch_dim=0)
        gt_labels = gt_labels.unsqueeze_batch_dim(0)
        gt_labels = gt_labels.repeat_samples(num_dec_layers, batch_dim=0)
        
        # ===== Compute batched loss over decoder layers =====
        #@NOTE 
        # Compute the loss for each decoder layer. Note that in the original implementation, the loss is 
        # computed for each decoder layer separately. Here, we batch the computation over all decoder layers. 
        # 
        # Batching over the individual samples in the batch is done both in the original and our 
        # implementations (although the approaches differ).
        losses_cls, losses_bbox = self.loss_batched(all_cls_scores, all_bbox_preds, gt_bboxes, gt_labels)

        # loss from the last decoder layer
        # ===== Assemble results =====
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_bbox'] = losses_bbox[-1]

        # loss from other decoder layers
        for ndl, (loss_cls_i, loss_bbox_i) in enumerate(zip(losses_cls[:-1], losses_bbox[:-1])):
            loss_dict[f'd{ndl}.loss_cls'] = loss_cls_i
            loss_dict[f'd{ndl}.loss_bbox'] = loss_bbox_i

        # ===== DN Loss (batched over layers) =====
        #@NOTE: Compute the DN loss. 
        if preds_dicts['dn_mask_dict'] is not None:
            known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = \
                self.prepare_for_loss(preds_dicts['dn_mask_dict'])
            #@NOTE
            # Duplicating the GT for each decoder layer to match the shape of the predictions, which
            # are also per-layer (but differ for each layer in contrast to the GT). In the original 
            # implementation, lists are used instead of `repeat()`, as the GT is not batched over the decoder 
            # layers and instead, the loss is computed for each layer individually. Here, the computation is 
            # batched, so we need the GT data to be repeated for each decoder layer.
            #   
            # Also, note that here, we use tensors and not `RaggedBatch` instances for the GT data, as the 
            # GT is of uniform size.
            all_known_bboxs = known_bboxs.unsqueeze(0).repeat((num_dec_layers, 1, 1))
            all_known_labels = known_labels.unsqueeze(0).repeat((num_dec_layers, 1))
            all_num_tgt = torch.full(
                (num_dec_layers,), num_tgt, dtype=torch.float32, device=all_known_labels.device
            )

            #@NOTE: Computing the DN loss batched over the decoder layers.
            dn_losses_cls, dn_losses_bbox = self.dn_loss_batched(
                output_known_class, output_known_coord, all_known_bboxs, all_known_labels, all_num_tgt
            )

            loss_dict['dn_loss_cls'] = dn_losses_cls[-1]
            loss_dict['dn_loss_bbox'] = dn_losses_bbox[-1]
            #@NOTE
            # Here, we split the loss for each decoder layer and store them in the loss dictionary
            # This results in a dictionary with the same structure as in the original implementation.
            for ndl, (dn_l_cls_i, dn_l_bbox_i) in enumerate(zip(dn_losses_cls[:-1], dn_losses_bbox[:-1])):
                loss_dict[f'd{ndl}.dn_loss_cls'] = dn_l_cls_i
                loss_dict[f'd{ndl}.dn_loss_bbox'] = dn_l_bbox_i
        elif self.with_dn:
            #@NOTE
            # In this case, the DN loss computation is the same as the loss computation, so we
            # re-use the results from the loss computation. Note that this is different from the original 
            # implementation, where the loss is re-computed. 
            # 
            # This branch is not used in the configurations used for the runtime experiments and
            # is only added for feature completeness w.r.t. the original implementation.
            loss_dict['dn_loss_cls'] = loss_dict['loss_cls'].detach().clone()
            loss_dict['dn_loss_bbox'] = loss_dict['loss_bbox'].detach().clone()
            for ndl in range(num_dec_layers - 1):
                loss_dict[f'd{ndl}.dn_loss_cls'] = loss_dict[f'd{ndl}.loss_cls'].detach().clone()
                loss_dict[f'd{ndl}.dn_loss_bbox'] = loss_dict[f'd{ndl}.loss_bbox'].detach().clone()

        # ===== Finalize (`loss()`) =====
        nvtx_wrp.range_pop("StreamPETRHeadBatched.loss")
        stopwatch.end_meas("StreamPETRHeadBatched.loss")

        return loss_dict
    

    def _compute_per_layer_loss(self, 
                                loss_func: Union[torch.nn.Module, Callable], 
                                gt_data: torch.Tensor, 
                                pred_data: torch.Tensor, 
                                weights: torch.Tensor, 
                                avg_factors: torch.Tensor,
                                num_batch_dims: int):
        ''' This functions computes the loss for all decoder layers.

        The computation is performed in a batched manner and the loss is computed for each decoder layer.
        To allow per-layer loss computation, the reduction of the loss over the objects is not performed
        inside the loss function (which would only allow computing the total loss). Instead,
        the following steps are performed:
          1. The batch dimensions are flattened away, as the loss function expests this format.
          2. The loss function is called with `reduction_override='none'`, so that no reduction is performed.
          3. The loss is reshaped to `[num_dec_layers, -1]`, i.e. the batch is flattened over the samples, but 
             not over the decoder layers.
          4. The reduction over the samples & objects is performed (if needed), obtaining a tensor of shape 
             `[num_dec_layers]` where each element is the loss for the corresponding decoder layer.

        This method assumes that the individual decoder layers correspond to the first dimension of the input 
        tensors.

        Args:
            loss_func: The loss function to be used. It will recieve the flattened predictions, weights and 
                ground truth data. The flattening is performed to combine the batch dimensions and the 
                dimension iterating over the objects into a single dimension. The shape of the data is 
                preserved in the remaining dimensions.
            gt_data: The ground truth data.
            pred_data: The predicted data.
            weights: The weights of the predicted data.
            avg_factors: The average factors. Need to be provided if `reduction_override='mean'`. The
                factors are the divisors which are used when computing the mean loss (instead of the number of 
                elements). This is consistent with the actual loss functions, which also expect such a factor. 
                The average factors used here are per-decoder-layer and correspond to the factors that would 
                be passed to the loss function for the corresponding decoder layer if the loss would be 
                computed for each decoder layer separately.
            num_batch_dims: The number of batch dimensions in the input data. This is used to determine the 
                number of dimensions to flatten.

        Returns:
            The loss for each decoder layer. Shape: [num_dec_layers].
        '''
        # Number of decoder layers (outer-most batch dimension)
        num_dec_layers = gt_data.shape[0]

        # Flatten the data, combining all batch dimensions and the dimension 
        # iterating over the objects
        num_dims_to_flatten = num_batch_dims + 1
        pred_data_flat = pred_data.reshape([-1, *pred_data.shape[num_dims_to_flatten:]])
        gt_data_flat = gt_data.reshape([-1, *gt_data.shape[num_dims_to_flatten:]])
        weights_flat = weights.reshape([-1, *weights.shape[num_dims_to_flatten:]])
        # Compute loss for the flattened list of objects
        loss = loss_func(
            pred_data_flat, gt_data_flat, weights_flat, reduction_override='none')
        # Perform reduction for the individual decoder layers
        if loss_func.reduction is not None:
            # If reduction is to be performed, reshape so that reduction is not performed 
            # for the individual decoders, but along the other dimensions
            shape_to_reduce = [num_dec_layers, -1]
            if loss_func.reduction == 'mean':
                loss = torch.sum(loss.reshape(shape_to_reduce), dim=1) / avg_factors
            elif loss_func.reduction == 'sum':
                loss = torch.sum(loss.reshape(shape_to_reduce), dim=1)
        return loss



    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts: Prediction results.
            img_metas: Point cloud and image's meta info.
            rescale: Unused (for compatibility with the original implementation).

        Returns:
            Decoded bbox, scores and labels after nms.
        """
        preds_dicts = self.bbox_coder.decode(preds_dicts)
        num_samples = len(preds_dicts)

        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
            bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
            scores = preds['scores']
            labels = preds['labels']
            ret_list.append([bboxes, scores, labels])
        return ret_list
    

    def get_targets(self, *args, **kwargs):
        '''
        This method is not implemented for the `StreamPETRHeadBatched` class.
        It is not needed internally and will result in an error if called from outside.
        It is included only to satisfy the interface expected by the base class.
        '''
        raise NotImplementedError("StreamPETRHeadBatched does not use/implement `get_targets()`")
