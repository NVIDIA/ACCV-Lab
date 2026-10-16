# ------------------------------------------------------------------------
# Modified from mmdetection (https://github.com/open-mmlab/mmdetection)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# Modified from DETR3D (https://github.com/WangYueFt/detr3d)
# Copyright (c) 2021 Wang, Yue
# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified in StreamPETR (https://github.com/exiawsh/StreamPETR).
# ------------------------------------------------------------------------
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: Based on the StreamPETR implementation of `HungarianAssigner3D` and
#          `hungarian_assigner_2d_batched.py`. Changes include:
#           - Added NVTX range-based profiling; adjusted the documentation
#           - Added batched processing of the cost matrix computation; adjusted the implementation and 
#             the interface accordingly.
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

from typing import Optional, Tuple

import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

import accvlab.batching_helpers as bhl

from accvlab.optim_test_tools import NVTXRangeWrapper
nvtx_wrp = NVTXRangeWrapper()

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3DBatched(BaseAssigner):
    '''Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. 

    The output is a pair of `RaggedBatch` instances, containing the indices (0-based) of the
    predicted and ground truth objects that are matched to each other.

    Args:
        cls_cost: Config describing of the classification cost to be used & the applied weight.
        reg_cost: Config describing of the regression cost to be used & the applied weight.
        iou_cost: Config describing of the iou cost to be used & the applied weight.
        pc_range: Spatial range of the point cloud to be considered.
    '''

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range


    @staticmethod
    def _flatten_batch_dims_for_tensor(tensor, num_batch_dims):
        res = tensor.reshape(
            torch.prod(torch.tensor(tensor.shape[:num_batch_dims])), *tensor.shape[num_batch_dims:]
        )
        return res
    

    def assign(self,
            bbox_pred: torch.Tensor,
            cls_pred: torch.Tensor,
            gt_bboxes: bhl.RaggedBatch,
            gt_labels: bhl.RaggedBatch,
            gt_bboxes_ignore: Optional[torch.Tensor] = None,
            code_weights: Optional[torch.Tensor] = None,
            with_velo: bool = False,
            eps: float = 1e-7 # to conform to interface
            ) -> Tuple[bhl.RaggedBatch, bhl.RaggedBatch]:
        ''' Perform the matching.

        Args:
            bbox_pred: Predicted normalized bounding boxes
                Shape: `[batch_size, num_preds, 10]`.
            cls_pred: Predicted classification logits.
                Shape: `[batch_size, num_preds, num_classes]`.
            gt_bboxes: Ground truth bounding boxes
                Shape: `[batch_size, max_num_gts, 9]`.
                Note that `max_num_gts` is the maximum number of ground truth objects in the batch.
            gt_labels: Ground truth labels.
                Shape: `[batch_size, max_num_gts]`.
                Note that `max_num_gts` is the maximum number of ground truth objects in the batch.
            gt_bboxes_ignore: Ignored ground truth bounding boxes (not supported and 
                must be set to `None`).
            code_weights: Code weights for the bounding boxes.
                Shape: `[10]`.
            with_velo: Whether to use velocity information.
            eps: Epsilon value for numerical stability (unused).

        Returns:
            pred_matched_indices: Predicted matched indices.
                Shape: `[batch_size, max_num_matches]`.
                Note that `max_num_matches` is the maximum number of matches in the batch.
            gt_matched_indices: Ground truth matched indices.
                Shape: `[batch_size, max_num_matches]`.
                Note that `max_num_matches` is the maximum number of matches in the batch.

        '''
        nvtx_wrp.range_push("HungarianAssigner3DBatched.assign")
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        
        #@NOTE
        #   This implementation is very similar to `hungarian_assigner_2d_batched.py` (and the the 
        # matcher implementation in the `batched-processing`
        # example). Please see the notes in `hungarian_assigner_2d_batched.py` for an explanation of
        #  the used approach.

        batch_shape_orig = gt_bboxes.batch_shape
        num_batch_dims_orig = len(batch_shape_orig)

        max_num_gts = gt_bboxes.max_sample_size
        num_bboxes = bbox_pred.size(num_batch_dims_orig)
        # Handle empty inputs & return without further processing.
        if max_num_gts == 0 or num_bboxes == 0:
            matched_indices = bhl.RaggedBatch.Empty(
                num_dims=num_batch_dims_orig + 1,
                non_uniform_dim=num_batch_dims_orig,
                device=bbox_pred.device,
                batch_shape=batch_shape_orig,
                dtype=torch.int64,
            )
            nvtx_wrp.range_pop("HungarianAssigner3DBatched.assign")
            return matched_indices, matched_indices

        needs_reshape = len(batch_shape_orig) > 1
        if needs_reshape:
            gt_bboxes = gt_bboxes.flatten_batch_dims()
            gt_labels = gt_labels.flatten_batch_dims()
            bbox_pred = self._flatten_batch_dims_for_tensor(bbox_pred, num_batch_dims_orig)
            cls_pred = self._flatten_batch_dims_for_tensor(cls_pred, num_batch_dims_orig)

        batch_size, max_num_gts, num_bboxes = gt_bboxes.size(0), gt_bboxes.size(1), bbox_pred.size(1)

        nvtx_wrp.range_push("HungarianAssigner3DBatched.assign:cost matrices")

        nvtx_wrp.range_push("HungarianAssigner3DBatched.assign:cls_cost")
        cls_cost = self.cls_cost(cls_pred, gt_labels.tensor)
        #@MARK
        nvtx_wrp.range_pop()
        nvtx_wrp.range_push("HungarianAssigner3DBatched.assign:bbox_norm")
        # regression L1 cost
        normalized_gt_bboxes = gt_bboxes.apply(lambda bbxs: normalize_bbox(bbxs, self.pc_range))
        if code_weights is not None:
            bbox_pred = bbox_pred * code_weights
            normalized_gt_bboxes = normalized_gt_bboxes.apply(lambda bbxs: bbxs * code_weights)
        #@MARK
        nvtx_wrp.range_pop()    
        nvtx_wrp.range_push("HungarianAssigner3DBatched.assign:reg_cost")
        if with_velo:
            reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes.tensor)
        else:
            reg_cost = self.reg_cost(bbox_pred[..., :8], normalized_gt_bboxes.tensor[..., :8])
        #@MARK
        nvtx_wrp.range_pop()

        cost = cls_cost + reg_cost
        cost = gt_bboxes.create_with_sample_sizes_like_self(cost, 2)

        nvtx_wrp.range_pop()

        nvtx_wrp.range_push("HungarianAssigner3D.assign:matching")
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.apply(lambda cst: torch.nan_to_num(cst, nan=100.0, posinf=100.0, neginf=-100.0))
        cost_cpu = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        
        per_sample_costs = cost_cpu.split()
        matched_gt_list = [None] * batch_size
        matched_pred_list = [None] * batch_size
        for i in range(batch_size):
            cost_i = per_sample_costs[i]
            nvtx_wrp.range_push("hungarian [HungarianAssigner3D]")
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost_i)
            nvtx_wrp.range_pop()
            matched_gt_list[i] = torch.from_numpy(matched_col_inds)
            matched_pred_list[i] = torch.from_numpy(matched_row_inds)
        #@MARK
        nvtx_wrp.range_pop()
        nvtx_wrp.range_push("HungarianAssigner3D.assign:combine data")
        gt_matched_indices = bhl.combine_data(matched_gt_list)
        pred_matched_indices = bhl.combine_data(
            matched_pred_list, other_with_same_sample_sizes=gt_matched_indices
        )
        nvtx_wrp.range_pop()

        device = gt_bboxes.device
        gt_matched_indices = gt_matched_indices.to_device(device)
        pred_matched_indices = pred_matched_indices.to_device(device)

        if needs_reshape:
            pred_matched_indices = pred_matched_indices.reshape_batch_dims(batch_shape_orig)
            gt_matched_indices = gt_matched_indices.reshape_batch_dims(batch_shape_orig)

        nvtx_wrp.range_pop("HungarianAssigner3DBatched.assign")
        return pred_matched_indices, gt_matched_indices
