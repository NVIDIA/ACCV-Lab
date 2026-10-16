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
# Changes: - Added NVTX range-based profiling; adjusted the documentation.
#          - Added possiblity to perform the `nan_to_num` operation on the GPU instead of the CPU (for 
#            improved performance and consistency with the batched implementation in 
#            `hungarian_assigner_2d_batched.py`).
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

import torch
from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import AssignResult
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from projects.mmdet3d_plugin.core.bbox.util import normalize_bbox

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from accvlab.optim_test_tools import NVTXRangeWrapper
nvtx_wrp = NVTXRangeWrapper()

@BBOX_ASSIGNERS.register_module()
class HungarianAssigner3D(BaseAssigner):
    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', weight=0.0),
                 pc_range=None,
                 nan_to_num_on_gpu=False):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.pc_range = pc_range
        self.nan_to_num_on_gpu = nan_to_num_on_gpu

    def assign(self,
               bbox_pred,
               cls_pred,
               gt_bboxes,
               gt_labels,
               gt_bboxes_ignore=None,
               code_weights=None,
               with_velo=False,
               eps=1e-7):
        #@MARK
        nvtx_wrp.range_push("HungarianAssigner3D.assign")
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)
        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)         
        # 2. compute the weighted costs
        # classification and bboxcost.
        #@MARK
        nvtx_wrp.range_push("HungarianAssigner3DBatched.assign:cost matrices")

        nvtx_wrp.range_push("HungarianAssigner3D.assign:cls_cost")
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        #@MARK
        nvtx_wrp.range_pop()
        nvtx_wrp.range_push("HungarianAssigner3D.assign:bbox_norm")
        # regression L1 cost
        normalized_gt_bboxes = normalize_bbox(gt_bboxes, self.pc_range)
        if code_weights is not None:
            bbox_pred = bbox_pred * code_weights
            normalized_gt_bboxes = normalized_gt_bboxes * code_weights
        #@MARK
        nvtx_wrp.range_pop()
        nvtx_wrp.range_push("HungarianAssigner3D.assign:reg_cost")
        if with_velo:
            reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes)
        else:
            reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])
        #@MARK
        nvtx_wrp.range_pop()
        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        nvtx_wrp.range_pop()
        
        #@MARK
        nvtx_wrp.range_push("HungarianAssigner3D.assign:matching")
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        if self.nan_to_num_on_gpu:
            nvtx_wrp.range_push("HungarianAssigner3D.assign:matching:nan_to_num")
            cost = torch.nan_to_num(cost.detach(), nan=100.0, posinf=100.0, neginf=-100.0)
            nvtx_wrp.range_pop()
            nvtx_wrp.range_push("HungarianAssigner3D.assign:matching:to_cpu")
            cost = cost.cpu()
            nvtx_wrp.range_pop()
        else:
            nvtx_wrp.range_push("HungarianAssigner3D.assign:matching:to_cpu")
            cost = cost.detach().cpu()
            nvtx_wrp.range_pop()
            nvtx_wrp.range_push("HungarianAssigner3D.assign:matching:nan_to_num")
            cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
            nvtx_wrp.range_pop()            
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        #@MARK
        nvtx_wrp.range_push("hungarian [HungarianAssigner3D]")
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        #@MARK
        nvtx_wrp.range_pop()
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)
        #@MARK
        nvtx_wrp.range_pop()
        nvtx_wrp.range_push("HungarianAssigner3D.assign:post_proc")

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]
        res = AssignResult(num_gts, assigned_gt_inds, None, labels=assigned_labels)
        #@MARK
        nvtx_wrp.range_pop()
        nvtx_wrp.range_pop()
        return res                       