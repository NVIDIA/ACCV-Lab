# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shihao Wang
# (StreamPETR: https://github.com/exiawsh/StreamPETR)
# ---------------------------------------------
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: This is based on `hungarian_assigner_2d.py`. Changes include:
#           - Added NVTX range-based profiling; adjusted the documentation
#           - Added batched processing of the cost matrix computation; adjusted the implementation and 
#             the interface accordingly.
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
The batched Hungarian 2D matcher (``HungarianAssigner2DBatched``) is used in the focal head with batched loss 
computation (``FocalHeadBatched``).

The matcher is designed to enable batched processing of parts of the matching process. Most notably, the 
cost matrix computation is performed in a batched manner, while the actual Hungarian matching is performed
on a per-sample basis on the CPU (as the used implementation of the Hungarian matching algorithm requires 
this).

The implementation is very similar to the matcher implementation in the ``batched-processing`` example
(see :doc:`/contained_package_docs_mirror/batching_helpers/docs/example` and the comments in ``matcher.py`` in
the implementation).

For implementation details, see notes in the ``assign()`` method of this class. 
'''

from typing import Optional, Tuple

import torch

from mmdet.core.bbox.builder import BBOX_ASSIGNERS
from mmdet.core.bbox.assigners import BaseAssigner
from mmdet.core.bbox.match_costs import build_match_cost
from mmdet.core import bbox_cxcywh_to_xyxy

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from accvlab.optim_test_tools import NVTXRangeWrapper
nvtx_wrp = NVTXRangeWrapper()

import accvlab.batching_helpers as bhl


@BBOX_ASSIGNERS.register_module()
class HungarianAssigner2DBatched(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost and regression iou cost. 

    The output is a pair of `RaggedBatch` instances, containing the indices (0-based) of the
    predicted and ground truth objects that are matched to each other.

    Args:
        cls_cost: Config describing of the classification cost to be used & the applied weight.
        reg_cost: Config describing of the regression cost to be used & the applied weight.
        iou_cost: Config describing of the iou cost to be used & the applied weight.
        centers2d_cost: Config describing of the center2d cost to be used & the applied weight.
    """

    def __init__(self,
                 cls_cost=dict(type='ClassificationCost', weight=1.),
                 reg_cost=dict(type='BBoxL1Cost', weight=1.0),
                 iou_cost=dict(type='IoUCost', iou_mode='giou', weight=1.0),
                 centers2d_cost=dict(type='BBox3DL1Cost', weight=1.0)):
        self.cls_cost = build_match_cost(cls_cost)
        self.reg_cost = build_match_cost(reg_cost)
        self.iou_cost = build_match_cost(iou_cost)
        self.centers2d_cost = build_match_cost(centers2d_cost)

    @staticmethod
    def _flatten_batch_dims_for_tensor(tensor, num_batch_dims):
        res = tensor.reshape(
            torch.prod(torch.tensor(tensor.shape[:num_batch_dims])), *tensor.shape[num_batch_dims:]
        )
        return res

    def assign(self,
               bbox_pred: torch.Tensor,
               cls_pred: torch.Tensor,
               pred_centers2d: torch.Tensor,
               gt_bboxes: bhl.RaggedBatch,
               gt_labels: bhl.RaggedBatch,
               centers2d: bhl.RaggedBatch,
               img_meta: dict,
               gt_bboxes_ignore: Optional[torch.Tensor] = None,
               eps: float = 1e-7
               ) -> Tuple[bhl.RaggedBatch, bhl.RaggedBatch]:
        
        ''' Perform the matching.

        Args:
            bbox_pred: Predicted bounding boxes as: (cx, cy, w, h), normalized to the image size.
                Shape: `[batch_size, num_preds, 4]`.
            cls_pred: Predicted classification logits.
                Shape: `[batch_size, num_preds, num_classes]`.
            pred_centers2d: Predicted center2d coordinates as: (cx, cy), normalized to the image size.
                Shape: `[batch_size, num_preds, 2]`.
            gt_bboxes: Ground truth bounding boxes as: (x_1, y_1, x_2, y_2)
                Shape: `[batch_size, max_num_gts, 4]`.
                Note that `max_num_gts` is the maximum number of ground truth objects in the batch.
            gt_labels: Ground truth labels.
                Shape: `[batch_size, max_num_gts]`.
                Note that `max_num_gts` is the maximum number of ground truth objects in the batch.
            centers2d: Ground truth center2d coordinates as: (x, y).
                Shape: `[batch_size, max_num_gts, 2]`.
                Note that `max_num_gts` is the maximum number of ground truth objects in the batch.
            img_meta: Meta information for the image.
            gt_bboxes_ignore: Ignored ground truth bounding boxes (not supported and 
                must be set to `None`).
            eps: Epsilon value for numerical stability (unused).

        Returns:
            pred_matched_indices: Predicted matched indices.
                Shape: `[batch_size, max_num_matches]`.
                Note that `max_num_matches` is the maximum number of matches in the batch.
            gt_matched_indices: Ground truth matched indices.
                Shape: `[batch_size, max_num_matches]`.
                Note that `max_num_matches` is the maximum number of matches in the batch.

        '''
        #@NOTE
        #   The implementation is similar to the matcher implementation in the `batched-processing` example.
        # Also see the comments in `matcher.py` in the `batched_processing` example for a detailed explanation 
        # of the basic approach used here.
        
        # ===== Pre-processing =====

        
        nvtx_wrp.range_push("HungarianAssigner2DBatched.assign")
        assert gt_bboxes_ignore is None, \
            'Only case when gt_bboxes_ignore is None is supported.'

        #@NOTE
        #   If multiple batch dimensions are used, we flatten them into a single dimension.
        # This is done as
        #   - We do not care about the batch shape, so we can flatten it.
        #   - The cost functions used to compute the cost matrices may expect a single batch dimension.
        # After the matching is done, we restore the original batch shape for the results.
        # Please also see the comments at the end of the function for more details on the restoration.
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
            nvtx_wrp.range_pop("HungarianAssigner2DBatched.assign")
            return matched_indices, matched_indices

        needs_reshape = len(batch_shape_orig) > 1
        if needs_reshape:
            #@NOTE: Flatten the batch dimensions.
            gt_bboxes = gt_bboxes.flatten_batch_dims()
            gt_labels = gt_labels.flatten_batch_dims()
            centers2d = centers2d.flatten_batch_dims()
            #@NOTE
            # Flatten the batch dimensions for the tensor inputs which
            #   - Are expected to have the same batch shape as the `RaggedBatch` instances.
            #   - Do not have built-in helper functions for this.
            bbox_pred = self._flatten_batch_dims_for_tensor(bbox_pred, num_batch_dims_orig)
            cls_pred = self._flatten_batch_dims_for_tensor(cls_pred, num_batch_dims_orig)
            pred_centers2d = self._flatten_batch_dims_for_tensor(pred_centers2d, num_batch_dims_orig)
            
        batch_size, max_num_gts, num_bboxes = gt_bboxes.size(0), gt_bboxes.size(1), bbox_pred.size(1)

        img_h, img_w, _ = img_meta[0]['pad_shape'][0]
        factor = gt_bboxes.tensor.new_tensor([img_w, img_h, img_w,
                                       img_h]).view(1, 1, -1)

        # ===== Compute the cost matrices =====

        #@NOTE
        #   Overall, the cost matrix computation implementation is very similar to the original implementation.
        # Note that we expect the individual cost functions to be able to handle batched data. However, they
        # do not need to handle non-uniform sample sizes and may assume a uniform batch size. This means that 
        # costs functions can be re-used as is if they support basic batching.
        # The output of the cost functions is a tensor of shape `[batch_size, num_preds, max_num_gts]`,
        # i.e. the cost matrices are padded to the same size (`max_num_gts`) for a given batch.
        #
        #   The handling of the non-uniform sample sizes is done after the cost matrix is computed by using
        # `gt_bboxes.create_with_sample_sizes_like_self(cost, 2)`. See comments below for more details).

        #@NOTE: Compute the cost matrices in a batched manner.
        nvtx_wrp.range_push("HungarianAssigner2DBatched.assign:cost matrices")
        nvtx_wrp.range_push("HungarianAssigner2DBatched.assign:cls_cost")
        cls_cost = self.cls_cost(cls_pred, gt_labels.tensor)
        nvtx_wrp.range_pop()
        # regression L1 cost
        #@NOTE 
        #   The `apply` method is a convenience method to apply a function to the underlying `tensor` 
        # data of the `RaggedBatch`. Note that while here, only the data tensor (i.e. `tensor`) is 
        # input to the lambda function, the passed function (lambda in this case) can accept more inputs. 
        # If more inputs are defined, the function will also receive the `sample_sizes` and `mask` 
        # of the `RaggedBatch`.
        normalized_gt_bboxes = gt_bboxes.apply(lambda bboxs: bboxs / factor)
        
        #@NOTE
        #  Here, the `normalized_gt_bboxes.tensor` is used as input to the cost function. The function expects
        #  tensor inputs and is not implmentented with the `RaggedBatch` format in mind.
        nvtx_wrp.range_push("HungarianAssigner2DBatched.assign:reg_cost")
        reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes.tensor)
        nvtx_wrp.range_pop()

        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor

        nvtx_wrp.range_push("HungarianAssigner2DBatched.assign:iou_cost")
        iou_cost = self.iou_cost(bboxes, gt_bboxes.tensor)
        nvtx_wrp.range_pop()

        # center2d L1 cost
        #@NOTE: The `apply` method is used again here.
        normalized_centers2d = centers2d.apply(lambda ctr2d: ctr2d / factor[..., 0:2])

        nvtx_wrp.range_push("HungarianAssigner2DBatched.assign:centers2d_cost")
        centers2d_cost = self.centers2d_cost(pred_centers2d, normalized_centers2d.tensor)
        nvtx_wrp.range_pop()

        # weighted sum of above four costs
        cost = cls_cost + reg_cost + iou_cost + centers2d_cost
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        
        #@NOTE
        #   During cost matrix computation, we assume uniform batch size (and use filler values). However, the
        # valid cost matrices are non-uniform in size. Along the GT object dimension, the sample sizes 
        # correspond to the sample sizes of the GT inputs. However, the dimension iterating over the GT 
        # objects (i.e. the `non_uniform_dim`) is `dim==2` for the matrices (`dim==1` corresponds to the 
        # predictions). Thererefore, we create a RaggedBatch containing the matrices, with the non-uniform 
        # dimension set to `dim==2` (instead of `dim==1` in `gt_bboxes`).
        cost = gt_bboxes.create_with_sample_sizes_like_self(cost, non_uniform_dim=2)

        nvtx_wrp.range_pop()

        # ===== Perform the matching =====

        nvtx_wrp.range_push("HungarianAssigner2DBatched.assign:matching")
        cost = cost.detach()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        
        #@NOTE 
        #   Move the cost matrices to the CPU. This is needed to run the Hungarian
        # matching on the CPU. Note that this is performed before splitting the cost matrices into individual
        # samples, so that data for all samples is moved in one go.
        cost_cpu = cost.to_device(device="cpu")
        #@NOTE 
        #   Split the cost matrices into individual samples. Note that `split()` removes the padding,
        # so that the valid matrices without any filler regions are returned.
        per_sample_costs = cost_cpu.split()
        #@NOTE: Perform matching for each sample individually in a loop.
        matched_gt_list = [None] * batch_size
        matched_pred_list = [None] * batch_size
        nvtx_wrp.range_push("assigner loop [HungarianAssigner2DBatched.assign]")
        for i in range(batch_size):
            cost_i = per_sample_costs[i]

            nvtx_wrp.range_push("hungarian [HungarianAssigner2DBatched]")
            matched_row_inds, matched_col_inds = linear_sum_assignment(cost_i)
            nvtx_wrp.range_pop()

            matched_gt_list[i] = torch.from_numpy(matched_col_inds)
            matched_pred_list[i] = torch.from_numpy(matched_row_inds)
        nvtx_wrp.range_pop()

        # ===== Combine the matching results & post-processing =====

        #@NOTE: Combine the matching results for all samples into a single `RaggedBatch`.
        gt_matched_indices = bhl.combine_data(matched_gt_list)
        #@NOTE 
        #   Note the `other_with_same_sample_sizes` argument, which is used to specify that the
        # second `RaggedBatch` (here `matched_pred_list`) has the same sample sizes as the first one
        # (here `gt_matched_indices`). This is not required, but results in a more efficient implementation
        # as the `sample_sizes` and `mask` tensors can be shared between the two `RaggedBatch` instances.
        pred_matched_indices = bhl.combine_data(
            matched_pred_list, other_with_same_sample_sizes=gt_matched_indices
        )

        #@NOTE: Convert to the expected format and transfer to the correct device.
        device = bbox_pred.device
        pred_matched_indices = pred_matched_indices.to_device(device=device)
        gt_matched_indices = gt_matched_indices.to_device(device=device)

        #@NOTE 
        # If the batch dimensions were flattened, restore the original shape.
        # Note that the overall shape of the results is different from the shape of the original.
        # However, as here, we are only concerned with the batch shape, it is the same and 
        # we can easily reshape into the needed format.
        if needs_reshape:
            pred_matched_indices = pred_matched_indices.reshape_batch_dims(batch_shape_orig)
            gt_matched_indices = gt_matched_indices.reshape_batch_dims(batch_shape_orig)

        #@NOTE: Types of range pops
        #@NOTE: A simple range pop to close the NVTX range, without mismatch checking.
        nvtx_wrp.range_pop()
        #@NOTE: This will check for range stack mismatches if checking is enabled (enable only for debugging).
        nvtx_wrp.range_pop("HungarianAssigner2DBatched.assign")

        #@NOTE 
        #   Return the matching results. In contrast to the original implementation, here, the results
        # of the matching (i.e. pairs of indices) are returned directly (as `RaggedBatch` instances). This 
        # is different from the original implementation, where an `AssignResult` instance is returned. 
        # See the comments in `focal_head_batched.py` for a discussion of the approach used here instead.
        return pred_matched_indices, gt_matched_indices
