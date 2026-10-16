# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
#  Modified by Shihao Wang
# (StreamPETR: https://github.com/exiawsh/StreamPETR)
# ------------------------------------------------------------------------
#
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Changes: This is based on `focal_head.py`. Changes include: 
#           - New more-fully-batched FocalHead loss/head variant (uses batching helpers).
#           - Added NVTX range-based profiling and runtime measurements.
#           - Added comments & notes describing the approach & implementation.          
#           - Added option to use the custom op implementation of the heatmap generation.
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
In the original implementation, the loss computation is partially batched: the loss functions are batched, 
while the bounding box matching/assignment is not.
Here, we leverage the :mod:`~accvlab.batching_helpers` package to extend batching further by computing 
the cost matrix for matching in a batched fashion (though we still apply the Hungarian matching procedure 
itself on a per-sample basis).

Crucially, we preserve the batch dimension throughout almost the entire pipeline, only flattening data across
samples and cameras at the final stage if the loss functions expect a flattened representation. 
Note that the flattening is only done to enable re-use of the already implemented loss functions, and 
is not necessary if a loss function is designed to process batched data.

Note that while the :class:`~accvlab.batching_helpers.RaggedBatch` class supports multiple batch dimensions, we
only use a single batch dimension iterating over both samples and cameras as the distinction between the 
samples and cameras are not needed for the loss computation and the losses are accumulated over both samples 
and cameras to obtain the final loss value.

The late flattening allows us to keep per-entry (sample & camera) data more clearly separated, making 
operations like cost matrix computation, heatmap generation, and other per-sample tasks much more 
straightforward. As some of the needed operations cannot be easily implemented in a batched manner using 
PyTorch build-in functionality, functionality of the :class:`~accvlab.batching_helpers.RaggedBatch` class and 
the :mod:`~accvlab.batching_helpers` package in general is used where applicable.

Compared to the original approach, this batching strategy differs in two key ways:
 1) Data is no longer flattened early across samples and cameras; instead, the batch dimension is maintained,
    making it possible to apply operations on a per-entry basis (while still processing the batch as a whole). 
    Only if and when the final loss computation implementation necessitates it, we flatten away the batch 
    dimension.
 2) In the original implementation, the ground truth data was aligned with the predictions in the 
    ``get_targets()`` function by creating prediction-sized tensors and filling them with the GT data 
    according to the matching. Instead, we acquire matching indices from the assigner (as 
    :class:`~accvlab.batching_helpers.RaggedBatch` objects) and perform alignment inside the loss 
    implementation itself (see the ``loss_batched()`` function for more details). In this way, we can easily 
    handle either matched-pairs-only alignment or align GT with all predictions for loss functions requiring 
    non-matched items as well (classification loss, for example) in a clearer fashion. This approach yields 
    smaller tensors for matched-only operations and keeps the code more modular and transparent.

For more details about the actual matching/assignment implementation including cost matrix computation, 
please also see the ``HungarianAssigner2DBatched`` implementation (see 
``demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py`` in the 
repository).

Note that the approach used here is an adaptation of the approach used in the :mod:`~accvlab.batching_helpers` 
:doc:`/contained_package_docs_mirror/batching_helpers/docs/example`, adjusted to the specifics of focal head 
implementation & the implementation of the used loss functions.

The implementation of the loss computation is done in the ``loss()`` function. Please see the notes there and 
in the related called functions for more details.

.. note::
    We refer to the reference implementation (in 
    ``demo/streampetr/projects/mmdet3d_plugin/models/dense_heads/focal_head.py``) in some of the 
    notes/code comments. Please note that this reference implementation mostly corresponds to the original
    implementation. However, we added an option there to use the custom op implementation of the heatmap 
    generation (for fairer comparison of the speedup achieved by our batching approach).

'''

from typing import List, Dict, Optional

import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler,
                        reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from projects.mmdet3d_plugin.models.utils.misc import apply_center_offset, apply_ltrb
from mmdet.core import bbox_overlaps
from mmdet3d.models.utils import clip_sigmoid
import random

from accvlab.draw_heatmap import draw_heatmap_batched

import accvlab.batching_helpers as bhl

from accvlab.optim_test_tools import NVTXRangeWrapper, Stopwatch
nvtx_wrp = NVTXRangeWrapper()
stopwatch = Stopwatch()


@HEADS.register_module()
class FocalHeadBatched(AnchorFreeHead):
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
    def __init__(self,
                 num_classes,
                 in_channels=256,
                 embed_dims=256,
                 stride=16,
                 use_hybrid_tokens=False,
                 train_ratio=1.0,
                 infer_ratio=1.0,
                 sync_cls_avg_factor=False,
                 loss_cls2d=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_centerness=dict(type='GaussianFocalLoss', reduction='mean'),
                 loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
                 loss_centers2d=dict(type='L1Loss', loss_weight=5.0),
                 train_cfg=dict(
                 assigner2d=dict(
                     type='HungarianAssigner2DBatched',
                     cls_cost=dict(type='FocalLossCostBatched', weight=2.0),
                     reg_cost=dict(type='BBoxL1CostBatched', weight=5.0, box_format='xywh'),
                     iou_cost=dict(type='IoUCostBatched', iou_mode='giou', weight=2.0),
                     centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        self.bg_cls_weight = 0
        self.sync_cls_avg_factor = sync_cls_avg_factor

        if train_cfg:
            assert 'assigner2d' in train_cfg, 'assigner2d should be provided '\
                'when train_cfg is set.'
            assigner2d = train_cfg['assigner2d']

            self.assigner2d = build_assigner(assigner2d)

            # DETR sampling=False, so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.stride=stride
        self.use_hybrid_tokens=use_hybrid_tokens
        self.train_ratio=train_ratio
        self.infer_ratio=infer_ratio

        super(FocalHeadBatched, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        self.loss_cls2d = build_loss(loss_cls2d)
        self.loss_bbox2d = build_loss(loss_bbox2d)
        self.loss_iou2d = build_loss(loss_iou2d)
        self.loss_centers2d = build_loss(loss_centers2d)
        self.loss_centerness = build_loss(loss_centerness)

        self._init_layers()

    def _init_layers(self):
        self.cls = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)

        self.shared_reg= nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)

        self.shared_cls = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)

        self.centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        self.ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
        self.center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

    def forward(self, location, **data):
        src = data['img_feats']
        bs, n, c, h, w= src.shape
        num_tokens = n*h*w
        
        # focal sampling
        if self.training:
            if self.use_hybrid_tokens:
                sample_ratio = random.uniform(0.2, 1.0)
            else:
                sample_ratio = self.train_ratio 
            num_sample_tokens = int(num_tokens * sample_ratio)
           
        else:
            sample_ratio = self.infer_ratio
            num_sample_tokens = int(num_tokens * sample_ratio)


        x = src.flatten(0, 1)
        cls_feat = self.shared_cls(x)
        cls = self.cls(cls_feat)
        centerness = self.centerness(cls_feat)
        cls_logits = cls.permute(0,2,3,1).reshape(bs*n,-1,self.num_classes)
        centerness = centerness.permute(0,2,3,1).reshape(bs*n,-1,1)
        pred_bboxes = None
        pred_centers2d = None
        
        reg_feat = self.shared_reg(x)
        ltrb = self.ltrb(reg_feat).permute(0,2,3,1).contiguous()
        ltrb = ltrb.sigmoid()
        centers2d_offset = self.center2d(reg_feat).permute(0,2,3,1).contiguous()

        centers2d = apply_center_offset(location, centers2d_offset)
        bboxes = apply_ltrb(location, ltrb)
            
        pred_bboxes = bboxes.view(bs*n,-1,4)
        pred_centers2d = centers2d.view(bs*n,-1,2)

        cls_score = cls_logits.topk(1, dim=2).values[..., 0].view(bs, -1, 1)

        sample_weight = cls_score.detach().sigmoid() * centerness.detach().view(bs,-1,1).sigmoid()

        _, topk_indexes = torch.topk(sample_weight, num_sample_tokens, dim=1)


        outs = {
                'enc_cls_scores': cls_logits,
                'enc_bbox_preds': pred_bboxes,
                'pred_centers2d': pred_centers2d,
                'centerness':centerness,
                'topk_indexes':topk_indexes,
            }

        return outs
    
    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes2d_list: List[List[torch.Tensor]],
             gt_labels2d_list: List[List[torch.Tensor]],
             centers2d: List[List[torch.Tensor]],
             depths: List[List[torch.Tensor]],
             preds_dicts: Dict[str, torch.Tensor],
             img_metas: List[dict],
             gt_bboxes_ignore: Optional[torch.Tensor] = None
             )->Dict[str, torch.Tensor]:
        """"Loss function.

        Note that in the argument description, `num_gts` is the number of ground truth objects and is
        different for each entry (i.e. each image in each sample). `B` denotes the batch size, which
        and `B = num_samples * num_cameras` holds. 
        
        Also note that `num_samples` is the number of samples. This does correspond to `B` as the batch 
        dimension iterates over both the samples samples and the cameras inside one sample.

        Note that for nested lists, the outer list iterates over samples and the inner list is over cameras.

        Args:
            gt_bboxes2d_list: Ground truth bboxes for each image
                Nested list of tensors with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
                The outer list is over samples and the inner list is over cameras.
            gt_labels2d_list: Ground truth class indexes for each image
                Nested list of tensors with shape (num_gts, ).
                The outer list is over samples and the inner list is over cameras.
            centers2d: Center coordinates of all points on the feature map
                Nested list of tensors with shape (num_gts, 2).
                The outer list is over samples and the inner list is over cameras.
            depths: Depth coordinates of all points on the feature map
                Nested list of tensors with shape (num_gts).
                The outer list is over samples and the inner list is over cameras.
            preds_dicts: Dictionary containing the following tensors:
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (B, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4).
                pred_centers2d (Tensor): Predicted center coordinates of all points on the feature map
                    Shape: (B, h*w, 2).
                centerness (Tensor): Centerness of all points on the feature map
                    Shape: (B, h*w, 1).
                topk_indexes (Tensor): Indexes of the topk elements in the sample weight
                    Shape: (num_samples, num_sample_tokens, 1).
            img_metas: Image meta information.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            loss_dict: A dictionary of loss components.
        """

        # @NOTE 
        # For an overview of the batching strategy used here, please see the note at the 
        # beginning of the file.

        # ===== Pre-processing (`loss()`) =====
        nvtx_wrp.range_push("FocalHeadBatched.loss")
        stopwatch.start_meas("FocalHeadBatched.loss")

        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        pred_centers2d = preds_dicts['pred_centers2d']
        centerness = preds_dicts['centerness']
        
        #@NOTE 
        # For each type of ground truth, we combine the data from all samples and cameras
        # into single RaggedBatch objects using the batching-helpers (bhl) package. This differs 
        # from the original implementation, where lists of tensors are used and the individual
        # elements in the lists are processed individually.
        #
        # Note that the `flatten_batch_dims` flag is set to `True` to flatten the batch dimensions 
        # (corresponding to samples and cameras). This is done as the distinction between the samples and 
        # cameras is not needed for the loss computation, as the loss is accumulated over both.
        gt_labels2d = bhl.combine_data(gt_labels2d_list, flatten_batch_dims=True)
        gt_bboxes_2d = bhl.combine_data(gt_bboxes2d_list, gt_labels2d, flatten_batch_dims=True)
        gt_centers2d = bhl.combine_data(centers2d, gt_labels2d, flatten_batch_dims=True)
        gt_depths = bhl.combine_data(depths, gt_labels2d, flatten_batch_dims=True)

        # ===== Compute batched loss =====
        #@NOTE 
        # The actual loss computation is done in the `loss_batched()` function, using the combined data.
        # Please see the notes there for more details on the approach. The main difference to the original 
        # implementation is that the ground truth data is batched. Note that while there, the called function
        # is called `loss_single()`, it also called once per batch and is provided with the data of all 
        # samples and cameras (with the GT data as lists containing per-sample and per-camera tensors).
        enc_loss_cls, enc_losses_bbox, enc_losses_iou, centers2d_losses, centerness_losses = \
            self.loss_batched(enc_cls_scores, enc_bbox_preds, pred_centers2d, centerness,
                              gt_bboxes_2d, gt_labels2d, gt_centers2d, gt_depths, img_metas)

        # ===== Assemble results =====
        #@NOTE: This is similar to the original implementation.
        loss_dict = dict()
        loss_dict['enc_loss_cls'] = enc_loss_cls
        loss_dict['enc_loss_bbox'] = enc_losses_bbox
        loss_dict['enc_loss_iou'] = enc_losses_iou
        loss_dict['centers2d_losses'] = centers2d_losses
        loss_dict['centerness_losses'] = centerness_losses
    
        nvtx_wrp.range_pop("FocalHeadBatched.loss")
        stopwatch.end_meas("FocalHeadBatched.loss")

        return loss_dict


    #@NOTE: Implementation of the batched loss computation.
    def loss_batched(self,
                    cls_scores: torch.Tensor,
                    bbox_preds: torch.Tensor,
                    pred_centers2d: torch.Tensor,
                    centerness: torch.Tensor,
                    gt_bboxes: bhl.RaggedBatch,
                    gt_labels: bhl.RaggedBatch,
                    gt_centers2d: bhl.RaggedBatch,
                    gt_depths: bhl.RaggedBatch,
                    img_metas: List[dict],
                    gt_bboxes_ignore_list=None):
        """"Optimized loss computation.

        Args:
            cls_scores: Classification scores of points on encode feature map
                Shape: (B, h*w, num_classes).
            bbox_preds: Regression results of each points on the encode feature map
                Shape: (B, h*w, 4).
            pred_centers2d: Predicted center coordinates of all points on the feature map
                Shape: (B, h*w, 2).
            centerness: Centerness of all points on the feature map
                Shape: (B, h*w, 1).
            gt_bboxes: Ground truth bounding boxes
                Shape: (B, max_num_gts, 4).
                Note that `max_num_gts` is the maximum number of ground truth objects for the samples in 
                the batch.
            gt_labels: Ground truth class indexes
                Shape: (B, max_num_gts).
                Note that `max_num_gts` is the maximum number of ground truth objects for the samples in 
                the batch.
            gt_centers2d: Ground truth center coordinates
                Shape: (B, max_num_gts, 2).
                Note that `max_num_gts` is the maximum number of ground truth objects for the samples in 
                the batch.
            gt_depths: Ground truth depths. Unuses in the loss computation.
                Kept for consistency with the original implementation (where they are passed to
                `get_targets()` and `get_targets_single()`, but also unused there).
                Shape: (B, max_num_gts).
                Note that `max_num_gts` is the maximum number of ground truth objects for the samples in 
                the batch. 
            img_metas: Image meta information
            gt_bboxes_ignore_list: List of ignored ground truth bounding boxes
                Note that ignore lists are not implemented and passing non-None values will 
                raise an error. This is consistent with the original implementation, where the 
                HungarianAssigner2D does not support ignore lists.
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """

        # ===== Pre-processing (`loss_batched()`) =====
        assert gt_bboxes_ignore_list is None, (
            'Using gt_bboxes_ignore_list is not supported. This is consistent with the original '
            'implementation, where setting ignore lists raises an error in the matcher.')
        
        num_imgs = cls_scores.size(0)
        num_preds = bbox_preds.size(1)

        #@NOTE 
        # Also see notes at the beginning of the file for an overview of the batched loss implementation 
        # approach.

        #@NOTE 
        # In the original implementation, the prediction tensors are split into individual samples and 
        # cameras to match the per-entry ground truth and perform the matching on a per-entry basis. 
        # Here, this step is omitted. The ground truth data is (already) batched instead, and we can directly 
        # proceed to the matching step below.

        # ===== Perform the matching =====
        #@NOTE 
        # In the original implementation, the assigner is called within `get_targets()`, which also performs
        # additional data processing. Here, we call the assigner directly since we need to return the matching 
        # indices. The batched assigner returns `RaggedBatch` instances containing indices for matched 
        # predictions and the corresponding GT bounding boxes. This is a modular approach, so that the 
        # matching does not need to be concerned with the format that the loss computation expects (beyond the
        # matching indices). This approach allows us to:
        #   - Maintain a clear separation of concerns between matching and loss computation
        #   - Use `batching-helpers` functionality for any additional processing of the data based on the 
        #     matching results later if/as needed depending on the computed loss type (e.g., depending on 
        #     whether the loss needs all predictions or only the matched pairs).
        pred_match_indices, gt_match_indices = self.assigner2d.assign(
            bbox_preds, cls_scores, pred_centers2d, gt_bboxes, gt_labels, gt_centers2d, img_metas
        )

        # DETR regress the relative position of boxes (cxcywh) in the image,
        # thus the learning target is normalized by the image size. So here
        # we need to re-scale them for calculating IoU loss
        # construct factors used for rescale bboxes
        # ===== Normalization factors =====
        img_h, img_w, _ = img_metas[0]['pad_shape'][0]

        #@NOTE 
        # The normalization factors are the same for all bboxes (both for GT and predictions). Therefore,
        # the first 2 dimensions have the size 1 and are implicitly broadcasted to the correct size
        # when applied to the bboxes. This is a difference from the original implementation mainly aimed at 
        # improving clarity and conciseness.
        factors = bbox_preds.new_tensor([img_w, img_h, img_w, img_h]).reshape(1,1,4)

        # ===== Align data by matches =====
        nvtx_wrp.range_push("Indexing [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # Get the aligned (i.e. indexed by indices established in the matching) versions of the 
        # bounding boxes. Note that in constrast to the targets in the original implementation 
        # (which are returned by the `get_targets()` function), the GT bboxes are still in the xyxy 
        # format and are not normalized. The normalization is done later for the matched targets only.
        #  
        # Here, aligned means that only the elements which have matches are preserved, and the indices
        # of the predictions and the mapped GT elements are the same, allowing for direct comparison of the 
        # tensors containing predictions to tensors containing GT.
        bboxes_preds_normed_matched = bhl.batched_indexing_access(bbox_preds, pred_match_indices)
        bboxes_gt_nonnormed_matched = bhl.batched_indexing_access(gt_bboxes, gt_match_indices)
        #@NOTE 
        # Similar to the bounding boxes, get the aligned versions of the centers.
        centers2d_gt_nonnormed_matched = bhl.batched_indexing_access(gt_centers2d, gt_match_indices)
        centers2d_pred_normed_matched = bhl.batched_indexing_access(pred_centers2d, pred_match_indices)

        #@NOTE 
        # Align/map the GT labels to the predictions. This is done by creating a tensor corresponding to the 
        # predictions in size and filling it with the GT labels according to the matching results (i.e. 
        # matched indices of both predictions and GT). This is different from the alignment above, as the 
        # classification loss is computed for all predictions, not only for matched predicitons. Therefore, 
        # all predictions need to be preserved, and the GT labels need to be mapped to them.
        #
        # Note that this corresponds to the mapping done in the original implementation by the 
        # `get_targets()` function.
        # 
        # As in the original implementation, the filler values corresponding to non-matched predictions are 
        # set to the number of classes, which is a special label indicating non-existence and is understood as 
        # such by the classification loss function implementation.
        gt_labels_mapped_to_pred = torch.full(
            (num_imgs, num_preds), fill_value=self.cls_out_channels, dtype=gt_labels.dtype, 
            device=gt_labels.device
        )
        #@NOTE: Insert the GT labels into the positions corresponding to the matched predictions.
        gt_labels_mapped_to_pred = bhl.batched_index_mapping(
            gt_labels, gt_match_indices, pred_match_indices, gt_labels_mapped_to_pred
        )
        nvtx_wrp.range_pop()

        # ===== Pre-process tensors =====
        nvtx_wrp.range_push("bboxes preproc. [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # In constrast to the bboxes in the original implementation (which are returned by the 
        # `get_targets()` function), the GT bboxes are still in the `xyxy` format and are not normalized. 
        # The predictions are in `xywh` and normalized format (same as original implementation). We need both 
        # GT and predictions in both formats. Here, we convert the normalized predictions to non-normalized 
        # format and the non-normalized GT bboxes to normalized format.
        # 
        # Note the `apply()` method, which is a convenience method for applying a function to 
        # the data in the RaggedBatch object. While here, only the data tensor (i.e. `tensor`) is input to the
        # lambda function, the passed function may accept more inputs, which allows access to the 
        # sample sizes and related mask (see `RaggedBatch.apply()` documentation for more details).
        bboxes_preds_nonnormed_matched = bboxes_preds_normed_matched.apply(
            lambda data: bbox_cxcywh_to_xyxy(data) * factors
        )
        bboxes_gt_normed_matched = bboxes_gt_nonnormed_matched.apply(
            lambda data: bbox_xyxy_to_cxcywh(data) / factors
        )
        nvtx_wrp.range_pop()

        nvtx_wrp.range_push("centers preproc [FocalHeadBatched.loss_batched]")
        #centerness
        #@NOTE 
        # Compute the normalized versions of the centers. Note that in the original implementation,
        # normalized centers are returned by the `get_targets()` function, so that no additional 
        # normalization is needed there.
        centers2d_gt_normed_matched = centers2d_gt_nonnormed_matched.apply(
            lambda ctr2d: ctr2d / factors[..., 0:2]
        )
        nvtx_wrp.range_pop()
        
        # ===== Existence weights & count valid and invalid elements =====
        nvtx_wrp.range_push("valid elements preproc [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # Get weights differentiating between valid (weight: 1.0) and invalid (weight: 0.0) elements.
        # In the original implementation, this is returned by the `get_targets()` function (in a different
        # format/shape, corresponding to the batching approach used there).
        bbox_weights_matched = bboxes_gt_nonnormed_matched.get_existence_weights(torch.float32)
        #@NOTE 
        # Get the total number of valid (i.e. existent, matched to GT) and invalid (i.e. non-existent, 
        # non-matched) elements. In the original implementation, this information is provided by the 
        # `get_targets()` function.
        num_total_pos = bboxes_gt_nonnormed_matched.total_num_entries
        num_total_neg = num_preds * num_imgs - num_total_pos
        nvtx_wrp.range_pop()

        # ===== Losses: IoU & IoU score =====
        # regression IoU loss, defaultly GIoU loss
        nvtx_wrp.range_push("loss_iou [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # Compute the IoU loss between predicted and ground truth bounding boxes. Unlike the original 
        # implementation, we compute this loss only for matched predictions. The original approach computes 
        # loss for all predictions (with GT targets aligned to predictions) and uses zero weights for 
        # unmatched predictions. Our implementation still contains filler values in the data tensors (with 
        # corresponding zero weights), but the number of fillers is minimized, and only as many fillers as 
        # needed to maintain uniform sample sizes (see `RaggedBatch` docs) are used.
        #
        # Note that in contrast to the original implementation, the inputs to this loss computation do 
        # include a batch dimension. The loss function handles this automatically and now reduces over both
        # the bounding boxes and over the entries in the batch dimension.
        loss_iou = self.loss_iou2d(
            bboxes_preds_nonnormed_matched.tensor, 
            bboxes_gt_nonnormed_matched.tensor, 
            bbox_weights_matched, 
            avg_factor=num_total_pos,
        )
        
        nvtx_wrp.range_pop()    

        nvtx_wrp.range_push("iou_score [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # Compute the IoU score between predicted and ground truth bounding boxes.
        # The same function is used as in the original implementation. Note that the input tensors
        # have a batch dimension in this implementation, while there is no batch dimension in the original 
        # implementation (and a sequence of all bounding boxes from all samples and cameras is used instead).
        # However, this function can handle both cases and does not need to be modified.
        iou_score = bbox_overlaps(
            bboxes_gt_nonnormed_matched.tensor, bboxes_preds_nonnormed_matched.tensor, is_aligned=True
        )
        nvtx_wrp.range_pop()

        # ===== Losses: Classification (setup & weights) =====
        nvtx_wrp.range_push("cls pre-proc [FocalHeadBatched.loss_batched]")
        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                cls_scores.new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)
        nvtx_wrp.range_pop()

        nvtx_wrp.range_push("labels pre-proc [FocalHeadBatched.loss_batched]")
        #@NOTE: Flatten `gt_labels_mapped_to_pred` as expected by the loss function.
        gt_labels_mapped_to_pred_flat = gt_labels_mapped_to_pred.view(-1)
        #@NOTE: Create a tensor of ones as the label weights (consistent with the original implementation).
        label_weights = torch.ones(
            gt_labels_mapped_to_pred_flat.shape, dtype=torch.float32, device = gt_labels.device
        )
        nvtx_wrp.range_pop()

        nvtx_wrp.range_push("iou score pre-proc [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # For the classification loss, we need to weight the predictions with the IoU score (for existent 
        # predictions). As the score is computed only for matched predictions, we need to map it back to match 
        # the original prediction tensor. This is done by using `batched_inverse_indexing_access()`. This is 
        # different from the original implementation, where all the data is mapped to the predictions inside 
        # the `get_targets()` function, and the IoU scores for the individual predictions, which are computed 
        # afterwards, are already located at the indices corresponding to the original prediction tensor. 
        #
        # Note that that for non-existent predictions, the IoU score is set to 0 in 
        # `batched_inverse_indexing_access()`. Again, this is consistent with the original implementation, and 
        # these values are ignored in any case.
        #
        # Also note that `batched_inverse_indexing_access()` creates a new tensor with constant values and
        # then fills the values at the indexed locations with the values from the original tensor. 
        # There is an alternative implementation (`batched_index_write()`) which fills the values
        # into an already existing tensor (analogous to `__setitem__()`). Please refer to the 
        # documentation of `batching-helpers` for more details and choose the appropriate method based 
        # on the use case.
        iou_score_mapped_to_pred = bhl.batched_inverse_indexing_access(
            iou_score, pred_match_indices, num_preds, filler_value=0.0
        )
        #@NOTE flatten the mapped IoU scores as needed by the loss function implementation.
        iou_score_mapped_to_pred_flat = iou_score_mapped_to_pred.view(-1)
        nvtx_wrp.range_pop()
        
        # ===== Losses: Classification =====
        nvtx_wrp.range_push("loss_cls [FocalHeadBatched.loss_batched]")
        #@NOTE
        # Compute the classification loss (for all predictions, and with the IoU scores as weights for 
        # existent predictions). After the inputs were flattened, the inputs to the loss function correspond
        # to the inputs in the original implementation, and the same loss functions can be used here.
        # Whether this flattening is needed or not (as in the IoU loss above) depends on the loss function 
        # implementation.
        loss_cls = self.loss_cls2d(
            cls_scores, 
            (gt_labels_mapped_to_pred_flat, iou_score_mapped_to_pred_flat.detach()), 
            label_weights, 
            avg_factor=cls_avg_factor,
        )
        nvtx_wrp.range_pop()

        # ===== Synchronize positives =====
        nvtx_wrp.range_push("num pos preproc. [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # The following lines are as in the original implementation and are done for cross-device 
        # synchronization.
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
        nvtx_wrp.range_pop()

        # ===== Heatmap generation =====
        nvtx_wrp.range_push("Heatmap generation[FocalHeadBatched.loss_batched]")
        img_shape = img_metas[0]['pad_shape'][0]
        img_h, img_w, _ = img_shape
        #@NOTE: Get the scaled centers and radii of the GT boxes using the `apply()` convenience method.
        gt_centers_heatmap, gt_radii_heatmap = centers2d_gt_nonnormed_matched.apply(
            lambda centers: self._get_scaled_centers_and_radii(centers, bboxes_gt_nonnormed_matched.tensor)
        )
        #@NOTE: Create the heatmap tensor (to be filled with the individual gaussian heatmaps).
        heatmaps = torch.zeros(
            (num_imgs, img_h // self.stride, img_w // self.stride), 
            dtype=torch.float32, 
            device=gt_centers_heatmap.device
        )
        nvtx_wrp.range_push("draw_heatmap call [FocalHeadBatched.loss_batched]")
        #@NOTE
        # Draw the heatmaps using `draw_heatmap_batched`. This function accepts
        # the data in the `RaggedBatch` object. Compared to the reference, this simplifies the implementation 
        # as we do not need to manually create additional tensors indicating to which image the individual 
        # bboxes belong. 
        #
        # Please compare to the heatmap generation implementation in `FocalHead.loss_single()` (and see 
        # the related noted there). There, we also add the option to use the custom op implementation of the 
        # heatmap generation. However, we need to use `draw_heatmap()` and manually construct a tensor 
        # indicating which bounding box is from which entry in the batch (i.e. which image in which sample), 
        # as the batch dimensions is flattened away and so this information is not available from the data 
        # tensors alone.
        draw_heatmap_batched(heatmaps, gt_centers_heatmap, gt_radii_heatmap)
        nvtx_wrp.range_pop()
        nvtx_wrp.range_pop()

        # ===== Losses: Centerness =====
        nvtx_wrp.range_push("loss_centerness [FocalHeadBatched.loss_batched]")
        centerness = clip_sigmoid(centerness)
        #@NOTE
        # Compute the centerness loss. Note that similar to the classification loss, 
        # the centerness loss is computed for all predictions.
        # The GT targets are not used directly here, and instead, the previously generated heatmap is used. 
        # The predictions correspond to individual pixels in the heatmap, so that they can be directly 
        # compared to the heatmap.
        #
        # As the non-uniformly-sized GT is not directly used for this loss, the loss computation 
        # is done as in the original implementation and no changes are needed to the format of the 
        # input data.
        loss_centerness = self.loss_centerness(
                centerness,
                heatmaps.view(num_imgs, -1, 1),
                avg_factor=max(num_total_pos, 1))
        nvtx_wrp.range_pop() 

        # ===== Losses: BBox and Centers2D =====
        # regression L1 loss
        nvtx_wrp.range_push("loss_bbox [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # Compute the bounding box loss. Similar to `loss_iou2d`, this loss is computed only for the 
        # predictions which have a match.
        loss_bbox = self.loss_bbox2d(
            bboxes_preds_normed_matched.tensor, 
            bboxes_gt_normed_matched.tensor, 
            bbox_weights_matched, 
            avg_factor=num_total_pos,
        )
        nvtx_wrp.range_pop()     

        pred_centers2d = pred_centers2d.view(-1, 2)
        # centers2d L1 loss
        nvtx_wrp.range_push("loss_centers2d [FocalHeadBatched.loss_batched]")
        #@NOTE 
        # Compute the centers2d loss. Note that similar to `loss_iou2d` and `loss_bbox`, this loss is computed
        # only for the existent/matched predictions.
        loss_centers2d = self.loss_centers2d(
            centers2d_pred_normed_matched.tensor, 
            centers2d_gt_normed_matched.tensor, 
            bbox_weights_matched[..., 0:2], 
            avg_factor=num_total_pos,
        )
        nvtx_wrp.range_pop()

        # ===== Finalize =====
        nvtx_wrp.range_pop("FocalHeadBatched.loss_batched")
        return loss_cls, loss_bbox, loss_iou, loss_centers2d, loss_centerness


    def _get_scaled_centers_and_radii(self, obj_centers2d, obj_bboxes):
        nvtx_wrp.range_push("FocalHeadBatched._get_scaled_centers_and_radii")
        l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
        t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
        r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
        b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
        bound = torch.cat([l, t, r, b], dim=-1)
        radii = torch.ceil(torch.min(bound, dim=-1)[0] / self.stride)
        radii = torch.clamp(radii, 1)
        centers = obj_centers2d / self.stride
        res = centers.to(torch.int32), radii.to(torch.int32)
        nvtx_wrp.range_pop()
        return res
    

    def get_targets(self, *args, **kwargs):
        '''
        This method is not implemented for the `FocalHeadBatched` class. 
        It is not needed internally and will result in an error if called from outside.
        It is included only to satisfy the interface expected by the base class.
        '''
        raise NotImplementedError("FocalHeadBatched does not use/implement `get_targets()`")
