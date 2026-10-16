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

from pathlib import Path
import sys

import pytest
import torch


pytest.importorskip("mmdet")
bhl = pytest.importorskip("accvlab.batching_helpers")

STREAM_PETR_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STREAM_PETR_ROOT))

from mmdet.core import build_assigner, build_sampler  # noqa: E402
from mmdet.models import build_loss  # noqa: E402
from projects.mmdet3d_plugin.models.dense_heads.focal_head import FocalHead  # noqa: E402
from projects.mmdet3d_plugin.models.dense_heads.focal_head_batched import (  # noqa: E402
    FocalHeadBatched,
)
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head import (  # noqa: E402
    StreamPETRHead,
)
from projects.mmdet3d_plugin.models.dense_heads.streampetr_head_batched import (  # noqa: E402
    StreamPETRHeadBatched,
)


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extensions are required")


def _make_focal_head(head_type, assigner_type, cost_suffix=""):
    head = head_type.__new__(head_type)
    torch.nn.Module.__init__(head)
    head.num_classes = 3
    head.cls_out_channels = 3
    head.stride = 4
    head.bg_cls_weight = 0
    head.sync_cls_avg_factor = False
    head.fp16_enabled = False
    head._use_cust_op_heatmap = True
    head.assigner2d = build_assigner(
        dict(
            type=assigner_type,
            cls_cost=dict(type=f"FocalLossCost{cost_suffix}", weight=2.0),
            reg_cost=dict(type=f"BBoxL1Cost{cost_suffix}", weight=5.0, box_format="xywh"),
            iou_cost=dict(type=f"IoUCost{cost_suffix}", iou_mode="giou", weight=2.0),
            centers2d_cost=dict(type="BBox3DL1Cost", weight=10.0),
        )
    )
    head.sampler = build_sampler(dict(type="PseudoSampler"), context=head)
    head.loss_cls2d = build_loss(
        dict(type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=2.0)
    )
    head.loss_centerness = build_loss(
        dict(type="GaussianFocalLoss", reduction="mean", loss_weight=1.0)
    )
    head.loss_bbox2d = build_loss(dict(type="L1Loss", loss_weight=5.0))
    head.loss_iou2d = build_loss(dict(type="GIoULoss", loss_weight=2.0))
    head.loss_centers2d = build_loss(dict(type="L1Loss", loss_weight=10.0))
    return head.cuda()


def _clone_with_grad(tensors):
    return tuple(tensor.detach().clone().requires_grad_() for tensor in tensors)


def _make_streampetr_head(head_type, assigner_type, cost_suffix=""):
    head = head_type.__new__(head_type)
    torch.nn.Module.__init__(head)
    head.num_classes = 3
    head.cls_out_channels = 3
    head.bg_cls_weight = 0
    head.sync_cls_avg_factor = False
    head.fp16_enabled = False
    head.pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    head.match_with_velo = False
    head.code_weights = torch.nn.Parameter(
        torch.tensor([2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        requires_grad=False,
    )
    head.match_costs = head.code_weights
    head.assigner = build_assigner(
        dict(
            type=assigner_type,
            cls_cost=dict(type=f"FocalLossCost{cost_suffix}", weight=2.0),
            reg_cost=dict(type="BBox3DL1Cost", weight=0.25),
            iou_cost=dict(type="IoUCost", weight=0.0),
            pc_range=head.pc_range,
        )
    )
    head.sampler = build_sampler(dict(type="PseudoSampler"), context=head)
    head.loss_cls = build_loss(
        dict(
            type="FocalLoss",
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0,
        )
    )
    head.loss_bbox = build_loss(dict(type="L1Loss", loss_weight=0.25))
    return head.cuda()


def test_focal_batched_losses_and_gradients_match_reference():
    torch.manual_seed(7)
    device = torch.device("cuda")
    batch_size = 3
    num_predictions = 16

    gt_bboxes = [
        torch.tensor([[1.0, 2.0, 7.0, 9.0], [8.0, 5.0, 14.0, 13.0]], device=device),
        torch.empty((0, 4), device=device),
        torch.tensor([[3.0, 4.0, 10.0, 12.0]], device=device),
    ]
    gt_labels = [
        torch.tensor([0, 2], dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.tensor([1], dtype=torch.int64, device=device),
    ]
    gt_centers = [
        torch.tensor([[4.0, 5.0], [11.0, 9.0]], device=device),
        torch.empty((0, 2), device=device),
        torch.tensor([[6.0, 8.0]], device=device),
    ]
    gt_depths = [torch.zeros(len(labels), device=device) for labels in gt_labels]
    img_metas = [{"pad_shape": [(16, 16, 3)]}]

    predictions = (
        torch.randn((batch_size, num_predictions, 3), device=device),
        torch.rand((batch_size, num_predictions, 4), device=device),
        torch.rand((batch_size, num_predictions, 2), device=device),
        torch.randn((batch_size, num_predictions, 1), device=device),
    )
    reference_predictions = _clone_with_grad(predictions)
    batched_predictions = _clone_with_grad(predictions)

    reference_head = _make_focal_head(FocalHead, "HungarianAssigner2D")
    batched_head = _make_focal_head(
        FocalHeadBatched, "HungarianAssigner2DBatched", cost_suffix="Batched"
    )

    reference_losses = reference_head.loss_single(
        *(prediction + 0 for prediction in reference_predictions),
        gt_bboxes,
        gt_labels,
        gt_centers,
        gt_depths,
        img_metas,
    )
    batched_losses = batched_head.loss_batched(
        *(prediction + 0 for prediction in batched_predictions),
        bhl.combine_data(gt_bboxes),
        bhl.combine_data(gt_labels),
        bhl.combine_data(gt_centers),
        bhl.combine_data(gt_depths),
        img_metas,
    )

    for actual, expected in zip(batched_losses, reference_losses):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    sum(reference_losses).backward()
    sum(batched_losses).backward()
    for actual, expected in zip(batched_predictions, reference_predictions):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-6)


def test_multilayer_3d_batched_losses_and_gradients_match_reference():
    torch.manual_seed(11)
    device = torch.device("cuda")
    num_dec_layers = 2
    batch_size = 3
    num_predictions = 12

    gt_bboxes = [
        torch.tensor(
            [
                [2.0, 3.0, 0.5, 1.8, 4.0, 1.6, 0.2, 0.0, 0.0],
                [-4.0, 1.0, -0.2, 2.2, 3.5, 1.4, -0.4, 0.1, -0.1],
            ],
            device=device,
        ),
        torch.empty((0, 9), device=device),
        torch.tensor(
            [[5.0, -2.0, 0.1, 1.6, 3.8, 1.5, 0.7, -0.2, 0.2]],
            device=device,
        ),
    ]
    gt_labels = [
        torch.tensor([0, 2], dtype=torch.int64, device=device),
        torch.empty((0,), dtype=torch.int64, device=device),
        torch.tensor([1], dtype=torch.int64, device=device),
    ]
    predictions = (
        torch.randn(
            (num_dec_layers, batch_size, num_predictions, 3), device=device
        ),
        torch.randn(
            (num_dec_layers, batch_size, num_predictions, 10), device=device
        ),
    )
    reference_predictions = _clone_with_grad(predictions)
    batched_predictions = _clone_with_grad(predictions)

    reference_head = _make_streampetr_head(StreamPETRHead, "HungarianAssigner3D")
    batched_head = _make_streampetr_head(
        StreamPETRHeadBatched,
        "HungarianAssigner3DBatched",
        cost_suffix="Batched",
    )

    reference_per_layer = [
        reference_head.loss_single(
            reference_predictions[0][layer],
            reference_predictions[1][layer],
            gt_bboxes,
            gt_labels,
        )
        for layer in range(num_dec_layers)
    ]
    reference_losses = tuple(
        torch.stack([layer_losses[loss_index] for layer_losses in reference_per_layer])
        for loss_index in range(2)
    )

    batched_gt_bboxes = bhl.combine_data(gt_bboxes).unsqueeze_batch_dim(0)
    batched_gt_labels = bhl.combine_data(gt_labels).unsqueeze_batch_dim(0)
    batched_gt_bboxes = batched_gt_bboxes.repeat_samples(num_dec_layers, batch_dim=0)
    batched_gt_labels = batched_gt_labels.repeat_samples(num_dec_layers, batch_dim=0)
    batched_losses = batched_head.loss_batched(
        *batched_predictions,
        batched_gt_bboxes,
        batched_gt_labels,
    )

    for actual, expected in zip(batched_losses, reference_losses):
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    sum(loss.sum() for loss in reference_losses).backward()
    sum(loss.sum() for loss in batched_losses).backward()
    for actual, expected in zip(batched_predictions, reference_predictions):
        torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-6, atol=1e-7)
