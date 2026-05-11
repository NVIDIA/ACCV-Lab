from __future__ import annotations

# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from typing import TYPE_CHECKING

import torch
from accvlab.draw_heatmap.draw_heatmap_ext import draw_heatmap_batched_impl
from accvlab.draw_heatmap.draw_heatmap_ext import draw_heatmap_batched_classwise_impl

if TYPE_CHECKING:
    from accvlab.batching_helpers import RaggedBatch


def draw_heatmap_batched(
    heatmap: torch.Tensor,
    centers: RaggedBatch,
    radii: RaggedBatch,
    diameter_to_sigma_factor: float = 6.0,
    k_scale: float = 1.0,
    labels: RaggedBatch = None,
):
    '''
    Draws heatmaps for a batch of samples.

    :gpu:

    Args:
        heatmap: Tensor of shape (batch_size, height, width) when labels is None.
            Otherwise with shape (batch_size, max_num_classes, height, width).
            The heatmap will be modified in place.
        centers: RaggedBatch of shape (batch_size, max_num_targets, 2).
            The centers of the heatmaps to draw. `max_num_targets` is the maximum number of targets across the batch.
        radii: RaggedBatch of shape (batch_size, max_num_targets).
            The radii of the heatmaps to draw. `max_num_targets` is the maximum number of targets across the batch.
        diameter_to_sigma_factor: Factor for converting diameter to sigma.
        k_scale: Scale factor for the Gaussian kernel
        labels: RaggedBatch of shape (batch_size, max_num_targets).
            The labels are denoted as the class index. `max_num_targets` is the maximum number of targets across the batch.
            If None, all classes of the sample will be drawn in one heatmap.
    '''
    centers_tensor = centers.tensor
    radii_tensor = radii.tensor
    assert (
        centers_tensor.shape[0] == radii_tensor.shape[0]
    ), "centers and radii must have the same size batch size"
    assert (
        centers_tensor.shape[1] == radii_tensor.shape[1]
    ), "centers and radii must have the same maximum number of objects"
    # TODO: This conversion can be replaced by type dispatching in the C++ implementation
    nums_targets = centers.sample_sizes.to(torch.int32)
    if labels is None:
        draw_heatmap_batched_impl(
            heatmap, centers_tensor, radii_tensor, nums_targets, diameter_to_sigma_factor, k_scale
        )
    else:
        labels_tensor = labels.tensor
        assert (
            centers_tensor.shape[0] == labels_tensor.shape[0]
        ), "centers and labels must have the same size batch size"
        assert (
            centers_tensor.shape[1] == labels_tensor.shape[1]
        ), "centers and labels must have the same maximum number of objects"
        draw_heatmap_batched_classwise_impl(
            heatmap,
            centers_tensor,
            radii_tensor,
            nums_targets,
            labels_tensor,
            diameter_to_sigma_factor,
            k_scale,
        )
