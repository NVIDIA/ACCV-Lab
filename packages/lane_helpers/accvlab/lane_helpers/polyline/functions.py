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

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .. import _polyline_sampling

if TYPE_CHECKING:
    from accvlab.batching_helpers import RaggedBatch


def interpolate(points: torch.Tensor, distances: torch.Tensor, *, relative: bool = False) -> torch.Tensor:
    """Interpolate batched polylines at requested distances.

    Args:
        points: CPU or CUDA tensor with shape ``(batch, num_points, num_dims)``.
        distances: Tensor with shape ``(batch, num_distances)`` on the same device as ``points``.
            Distances below zero are clamped to the first point of the polyline. Distances beyond the
            total polyline length are clamped to the last point. When ``relative=True``, this corresponds
            to clamping values below ``0`` and above ``1``.
        relative: If ``True``, interpret ``distances`` as fractions of each polyline's total length.
            If ``False``, interpret them as absolute distances from the start of each polyline.

    Returns:
        Tensor with shape ``(batch, num_distances, num_dims)`` on the same device as ``points``.
    """
    result = _polyline_sampling.polyline_interpolation(points, distances, relative=relative)
    return result


def lengths(points: torch.Tensor) -> torch.Tensor:
    """Compute the total length of each polyline in a fixed-size batch.

    Args:
        points: CPU or CUDA tensor with shape ``(batch, num_points, num_dims)``.

    Returns:
        Tensor with shape ``(batch,)`` on the same device as ``points``.
    """
    result = _polyline_sampling._polyline_lengths(points)
    return result


def interpolate_var_size_batch(
    points: RaggedBatch, distances: RaggedBatch, *, relative: bool = False
) -> RaggedBatch:
    """Interpolate variable-length batched polylines at requested distances.

    Args:
        points: RaggedBatch-like object with tensor data on CPU or CUDA and shape
            ``(batch, max_num_points, num_dims)``.
        distances: RaggedBatch-like object with shape ``(batch, max_num_distances)`` and tensor data
            on the same device as ``points``. Distances below zero are clamped to the first point of the
            polyline. Distances beyond the total polyline length are clamped to the last point. When
            ``relative=True``, this corresponds to clamping values below ``0`` and above ``1``.
        relative: If ``True``, interpret ``distances`` as fractions of each polyline's total length.
            If ``False``, interpret them as absolute distances from the start of each polyline.

    Returns:
        RaggedBatch-like object with shape ``(batch, max_num_distances, num_dims)`` and tensor data
        on the same device as ``points``.
    """
    assert points.num_batch_dims == 1, "points must have exactly one batch dimension"
    assert distances.num_batch_dims == 1, "distances must have exactly one batch dimension"
    assert (
        points.non_uniform_dim == 1
    ), "points.non_uniform_dim must be 1 for shape (batch, max_num_points, num_dims)"
    assert (
        distances.non_uniform_dim == 1
    ), "distances.non_uniform_dim must be 1 for shape (batch, max_num_distances)"

    result = _polyline_sampling._polyline_interpolation_var_size_batch(
        points.tensor,
        distances.tensor,
        points.sample_sizes,
        distances.sample_sizes,
        relative=relative,
    )
    result_batch = distances.create_with_sample_sizes_like_self(result)
    return result_batch


def lengths_var_size_batch(points: RaggedBatch) -> torch.Tensor:
    """Compute the total length of each polyline in a variable-size batch.

    Args:
        points: RaggedBatch-like object with tensor data on CPU or CUDA and shape
            ``(batch, max_num_points, num_dims)``.

    Returns:
        Tensor with shape ``(batch,)`` on the same device as ``points``.
    """
    assert points.num_batch_dims == 1, "points must have exactly one batch dimension"
    assert (
        points.non_uniform_dim == 1
    ), "points.non_uniform_dim must be 1 for shape (batch, max_num_points, num_dims)"
    result = _polyline_sampling._polyline_lengths_var_size_batch(points.tensor, points.sample_sizes)
    return result
