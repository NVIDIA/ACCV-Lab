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

import pytest
import torch

from accvlab.batching_helpers import RaggedBatch
from accvlab.lane_helpers import polyline

from polyline_test_utils import (
    DEVICES,
    make_padded_ragged_polyline_case,
    make_random_ragged_polyline_case,
    polyline_lengths_cpu,
    polyline_lengths_var_size_cpu,
)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_rectangle_and_single_point(device: str):
    rectangle = torch.tensor(
        [
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 2.0],
                [0.0, 2.0],
                [0.0, 0.0],
            ]
        ],
        device=device,
        dtype=torch.float32,
    )
    single_point = torch.tensor([[[1.0, 2.0]]], device=device, dtype=torch.float32)

    assert torch.allclose(polyline.lengths(rectangle).cpu(), torch.tensor([6.0]), atol=1e-5, rtol=0.0)
    assert torch.allclose(polyline.lengths(single_point).cpu(), torch.tensor([0.0]), atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_zero_point_batch_returns_nan(device: str):
    points = torch.empty((3, 0, 2), device=device, dtype=torch.float32)

    result = polyline.lengths(points)

    assert result.shape == (3,)
    assert torch.isnan(result).all()


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_random_nd_matches_cpu_reference(device: str):
    generator = torch.Generator().manual_seed(1)
    num_iters = 100
    for _ in range(num_iters):
        points_cpu = torch.rand((5, 37, 4), generator=generator, dtype=torch.float32)

        expected = polyline_lengths_cpu(points_cpu)
        result = polyline.lengths(points_cpu.to(device))

        assert torch.allclose(result.cpu(), expected, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_accepts_non_contiguous_points(device: str):
    points_storage = torch.tensor(
        [
            [[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 2.0, 2.0]],
            [[2.0, 3.0, 3.0, 2.0], [2.0, 2.0, 4.0, 4.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    points = points_storage.transpose(1, 2)
    assert not points.is_contiguous()

    result = polyline.lengths(points)
    expected = polyline_lengths_cpu(points.cpu())

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_var_size_batch_random_matches_cpu_reference(device: str):
    num_iters = 100
    for i in range(num_iters):
        points_batch_cpu, _ = make_random_ragged_polyline_case(seed=i * 100)
        points_batch = points_batch_cpu.to(device)

        result = polyline.lengths_var_size_batch(points_batch)
        expected = polyline_lengths_var_size_cpu(points_batch_cpu.tensor, points_batch_cpu.sample_sizes)

        assert result.shape == (points_batch.tensor.shape[0],)
        assert torch.allclose(result.cpu(), expected, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_var_size_batch_matches_cpu_reference_and_ignores_padding(device: str):
    points_batch, _ = make_padded_ragged_polyline_case(device)

    result = polyline.lengths_var_size_batch(points_batch)
    expected = polyline_lengths_var_size_cpu(points_batch.tensor.cpu(), points_batch.sample_sizes.cpu())

    assert result.shape == (4,)
    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_var_size_batch_zero_point_row_returns_nan(device: str):
    points = torch.tensor(
        [
            [[9999.0, 9999.0], [9999.0, 9999.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 3.0], [9999.0, 9999.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    sample_sizes = torch.tensor([0, 2, 1], device=device, dtype=torch.int32)

    result = polyline.lengths_var_size_batch(RaggedBatch(points, sample_sizes=sample_sizes))
    expected = polyline_lengths_var_size_cpu(points.cpu(), sample_sizes.cpu())

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0, equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_var_size_batch_all_zero_point_rows_return_nan(device: str):
    points = torch.empty((3, 0, 2), device=device, dtype=torch.float32)
    sample_sizes = torch.zeros(3, device=device, dtype=torch.int32)

    result = polyline.lengths_var_size_batch(RaggedBatch(points, sample_sizes=sample_sizes))

    assert result.shape == (3,)
    assert torch.isnan(result).all()


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_var_size_batch_accepts_int32_sample_sizes_and_non_contiguous_points(device: str):
    points_storage = torch.tensor(
        [
            [[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 2.0, 2.0]],
            [[2.0, 3.0, 3.0, 2.0], [2.0, 2.0, 4.0, 4.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    points = points_storage.transpose(1, 2)
    sample_sizes = torch.tensor([4, 3], device=device, dtype=torch.int32)
    assert not points.is_contiguous()

    result = polyline.lengths_var_size_batch(RaggedBatch(points, sample_sizes=sample_sizes))
    expected = polyline_lengths_var_size_cpu(points.cpu(), sample_sizes.cpu())

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


def test_polyline_lengths_var_size_batch_handles_inactive_cuda_rows():
    num_samples = 33
    points = torch.empty((num_samples, 2, 2), device="cuda", dtype=torch.float32)
    points[:, 0, 0] = torch.arange(num_samples, device="cuda", dtype=torch.float32)
    points[:, 0, 1] = 0.0
    points[:, 1, 0] = points[:, 0, 0] + 1.0
    points[:, 1, 1] = 0.0
    sample_sizes = torch.full((num_samples,), 2, device="cuda")

    result = polyline.lengths_var_size_batch(RaggedBatch(points, sample_sizes=sample_sizes))

    assert torch.allclose(result.cpu(), torch.ones(num_samples), atol=1e-5, rtol=0.0)
