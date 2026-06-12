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

from polyline_test_utils import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_rejects_invalid_ragged_layout(device: str):
    points = torch.randn((2, 3, 4), device=device)
    distances = torch.randn((2, 4), device=device)
    points_batch = RaggedBatch(
        points.transpose(1, 2).contiguous(),
        sample_sizes=torch.tensor([3, 2], device=device, dtype=torch.int32),
        non_uniform_dim=2,
    )
    distances_batch = RaggedBatch(
        distances, sample_sizes=torch.tensor([4, 2], device=device, dtype=torch.int32)
    )

    # Polyline points must use dimension 1 as the non-uniform point dimension.
    with pytest.raises(AssertionError, match="points.non_uniform_dim"):
        polyline.interpolate_var_size_batch(points_batch, distances_batch)


@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_validates_inputs(device: str):
    points = torch.randn((1, 3, 2), device=device)
    distances = torch.randn((1, 4), device=device)
    points_batch = RaggedBatch(points, sample_sizes=torch.tensor([3], device=device))
    distances_batch = RaggedBatch(distances, sample_sizes=torch.tensor([4], device=device))

    # Points sample sizes must not be negative.
    bad_points_sizes = RaggedBatch(points, sample_sizes=torch.tensor([-1], device=device))
    with pytest.raises(RuntimeError, match="points.sample_sizes"):
        polyline.interpolate_var_size_batch(bad_points_sizes, distances_batch)

    # Distance sample sizes must not exceed the padded distance dimension.
    bad_distances_sizes = RaggedBatch(distances, sample_sizes=torch.tensor([5], device=device))
    with pytest.raises(RuntimeError, match="distances.sample_sizes"):
        polyline.interpolate_var_size_batch(points_batch, bad_distances_sizes)

    # Points and distances must have the same dtype.
    distances_double = distances_batch.double()
    with pytest.raises(RuntimeError, match="same dtype"):
        polyline.interpolate_var_size_batch(points_batch, distances_double)

    # Points and distances must have the same sample size dtype.
    mismatched_sample_size_dtype = RaggedBatch(
        distances,
        sample_sizes=torch.tensor([4], device=device, dtype=torch.int32),
    )
    with pytest.raises(RuntimeError, match="same dtype"):
        polyline.interpolate_var_size_batch(points_batch, mismatched_sample_size_dtype)


def test_polyline_functions_reject_mixed_cpu_cuda_inputs():
    points_cpu = torch.randn((1, 3, 2), device="cpu")
    distances_cpu = torch.randn((1, 4), device="cpu")
    points_cuda = points_cpu.cuda()
    distances_cuda = distances_cpu.cuda()

    # Fixed-size points and distances must live on the same device.
    with pytest.raises(RuntimeError, match="same device"):
        polyline.interpolate(points_cpu, distances_cuda)

    # Ragged points and distances must live on the same device.
    with pytest.raises(RuntimeError, match="same device"):
        polyline.interpolate_var_size_batch(
            RaggedBatch(points_cpu, sample_sizes=torch.tensor([3], device="cpu")),
            RaggedBatch(distances_cuda, sample_sizes=torch.tensor([4], device="cuda")),
        )

    # Ragged sample sizes must live on the same device as their data tensor.
    with pytest.raises(RuntimeError, match="same device"):
        polyline.interpolate_var_size_batch(
            RaggedBatch(points_cuda, sample_sizes=torch.tensor([3], device="cpu")),
            RaggedBatch(distances_cuda, sample_sizes=torch.tensor([4], device="cuda")),
        )

    # Lengths use only points, but points.sample_sizes must still match the points device.
    with pytest.raises(RuntimeError, match="same device"):
        polyline.lengths_var_size_batch(
            RaggedBatch(points_cuda, sample_sizes=torch.tensor([3], device="cpu"))
        )


def test_cpu_polyline_functions_reject_low_precision_dtypes():
    for dtype in (torch.float16, torch.bfloat16):
        # CPU kernels intentionally support only float32 and float64.
        points = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], dtype=dtype)
        distances = torch.tensor([[0.0, 1.0]], dtype=dtype)
        points_batch = RaggedBatch(points, sample_sizes=torch.tensor([2]))
        distances_batch = RaggedBatch(distances, sample_sizes=torch.tensor([2]))

        with pytest.raises(RuntimeError, match="float32 or float64 on CPU"):
            polyline.interpolate(points, distances)
        with pytest.raises(RuntimeError, match="float32 or float64 on CPU"):
            polyline.lengths(points)
        with pytest.raises(RuntimeError, match="float32 or float64 on CPU"):
            polyline.interpolate_var_size_batch(points_batch, distances_batch)
        with pytest.raises(RuntimeError, match="float32 or float64 on CPU"):
            polyline.lengths_var_size_batch(points_batch)


def test_cuda_polyline_functions_accept_low_precision_dtypes():
    for dtype in (torch.float16, torch.bfloat16):
        points = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], device="cuda", dtype=dtype)
        distances = torch.tensor([[0.0, 1.0]], device="cuda", dtype=dtype)
        points_batch = RaggedBatch(points, sample_sizes=torch.tensor([2], device="cuda"))
        distances_batch = RaggedBatch(distances, sample_sizes=torch.tensor([2], device="cuda"))

        expected_points = torch.tensor([[[0.0, 0.0], [1.0, 0.0]]], device="cuda", dtype=dtype)
        expected_lengths = torch.tensor([1.0], device="cuda", dtype=dtype)

        assert torch.equal(polyline.interpolate(points, distances), expected_points)
        assert torch.equal(polyline.lengths(points), expected_lengths)
        assert torch.equal(
            polyline.interpolate_var_size_batch(points_batch, distances_batch).tensor, expected_points
        )
        assert torch.equal(polyline.lengths_var_size_batch(points_batch), expected_lengths)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_var_size_batch_rejects_invalid_ragged_layout(device: str):
    points = torch.randn((2, 3, 4), device=device)
    points_batch = RaggedBatch(
        points.transpose(1, 2).contiguous(),
        sample_sizes=torch.tensor([3, 2], device=device, dtype=torch.int32),
        non_uniform_dim=2,
    )

    # Polyline points must use dimension 1 as the non-uniform point dimension.
    with pytest.raises(AssertionError, match="points.non_uniform_dim"):
        polyline.lengths_var_size_batch(points_batch)


@pytest.mark.parametrize("device", DEVICES)
def test_polyline_lengths_var_size_batch_validates_sample_sizes(device: str):
    points = torch.randn((1, 3, 2), device=device)

    # Length sample sizes must not be negative.
    bad_small = RaggedBatch(points, sample_sizes=torch.tensor([-1], device=device))
    with pytest.raises(RuntimeError, match="points.sample_sizes"):
        polyline.lengths_var_size_batch(bad_small)

    # Length sample sizes must not exceed the padded point dimension.
    bad_large = RaggedBatch(points, sample_sizes=torch.tensor([4], device=device))
    with pytest.raises(RuntimeError, match="points.sample_sizes"):
        polyline.lengths_var_size_batch(bad_large)
