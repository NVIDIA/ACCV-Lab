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
    assert_ragged_matches_cpu,
    make_padded_ragged_polyline_case,
    make_random_ragged_polyline_case,
    ragged_distances_for_mode,
)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_matches_cpu_reference(relative: bool, device: str):
    points_batch, distances_batch = make_padded_ragged_polyline_case(device)
    distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

    result = polyline.interpolate_var_size_batch(points_batch, distances_input_batch, relative=relative)

    assert isinstance(result, RaggedBatch)
    assert result.tensor.shape == (4, 11, 2)
    assert result.non_uniform_dim == 1
    assert_ragged_matches_cpu(
        result,
        points_batch.tensor,
        distances_batch.tensor,
        points_batch.sample_sizes,
        distances_batch.sample_sizes,
    )


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_random_matches_cpu_reference(relative: bool, device: str):
    num_iters = 100
    for i in range(num_iters):
        points_batch_cpu, distances_batch_cpu = make_random_ragged_polyline_case(seed=i)
        points_batch = points_batch_cpu.to(device)
        distances_batch = distances_batch_cpu.to(device)
        distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

        result = polyline.interpolate_var_size_batch(
            points_batch,
            distances_input_batch,
            relative=relative,
        )

        assert isinstance(result, RaggedBatch)
        assert result.tensor.shape == (
            points_batch.tensor.shape[0],
            distances_batch.tensor.shape[1],
            points_batch.tensor.shape[2],
        )
        assert_ragged_matches_cpu(
            result,
            points_batch.tensor,
            distances_batch.tensor,
            points_batch.sample_sizes,
            distances_batch.sample_sizes,
            atol=1e-4,
        )


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_matches_fixed_size_when_uniform(relative: bool, device: str):
    points = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0]],
            [[2.0, 2.0], [3.0, 2.0], [3.0, 4.0], [2.0, 4.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    distances = torch.tensor(
        [[0.0, 0.5, 2.0, 4.0], [4.0, 2.0, 0.5, 0.0]],
        device=device,
        dtype=torch.float32,
    )
    sample_sizes = torch.tensor([points.shape[1], points.shape[1]], device=device, dtype=torch.int32)
    distances_sample_sizes = torch.tensor(
        [distances.shape[1], distances.shape[1]], device=device, dtype=torch.int32
    )
    points_batch = RaggedBatch(points, sample_sizes=sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)
    distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

    result = polyline.interpolate_var_size_batch(
        points_batch,
        distances_input_batch,
        relative=relative,
    )
    expected = polyline.interpolate(
        points.contiguous(), distances_input_batch.tensor.contiguous(), relative=relative
    )

    assert torch.allclose(result.tensor, expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_accepts_non_contiguous_inputs(relative: bool, device: str):
    points_storage = torch.tensor(
        [
            [[0.0, 1.0, 1.0, 0.0], [0.0, 0.0, 2.0, 2.0]],
            [[2.0, 3.0, 3.0, 2.0], [2.0, 2.0, 4.0, 4.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    points = points_storage.transpose(1, 2)
    distances = torch.tensor(
        [[0.0, 4.0], [0.5, 2.0], [2.0, 0.5], [4.0, 0.0]],
        device=device,
        dtype=torch.float32,
    ).transpose(0, 1)
    assert not points.is_contiguous()
    assert not distances.is_contiguous()

    points_sample_sizes = torch.tensor([4, 3], device=device, dtype=torch.int32)
    distances_sample_sizes = torch.tensor([4, 2], device=device, dtype=torch.int32)
    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)
    distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

    result = polyline.interpolate_var_size_batch(
        points_batch,
        distances_input_batch,
        relative=relative,
    )

    assert_ragged_matches_cpu(
        result,
        points,
        distances,
        points_sample_sizes,
        distances_sample_sizes,
    )


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_zero_point_row_returns_nan(relative: bool, device: str):
    points = torch.tensor(
        [
            [[9999.0, 9999.0], [9999.0, 9999.0]],
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 3.0], [9999.0, 9999.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    distances = torch.tensor(
        [[0.0, 1.0], [0.0, 0.5], [-1.0, 2.0]],
        device=device,
        dtype=torch.float32,
    )
    points_sample_sizes = torch.tensor([0, 2, 1], device=device, dtype=torch.int32)
    distances_sample_sizes = torch.tensor([2, 2, 2], device=device, dtype=torch.int32)
    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)
    distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

    result = polyline.interpolate_var_size_batch(
        points_batch,
        distances_input_batch,
        relative=relative,
    )

    assert_ragged_matches_cpu(
        result,
        points,
        distances,
        points_sample_sizes,
        distances_sample_sizes,
    )
    assert torch.isnan(result.tensor[0, :2]).all()


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_zero_max_distances_returns_empty(relative: bool, device: str):
    points = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0]],
            [[2.0, 3.0], [9999.0, 9999.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    distances = torch.empty((2, 0), device=device, dtype=torch.float32)
    points_sample_sizes = torch.tensor([2, 1], device=device, dtype=torch.int32)
    distances_sample_sizes = torch.tensor([0, 0], device=device, dtype=torch.int32)
    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)
    distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

    result = polyline.interpolate_var_size_batch(
        points_batch,
        distances_input_batch,
        relative=relative,
    )

    assert isinstance(result, RaggedBatch)
    assert result.tensor.shape == (2, 0, 2)
    assert torch.equal(result.sample_sizes.cpu(), torch.zeros(2, dtype=torch.int32))


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_variable_size_polyline_interpolation_all_zero_point_rows_return_nan(relative: bool, device: str):
    points = torch.empty((2, 0, 2), device=device, dtype=torch.float32)
    distances = torch.tensor([[0.0, 1.0, 2.0], [-1.0, 0.5, 3.0]], device=device, dtype=torch.float32)
    points_sample_sizes = torch.zeros(2, device=device, dtype=torch.int32)
    distances_sample_sizes = torch.full((2,), 3, device=device, dtype=torch.int32)
    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)
    distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

    result = polyline.interpolate_var_size_batch(
        points_batch,
        distances_input_batch,
        relative=relative,
    )

    assert result.tensor.shape == (2, 3, 2)
    assert torch.isnan(result.tensor).all()


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_variable_size_large_polyline_interpolation_external_distance_buffer(relative: bool):
    # Create a large polyline to ensure that the external distance buffer is used.
    num_points = 200_000
    x = torch.linspace(0.0, 1.0, num_points, device="cuda", dtype=torch.float32)
    first_polyline = torch.stack((x, torch.zeros_like(x)), dim=1)
    second_polyline = torch.stack((x, torch.ones_like(x)), dim=1)
    points = torch.stack((first_polyline, second_polyline), dim=0)
    distances = torch.tensor(
        # Note that 9999.0 is a filler is not not part of the distances used for interpolation (due to `distances_sample_sizes`)
        [[0.0, 0.25, 0.5, 1.0, 2.0], [1.0, 0.5, 0.0, -1.0, 9999.0]],
        device="cuda",
        dtype=torch.float32,
    )
    points_sample_sizes = torch.full((2,), num_points, device="cuda", dtype=torch.int32)
    distances_sample_sizes = torch.tensor([5, 4], device="cuda", dtype=torch.int32)
    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)
    expected = torch.tensor(
        [
            [[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.0]],
            # Note that 9999.0 is a filler and is not checked for equality in the test.
            [[1.0, 1.0], [0.5, 1.0], [0.0, 1.0], [0.0, 1.0], [9999.0, 9999.0]],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = polyline.interpolate_var_size_batch(points_batch, distances_batch, relative=relative)
    stream.synchronize()

    assert torch.equal(result.sample_sizes.cpu(), distances_sample_sizes.cpu())
    assert torch.allclose(result.tensor[0, :5], expected[0, :5], atol=1e-4, rtol=0.0)
    assert torch.allclose(result.tensor[1, :4], expected[1, :4], atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_variable_size_polyline_interpolation_handles_inactive_cuda_rows(relative: bool):
    num_samples = 33
    points = torch.empty((num_samples, 2, 2), device="cuda", dtype=torch.float32)
    points[:, 0, 0] = torch.arange(num_samples, device="cuda", dtype=torch.float32)
    points[:, 0, 1] = 0.0
    points[:, 1, 0] = points[:, 0, 0] + 1.0
    points[:, 1, 1] = 0.0
    distances = (
        torch.tensor([[0.0, 0.25, 1.0]], device="cuda", dtype=torch.float32).expand(num_samples, -1).clone()
    )
    points_sample_sizes = torch.full((num_samples,), 2, device="cuda")
    distances_sample_sizes = torch.full((num_samples,), 3, device="cuda")
    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)
    distances_input_batch = ragged_distances_for_mode(points_batch, distances_batch, relative=relative)

    result = polyline.interpolate_var_size_batch(
        points_batch,
        distances_input_batch,
        relative=relative,
    )

    assert_ragged_matches_cpu(result, points, distances, points_sample_sizes, distances_sample_sizes)


if __name__ == "__main__":
    pytest.main([__file__])
