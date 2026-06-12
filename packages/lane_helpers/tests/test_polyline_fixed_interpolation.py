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

from accvlab.lane_helpers import polyline

from polyline_test_utils import DEVICES, distances_for_mode, sample_batch_cpu


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_rectangle_polyline_interpolation(relative: bool, device: str):
    points = torch.tensor(
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
    distances = torch.tensor(
        [[0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0]],
        device=device,
    )

    distances_input = distances_for_mode(points, distances, relative=relative)

    expected = sample_batch_cpu(points.cpu(), distances.cpu())
    result = polyline.interpolate(points, distances_input, relative=relative)

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_batched_polyline_interpolation(relative: bool, device: str):
    base_points = torch.tensor(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 2.0],
            [0.0, 2.0],
            [0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    offsets = torch.tensor([[0.0, 0.0], [3.5, -1.25]], dtype=torch.float32)
    points = (base_points.unsqueeze(0) + offsets.unsqueeze(1)).to(device)
    distances = torch.tensor(
        [
            [0.0, 0.5, 1.0, 3.0, 6.0],
            [6.0, 5.0, 3.0, 1.0, 0.0],
        ],
        device=device,
        dtype=torch.float32,
    )

    distances_input = distances_for_mode(points, distances, relative=relative)

    expected = sample_batch_cpu(points.cpu(), distances.cpu())
    result = polyline.interpolate(points.contiguous(), distances_input.contiguous(), relative=relative)

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_polyline_interpolation_accepts_non_contiguous_inputs(relative: bool, device: str):
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

    distances_input = distances_for_mode(points, distances, relative=relative)

    expected = sample_batch_cpu(points.cpu(), distances.cpu())
    result = polyline.interpolate(points, distances_input, relative=relative)

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_out_of_range_distances_clamp_to_endpoints(relative: bool, device: str):
    points = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 2.0]]], device=device, dtype=torch.float32)
    distances = torch.tensor([[-4.0, -1.0, 0.0, 3.0, 4.0]], device=device, dtype=torch.float32)

    distances_input = distances_for_mode(points, distances, relative=relative)

    expected = sample_batch_cpu(points.cpu(), distances.cpu())
    result = polyline.interpolate(points, distances_input, relative=relative)

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_single_point_polyline(relative: bool, device: str):
    points = torch.tensor([[[1.0, 2.0]]], device=device, dtype=torch.float32)
    distances = torch.tensor([[-1.0, 0.0, 1.0]], device=device, dtype=torch.float32)

    distances_input = distances_for_mode(points, distances, relative=relative)

    expected = sample_batch_cpu(points.cpu(), distances.cpu())
    result = polyline.interpolate(points, distances_input, relative=relative)

    assert torch.allclose(result.cpu(), expected, atol=1e-5, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_zero_point_polyline_returns_nan(relative: bool, device: str):
    points = torch.empty((2, 0, 3), device=device, dtype=torch.float32)
    distances = torch.tensor([[0.0, 1.0], [-1.0, 2.0]], device=device, dtype=torch.float32)
    distances_input = distances_for_mode(points, distances, relative=relative)

    result = polyline.interpolate(points, distances_input, relative=relative)

    assert result.shape == (2, 2, 3)
    assert torch.isnan(result).all()


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_zero_point_polyline_with_zero_distances_returns_empty(relative: bool, device: str):
    points = torch.empty((2, 0, 3), device=device, dtype=torch.float32)
    distances = torch.empty((2, 0), device=device, dtype=torch.float32)
    distances_input = distances_for_mode(points, distances, relative=relative)

    result = polyline.interpolate(points, distances_input, relative=relative)

    assert result.shape == (2, 0, 3)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
@pytest.mark.parametrize("device", DEVICES)
def test_random_polyline_matches_cpu_reference(relative: bool, device: str):
    num_iters = 100
    generator = torch.Generator().manual_seed(0)
    for _ in range(num_iters):
        num_points = int(torch.randint(15, 61, (), generator=generator).item())
        num_distances = int(torch.randint(15, 61, (), generator=generator).item())
        points_cpu = torch.rand((3, num_points, 2), generator=generator, dtype=torch.float32)
        distances_cpu = torch.rand((3, num_distances), generator=generator, dtype=torch.float32)

        segment_lengths = torch.linalg.vector_norm(points_cpu[:, 1:] - points_cpu[:, :-1], dim=2)
        total_lengths = torch.sum(segment_lengths, dim=1)
        distances_cpu = distances_cpu * total_lengths[:, None]

        distances_input_cpu = distances_for_mode(points_cpu, distances_cpu, relative=relative)

        expected = sample_batch_cpu(points_cpu, distances_cpu)
        result = polyline.interpolate(
            points_cpu.to(device), distances_input_cpu.to(device), relative=relative
        )

        assert torch.allclose(result.cpu(), expected, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize("relative", [False, True], ids=["absolute", "relative"])
def test_large_polyline_interpolation_external_distance_buffer(relative: bool):
    # Create a large polyline to ensure that the external distance buffer is used.
    num_points = 200_000
    x = torch.linspace(0.0, 1.0, num_points, device="cuda", dtype=torch.float32)
    points = torch.stack((x, torch.zeros_like(x)), dim=1).unsqueeze(0)
    distances = torch.tensor([[0.0, 0.25, 0.5, 1.0, 2.0]], device="cuda", dtype=torch.float32)
    expected = torch.tensor(
        [[[0.0, 0.0], [0.25, 0.0], [0.5, 0.0], [1.0, 0.0], [1.0, 0.0]]],
        device="cuda",
        dtype=torch.float32,
    )

    torch.cuda.synchronize()
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = polyline.interpolate(points, distances, relative=relative)
    stream.synchronize()

    assert torch.allclose(result, expected, atol=1e-4, rtol=0.0)
