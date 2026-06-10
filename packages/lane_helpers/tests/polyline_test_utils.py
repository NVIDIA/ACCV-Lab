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

import torch

from accvlab.batching_helpers import RaggedBatch

DEVICES = ["cpu", "cuda"]


def sample_polyline_cpu(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    # For no distances, the result is an empty tensor.
    if distances.shape[0] == 0:
        sampled_points = points.new_empty((0, points.shape[1]))
        return sampled_points
    # For no points, the result is NaN for every requested point coordinate.
    if points.shape[0] == 0:
        sampled_points = points.new_full((distances.shape[0], points.shape[1]), torch.nan)
        return sampled_points

    segment_lengths = torch.linalg.vector_norm(points[1:] - points[:-1], dim=1)
    accum = torch.cat([segment_lengths.new_zeros(1), torch.cumsum(segment_lengths, dim=0)])
    total_length = accum[-1]

    out = []
    for distance in distances:
        d = torch.clamp(distance, min=0.0, max=total_length)
        lower_idx = int(torch.nonzero(accum <= d, as_tuple=False)[-1])
        if lower_idx >= points.shape[0] - 1:
            out.append(points[-1])
            continue

        upper_idx = lower_idx + 1
        lower_dist = accum[lower_idx]
        upper_dist = accum[upper_idx]
        segment_dist = upper_dist - lower_dist
        if segment_dist <= torch.finfo(points.dtype).eps:
            out.append(points[lower_idx])
            continue

        weight_upper = (d - lower_dist) / segment_dist
        weight_lower = (upper_dist - d) / segment_dist
        out.append(points[lower_idx] * weight_lower + points[upper_idx] * weight_upper)

    sampled_points = torch.stack(out)
    return sampled_points


def sample_batch_cpu(points: torch.Tensor, distances: torch.Tensor) -> torch.Tensor:
    sampled_points = torch.stack(
        [
            sample_polyline_cpu(points_sample, distances_sample)
            for points_sample, distances_sample in zip(points, distances)
        ]
    )
    return sampled_points


def polyline_lengths_cpu(points: torch.Tensor) -> torch.Tensor:
    # For no points, the length is undefined.
    if points.shape[1] == 0:
        lengths = points.new_full((points.shape[0],), torch.nan)
        return lengths
    # For a single point, the length is 0.
    if points.shape[1] == 1:
        lengths = points.new_zeros((points.shape[0],))
        return lengths

    lengths = torch.linalg.vector_norm(points[:, 1:] - points[:, :-1], dim=2).sum(dim=1)
    return lengths


def polyline_lengths_var_size_cpu(points: torch.Tensor, sample_sizes: torch.Tensor) -> torch.Tensor:
    lengths = []
    for sample_idx in range(points.shape[0]):
        num_points = int(sample_sizes[sample_idx].item())
        lengths.append(polyline_lengths_cpu(points[sample_idx : sample_idx + 1, :num_points])[0])
    lengths = torch.stack(lengths)
    return lengths


def assert_ragged_matches_cpu(
    result: RaggedBatch,
    points: torch.Tensor,
    distances: torch.Tensor,
    points_sample_sizes: torch.Tensor,
    distances_sample_sizes: torch.Tensor,
    *,
    atol: float = 1e-5,
) -> None:
    assert torch.equal(result.sample_sizes.cpu(), distances_sample_sizes.cpu())

    for sample_idx in range(points.shape[0]):

        num_points = int(points_sample_sizes[sample_idx].item())
        num_distances = int(distances_sample_sizes[sample_idx].item())
        expected = sample_polyline_cpu(
            points[sample_idx, :num_points].cpu(),
            distances[sample_idx, :num_distances].cpu(),
        )

        actual = result.tensor[sample_idx, :num_distances].cpu()

        assert torch.allclose(actual, expected, atol=atol, rtol=0.0, equal_nan=True)


def make_random_ragged_polyline_case(
    *,
    seed: int,
    batch_size: int = 7,
    max_num_points: int = 12,
    max_num_distances: int = 17,
    num_dims: int = 3,
) -> tuple[RaggedBatch, RaggedBatch]:
    generator = torch.Generator().manual_seed(seed)
    points_sample_sizes = torch.randint(1, max_num_points + 1, (batch_size,), generator=generator)
    distances_sample_sizes = torch.randint(0, max_num_distances + 1, (batch_size,), generator=generator)

    max_points_in_batch = int(points_sample_sizes.max().item())
    max_distances_in_batch = int(distances_sample_sizes.max().item())

    points = torch.full((batch_size, max_points_in_batch, num_dims), 9999.0, dtype=torch.float32)
    distances = torch.full((batch_size, max_distances_in_batch), -9999.0, dtype=torch.float32)

    for sample_idx in range(batch_size):
        num_points = int(points_sample_sizes[sample_idx].item())
        num_distances = int(distances_sample_sizes[sample_idx].item())
        points[sample_idx, :num_points] = torch.rand((num_points, num_dims), generator=generator)
        total_length = polyline_lengths_cpu(points[sample_idx : sample_idx + 1, :num_points])[0]
        distances[sample_idx, :num_distances] = (
            torch.rand((num_distances,), generator=generator) * total_length
        )

    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)

    return points_batch, distances_batch


def make_padded_ragged_polyline_case(
    device: str,
) -> tuple[RaggedBatch, RaggedBatch]:
    # Poitns data
    points = torch.tensor(
        [
            [[0.0, 0.0], [1.0, 0.0], [1.0, 2.0], [0.0, 2.0], [0.0, 0.0]],
            [[3.5, -1.25], [4.5, -1.25], [4.5, 0.75], [9999.0, 9999.0], [9999.0, 9999.0]],
            [[-2.0, 3.0], [9999.0, 9999.0], [9999.0, 9999.0], [9999.0, 9999.0], [9999.0, 9999.0]],
            [[10.0, 0.0], [12.0, 0.0], [9999.0, 9999.0], [9999.0, 9999.0], [9999.0, 9999.0]],
        ],
        device=device,
        dtype=torch.float32,
    )
    points_sample_sizes = torch.tensor([5, 3, 1, 2], device=device)
    # Distances data
    distances = torch.tensor(
        [
            [0.0, 0.5, 1.0, 2.0, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0],
            [3.0, 2.0, 1.0, 0.0, -1.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0],
            [9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0],
            [-5.0, 1.0, 5.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0, 9999.0],
        ],
        device=device,
        dtype=torch.float32,
    )
    distances_sample_sizes = torch.tensor([11, 5, 0, 3], device=device)

    points_batch = RaggedBatch(points, sample_sizes=points_sample_sizes)
    distances_batch = RaggedBatch(distances, sample_sizes=distances_sample_sizes)

    return points_batch, distances_batch


def distances_for_mode(
    points: torch.Tensor, absolute_distances: torch.Tensor, *, relative: bool
) -> torch.Tensor:

    if not relative:
        return absolute_distances

    lengths = polyline_lengths_cpu(points.cpu()).to(
        device=absolute_distances.device, dtype=absolute_distances.dtype
    )

    # For zero length, use 1.0 to avoid division by zero.
    safe_lengths = torch.where(lengths > 0, lengths, torch.ones_like(lengths))
    relative_distances = absolute_distances / safe_lengths[:, None]

    return relative_distances


def ragged_distances_for_mode(
    points: RaggedBatch,
    absolute_distances: RaggedBatch,
    *,
    relative: bool,
) -> RaggedBatch:
    if not relative:
        return absolute_distances

    relative_distances = absolute_distances.tensor.clone()
    lengths = polyline_lengths_var_size_cpu(points.tensor.cpu(), points.sample_sizes.cpu()).to(
        device=absolute_distances.tensor.device, dtype=absolute_distances.tensor.dtype
    )
    for sample_idx in range(points.tensor.shape[0]):
        num_distances = int(absolute_distances.sample_sizes[sample_idx].item())
        if num_distances == 0:
            continue
        length = lengths[sample_idx]
        if length > 0:
            relative_distances[sample_idx, :num_distances] /= length
        else:
            relative_distances[sample_idx, :num_distances] = 0.0
    relative_distances_batch = absolute_distances.create_with_sample_sizes_like_self(relative_distances)
    return relative_distances_batch
