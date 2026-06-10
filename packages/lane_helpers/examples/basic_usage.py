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

from accvlab.lane_helpers import polyline


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("This example requires a CUDA-capable PyTorch installation.")

    # @NOTE Use one rectangle polyline with shape (batch=1, num_points=5, num_dims=2).
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
        device="cuda",
        dtype=torch.float32,
    )

    # @NOTE Use a tensor of distances to sample the polyline at (batch=1, num_distances=5).
    distances = torch.tensor([[0.0, 0.5, 1.0, 3.0, 6.0]], device="cuda", dtype=torch.float32)

    # @NOTE Interpolate the polyline at the given distances.
    sampled_points = polyline.interpolate(points, distances)
    # @NOTE Compute the length of the polyline.
    line_lengths = polyline.lengths(points)

    # @NOTE Print the results.
    print(f"Interpolated points:\n{sampled_points}")
    print(f"Line length(s): {line_lengths}")


if __name__ == "__main__":
    main()
