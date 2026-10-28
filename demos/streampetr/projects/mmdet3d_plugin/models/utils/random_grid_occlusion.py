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

"""NVIDIA-authored periodic image occlusion used by the StreamPETR demo.

The public API retains the parameters expected by StreamPETR, but only the
configuration actually used by this demo is implemented. Unsupported options
fail during construction instead of silently producing different
augmentation behavior.
"""

import torch
from torch import Tensor, nn


class RandomGridOcclusion(nn.Module):
    """Apply the periodic image occlusion configured by the StreamPETR demo.

    Args mirror the original StreamPETR call site for configuration
    compatibility. The supported configuration is ``use_h=True``,
    ``use_w=True``, ``rotate=1``, ``offset=False``, ``ratio=0.5``, and
    ``mode=1``. ``prob`` may be any value in ``[0, 1]``.
    """

    def __init__(
        self,
        use_h: bool,
        use_w: bool,
        rotate: int = 1,
        offset: bool = False,
        ratio: float = 0.5,
        mode: int = 0,
        prob: float = 1.0,
    ) -> None:
        super().__init__()
        unsupported = []
        if not use_h or not use_w:
            unsupported.append("use_h and use_w must both be True")
        if rotate != 1:
            unsupported.append("rotate must be 1 (the value 1 means zero-degree rotation)")
        if offset:
            unsupported.append("offset must be False (random offset filling is not used by this demo)")
        if ratio != 0.5:
            unsupported.append("ratio must be 0.5")
        if mode != 1:
            unsupported.append("mode must be 1 (retain periodic bands)")
        if unsupported:
            details = "; ".join(unsupported)
            raise NotImplementedError(
                "The NVIDIA-authored RandomGridOcclusion replacement for the GridMask implementation intentionally supports "
                f"only the configuration used by the StreamPETR demo: {details}."
            )
        if not 0.0 <= prob <= 1.0:
            raise ValueError("prob must be between 0 and 1")

        self.initial_probability = prob
        self.probability = prob

    def set_prob(self, epoch: int, max_epoch: int) -> None:
        """Retain the reference scheduling API and linearly ramp probability."""
        if max_epoch <= 0:
            raise ValueError("max_epoch must be positive")
        if not 0 <= epoch <= max_epoch:
            raise ValueError("epoch must be between zero and max_epoch")
        self.probability = self.initial_probability * epoch / max_epoch

    def forward(self, images: Tensor) -> Tensor:
        """Return augmented ``NCHW`` images without changing shape or dtype."""
        if images.ndim != 4:
            raise ValueError(f"Expected an NCHW tensor, got {images.ndim} dimensions")
        if not self.training or self.probability == 0.0:
            return images
        if self.probability < 1.0 and torch.rand((), device=images.device) >= self.probability:
            return images

        height, width = images.shape[-2:]
        if height < 3:
            return images

        period = int(torch.randint(2, height, (), device=images.device).item())
        band_width = min(period - 1, max(1, int(period * 0.5 + 0.5)))
        row_phase = int(torch.randint(period, (), device=images.device).item())
        column_phase = int(torch.randint(period, (), device=images.device).item())

        # Define the finite bands on a centered 1.5x canvas, then select the
        # visible coordinates. This preserves the supported reference behavior
        # at the image edges.
        canvas_height = int(1.5 * height)
        canvas_width = int(1.5 * width)
        row_crop = (canvas_height - height) // 2
        column_crop = (canvas_width - width) // 2
        rows = torch.arange(height, device=images.device) + row_crop
        columns = torch.arange(width, device=images.device) + column_crop

        row_delta = rows - row_phase
        column_delta = columns - column_phase
        in_horizontal_band = (
            (row_delta >= 0)
            & (torch.div(row_delta, period, rounding_mode="floor") < canvas_height // period)
            & (row_delta % period < band_width)
        ).unsqueeze(1)
        in_vertical_band = (
            (column_delta >= 0)
            & (torch.div(column_delta, period, rounding_mode="floor") < canvas_width // period)
            & (column_delta % period < band_width)
        ).unsqueeze(0)

        # mode=1 retains the union of the periodic horizontal and vertical bands.
        keep = (in_horizontal_band | in_vertical_band).to(dtype=images.dtype)
        augmented_images = images * keep.view(1, 1, height, width)
        return augmented_images
