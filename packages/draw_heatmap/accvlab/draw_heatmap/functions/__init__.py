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

"""
GPU-accelerated Gaussian heatmap drawing for object detection.

This package provides CUDA kernels for efficiently drawing Gaussian heatmaps
based on bounding box centers and radii, significantly outperforming CPU-based
implementations.
"""

# ensure torch is available before importing `draw_heatmap_ext`
import torch

from accvlab.draw_heatmap.draw_heatmap_ext import draw_heatmap

from .draw_heatmap_batched import draw_heatmap_batched

__all__ = ["draw_heatmap", "draw_heatmap_batched"]
