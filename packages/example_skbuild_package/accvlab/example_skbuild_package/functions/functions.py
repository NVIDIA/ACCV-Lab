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
ACCV-Lab Examples Functions
Wrapper functions for C++ and CUDA extensions.
"""

# Import torch before loading the extension so its shared libraries are available.
import torch

HAS_CUDA_EXTENSIONS = False

try:
    import accvlab.example_skbuild_package._ext as _ext

    HAS_CUDA_EXTENSIONS = True
except ImportError:
    # Extension not available (e.g., during development)
    _ext = None


def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Element-wise vector addition using the CUDA extension.

    :gpu:

    Args:
        a: First input 1D tensor on a CUDA device.
        b: Second input 1D tensor on a CUDA device.

    Returns:
        Tensor on CUDA containing the element-wise sum of ``a`` and ``b``.
    """
    if not HAS_CUDA_EXTENSIONS or _ext is None:
        raise ImportError("CUDA extension not available. Please install the package.")
    return _ext.external_vector_add_cuda(a, b)


def vector_scale(a: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Element-wise vector scaling using the CUDA extension.

    :gpu:

    Args:
        a: Input 1D tensor on a CUDA device.
        scale: Scalar factor used to scale all elements of ``a``.

    Returns:
        Tensor on CUDA containing the scaled values of ``a``.
    """
    if not HAS_CUDA_EXTENSIONS or _ext is None:
        raise ImportError("CUDA extension not available. Please install the package.")
    return _ext.external_vector_scale_cuda(a, scale)
