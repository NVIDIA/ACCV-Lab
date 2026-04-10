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
ACCV-Lab Examples Module
Demonstrates manual build configuration with C++/CUDA extensions.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("accvlab.example_package")
except PackageNotFoundError:
    __version__ = "0.0.0"

from typing import Callable

from torch import Tensor

from .functions import (
    cpp_vector_sum,
    cpp_matrix_transpose,
    cuda_vector_multiply,
    cuda_reduce_sum,
)

# Import the external module
from accvlab.example_package.accvlab_example_package_ext import (
    external_vector_add_cuda,
    external_vector_scale_cuda,
)

# Type annotations for externally provided CUDA functions
external_vector_add_cuda: Callable[[Tensor, Tensor], Tensor]
external_vector_scale_cuda: Callable[[Tensor, float], Tensor]


def hello_examples() -> str:
    """Simple function to test the examples module."""
    return "Hello from ACCV-Lab Example Package!"


__all__ = [
    '__version__',
    'hello_examples',
    'cpp_vector_sum',
    'cpp_matrix_transpose',
    'cuda_vector_multiply',
    'cuda_reduce_sum',
    'external_vector_add_cuda',
    'external_vector_scale_cuda',
]
