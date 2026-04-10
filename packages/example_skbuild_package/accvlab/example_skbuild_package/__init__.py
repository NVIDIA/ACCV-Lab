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
Demonstrates manual build configuration with C++/CUDA extensions
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("accvlab.example_skbuild_package")
except PackageNotFoundError:
    __version__ = "0.0.0"

from .functions import (
    vector_add,
    vector_scale,
)

# Import the external module
from ._ext import external_vector_add_cuda, external_vector_scale_cuda


def hello_examples():
    """Simple function to test the examples module"""
    return "Hello from ACCV-Lab SKBuild Example Package!"


__all__ = [
    '__version__',
    'hello_examples',
    'vector_add',
    'vector_scale',
]
