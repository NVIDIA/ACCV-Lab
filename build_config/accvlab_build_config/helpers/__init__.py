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
Build configuration helpers for accvlab packages
"""

from .build_utils import (
    load_config,
    detect_cuda_info,
    get_compile_flags,
    get_abs_setup_dir,
)
from .cmake_args import (
    CUDA_ARCH_STRATEGY_CMAKE,
    CUDA_ARCH_STRATEGY_TORCH,
    build_cmake_args,
    get_project_root,
)
from .cpp_unit_test_config import (
    CppUnitTestPackage,
    load_cpp_unit_test_package_configs,
    select_cpp_unit_test_packages,
)
from .yaml_config import YamlConfigError

__all__ = [
    'load_config',
    'detect_cuda_info',
    'get_compile_flags',
    'get_abs_setup_dir',
    'CUDA_ARCH_STRATEGY_CMAKE',
    'CUDA_ARCH_STRATEGY_TORCH',
    'build_cmake_args',
    'get_project_root',
    'CppUnitTestPackage',
    'load_cpp_unit_test_package_configs',
    'select_cpp_unit_test_packages',
    'YamlConfigError',
]
