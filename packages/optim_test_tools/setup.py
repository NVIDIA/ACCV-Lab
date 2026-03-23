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

from skbuild import setup
from setuptools import find_namespace_packages

from accvlab_build_config import build_cmake_args_from_env

_cmake_args = build_cmake_args_from_env()

setup(
    name="accvlab.optim_test_tools",
    version="0.1.0",
    description="Optimization Testing Tools Package (part of the ACCV-Lab package).",
    packages=find_namespace_packages(include=["accvlab.optim_test_tools*"]),
    include_package_data=True,
    zip_safe=False,
    cmake_source_dir="ext_impl",
    cmake_install_dir="accvlab/optim_test_tools/numba_nvtx",
    cmake_args=_cmake_args,
)
