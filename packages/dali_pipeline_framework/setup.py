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
from accvlab_build_config import build_cmake_args  # type: ignore

_cmake_args = build_cmake_args()


setup(
    name="accvlab.dali_pipeline_framework",
    description="DALI Pipeline Framework Package (part of the ACCV-Lab package).",
    packages=find_namespace_packages(include=["accvlab.dali_pipeline_framework*"]),
    include_package_data=True,
    zip_safe=False,
    cmake_source_dir="ext_impl",
    cmake_install_dir="accvlab/dali_pipeline_framework",
    cmake_args=_cmake_args,
)
