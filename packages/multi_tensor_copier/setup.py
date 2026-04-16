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

from __future__ import annotations

from setuptools import find_namespace_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

from accvlab_build_config import detect_cuda_info, get_compile_flags, load_config


def get_extensions():
    config = load_config()
    cuda_info = detect_cuda_info()
    compile_flags = get_compile_flags(config, cuda_info)

    return [
        CUDAExtension(
            name="accvlab.multi_tensor_copier._ext",
            sources=[
                "accvlab/multi_tensor_copier/csrc/multi_tensor_copier.cpp",
            ],
            extra_compile_args={
                "cxx": compile_flags["cxx"],
                "nvcc": compile_flags["nvcc"],
            },
            define_macros=[
                ("TORCH_EXTENSION_NAME", "_ext"),
            ],
            include_dirs=compile_flags["include_dirs"],
        ),
    ]


setup(
    name="accvlab.multi_tensor_copier",
    description="Async copying of nested tensor structures (ACCV-Lab).",
    packages=find_namespace_packages(include=["accvlab.multi_tensor_copier*"]),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    zip_safe=False,
)
