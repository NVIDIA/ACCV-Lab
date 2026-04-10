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

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import sys
from pathlib import Path
from accvlab_build_config import (
    load_config,
    detect_cuda_info,
    get_compile_flags,
)

from accvlab_build_config import run_external_build


def get_extensions():
    """Return all extensions"""

    cpp_filenames = [
        'cpp_impl/batched_indexing_access_cpu.cpp',
        'cpp_impl/batched_indexing_access_cpu_impl.cpp',
    ]
    # For CUDA extension, include both C++ and CUDA files
    cuda_filenames = [
        'cpp_impl/batched_indexing_access_cuda.cpp',
        'cpp_impl/batched_indexing_access_cuda_impl.cu',
    ]

    config = load_config()
    cuda_info = detect_cuda_info()

    # Get relative path for source files (required by setuptools)
    # Source directory is relative to setup.py
    source_dir = 'accvlab/batching_helpers'

    compile_flags = get_compile_flags(config, cuda_info, None)

    extensions = []

    # Fail if CUDA is not available
    if not cuda_info['cuda_available']:
        raise RuntimeError("CUDA is not available")

    # Always try to build C++ extension - fail if it cannot be created
    cpp_sources = [str(Path(source_dir) / f) for f in cpp_filenames]
    cpp_ext = CppExtension(
        name='accvlab.batching_helpers.batched_indexing_access_cpu',
        sources=cpp_sources,
        extra_compile_args=compile_flags['cxx'],
        language='c++',
        verbose=config['VERBOSE_BUILD'],
    )
    extensions.append(cpp_ext)

    # Always try to build CUDA extension - fail if it cannot be created
    cuda_sources = [str(Path(source_dir) / f) for f in cuda_filenames]
    cu_ext = CUDAExtension(
        name='accvlab.batching_helpers.batched_indexing_access_cuda',
        sources=cuda_sources,
        extra_compile_args={
            'cxx': compile_flags['cxx'],
            'nvcc': compile_flags['nvcc'],
        },
        language='c++',
        verbose=config['VERBOSE_BUILD'],
    )
    extensions.append(cu_ext)

    return extensions


# Run external build before setup
run_external_build(str(Path(__file__).parent))

setup(
    name="accvlab.batching_helpers",
    description="Batching Helpers Package (part of the ACCV-Lab package).",
    ext_modules=get_extensions(),
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    zip_safe=False,
    options={
        'build_ext': {
            'use_ninja': True,  # Use Ninja for faster builds
        }
    },
)
