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
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import sys
from pathlib import Path
from accvlab_build_config import (
    load_config,
    detect_cuda_info,
    get_compile_flags,
    get_abs_setup_dir,
)

from accvlab_build_config import run_external_build


def get_extensions():
    """Return all extensions"""
    config = load_config()
    cuda_info = detect_cuda_info()

    compile_flags = get_compile_flags(config, cuda_info)

    # Source directory is relative to setup.py
    source_dir = 'accvlab/draw_heatmap'
    # Note that include directories need to be global, while source directories are relative
    include_dirs = [str(get_abs_setup_dir(__file__) / source_dir / 'include')]

    extensions = []

    # CUDA extension
    cuda_sources = [
        str(Path(source_dir) / 'csrc' / 'draw_heatmap.cpp'),
        str(Path(source_dir) / 'csrc' / 'draw_heatmap_cuda.cu'),
    ]
    cu_ext = CUDAExtension(
        name='accvlab.draw_heatmap.draw_heatmap_ext',
        sources=cuda_sources,
        include_dirs=include_dirs,
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
    name="accvlab.draw_heatmap",
    description="Draw Heatmap Package (part of the ACCV-Lab package).",
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
