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
Shared build utilities for accvlab packages
Provides reusable functions for configuration, CUDA detection, and extension creation
"""

import os
import re
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List, NamedTuple, Optional


class CudaArchitectureSelection(NamedTuple):
    """CUDA architecture selection compatible with the available ``nvcc``.

    Attributes:
        architectures: CUDA architectures to build as cubin targets.
        ptx_architectures: CUDA architectures to build as PTX targets because
            detected GPU architectures were not exact ``nvcc`` cubin targets.
    """

    architectures: List[str]
    ptx_architectures: List[str]


def _find_nvcc() -> Optional[str]:
    """
    Locate the CUDA compiler used to determine supported target architectures.
    """
    candidate = os.environ.get("CUDACXX")
    if candidate:
        return candidate

    for env_var in ("CUDA_HOME", "CUDA_PATH"):
        cuda_root = os.environ.get(env_var)
        if cuda_root:
            candidate = os.path.join(cuda_root, "bin", "nvcc")
            if os.path.exists(candidate):
                return candidate

    return shutil.which("nvcc")


def _detect_nvcc_supported_architectures() -> List[str]:
    """
    Ask nvcc which virtual GPU architectures it supports.
    Returns values like ['70', '75', '80', '90'].
    """
    nvcc = _find_nvcc()
    if not nvcc:
        return []

    try:
        result = subprocess.run(
            [nvcc, "--list-gpu-arch"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except Exception:
        return []

    archs: List[str] = []
    for match in re.finditer(r"compute_([0-9]+)", result.stdout):
        arch = match.group(1)
        if arch not in archs:
            archs.append(arch)

    return sorted(archs, key=int)


def _split_cuda_architectures(value: str) -> List[str]:
    return [arch.strip() for arch in re.split(r"[,;]", value) if arch.strip()]


def _supported_ptx_fallback_architecture(
    supported_architectures: List[str], detected_architecture: int
) -> Optional[str]:
    forward_compatible_archs: List[str] = []
    fallback_archs: List[str] = []
    for arch in supported_architectures:
        try:
            arch_int = int(arch)
        except ValueError:
            continue

        if arch_int > detected_architecture:
            continue

        fallback_archs.append(arch)
        if arch_int % 10 == 0:
            forward_compatible_archs.append(arch)

    if forward_compatible_archs:
        return max(forward_compatible_archs, key=int)
    if fallback_archs:
        return max(fallback_archs, key=int)
    return None


def select_cuda_architectures_for_nvcc(
    cuda_architectures: List[str],
) -> CudaArchitectureSelection:
    """Select CUDA cubin and PTX targets supported by the installed ``nvcc``.

    A detected architecture is emitted as a cubin target only when
    ``nvcc --list-gpu-arch`` reports that exact architecture. Unsupported
    detected architectures use a PTX fallback at or below the detected
    architecture, preferring the newest supported base architecture where the
    architecture number is divisible by 10.

    Args:
        cuda_architectures: CUDA architecture numbers to select from, for
            example ``["80", "90", "103"]``.

    Returns:
        CudaArchitectureSelection: The exact cubin architectures and any PTX
        fallback architectures. If ``nvcc`` cannot be found or queried, the
        input architectures are returned unchanged and no PTX targets are added.
    """
    supported_archs = _detect_nvcc_supported_architectures()
    if not cuda_architectures or not supported_archs:
        return CudaArchitectureSelection(cuda_architectures, [])

    supported_arch_set = set(supported_archs)
    selected_archs: List[str] = []
    ptx_archs: List[str] = []

    for arch in cuda_architectures:
        if arch in supported_arch_set:
            if arch not in selected_archs:
                selected_archs.append(arch)
            continue

        try:
            arch_int = int(arch)
        except ValueError:
            if arch not in selected_archs:
                selected_archs.append(arch)
            continue

        ptx_arch = _supported_ptx_fallback_architecture(supported_archs, arch_int)
        if ptx_arch is None:
            if arch not in selected_archs:
                selected_archs.append(arch)
            continue

        if ptx_arch not in ptx_archs:
            ptx_archs.append(ptx_arch)

    return CudaArchitectureSelection(selected_archs, ptx_archs)


def missing_torch_error() -> RuntimeError:
    return RuntimeError("""
#########################################################################################
# Missing build dependency: torch.                                                      #
#                                                                                       #
# ACCV-Lab CUDA extension builds require PyTorch with CUDA support.                     #
#                                                                                       #
# Install a CUDA-enabled PyTorch wheel and retry. When using pip build isolation,       #
# configure PIP_INDEX_URL/PIP_EXTRA_INDEX_URL so the isolated build environment         #
# resolves a CUDA-enabled torch wheel. See:                                             #
#                                                                                       #
#     docs/guides/INSTALLATION_GUIDE.md#installing-with-build-isolation                 #
#########################################################################################
""")


def require_torch_cuda_support(torch_module) -> None:
    if getattr(torch_module.version, "cuda", None) is not None:
        return

    torch_version = getattr(torch_module, "__version__", "unknown")
    raise RuntimeError(f"""
#########################################################################################
# PyTorch was installed without CUDA support.                                           #
#                                                                                       #
# ACCV-Lab CUDA extension builds require a CUDA-enabled PyTorch wheel.                  #
#                                                                                       #
# Detected PyTorch build:                                                               #
#                                                                                       #
#     torch.__version__ = {torch_version!r:<62}#
#     torch.version.cuda = None                                                         #
#                                                                                       #
# Install PyTorch with CUDA support and retry. When using pip build isolation,          #
# configure PIP_INDEX_URL/PIP_EXTRA_INDEX_URL so the isolated build environment         #
# resolves a CUDA-enabled torch wheel. See:                                             #
#                                                                                       #
#     docs/guides/INSTALLATION_GUIDE.md#installing-with-build-isolation                 #
#########################################################################################
""")


def load_config(default_config: Optional[dict] = None) -> dict:
    """Load configuration from environment variables or use defaults

    If no default configuration is provided, a default configuration is used. In this case, the default
    configuration is:

        'DEBUG_BUILD': False,
        'OPTIMIZE_LEVEL': 3,
        'CPP_STANDARD': 'c++17',
        'VERBOSE_BUILD': False,
        'CUSTOM_CUDA_ARCHS': None,
        'USE_FAST_MATH': True,
        'ENABLE_PROFILING': False,

    Args:
        default_config: Default configuration dictionary. If `None`, a default configuration is used.

    Returns:
        dict: Configuration with environment variable overrides
    """
    if default_config is None:
        default_config = {
            'DEBUG_BUILD': False,
            'OPTIMIZE_LEVEL': 3,
            'CPP_STANDARD': 'c++17',
            'VERBOSE_BUILD': False,
            'CUSTOM_CUDA_ARCHS': None,
            'USE_FAST_MATH': True,
            'ENABLE_PROFILING': False,
        }

    config = default_config.copy()

    # Override with environment variables if present
    for key in config:
        if key in os.environ:
            env_val = os.environ[key]
            if isinstance(config[key], bool):
                config[key] = env_val.lower() in ('1', 'true', 'yes', 'on')
            elif isinstance(config[key], int):
                config[key] = int(env_val)
            elif key == 'CUSTOM_CUDA_ARCHS':
                config[key] = _split_cuda_architectures(env_val) if env_val else None
            else:
                config[key] = env_val

    return config


def detect_cuda_info():
    """Detect CUDA availability and GPU architectures.

    Missing or CPU-only PyTorch is treated as a build configuration error:
    ACCV-Lab CUDA extension builds require a CUDA-enabled PyTorch wheel.

    Returns:
        dict: CUDA information including availability and GPU architectures

    Raises:
        RuntimeError: If PyTorch is not installed or is installed without CUDA support.
    """
    cuda_info = {
        'cuda_available': False,
        'gpu_architectures': [],
    }

    try:
        import torch

        require_torch_cuda_support(torch)

        if torch.cuda.is_available():
            cuda_info['cuda_available'] = True

            # Get GPU architectures
            for i in range(torch.cuda.device_count()):
                capability = torch.cuda.get_device_capability(i)
                arch = f"{capability[0]}{capability[1]}"
                if arch not in cuda_info['gpu_architectures']:
                    cuda_info['gpu_architectures'].append(arch)
    except ImportError as exc:
        raise missing_torch_error() from exc

    return cuda_info


def get_compile_flags(config, cuda_info, include_dirs=None):
    """Construct compilation flags.

    If ``CUSTOM_CUDA_ARCHS`` is unset, detected CUDA architectures are emitted
    as cubin targets only when ``nvcc`` reports exact support. Unsupported
    detections fall back to supported PTX at or below the detected architecture.
    If no architecture can be detected, no explicit CUDA architecture flags are
    generated.

    Args:
        config (dict): Build configuration
        cuda_info (dict): CUDA information from detect_cuda_info()
        include_dirs (list): Additional include directories

    Returns:
        dict: Compilation flags for cxx, nvcc, and include_dirs
    """
    flags = {
        'cxx': [],
        'nvcc': [],
        'include_dirs': [],
    }

    # Base C++ flags
    flags['cxx'].extend(
        [
            f'-std={config["CPP_STANDARD"]}',
            f'-O{config["OPTIMIZE_LEVEL"]}',
            '-fPIC',
            '-Wall',
            '-Wextra',
            '-pthread',
        ]
    )

    # Add fast math for C++ if enabled
    if config['USE_FAST_MATH']:
        flags['cxx'].append('-ffast-math')

    # Debug flags
    if config['DEBUG_BUILD']:
        flags['cxx'].extend(['-g', '-DDEBUG'])
        flags['nvcc'].extend(['-g', '-G', '-DDEBUG'])

    # Profiling flags
    if config['ENABLE_PROFILING']:
        flags['cxx'].append('-pg')

    # Include directories
    if include_dirs:
        flags['include_dirs'].extend(include_dirs)

    # Default include directories
    flags['include_dirs'].extend(['/usr/local/include'])

    # CUDA flags (only if CUDA is available)
    if cuda_info['cuda_available']:
        ptx_archs: List[str] = []
        if config['CUSTOM_CUDA_ARCHS'] is not None:
            cuda_archs = config['CUSTOM_CUDA_ARCHS']
        else:
            arch_selection = select_cuda_architectures_for_nvcc(cuda_info['gpu_architectures'])
            cuda_archs = arch_selection.architectures
            ptx_archs = arch_selection.ptx_architectures

        # Generate architecture flags
        for arch in cuda_archs:
            flags['nvcc'].extend([f'-gencode=arch=compute_{arch},code=sm_{arch}'])
        for arch in ptx_archs:
            flags['nvcc'].extend([f'-gencode=arch=compute_{arch},code=compute_{arch}'])

        # CUDA compilation flags
        flags['nvcc'].extend(
            [
                f'-std={config["CPP_STANDARD"]}',
                f'-O{config["OPTIMIZE_LEVEL"]}',
                '--use_fast_math' if config['USE_FAST_MATH'] else '--ftz=false',
                '--generate-line-info',
                '-Xptxas=-v' if config['VERBOSE_BUILD'] else '',
            ]
        )

        # Remove empty flags
        flags['nvcc'] = [f for f in flags['nvcc'] if f]

    return flags


def run_external_build(
    package_dir: str, ext_impl_dir: str = 'ext_impl', build_script_name: str = 'build_and_copy.sh'
):
    """
    Run the external build script if it exists.

    Args:
        package_dir (str): Path to the package directory.
        ext_impl_dir (str): Path to the external implementation directory (relative to `package_dir`).
        build_script_name (str): Name of the build script.
    """

    build_script = Path(package_dir) / ext_impl_dir / build_script_name
    if build_script.exists():
        try:
            subprocess.run(['bash', str(build_script)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running external build script: {e}")
            raise
    else:
        print(f"No external build script found at {build_script}")


def get_abs_setup_dir(filename: str) -> Path:
    """
    Get the absolute path of the setup.py file's parent directory.

    Args:
        filename (str): Path to the setup.py file (as set in the __file__ variable of the setup.py file).

    Returns:
        Path: Absolute path of the setup.py file's parent directory.
    """
    try:
        path = os.path.abspath(filename)
    except NameError:
        path = os.path.abspath(sys.argv[0])
    return Path(path).parent
