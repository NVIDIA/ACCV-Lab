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

""" """

__author__ = "NVIDIA"
__copyright__ = "Copyright 2024, NVIDIA"
__credits__ = []
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = "NVIDIA"
__email__ = "TODO"
__status__ = "Production"


from ast import Str
import string
import os
import ctypes


def _preload_local_ffmpeg() -> None:
    base_dir = os.path.dirname(__file__)
    required = [
        "libavutil.so.56",
        "libswresample.so.3",
        "libavcodec.so.58",
        "libavformat.so.58",
    ]
    missing = [name for name in required if not os.path.exists(os.path.join(base_dir, name))]
    if missing:
        raise ImportError(f"Missing local FFmpeg libs next to module: {missing}")

    mode = getattr(ctypes, "RTLD_GLOBAL", 0) | getattr(ctypes, "RTLD_NOW", 0)
    for name in ["libavutil.so.56", "libswresample.so.3", "libavcodec.so.58", "libavformat.so.58"]:
        path = os.path.join(base_dir, name)
        ctypes.CDLL(path, mode=mode)


try:
    # Temporarily clear LD_LIBRARY_PATH so RPATH ($ORIGIN) + our preloads are honored
    _prev_ld_library_path = os.environ.pop("LD_LIBRARY_PATH", None)
    try:
        _preload_local_ffmpeg()
        from ._PyNvOnDemandDecoder import *  # noqa
    finally:
        if _prev_ld_library_path is not None:
            os.environ["LD_LIBRARY_PATH"] = _prev_ld_library_path
    from enum import Enum
except ImportError:
    import distutils.sysconfig
    from os.path import join, dirname

    raise RuntimeError(
        "Failed to import native module _PyNvOnDemandDecoder! "
        f"Please check whether \"{join(dirname(__file__), '_PyNvOnDemandDecoder' + distutils.sysconfig.get_config_var('EXT_SUFFIX'))}\""  # noqa
        " exists and can find all library dependencies (CUDA, local FFmpeg libraries).\n"
    )


class Codec(Enum):
    h264 = 4
    hevc = 8
    av1 = 11


# Import Python utility functions
from .utils import drop_videos_cache, DropCacheStatus

__all__ = [
    # C++ core interfaces
    'PyNvGopDecoder',
    'PyNvSampleReader',
    'FastStreamInfo',
    'DecodedFrameExt',
    'RGBFrame',
    'CreateGopDecoder',
    'CreateSampleReader',
    'GetFastInitInfo',
    'SavePacketsToFile',
    'Codec',
    # Python utility functions
    'drop_videos_cache',
    'DropCacheStatus',
]
