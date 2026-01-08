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
Type definitions and enumerations for the video decoder module.
"""

from enum import Enum


class Codec(Enum):
    """
    Video codec enumeration matching CUDA Video Codec SDK codec IDs.

    These values correspond to cudaVideoCodec enum values used by
    the underlying NVIDIA hardware decoder.
    """

    h264 = 4
    hevc = 8
    av1 = 11
