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
from typing import NamedTuple


class GopRef(NamedTuple):
    """Lightweight, picklable reference to GOP data in shared memory.

    Designed to be passed through DataLoader IPC queues (tens of bytes)
    instead of the actual GOP packet data (tens of KB).  The main process
    calls ``SharedGopStore.get_batch()`` to read the referenced shm blocks
    as zero-copy numpy views.

    Attributes:
        shm_name: POSIX SharedMemory name for the data block.
        data_size: Number of bytes of GOP packet data.
        first_frame_id: First frame index covered by this GOP.
        gop_len: Number of frames in this GOP.
    """
    shm_name: str
    data_size: int
    first_frame_id: int
    gop_len: int


class Codec(Enum):
    """
    Video codec enumeration matching CUDA Video Codec SDK codec IDs.

    These values correspond to cudaVideoCodec enum values used by
    the underlying NVIDIA hardware decoder.
    """

    h264 = 4
    hevc = 8
    av1 = 11
