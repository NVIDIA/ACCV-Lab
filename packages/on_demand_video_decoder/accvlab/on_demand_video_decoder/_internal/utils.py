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
Python utility functions module.

Provides auxiliary functionality related to video decoding, such as Linux paged cache management.
"""

import os
import sys
from enum import Enum
from typing import List


class DropCacheStatus(Enum):
    """Status codes for drop_videos_cache operations."""

    SUCCESS = 0
    PLATFORM_ERROR = 1
    FILE_OPEN_FAILED = 2
    FADVISE_FAILED = 3
    UNKNOWN_ERROR = 4


def _drop_single_video_cache(filepath: str) -> DropCacheStatus:
    """
    Internal function to evict a single video file's cached pages from Linux paged cache.

    Args:
        filepath: Path to the video file.

    Returns:
        DropCacheStatus enum:
        - SUCCESS: cache dropped successfully
        - PLATFORM_ERROR: not Linux platform
        - FILE_OPEN_FAILED: failed to open file (file not found, permission denied, etc.)
        - FADVISE_FAILED: posix_fadvise call failed
        - UNKNOWN_ERROR: unexpected error occurred
    """
    # Only Linux supports posix_fadvise with POSIX_FADV_DONTNEED
    if not sys.platform.startswith('linux'):
        return DropCacheStatus.PLATFORM_ERROR

    try:
        fd = os.open(filepath, os.O_RDONLY)
    except (OSError, IOError):
        return DropCacheStatus.FILE_OPEN_FAILED
    except Exception:
        return DropCacheStatus.UNKNOWN_ERROR

    try:
        file_size = os.fstat(fd).st_size
        # POSIX_FADV_DONTNEED advises the kernel that these pages are no longer needed
        os.posix_fadvise(fd, 0, file_size, os.POSIX_FADV_DONTNEED)
        return DropCacheStatus.SUCCESS
    except (OSError, IOError):
        return DropCacheStatus.FADVISE_FAILED
    except Exception:
        return DropCacheStatus.UNKNOWN_ERROR
    finally:
        os.close(fd)


def drop_videos_cache(filepaths: List[str]) -> DropCacheStatus:
    """
    Evict cached pages for multiple video files from Linux paged cache.

    Uses posix_fadvise with POSIX_FADV_DONTNEED flag to advise the kernel that
    these pages are no longer needed, causing the kernel to remove them from
    the page cache. This is useful when switching video datasets during training
    to release memory cache occupied by old videos.

    This function uses fail-fast mode: it stops processing and returns immediately
    when the first error occurs.

    Args:
        filepaths: List of video file paths.

    Returns:
        DropCacheStatus enum

        - SUCCESS: all files processed successfully
        - PLATFORM_ERROR: not Linux platform
        - FILE_OPEN_FAILED: failed to open file (file not found, permission denied, etc.)
        - FADVISE_FAILED: posix_fadvise call failed
        - UNKNOWN_ERROR: unexpected error occurred

    Note:
        - This is an advisory operation; the kernel may ignore the request
          depending on system state.
        - This function only works on Linux systems; other platforms return PLATFORM_ERROR.
        - Processing stops at the first error (fail-fast mode).
        - File contents are not affected; only the in-memory cached copy is released.

    Warning:
        The actual cache eviction behavior is influenced by system environment factors:

        - System memory pressure and kernel memory management policies affect
          whether the advisory is honored.
        - On shared systems with many concurrent processes, cache state may change
          due to other processes' I/O activities.

    Example:
        >>> import accvlab.on_demand_video_decoder as nvc
        >>> from accvlab.on_demand_video_decoder import DropCacheStatus
        >>> video_files = ["/path/to/video1.mp4", "/path/to/video2.mp4"]
        >>> status = nvc.drop_videos_cache(video_files)
        >>> if status == DropCacheStatus.SUCCESS:
        ...     print("Successfully dropped cache for all files")
        >>> elif status == DropCacheStatus.PLATFORM_ERROR:
        ...     print("Platform not supported (not Linux)")
        >>> elif status == DropCacheStatus.FILE_OPEN_FAILED:
        ...     print("Failed to open file (not found or permission denied)")
        >>> elif status == DropCacheStatus.FADVISE_FAILED:
        ...     print("posix_fadvise call failed")
        >>> elif status == DropCacheStatus.UNKNOWN_ERROR:
        ...     print("Unexpected error occurred")
    """
    for filepath in filepaths:
        status = _drop_single_video_cache(filepath)
        if status != DropCacheStatus.SUCCESS:
            return status
    return DropCacheStatus.SUCCESS
