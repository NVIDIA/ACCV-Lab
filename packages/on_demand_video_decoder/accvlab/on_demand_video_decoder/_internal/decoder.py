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
GOP decoder with Python-side caching support.

This module provides the CachedGopDecoder class and CreateGopDecoder factory function
that wrap the underlying C++ PyNvGopDecoder with transparent GOP caching functionality.
"""

from typing import List, Tuple, Any
import numpy as np

from .. import _CreateGopDecoderCpp


class CachedGopDecoder:
    """
    Python wrapper for PyNvGopDecoder with GOP caching support.

    This wrapper provides transparent GOP caching functionality that can significantly
    reduce redundant demuxing operations when the same GOP data is requested multiple times.

    The caching is controlled by the ``useGOPCache`` parameter in the GetGOP method:

    - When useGOPCache=False (default): Behaves exactly like the original C++ decoder
    - When useGOPCache=True: Caches GOP data and returns cached results when the requested
      frame_id falls within a previously cached GOP range

    Cache hit condition for each file: ``first_frame_id <= frame_id < first_frame_id + gop_len``

    Attributes:
        _decoder: The underlying C++ PyNvGopDecoder instance
        _gop_cache: Dictionary mapping filepath to cached (packets, first_frame_id, gop_len)
    """

    def __init__(self, decoder):
        """
        Initialize the cached GOP decoder wrapper.

        Args:
            decoder: The underlying C++ PyNvGopDecoder instance to wrap
        """
        self._decoder = decoder
        # Cache structure: {filepath: (packets_numpy, first_frame_id, gop_len)}
        self._gop_cache = {}
        # Track cache hit status for each file in the last GetGOP call
        self._last_cache_hits = []

    def _is_cache_hit(self, filepath: str, frame_id: int) -> bool:
        """
        Check if the requested frame_id is within the cached GOP range for the given filepath.

        Args:
            filepath: The video file path to check
            frame_id: The target frame index

        Returns:
            True if cache hit (frame_id is within cached GOP range), False otherwise
        """
        if filepath not in self._gop_cache:
            return False
        _, first_frame_id, gop_len = self._gop_cache[filepath]
        return first_frame_id <= frame_id < first_frame_id + gop_len

    def GetGOP(
        self,
        filepaths: List[str],
        frame_ids: List[int],
        fastStreamInfos: List[Any] = [],
        useGOPCache: bool = False,
    ) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Extract GOP data from video files with optional caching support.

        This method wraps the underlying C++ GetGOP with Python-side caching.
        When useGOPCache=True, it checks if all requested files have cache hits
        (i.e., the requested frame_ids fall within previously cached GOP ranges).
        If all hit, cached data is returned directly without calling C++ code.

        Args:
            filepaths: List of video file paths to extract GOP data from
            frame_ids: List of frame IDs to extract GOP data for (one per file)
            fastStreamInfos: Optional list of FastStreamInfo objects for fast initialization
            useGOPCache: If True, enables GOP caching. Default is False.

        Returns:
            Tuple containing:
            - numpy array with serialized GOP data (merged if multiple files)
            - list of first frame IDs for each GOP
            - list of GOP lengths for each GOP

        Example:
            >>> decoder = CreateGopDecoder(maxfiles=6, iGpu=0)
            >>> # First call - fetches from video files
            >>> packets, first_ids, gop_lens = decoder.GetGOP(files, [77, 77], useGOPCache=True)
            >>> # Second call with frame_id in same GOP range - returns from cache
            >>> packets, first_ids, gop_lens = decoder.GetGOP(files, [80, 80], useGOPCache=True)
        """
        if not useGOPCache:
            # No caching, directly call C++ implementation
            self._last_cache_hits = [False] * len(filepaths)
            return self._decoder.GetGOP(filepaths, frame_ids, fastStreamInfos)

        # Check cache hits for each file
        cache_hits = [self._is_cache_hit(fp, fid) for fp, fid in zip(filepaths, frame_ids)]
        self._last_cache_hits = cache_hits

        if all(cache_hits):
            # All cache hits - return merged cached data
            return self._get_from_cache(filepaths)

        # At least one cache miss - need to fetch from C++
        # Use GetGOPList to get per-file data for individual caching
        results = self._decoder.GetGOPList(filepaths, frame_ids, fastStreamInfos)

        # Update cache with new data
        for filepath, (packets, first_frame_ids, gop_lens) in zip(filepaths, results):
            # Each result contains data for a single file
            # first_frame_ids and gop_lens are lists with single element
            self._gop_cache[filepath] = (packets, first_frame_ids[0], gop_lens[0])

        # Merge and return in GetGOP format
        return self._merge_cached_data(filepaths)

    def _get_from_cache(self, filepaths: List[str]) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Retrieve and merge cached GOP data for the given filepaths.

        Args:
            filepaths: List of video file paths to retrieve from cache

        Returns:
            Tuple of (merged_packets, first_frame_ids, gop_lens)
        """
        return self._merge_cached_data(filepaths)

    def _merge_cached_data(self, filepaths: List[str]) -> Tuple[np.ndarray, List[int], List[int]]:
        """
        Merge cached GOP data from multiple files into a single result.

        Uses the C++ MergePacketDataToOne method for efficient binary merging.

        Args:
            filepaths: List of video file paths to merge data from

        Returns:
            Tuple of (merged_packets, first_frame_ids, gop_lens)
        """
        packets_list = []
        first_frame_ids = []
        gop_lens = []

        for filepath in filepaths:
            packets, first_fid, gop_len = self._gop_cache[filepath]
            packets_list.append(packets)
            first_frame_ids.append(first_fid)
            gop_lens.append(gop_len)

        if len(packets_list) == 1:
            # Single file, no merge needed
            return packets_list[0], first_frame_ids, gop_lens

        # Merge multiple packet arrays using C++ implementation
        merged_packets = self._decoder.MergePacketDataToOne(packets_list)
        return merged_packets, first_frame_ids, gop_lens

    def clear_cache(self) -> None:
        """
        Clear all cached GOP data.

        Call this method to free memory when cached data is no longer needed.
        """
        self._gop_cache.clear()

    def get_cache_info(self) -> dict:
        """
        Get information about the current cache state.

        Returns:
            Dictionary with cache statistics and per-file information
        """
        info = {"cached_files_count": len(self._gop_cache), "cached_files": {}}
        for filepath, (packets, first_fid, gop_len) in self._gop_cache.items():
            info["cached_files"][filepath] = {
                "first_frame_id": first_fid,
                "gop_len": gop_len,
                "frame_range": (first_fid, first_fid + gop_len - 1),
                "packets_size_bytes": packets.nbytes if hasattr(packets, 'nbytes') else len(packets),
            }
        return info

    def isCacheHit(self) -> List[bool]:
        """
        Get cache hit status for each file in the last GetGOP/GetGOPList call.

        Returns:
            List of booleans, one per file in the last GetGOP/GetGOPList call.
            True indicates cache hit, False indicates cache miss.
            Returns empty list if GetGOP/GetGOPList has not been called yet.

        Example:
            >>> decoder = CreateGopDecoder(maxfiles=6, iGpu=0)
            >>> files = ['video1.mp4', 'video2.mp4', 'video3.mp4']
            >>> packets, first_ids, gop_lens = decoder.GetGOP(files, [77, 77, 77], useGOPCache=True)
            >>> cache_hits = decoder.isCacheHit()
            >>> # cache_hits = [False, False, False]  # First call, all miss
            >>>
            >>> packets, first_ids, gop_lens = decoder.GetGOP(files, [80, 80, 80], useGOPCache=True)
            >>> cache_hits = decoder.isCacheHit()
            >>> # cache_hits = [True, True, True]  # Second call in same GOP range, all hit
        """
        return self._last_cache_hits

    def GetGOPList(
        self,
        filepaths: List[str],
        frame_ids: List[int],
        fastStreamInfos: List[Any] = [],
        useGOPCache: bool = False,
    ) -> List[Tuple[np.ndarray, List[int], List[int]]]:
        """
        Extract per-video GOP data with optional caching support.

        Unlike GetGOP which returns merged data, GetGOPList returns separate
        GOP data for each video, enabling more granular control and caching.

        When useGOPCache=True, this method:
        1. Checks cache hits for each file individually
        2. Only fetches data from C++ for cache misses
        3. Updates cache with new data
        4. Returns results from cache (preserving original order)

        Args:
            filepaths: List of video file paths to extract GOP data from
            frame_ids: List of frame IDs to extract GOP data for (one per file)
            fastStreamInfos: Optional list of FastStreamInfo objects for fast initialization
            useGOPCache: If True, enables GOP caching. Default is False.

        Returns:
            List of tuples, one per video file, each containing:
            - numpy array with serialized GOP data for that video
            - list of first frame IDs for each GOP in that video
            - list of GOP lengths for each GOP in that video

        Example:
            >>> decoder = CreateGopDecoder(maxfiles=6, iGpu=0)
            >>> files = ['video1.mp4', 'video2.mp4']
            >>> # First call - fetches from video files
            >>> gop_list = decoder.GetGOPList(files, [77, 77], useGOPCache=True)
            >>> print(decoder.isCacheHit())  # [False, False]
            >>>
            >>> # Second call with frame_id in same GOP range - returns from cache
            >>> gop_list = decoder.GetGOPList(files, [80, 80], useGOPCache=True)
            >>> print(decoder.isCacheHit())  # [True, True]
            >>>
            >>> # Use with DecodeFromGOPListRGB
            >>> gop_data_list = [data for data, _, _ in gop_list]
            >>> frames = decoder.DecodeFromGOPListRGB(gop_data_list, files, [80, 80], True)
        """
        if not useGOPCache:
            # No caching, directly call C++ implementation
            self._last_cache_hits = [False] * len(filepaths)
            return self._decoder.GetGOPList(filepaths, frame_ids, fastStreamInfos)

        # Check cache hits for each file
        cache_hits = [self._is_cache_hit(fp, fid) for fp, fid in zip(filepaths, frame_ids)]
        self._last_cache_hits = cache_hits

        # Find indices of cache misses
        miss_indices = [i for i, hit in enumerate(cache_hits) if not hit]

        if miss_indices:
            # Fetch data for cache misses only
            miss_filepaths = [filepaths[i] for i in miss_indices]
            miss_frame_ids = [frame_ids[i] for i in miss_indices]
            miss_fast_infos = [fastStreamInfos[i] for i in miss_indices] if fastStreamInfos else []

            miss_results = self._decoder.GetGOPList(miss_filepaths, miss_frame_ids, miss_fast_infos)

            # Update cache with new data
            for idx, (packets, first_frame_ids_list, gop_lens_list) in zip(miss_indices, miss_results):
                filepath = filepaths[idx]
                # Each result contains data for a single file
                # first_frame_ids_list and gop_lens_list are lists with single element
                self._gop_cache[filepath] = (packets, first_frame_ids_list[0], gop_lens_list[0])

        # Build results from cache in original order
        results = []
        for filepath in filepaths:
            packets, first_fid, gop_len = self._gop_cache[filepath]
            # Return in GetGOPList format: (packets, [first_frame_id], [gop_len])
            results.append((packets, [first_fid], [gop_len]))

        return results

    def __getattr__(self, name: str) -> Any:
        """
        Proxy all other attribute accesses to the underlying C++ decoder.

        This ensures that all methods not explicitly overridden (like DecodeFromGOPRGB,
        DecodeFromGOPListRGB, etc.) are transparently forwarded to the C++ implementation.

        Args:
            name: The attribute name to access

        Returns:
            The attribute from the underlying decoder
        """
        return getattr(self._decoder, name)


def CreateGopDecoder(
    maxfiles: int, iGpu: int = 0, suppressNoColorRangeWarning: bool = False
) -> CachedGopDecoder:
    """
    Initialize GOP decoder with set of particular parameters.

    This factory function creates a CachedGopDecoder instance that wraps the
    underlying C++ PyNvGopDecoder with Python-side GOP caching support.

    Args:
        maxfiles: Maximum number of unique files that can be processed concurrently
        iGpu: GPU device ID to use for decoding (0 for primary GPU)
        suppressNoColorRangeWarning: Suppress warning when no color range can be
                                     extracted from video files (limited/MPEG range is assumed)

    Returns:
        CachedGopDecoder instance configured with the specified parameters

    Raises:
        RuntimeError: If GPU initialization fails or parameters are invalid

    Example:
        >>> decoder = CreateGopDecoder(maxfiles=3, iGpu=0)
        >>> # Use with caching enabled
        >>> packets, fids, glens = decoder.GetGOP(['v0.mp4'], [10], useGOPCache=True)
        >>> # Subsequent calls with frame_id in same GOP return cached data
        >>> packets, fids, glens = decoder.GetGOP(['v0.mp4'], [15], useGOPCache=True)
    """
    cpp_decoder = _CreateGopDecoderCpp(maxfiles, iGpu, suppressNoColorRangeWarning)
    return CachedGopDecoder(cpp_decoder)
