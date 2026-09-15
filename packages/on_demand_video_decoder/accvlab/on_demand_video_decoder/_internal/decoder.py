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
GOP decoder with caching support.

This module provides the CachedGopDecoder class and CreateGopDecoder factory function
for video GOP extraction with transparent caching functionality.
"""

from collections import OrderedDict
from typing import List, Tuple, Any, Optional
import numpy as np

from .. import _CreateGopDecoderCpp, PyNvGopDecoder

# Private key to prevent direct instantiation of CachedGopDecoder
_CREATION_KEY = object()


class CachedGopDecoder:
    """
    GOP decoder with transparent GOP caching.

    This class extends :class:`PyNvGopDecoder`: all of its methods are available
    on this class, and :meth:`GetGOPList` additionally accepts a ``useGOPCache``
    parameter that caches serialized GOP bundles to avoid redundant demuxing when
    frames from the same GOP are requested multiple times. See :meth:`GetGOPList`
    for the caching behavior.

    Do not instantiate this class directly. Use :func:`CreateGopDecoder` to
    obtain an instance.

    See Also:
        :class:`PyNvGopDecoder`: The underlying decoder class with full method documentation.
    """

    def __init__(self, decoder: PyNvGopDecoder, cache_capacity: int, *, _key=None) -> None:
        """
        Initialize the cached GOP decoder.

        Note:
            Do not instantiate this class directly.
            Use :func:`CreateGopDecoder` instead.

        Args:
            decoder: The internal decoder instance

        Raises:
            RuntimeError: If called directly instead of using CreateGopDecoder()
        """
        if _key is not _CREATION_KEY:
            raise RuntimeError(
                "CachedGopDecoder cannot be instantiated directly. " "Use CreateGopDecoder() instead."
            )
        if isinstance(cache_capacity, bool) or not isinstance(cache_capacity, int):
            raise TypeError("cache_capacity must be a positive integer")
        if cache_capacity < 1:
            raise ValueError("cache_capacity must be positive")
        self._decoder = decoder
        # Cache structure:
        # {filepath: (packets_numpy, first_frame_id, gop_len, graph_optimization_enabled)}.
        # Each filepath stores only one GOP. The OrderedDict keeps LRU order and
        # is bounded by gopCacheCapacity from CreateGopDecoder().
        self._gop_cache = OrderedDict()
        self._cache_capacity = cache_capacity
        # Track cache hit status for each file in the last GetGOPList call
        self._last_cache_hits = []

    def _is_cache_hit(
        self,
        filepath: str,
        frame_id: int,
        enable_gop_dependency_graph_optimization: bool,
    ) -> bool:
        """
        Check if the requested frame_id is within the cached GOP range for the given filepath.

        Args:
            filepath: The video file path to check
            frame_id: The target frame index

        Returns:
            True if cache hit (frame_id is within cached GOP range), False otherwise
        """
        entry = self._gop_cache.get(filepath)
        if entry is None:
            return False
        _, first_frame_id, gop_len, cached_with_dependency_graph = entry
        hit = (
            cached_with_dependency_graph == enable_gop_dependency_graph_optimization
            and first_frame_id <= frame_id < first_frame_id + gop_len
        )
        if hit:
            self._gop_cache.move_to_end(filepath)
        return hit

    def _update_cache(
        self,
        filepath: str,
        packets: np.ndarray,
        first_frame_id: int,
        gop_len: int,
        enable_gop_dependency_graph_optimization: bool,
    ) -> None:
        self._gop_cache[filepath] = (
            packets,
            first_frame_id,
            gop_len,
            enable_gop_dependency_graph_optimization,
        )
        self._gop_cache.move_to_end(filepath)
        while len(self._gop_cache) > self._cache_capacity:
            self._gop_cache.popitem(last=False)

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
        info = {
            "cache_capacity": self._cache_capacity,
            "cached_files_count": len(self._gop_cache),
            "cached_files": {},
        }
        for filepath, (packets, first_fid, gop_len, graph_enabled) in self._gop_cache.items():
            info["cached_files"][filepath] = {
                "first_frame_id": first_fid,
                "gop_len": gop_len,
                "frame_range": (first_fid, first_fid + gop_len - 1),
                "packets_size_bytes": packets.nbytes if hasattr(packets, "nbytes") else len(packets),
                "enable_gop_dependency_graph_optimization": graph_enabled,
            }
        return info

    def isCacheHit(self) -> List[bool]:
        """
        Get cache hit status for each file in the last :meth:`GetGOPList` call.

        Returns:
            List of booleans, one per file in the last :meth:`GetGOPList` call.
            True indicates cache hit, False indicates cache miss.
            Returns empty list if :meth:`GetGOPList` has not been called yet.

        Example:
            >>> decoder = CreateGopDecoder(maxfiles=6, iGpu=0)
            >>> files = ['video1.mp4', 'video2.mp4', 'video3.mp4']
            >>> gops, first_ids, gop_lens = zip(*decoder.GetGOPList(files, [77, 77, 77], useGOPCache=True))
            >>> cache_hits = decoder.isCacheHit()
            >>> # cache_hits = [False, False, False]  # First call, all miss
            >>>
            >>> gops, first_ids, gop_lens = zip(*decoder.GetGOPList(files, [80, 80, 80], useGOPCache=True))
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
        enable_gop_dependency_graph_optimization: bool = False,
    ) -> List[Tuple[np.ndarray, List[int], List[int]]]:
        """
        Extract serialized GOP bundles with optional caching support.

        Same as :meth:`PyNvGopDecoder.GetGOPList`, with an additional ``useGOPCache``
        parameter.

        When ``useGOPCache=True``, cache hits are checked per file: only cache
        misses are demuxed, the cache is updated with the newly extracted bundles,
        and results are assembled in the same order as the input ``filepaths``. A cache hit for a file
        occurs when the requested frame_id falls within that file's previously
        cached GOP range (``first_frame_id <= frame_id < first_frame_id + gop_len``).
        When ``useGOPCache=False`` (default), the cache is bypassed.

        Args:
            filepaths: List of video file paths to extract GOP data from
            frame_ids: List of frame IDs to extract GOP data for (one per file)
            fastStreamInfos: Optional list of FastStreamInfo objects for fast initialization
            useGOPCache: If True, enables GOP caching. Default is False.
            enable_gop_dependency_graph_optimization: Enable GOP dependency
                optimization and write the GOP dependency graph into the returned
                GOP data. Defaults to False. Only applies to HEVC inputs.

        Returns:
            List of tuples, one per video file, each containing

            - serialized GOP bundle (numpy array) for that video; when dependency
              optimization is enabled for HEVC, the bundle includes the GOP
              dependency graph
            - list with the first frame ID of the extracted GOP
            - list with the length (frame count) of the extracted GOP

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
            >>> # Convert to PyTorch tensors on GPU (shape (height, width, 3), uint8)
            >>> rgb_tensors = [torch.as_tensor(frame).clone() for frame in frames]
        """
        if not useGOPCache:
            # No caching, directly call C++ implementation
            self._last_cache_hits = [False] * len(filepaths)
            return self._decoder.GetGOPList(
                filepaths,
                frame_ids,
                fastStreamInfos,
                enable_gop_dependency_graph_optimization,
            )

        # Check cache hits for each file
        cache_hits = [
            self._is_cache_hit(
                filepath,
                frame_id,
                enable_gop_dependency_graph_optimization,
            )
            for filepath, frame_id in zip(filepaths, frame_ids)
        ]
        self._last_cache_hits = cache_hits

        # Find indices of cache misses
        miss_indices = [i for i, hit in enumerate(cache_hits) if not hit]

        if miss_indices:
            # Fetch data for cache misses only
            miss_filepaths = [filepaths[i] for i in miss_indices]
            miss_frame_ids = [frame_ids[i] for i in miss_indices]
            miss_fast_infos = [fastStreamInfos[i] for i in miss_indices] if fastStreamInfos else []

            miss_results = self._decoder.GetGOPList(
                miss_filepaths,
                miss_frame_ids,
                miss_fast_infos,
                enable_gop_dependency_graph_optimization,
            )

            # Update cache with new data
            for idx, (packets, first_frame_ids_list, gop_lens_list) in zip(miss_indices, miss_results):
                filepath = filepaths[idx]
                # Each result contains data for a single file
                # first_frame_ids_list and gop_lens_list are lists with single element
                self._update_cache(
                    filepath,
                    packets,
                    first_frame_ids_list[0],
                    gop_lens_list[0],
                    enable_gop_dependency_graph_optimization,
                )

        # Build results from cache in original order
        results = []
        for filepath in filepaths:
            packets, first_fid, gop_len, _ = self._gop_cache[filepath]
            # Return in GetGOPList format: (packets, [first_frame_id], [gop_len])
            results.append((packets, [first_fid], [gop_len]))

        return results

    def __getattr__(self, name: str) -> Any:
        """
        Proxy all other attribute accesses to the internal decoder.

        This ensures that all methods not explicitly overridden (like :meth:`~PyNvGopDecoder.DecodeFromGOPListRGB`,
        etc.) are transparently forwarded.

        Args:
            name: The attribute name to access

        Returns:
            The attribute from the internal decoder
        """
        return getattr(self._decoder, name)


def CreateGopDecoder(
    maxfiles: int,
    iGpu: int = 0,
    suppressNoColorRangeWarning: bool = False,
    gopCacheCapacity: Optional[int] = None,
) -> CachedGopDecoder:
    """
    Create a GPU-accelerated video decoder with GOP-level random access.

    This factory function creates a :class:`CachedGopDecoder` instance with
    transparent GOP caching support.

    Args:
        maxfiles: Maximum number of unique files that can be processed concurrently
        iGpu: GPU device ID to use for decoding (0 for primary GPU)
        suppressNoColorRangeWarning: Suppress the warning emitted during RGB/BGR conversion
                                     when the input color range is unspecified. Limited/MPEG
                                     range is assumed regardless of this option.
        gopCacheCapacity: Maximum number of filepath entries kept in the Python GOP cache.
                          ``None`` defaults to ``maxfiles``. This capacity only affects
                          calls with ``useGOPCache=True``; each filepath stores the most
                          recently requested serialized GOP bundle, and least recently used
                          filepaths are evicted when the limit is exceeded.

    Returns:
        :class:`CachedGopDecoder` instance configured with the specified parameters

    Raises:
        RuntimeError: If parameters are invalid

    Example:
        >>> decoder = CreateGopDecoder(maxfiles=3, iGpu=0)
        >>> # Use with caching enabled
        >>> (gops, first_ids, gop_lens), = decoder.GetGOPList(['v0.mp4'], [10], useGOPCache=True)
        >>> # Subsequent calls with frame_id in same GOP return cached data
        >>> (gops, first_ids, gop_lens), = decoder.GetGOPList(['v0.mp4'], [15], useGOPCache=True)
    """
    if gopCacheCapacity is None:
        cache_capacity = maxfiles
    else:
        if isinstance(gopCacheCapacity, bool) or not isinstance(gopCacheCapacity, int):
            raise TypeError("gopCacheCapacity must be a positive integer or None")
        cache_capacity = gopCacheCapacity

    cpp_decoder = _CreateGopDecoderCpp(maxfiles, iGpu, suppressNoColorRangeWarning)
    return CachedGopDecoder(cpp_decoder, cache_capacity, _key=_CREATION_KEY)
