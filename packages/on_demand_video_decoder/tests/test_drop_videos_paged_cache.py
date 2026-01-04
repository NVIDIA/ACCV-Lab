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
Tests for drop_videos_cache utility function.

This module tests the Linux paged cache eviction functionality provided by
the drop_videos_cache API. The function uses posix_fadvise with POSIX_FADV_DONTNEED
to advise the kernel to release cached pages for specified video files.

Status codes (DropCacheStatus enum):
    - success: operation completed successfully
    - platform_error: not Linux
    - fadvise_failed: file not found, permission denied, etc.

Note:
    - These tests are primarily designed for Linux systems.
    - On non-Linux systems, the function returns DropCacheStatus.platform_error.
    - The function uses fail-fast mode: stops at first error.
"""

import pytest
import sys
import os

import utils as test_utils
import accvlab.on_demand_video_decoder as nvc
from accvlab.on_demand_video_decoder import DropCacheStatus


def get_test_video_files():
    """
    Get test video files from the data directory.

    Returns:
        List of video file paths, or empty list if not found.
    """
    path_base = test_utils.get_data_dir()
    if not os.path.exists(path_base):
        return []

    files = test_utils.select_random_clip(path_base)
    return files if files else []


class TestDropVideosCacheAPI:
    """
    Test class for drop_videos_cache API functionality.

    Tests cover:
    - Handling of non-existent files
    - Handling of empty file list
    - Mixed valid/invalid files (fail-fast behavior)
    - Integration with video decoding
    """

    def test_drop_videos_cache_with_nonexistent_files(self):
        """
        Test: Call drop_videos_cache with non-existent file paths.

        On Linux: Should return status code 2 (fadvise failed) for first file.
        On non-Linux: Should return status code 1 (platform error).
        """
        print("\n=== Test: drop_videos_cache with non-existent files ===")

        nonexistent_files = [
            "/nonexistent/path/video1.mp4",
            "/nonexistent/path/video2.mp4",
            "/tmp/definitely_not_a_real_file_12345.mp4",
        ]

        result = nvc.drop_videos_cache(nonexistent_files)

        print(f"drop_videos_cache returned: {result}")

        if sys.platform.startswith('linux'):
            # On Linux, should fail with fadvise error for non-existent file
            assert (
                result == DropCacheStatus.fadvise_failed
            ), f"Expected {DropCacheStatus.fadvise_failed} for non-existent files, got {result}"
            print("Test passed: Returned fadvise_failed for non-existent files")
        else:
            # On non-Linux systems, should return platform error
            assert (
                result == DropCacheStatus.platform_error
            ), f"Expected {DropCacheStatus.platform_error} on non-Linux system, got {result}"
            print(f"Test passed: Returned platform_error on {sys.platform}")

    def test_drop_videos_cache_with_empty_list(self):
        """
        Test: Call drop_videos_cache with an empty file list.

        Should return status code 0 (success) as there are no files to process.
        """
        print("\n=== Test: drop_videos_cache with empty list ===")

        result = nvc.drop_videos_cache([])

        print(f"drop_videos_cache returned: {result}")

        assert result == DropCacheStatus.success, f"Expected {DropCacheStatus.success} for empty list, got {result}"

        print("Test passed: Returned success for empty list")

    @pytest.mark.parametrize("invalid_first", [False, True], ids=["valid_first", "invalid_first"])
    def test_drop_videos_cache_with_mixed_files(self, invalid_first):
        """
        Test: Call drop_videos_cache with mixed valid and invalid files.

        Tests two scenarios using parametrization:
        1. invalid_first=False: valid files first, then invalid file (tests fail-fast after processing valid files)
        2. invalid_first=True: invalid file first, then valid files (tests immediate fail-fast)

        Due to fail-fast behavior:
        - On Linux: Should return fadvise_failed on first invalid file encountered
        - On non-Linux: Should return platform_error immediately
        """
        print(f"\n=== Test: drop_videos_cache with mixed files (invalid_first={invalid_first}) ===")

        valid_files = get_test_video_files()
        if not valid_files:
            pytest.skip("No test video files available")

        invalid_file = "/nonexistent/video.mp4"

        # Construct file list based on parameter
        if invalid_first:
            mixed_files = [invalid_file] + valid_files
            print(f"Mixed files (invalid first): {mixed_files}")
        else:
            mixed_files = valid_files + [invalid_file]
            print(f"Mixed files (valid first): {mixed_files}")

        result = nvc.drop_videos_cache(mixed_files)

        print(f"drop_videos_cache returned: {result}")

        if sys.platform.startswith('linux'):
            # On Linux, should fail on the first invalid file encountered
            assert (
                result == DropCacheStatus.fadvise_failed
            ), f"Expected {DropCacheStatus.fadvise_failed} (fail on invalid file), got {result}"
            if invalid_first:
                print("Test passed: Fail-fast returned fadvise_failed immediately on first invalid file")
            else:
                print("Test passed: Fail-fast returned fadvise_failed after processing valid files")
        else:
            # On non-Linux systems, should return platform error immediately
            assert (
                result == DropCacheStatus.platform_error
            ), f"Expected {DropCacheStatus.platform_error} on non-Linux system, got {result}"
            print(f"Test passed: Returned platform_error on {sys.platform}")

    def test_drop_videos_cache_after_decode(self):
        """
        Test: Integration test - decode video then drop its cache.

        This test verifies that drop_videos_cache can be called after video decoding
        and returns success status. Note that actual cache eviction depends on
        system environment and kernel decisions.
        """
        print("\n=== Test: drop_videos_cache after video decode ===")

        if not sys.platform.startswith('linux'):
            print(f"Non-Linux platform ({sys.platform}), skipping test")
            pytest.skip("This test only runs on Linux")

        files = get_test_video_files()
        if not files:
            pytest.skip("No test video files available")

        print(f"Test files: {files}")

        # Create decoder and decode frames
        reader = nvc.CreateSampleReader(
            num_of_set=1,
            num_of_file=len(files),
            iGpu=0,
        )

        # Decode multiple frames
        num_frames_to_decode = 30
        for frame_idx in range(num_frames_to_decode):
            frame_ids = [frame_idx] * len(files)
            try:
                _ = reader.DecodeN12ToRGB(files, frame_ids, True)
            except Exception as e:
                print(f"Decode stopped at frame {frame_idx}: {e}")
                break

        print(f"Decoded {num_frames_to_decode} frames from {len(files)} files")

        # Release decoder file handles before dropping cache
        reader.clearAllReaders()

        # Drop the page cache for these videos
        result = nvc.drop_videos_cache(files)
        print(f"drop_videos_cache returned: {result}")

        assert result == DropCacheStatus.success, f"Expected {DropCacheStatus.success}, got {result}"

        print("Test passed: drop_videos_cache returned success after decode")

        # Clean up
        del reader


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
