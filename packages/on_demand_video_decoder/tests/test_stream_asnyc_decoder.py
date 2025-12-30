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

import pytest
import sys
import torch
import random
import time

import utils
import accvlab.on_demand_video_decoder as nvc


def test_async_decode_basic_flow():
    """
    Test 1.1: Basic async decode flow
    Test that async decode works: start async -> wait -> get result
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

    frame_id_list = [0] * len(files)

    # Start async decode
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Get result (will wait for async decode to complete)
    decoded_frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)

    # Verify results
    assert decoded_frames is not None, "decoded_frames is None"
    assert len(decoded_frames) == len(files), f"Expected {len(files)} frames, got {len(decoded_frames)}"

    # Verify each frame is valid RGBFrame
    for i, frame in enumerate(decoded_frames):
        assert frame is not None, f"Frame {i} is None"
        assert hasattr(frame, 'shape'), f"Frame {i} has no shape attribute"
        assert hasattr(frame, 'data'), f"Frame {i} has no data attribute"


def test_async_decode_prefetch_flow():
    """
    Test 1.2: Prefetch mechanism
    Test that prefetching works: process frame N while prefetching frame N+1
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

    # First frame: start async and get immediately
    frame_0 = [0] * len(files)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_0, False)
    frames_0 = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_0, False)
    assert frames_0 is not None and len(frames_0) == len(files)

    # Prefetch frame 3 while processing frame 0
    frame_3 = [3] * len(files)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_3, False)

    # Do some processing (simulate)
    tensor_list_0 = [torch.as_tensor(frame, device='cuda').clone() for frame in frames_0]
    assert len(tensor_list_0) == len(files)

    # Get prefetched frame 3
    frames_3 = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_3, False)
    assert frames_3 is not None and len(frames_3) == len(files)


def test_async_decode_parameter_validation_filepath_mismatch():
    """
    Test 2.1: Parameter validation - filepath mismatch
    Test that GetBuffer throws error when filepaths don't match
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode with original files
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Try to get with different file (should fail)
    wrong_files = files[:]  # Copy list
    if len(wrong_files) > 0:
        wrong_files[0] = "wrong_file.mp4"  # Change first file

    with pytest.raises(RuntimeError, match="Request parameters do not match"):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(wrong_files, frame_id_list, False)


def test_async_decode_parameter_validation_frameid_mismatch():
    """
    Test 2.2: Parameter validation - frame_id mismatch
    Test that GetBuffer throws error when frame_ids don't match
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode with frame 0
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Try to get with different frame_id (should fail)
    wrong_frame_ids = [10] * len(files)  # Different frame ID

    with pytest.raises(RuntimeError, match="Request parameters do not match"):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, wrong_frame_ids, False)


def test_async_decode_parameter_validation_bgr_mismatch():
    """
    Test 2.3: Parameter validation - as_bgr flag mismatch
    Test that GetBuffer throws error when as_bgr flag doesn't match
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode with as_bgr=False (RGB)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Try to get with as_bgr=True (BGR) - should fail
    with pytest.raises(RuntimeError, match="Request parameters do not match"):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, True)


def test_async_decode_parameter_validation_list_size_mismatch():
    """
    Test 2.4: Parameter validation - list size mismatch
    Test that GetBuffer throws error when list sizes don't match
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Try to get with different number of files (should fail)
    wrong_frame_ids = [0] * (len(files) + 1)  # Different size

    with pytest.raises(RuntimeError, match="Request parameters do not match"):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, wrong_frame_ids, False)


def test_async_decode_getbuffer_on_empty_buffer_throws_error():
    """
    Test 3.1: GetBuffer on empty buffer throws error instead of deadlock
    Test that calling GetBuffer when no pending task and buffer is empty throws RuntimeError
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_0 = [0] * len(files)
    frame_3 = [3] * len(files)

    # Start async decode for frame 0
    print("Start async decode for frame 0")
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_0, False)

    # Don't retrieve frame 0, start new async decode for frame 3
    # This should clear the buffer and discard frame 0
    # Note: This will wait for frame 0 to complete, then clear it
    print("Start async decode for frame 3")
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_3, False)

    # Now we can only get frame 3, frame 0 is lost

    frames_3 = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_3, False)
    assert frames_3 is not None and len(frames_3) == len(files)

    with pytest.raises(RuntimeError, match="No pending decode task"):
        print("Try to get frame 0 again")
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_0, False)


def test_async_decode_error_handling_invalid_file():
    """
    Test 4.1: Error handling - invalid file path
    Test that decoding failure propagates correctly
    """
    max_num_files_to_use = 6

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    # Use invalid file path
    invalid_files = ["/nonexistent/path/to/video.mp4"]
    frame_id_list = [0]

    # Start async decode with invalid file (should fail in background)
    nv_stream_dec.DecodeN12ToRGBAsync(invalid_files, frame_id_list, False)

    # GetBuffer should rethrow the exception
    with pytest.raises(RuntimeError):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(invalid_files, frame_id_list, False)


def test_async_decode_error_message_is_preserved():
    """
    Test 4.1b: Error handling - exception message is preserved from async thread
    Test that the exception message from the async thread is correctly propagated
    to the caller, not lost or replaced with a generic message.
    """
    max_num_files_to_use = 6

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    # Use a specific invalid file path
    invalid_filepath = "/nonexistent/path/to/invalid_video_test_12345.mp4"
    invalid_files = [invalid_filepath]
    frame_id_list = [0]

    # Start async decode with invalid file (should fail in background)
    nv_stream_dec.DecodeN12ToRGBAsync(invalid_files, frame_id_list, False)

    # GetBuffer should rethrow the exception with the original error message preserved
    try:
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(invalid_files, frame_id_list, False)
        pytest.fail("Expected RuntimeError was not raised")
    except RuntimeError as e:
        error_message = str(e)

        # Verify the exception message is not empty and contains meaningful info
        # The error comes from NVIDIA decoder when it fails to parse invalid input
        assert len(error_message) > 0, "Exception message should not be empty"

        # Verify the error message contains some technical context
        # (either from FFmpeg, NvDecoder, or file path)
        has_meaningful_context = (
            "NvDecoder" in error_message
            or "error" in error_message.lower()
            or "failed" in error_message.lower()
            or "invalid" in error_message.lower()
            or "FFmpeg" in error_message
            or invalid_filepath in error_message
        )
        assert (
            has_meaningful_context
        ), f"Exception message should contain technical error context, got: {error_message}"

        # Print the error message for debugging (visible in pytest -v output)
        print(f"Caught expected exception with message: {error_message}")

        # NOTE: Ideally, the error message should include the file path for easier debugging.
        # Current behavior returns the underlying NvDecoder error which doesn't include the path.
        # This could be improved in the future by wrapping exceptions with file context.


def test_async_decode_error_handling_invalid_frame_id():
    """
    Test 4.2: Error handling - invalid frame_id
    Test that invalid frame_id causes exception
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    # Use very large frame_id that doesn't exist
    invalid_frame_ids = [999999] * len(files)

    # Start async decode with invalid frame_id
    nv_stream_dec.DecodeN12ToRGBAsync(files, invalid_frame_ids, False)

    # GetBuffer should rethrow the exception
    with pytest.raises(RuntimeError):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, invalid_frame_ids, False)


def test_async_decode_vs_sync_decode_result_comparison():
    """
    Test 9.2: Compare async decode results with sync decode results
    Test that async decode produces identical results to sync decode
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    # Create two readers: one for async, one for sync
    nv_stream_dec_async = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    nv_stream_dec_sync = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Async decode
    nv_stream_dec_async.DecodeN12ToRGBAsync(files, frame_id_list, False)
    async_frames = nv_stream_dec_async.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)

    # Sync decode
    sync_frames = nv_stream_dec_sync.DecodeN12ToRGB(files, frame_id_list, False)

    # Verify same number of frames
    assert len(async_frames) == len(
        sync_frames
    ), f"Frame count mismatch: async={len(async_frames)}, sync={len(sync_frames)}"

    # Compare each frame pixel by pixel
    for i, (async_frame, sync_frame) in enumerate(zip(async_frames, sync_frames)):
        # Convert to tensors and deep copy for comparison
        async_tensor = torch.as_tensor(async_frame, device='cuda').clone()
        sync_tensor = torch.as_tensor(sync_frame, device='cuda').clone()

        # Move to CPU for comparison
        async_tensor_cpu = async_tensor.cpu()
        sync_tensor_cpu = sync_tensor.cpu()

        # Compare shapes
        assert (
            async_tensor_cpu.shape == sync_tensor_cpu.shape
        ), f"Frame {i} shape mismatch: async={async_tensor_cpu.shape}, sync={sync_tensor_cpu.shape}"

        # Compare pixel values (should be identical)
        max_diff = (async_tensor_cpu.float() - sync_tensor_cpu.float()).abs().max().item()
        assert max_diff < 1.0, f"Frame {i} pixel values differ: max_diff={max_diff}"


def test_async_decode_multiple_frames_sequential():
    """
    Test: Sequential processing of multiple frames with prefetching
    Test the complete workflow: decode frame 0, then prefetch and process frame 3, then frame 6
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    # Process frame 0
    frame_0 = [0] * len(files)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_0, False)
    frames_0 = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_0, False)
    assert frames_0 is not None

    # Deep copy frame 0 (as required by documentation)
    tensor_0 = [torch.as_tensor(frame, device='cuda').clone() for frame in frames_0]

    # Prefetch frame 3
    frame_3 = [3] * len(files)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_3, False)

    # Process frame 3
    frames_3 = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_3, False)
    assert frames_3 is not None

    # Deep copy frame 3
    tensor_3 = [torch.as_tensor(frame, device='cuda').clone() for frame in frames_3]

    # Prefetch frame 6
    frame_6 = [6] * len(files)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_6, False)

    # Process frame 6
    frames_6 = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_6, False)
    assert frames_6 is not None

    # Deep copy frame 6
    tensor_6 = [torch.as_tensor(frame, device='cuda').clone() for frame in frames_6]

    # Verify all frames are valid
    assert len(tensor_0) == len(files)
    assert len(tensor_3) == len(files)
    assert len(tensor_6) == len(files)


@pytest.mark.timeout(30)
def test_destructor_with_pending_task():
    """
    Test: Destructor with pending async task should not deadlock
    Test that destroying reader while async task is running doesn't cause deadlock
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode but don't retrieve the result
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Delete the reader while async task may still be running
    # This should not deadlock
    del nv_stream_dec

    # If we reach here, the test passed (no deadlock)


def test_getbuffer_without_async_call():
    """
    Test: GetBuffer without calling Async first should throw error
    Test that calling GetBuffer before any Async call throws RuntimeError
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Try to get buffer without calling Async first - should fail
    with pytest.raises(RuntimeError, match="No pending decode task"):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)


def test_recovery_after_exception():
    """
    Test: Reader should be usable after exception
    Test that reader can continue working after an exception occurred
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    # Trigger an exception with invalid file
    nv_stream_dec.DecodeN12ToRGBAsync(["/invalid/nonexistent/path.mp4"], [0], False)
    with pytest.raises(RuntimeError):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(["/invalid/nonexistent/path.mp4"], [0], False)

    # Verify reader can still work after exception
    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)
    frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)

    assert frames is not None
    assert len(frames) == len(files)


def test_multiple_async_without_getbuffer():
    """
    Test: Multiple consecutive Async calls without GetBuffer
    Test that calling Async multiple times (without GetBuffer) works correctly
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_0 = [0] * len(files)
    frame_1 = [1] * len(files)
    frame_2 = [2] * len(files)

    # Call Async 3 times consecutively without GetBuffer
    # Each call should wait for the previous one to complete and clear the queue
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_0, False)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_1, False)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_2, False)

    # Only the last result should be available
    frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_2, False)
    assert frames is not None
    assert len(frames) == len(files)


def test_mixed_sync_async_usage():
    """
    Test: Mixed usage of sync and async API on same reader
    Test that using both DecodeN12ToRGB and DecodeN12ToRGBAsync on same reader works
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    # Sync decode first
    frame_0 = [0] * len(files)
    sync_frames_1 = nv_stream_dec.DecodeN12ToRGB(files, frame_0, False)
    assert sync_frames_1 is not None
    assert len(sync_frames_1) == len(files)

    # Then async decode
    frame_1 = [1] * len(files)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_1, False)
    async_frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_1, False)
    assert async_frames is not None
    assert len(async_frames) == len(files)

    # Then sync decode again
    frame_2 = [2] * len(files)
    sync_frames_2 = nv_stream_dec.DecodeN12ToRGB(files, frame_2, False)
    assert sync_frames_2 is not None
    assert len(sync_frames_2) == len(files)


def test_double_getbuffer_same_params():
    """
    Test: Calling GetBuffer twice with same params should fail on second call
    Test that the second GetBuffer call throws error when buffer is empty
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # First GetBuffer should succeed
    frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)
    assert frames is not None
    assert len(frames) == len(files)

    # Second GetBuffer should fail (buffer is now empty)
    with pytest.raises(RuntimeError, match="No pending decode task"):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)


def test_empty_list_params():
    """
    Test: Empty list parameters should be handled gracefully
    Test that empty filepaths/frame_ids don't cause crashes
    """
    max_num_files_to_use = 6

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    # Empty lists - return empty
    nv_stream_dec.DecodeN12ToRGBAsync([], [], False)
    frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer([], [], False)
    # If it succeeds, result should be empty
    assert len(frames) == 0


@pytest.mark.timeout(30)
def test_sync_api_waits_for_async():
    """
    Test: Sync API should wait for pending async task to complete
    Test that calling sync DecodeN12ToRGB while async is running is safe
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    # Start async decode
    frame_0 = [0] * len(files)
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_0, False)

    # Immediately call sync decode - should wait for async to complete first
    frame_1 = [1] * len(files)
    sync_frames = nv_stream_dec.DecodeN12ToRGB(files, frame_1, False)

    # Sync decode should succeed
    assert sync_frames is not None
    assert len(sync_frames) == len(files)

    # After sync decode, the async buffer is cleared (invalidated).
    # This prevents users from accidentally retrieving stale async results.
    # Calling GetBuffer should fail because buffer is empty.
    with pytest.raises(RuntimeError, match="No pending decode task"):
        nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_0, False)


@pytest.mark.timeout(30)
def test_clear_readers_waits_for_async():
    """
    Test: clearAllReaders should wait for pending async task to complete
    Test that calling clearAllReaders while async is running is safe
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Immediately call clearAllReaders - should wait for async to complete first
    # This should not crash or cause undefined behavior
    nv_stream_dec.clearAllReaders()

    # Reader should still be usable after clearing
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)
    frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)
    assert frames is not None


@pytest.mark.timeout(30)
def test_release_memory_waits_for_async():
    """
    Test: release_device_memory should wait for pending async task to complete
    Test that calling release_device_memory while async is running is safe
    """
    max_num_files_to_use = 6
    path_base = utils.get_data_dir()

    nv_stream_dec = nvc.CreateSampleReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    files = utils.select_random_clip(path_base)
    assert files is not None

    frame_id_list = [0] * len(files)

    # Start async decode
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)

    # Immediately call release_device_memory - should wait for async to complete first
    # This should not crash or cause undefined behavior
    nv_stream_dec.release_device_memory()

    # Reader should still be usable after releasing memory
    nv_stream_dec.DecodeN12ToRGBAsync(files, frame_id_list, False)
    frames = nv_stream_dec.DecodeN12ToRGBAsyncGetBuffer(files, frame_id_list, False)
    assert frames is not None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
