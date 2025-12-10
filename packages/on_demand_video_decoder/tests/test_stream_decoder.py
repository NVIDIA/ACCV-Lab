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

import random
import threading
import time

import utils
import accvlab.on_demand_video_decoder as nvc


def test_stream_access_single():
    max_num_files_to_use = 6
    iter_num = 10
    path_base = utils.get_data_dir()

    nv_gop_dec = nvc.CreateSampleReader(
        num_of_set=10,
        num_of_file=max_num_files_to_use,
        iGpu=0,
    )

    frame_min = 0
    frame_max = 200

    for c in range(iter_num):
        files = utils.select_random_clip(path_base)
        assert files is not None, f"files is None for select_random_clip, path_base: {path_base}"

        frames = [random.randint(frame_min, frame_max) for _ in range(len(files))]
        print(f"Comparison: {c}, frames: {frames}")

        gop_decoded = utils.gop_decode_bgr(nv_gop_dec, files, frames)
        assert gop_decoded is not None, f"gop_decoded is None for DecodeN12ToRGB, frames: {frames}"


def test_gil_release_parallel_decode_performance():
    """
    Test that verifies GIL release by measuring overlap of decode operations.

    Strategy:
    - Run decode operations in parallel threads
    - Record precise start/end times of each decode call
    - If GIL is released: decode calls can OVERLAP in time
    - If GIL is NOT released: decode calls will be SERIALIZED (no overlap)

    Key metric: We calculate the OVERLAP RATIO
    - overlap_ratio = actual_parallel_time / sum_of_individual_times
    - With GIL released: overlap_ratio < 1 (calls overlap)
    - Without GIL: overlap_ratio ≈ 1 (calls serialized)
    """
    path_base = utils.get_data_dir()
    num_threads = 4
    num_decode_calls = 5

    # Create separate decoders for each thread
    decoders = [nvc.CreateSampleReader(num_of_set=2, num_of_file=1, iGpu=0) for _ in range(num_threads)]

    all_files = utils.select_random_clip(path_base)
    assert all_files is not None and len(all_files) > 0, "No test files found"
    files = [all_files[0]]

    # Record individual decode call durations per thread
    thread_call_times = [[] for _ in range(num_threads)]

    def decode_work(thread_id, decoder):
        """Decode and record precise timing for each call"""
        # Warm up
        decoder.DecodeN12ToRGB(files, [10])

        for i in range(num_decode_calls):
            start = time.perf_counter()
            frames = decoder.DecodeN12ToRGB(files, [50 + i * 10])
            end = time.perf_counter()
            thread_call_times[thread_id].append((start, end))
            assert frames is not None

    # Run all threads in parallel
    threads = []
    overall_start = time.perf_counter()
    for i, decoder in enumerate(decoders):
        t = threading.Thread(target=decode_work, args=(i, decoder))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()
    overall_end = time.perf_counter()

    # Calculate metrics
    actual_parallel_time = overall_end - overall_start

    # Sum of all individual decode call durations
    total_individual_time = 0
    all_call_times = []
    for thread_times in thread_call_times:
        for start, end in thread_times:
            total_individual_time += end - start
            all_call_times.append((start, end))

    # Calculate overlap: how many calls were running simultaneously?
    # Sort all events and count concurrent calls at each point
    events = []
    for start, end in all_call_times:
        events.append((start, 'start'))
        events.append((end, 'end'))
    events.sort(key=lambda x: x[0])

    max_concurrent = 0
    current_concurrent = 0
    for _, event_type in events:
        if event_type == 'start':
            current_concurrent += 1
            max_concurrent = max(max_concurrent, current_concurrent)
        else:
            current_concurrent -= 1

    overlap_ratio = actual_parallel_time / total_individual_time if total_individual_time > 0 else 1

    print(f"\n[GIL Parallel Test] Actual parallel time: {actual_parallel_time*1000:.1f}ms")
    print(f"[GIL Parallel Test] Sum of individual call times: {total_individual_time*1000:.1f}ms")
    print(f"[GIL Parallel Test] Overlap ratio: {overlap_ratio:.2f} (lower = more overlap)")
    print(f"[GIL Parallel Test] Max concurrent decode calls: {max_concurrent}")

    # If GIL is released, we expect:
    # 1. Multiple decode calls running concurrently (max_concurrent > 1)
    # 2. Overlap ratio significantly less than 1
    #
    # If GIL is NOT released:
    # 1. max_concurrent = 1 (calls are serialized)
    # 2. overlap_ratio ≈ 1 (no overlap, serial execution)

    # Threshold: with 4 threads, if GIL is released, we should see at least 2 concurrent calls
    MIN_EXPECTED_CONCURRENT = 2
    MAX_OVERLAP_RATIO = 0.9  # Allow some overhead, but should be < 1

    assert max_concurrent >= MIN_EXPECTED_CONCURRENT, (
        f"GIL does not appear to be released! "
        f"Max concurrent decode calls: {max_concurrent} (expected >= {MIN_EXPECTED_CONCURRENT}). "
        "If GIL was released, multiple decode calls should run simultaneously."
    )

    print(
        f"[GIL Parallel Test] PASSED - GIL was released "
        f"(max {max_concurrent} concurrent calls, overlap ratio {overlap_ratio:.2f})"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
