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

import ctypes
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


def _heartbeats_during(call, margin=0.02):
    """
    Run `call` while a background thread records timestamps in a tight pure-Python loop.

    Returns (interior_beats, duration):
    - interior_beats: heartbeats observed strictly inside (start + margin, end - margin),
      i.e. while `call` was executing.
    - duration: wall-clock duration of `call` in seconds.

    This is a deterministic GIL-release probe. If `call` holds the GIL for its entire
    duration, the heartbeat thread cannot execute any Python bytecode until `call`
    returns, so interior_beats is exactly 0. If `call` releases the GIL, the heartbeat
    loop runs throughout and interior_beats is in the thousands. The margin excludes
    beats from thread-switch slices at the call boundaries (default switch interval
    is 5ms, so 20ms margin is ample).
    """
    beats = []
    stop = threading.Event()
    started = threading.Event()

    def heartbeat():
        append = beats.append
        clock = time.perf_counter
        started.set()
        while not stop.is_set():
            append(clock())

    t = threading.Thread(target=heartbeat, daemon=True)
    t.start()
    started.wait()

    start = time.perf_counter()
    call()
    end = time.perf_counter()

    stop.set()
    t.join()

    lo, hi = start + margin, end - margin
    interior_beats = sum(1 for ts in beats if lo < ts < hi)
    return interior_beats, end - start


# With the GIL released, the heartbeat loop runs at millions of iterations per second;
# even on a heavily loaded machine it gets scheduled often enough to record thousands
# of beats per second. With the GIL held, interior beats are exactly 0. A threshold of
# 50 sits orders of magnitude away from both outcomes.
MIN_INTERIOR_HEARTBEATS = 50


def test_gil_probe_negative_control():
    """
    Validate the probe itself: a C call that HOLDS the GIL must be detected as such.

    ctypes.PyDLL calls foreign functions WITHOUT releasing the GIL, so a blocking
    usleep(0.5s) through PyDLL freezes all other Python threads for its duration.
    The probe must report (near) zero interior heartbeats.
    """
    libc_hold_gil = ctypes.PyDLL("libc.so.6")
    interior_beats, duration = _heartbeats_during(lambda: libc_hold_gil.usleep(500_000))

    print(f"\n[GIL Probe Negative Control] duration: {duration*1000:.1f}ms, interior beats: {interior_beats}")
    assert duration >= 0.4, "usleep did not block as expected; control is inconclusive"
    assert interior_beats < MIN_INTERIOR_HEARTBEATS, (
        f"Probe failed to detect a held GIL: {interior_beats} interior heartbeats "
        f"observed during a GIL-holding C call (expected < {MIN_INTERIOR_HEARTBEATS})."
    )


def test_gil_probe_positive_control():
    """
    Validate the probe itself: a C call that RELEASES the GIL must be detected as such.

    ctypes.CDLL releases the GIL around foreign calls, so the same usleep(0.5s)
    through CDLL lets the heartbeat thread run freely throughout.
    """
    libc_release_gil = ctypes.CDLL("libc.so.6")
    interior_beats, duration = _heartbeats_during(lambda: libc_release_gil.usleep(500_000))

    print(f"\n[GIL Probe Positive Control] duration: {duration*1000:.1f}ms, interior beats: {interior_beats}")
    assert duration >= 0.4, "usleep did not block as expected; control is inconclusive"
    assert interior_beats >= MIN_INTERIOR_HEARTBEATS, (
        f"Probe failed to detect a released GIL: only {interior_beats} interior "
        f"heartbeats observed during a GIL-releasing C call "
        f"(expected >= {MIN_INTERIOR_HEARTBEATS})."
    )


def test_gil_release_during_decode():
    """
    Verify DecodeN12ToRGB releases the GIL while decoding.

    A heartbeat thread increments in a pure-Python loop while a single long decode
    call runs. If the extension holds the GIL, the heartbeat records exactly 0 beats
    inside the call window; if it releases the GIL, thousands. The two outcomes are
    orders of magnitude apart, so the assertion is insensitive to machine load
    (validated by the negative/positive control tests above).
    """
    path_base = utils.get_data_dir()
    base_files = utils.select_random_clip(path_base)
    assert base_files is not None and len(base_files) > 0, "No test files found"

    # The probe needs the decode call to be long enough that margin trimming leaves
    # a meaningful interior window. Grow the per-call workload until it is.
    MIN_PROBE_DURATION = 0.08
    reps = 4
    interior_beats = duration = None
    for _ in range(5):
        call_files = base_files * reps
        # Scattered frame ids across different GOPs so the small cache (num_of_set=2)
        # cannot shortcut the decode work.
        frame_ids = [30 + (i * 13) % 170 for i in range(len(call_files))]

        decoder = nvc.CreateSampleReader(num_of_set=2, num_of_file=len(call_files), iGpu=0)
        # Warm up: CUDA context / NVDEC session creation must not pollute the measurement.
        decoder.DecodeN12ToRGB(call_files, [10] * len(call_files))

        result = {}

        def decode_call():
            result["frames"] = decoder.DecodeN12ToRGB(call_files, frame_ids)

        interior_beats, duration = _heartbeats_during(decode_call)
        assert result["frames"] is not None

        print(
            f"\n[GIL Decode Test] reps: {reps}, decode duration: {duration*1000:.1f}ms, interior beats: {interior_beats}"
        )
        del decoder
        if duration >= MIN_PROBE_DURATION:
            break
        reps *= 2
    else:
        pytest.fail(
            f"Could not build a decode call longer than {MIN_PROBE_DURATION*1000:.0f}ms "
            f"(last attempt: {duration*1000:.1f}ms with reps={reps}); probe inconclusive."
        )

    assert interior_beats >= MIN_INTERIOR_HEARTBEATS, (
        f"GIL does not appear to be released during DecodeN12ToRGB: only "
        f"{interior_beats} heartbeats observed inside a {duration*1000:.1f}ms decode "
        f"call (expected >= {MIN_INTERIOR_HEARTBEATS}). A heartbeat thread should run "
        "freely while the decoder works if the GIL is released."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
