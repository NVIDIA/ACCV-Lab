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
Tests for ``PyNvBatchAsyncStreamReader`` (2D async stream decoder).

Layout:
    Section A — construction / module exports
    Section B — Decode() input validation (independent of decode impl)
    Section C — maintenance methods
    Section D — functional 2D decode
    Section E — precision: 2D output must bit-match sequential 1D calls
    Section F — async behavior: in-flight slot, request validation, error paths
"""

import pytest
import torch

import utils
import accvlab.on_demand_video_decoder as nvc

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _select_sample_videos():
    """Return a list of sample video file paths from the project's test data.

    No fallback — the data dir is guaranteed to exist by the project layout.
    """
    files = utils.select_random_clip(utils.get_data_dir())
    assert files is not None and len(files) > 0, "test data missing"
    return files


def _reference_decode_via_1d(files, frame_ids_2d, as_bgr):
    """Compute the ground-truth 2D decode result by calling the synchronous
    1D API once per frame index, accumulating cloned tensors.

    Returns out[v][f] as a torch.Tensor (already cloned, safe to keep).
    """
    V = len(files)
    F = len(frame_ids_2d[0])
    reader_1d = nvc.CreateSampleReader(num_of_set=1, num_of_file=V, iGpu=0)

    out = [[None] * F for _ in range(V)]
    for f in range(F):
        fids_at_f = [frame_ids_2d[v][f] for v in range(V)]
        frames_1d = reader_1d.DecodeN12ToRGB(files, fids_at_f, as_bgr)
        # The per-reader pool inside PyNvSampleReader is overwritten on the next
        # call, so we MUST clone each frame before the loop advances.
        for v in range(V):
            out[v][f] = torch.as_tensor(frames_1d[v], device="cuda").clone()
    return out


def _make_reader(V=4, F=8):
    return nvc.CreateBatchAsyncStreamReader(num_of_set=1, num_of_file=V, max_frames_per_decode_call=F, iGpu=0)


# ===========================================================================
# Section A — construction / module exports
# ===========================================================================


def test_module_exports():
    """Factory and class are re-exported at the package top level."""
    assert hasattr(nvc, "CreateBatchAsyncStreamReader")
    assert hasattr(nvc, "PyNvBatchAsyncStreamReader")


def test_construct_valid():
    """Construction with valid args succeeds and exposes the expected methods."""
    r = _make_reader()
    methods = {m for m in dir(r) if not m.startswith("_")}
    expected = {"Decode", "GetBuffer", "clearAllReaders", "release_device_memory", "release_decoder"}
    assert expected.issubset(methods), f"missing methods: {expected - methods}"


def test_destructor_clean():
    """Reader destructs cleanly with no pending task."""
    r = _make_reader()
    del r


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_of_set=0, num_of_file=1, max_frames_per_decode_call=1),
        dict(num_of_set=-1, num_of_file=1, max_frames_per_decode_call=1),
        dict(num_of_set=1, num_of_file=0, max_frames_per_decode_call=1),
        dict(num_of_set=1, num_of_file=-1, max_frames_per_decode_call=1),
        dict(num_of_set=1, num_of_file=1, max_frames_per_decode_call=0),
        dict(num_of_set=1, num_of_file=1, max_frames_per_decode_call=-1),
    ],
)
def test_construct_rejects_invalid_args(kwargs):
    """Non-positive sizing arguments are rejected at construction."""
    with pytest.raises((ValueError, RuntimeError)):
        nvc.CreateBatchAsyncStreamReader(**kwargs)


# ===========================================================================
# Section B — Decode() input validation
# ===========================================================================


def test_validate_size_mismatch():
    """filepaths.size() != frame_ids_2d.size() rejected at entry."""
    files = _select_sample_videos()
    r = _make_reader(V=len(files))
    bad_frame_ids = [[0]]  # length 1, but files has more entries
    with pytest.raises(RuntimeError, match=r"filepaths\.size\(\).*frame_ids_2d\.size\(\)"):
        r.Decode(files, bad_frame_ids, False)


def test_validate_empty_filepaths():
    """Empty filepaths list is rejected."""
    r = _make_reader()
    with pytest.raises(RuntimeError, match=r"filepaths must not be empty"):
        r.Decode([], [], False)


def test_validate_too_many_files():
    """Exceeding num_of_file is rejected at entry."""
    files = _select_sample_videos()
    r = _make_reader(V=1)  # num_of_file=1, but we'll pass len(files) > 1
    if len(files) <= 1:
        pytest.skip("need at least 2 sample videos for this test")
    with pytest.raises(RuntimeError, match=r"exceeds num_of_file"):
        r.Decode(files, [[0]] * len(files), False)


def test_validate_too_many_frames():
    """Exceeding max_frames_per_decode_call is rejected at entry."""
    files = _select_sample_videos()
    r = _make_reader(V=len(files), F=4)
    with pytest.raises(RuntimeError, match=r"exceeds max_frames_per_decode_call"):
        r.Decode(files, [list(range(100))] * len(files), False)


def test_validate_jagged_inner_lengths():
    """Per-video inner lists must be the same length across V."""
    files = _select_sample_videos()
    if len(files) < 2:
        pytest.skip("need at least 2 sample videos for this test")
    r = _make_reader(V=len(files))
    # First video asks for 3 frames, second asks for 2.
    jagged = [[0, 7, 14], [0, 7]] + [[0, 7, 14]] * (len(files) - 2)
    with pytest.raises(RuntimeError, match=r"jagged inner lengths are not supported"):
        r.Decode(files, jagged, False)


def test_validate_empty_inner_list():
    """Inner frame_ids list must be non-empty."""
    files = _select_sample_videos()
    r = _make_reader(V=len(files))
    with pytest.raises(RuntimeError, match=r"frame_ids_2d\[0\] must not be empty"):
        r.Decode(files, [[]] * len(files), False)


# ===========================================================================
# Section C — maintenance methods
# ===========================================================================


def test_maintenance_idle_callable():
    """Maintenance methods are safe no-ops when no task is pending."""
    r = _make_reader()
    r.clearAllReaders()
    r.release_device_memory()
    r.release_decoder()
    # Order-independent and idempotent.
    r.release_decoder()
    r.clearAllReaders()
    r.release_device_memory()


# ===========================================================================
# Section D — functional 2D decode
# ===========================================================================


def test_decode_basic_2d_shape():
    """Decode returns List[List[RGBFrame]] with the expected outer/inner shape."""
    files = _select_sample_videos()
    V = len(files)
    F = 4
    frame_ids_2d = [[0, 7, 14, 21]] * V

    r = _make_reader(V=V, F=F)
    r.Decode(files, frame_ids_2d, as_bgr=False)
    out = r.GetBuffer(files, frame_ids_2d, as_bgr=False)

    assert len(out) == V, f"outer len should be V={V}, got {len(out)}"
    for v in range(V):
        assert len(out[v]) == F, f"out[{v}] inner len should be F={F}, got {len(out[v])}"


def test_decode_basic_2d_dtype_and_device():
    """Each frame is a uint8, 3-channel tensor on CUDA."""
    files = _select_sample_videos()
    V = len(files)
    F = 2
    frame_ids_2d = [[0, 7]] * V

    r = _make_reader(V=V, F=F)
    r.Decode(files, frame_ids_2d, as_bgr=False)
    out = r.GetBuffer(files, frame_ids_2d, as_bgr=False)

    for v in range(V):
        for f in range(F):
            t = torch.as_tensor(out[v][f], device="cuda")
            assert t.dtype == torch.uint8, f"out[{v}][{f}].dtype = {t.dtype}"
            assert t.ndim == 3, f"out[{v}][{f}].ndim = {t.ndim}"
            assert t.shape[-1] == 3, f"out[{v}][{f}].shape = {tuple(t.shape)}"
            assert t.device.type == "cuda"


def test_decode_single_frame_per_video():
    """F=1 is supported (degenerates to behavior similar to 1D API)."""
    files = _select_sample_videos()
    V = len(files)
    frame_ids_2d = [[0]] * V

    r = _make_reader(V=V, F=1)
    r.Decode(files, frame_ids_2d, as_bgr=False)
    out = r.GetBuffer(files, frame_ids_2d, as_bgr=False)

    assert len(out) == V
    for v in range(V):
        assert len(out[v]) == 1


def test_decode_single_video_multi_frame():
    """V=1 with multiple frames is supported."""
    files = _select_sample_videos()
    single = [files[0]]
    F = 4
    frame_ids_2d = [[0, 7, 14, 21]]

    r = _make_reader(V=1, F=F)
    r.Decode(single, frame_ids_2d, as_bgr=False)
    out = r.GetBuffer(single, frame_ids_2d, as_bgr=False)

    assert len(out) == 1
    assert len(out[0]) == F


# ===========================================================================
# Section E — precision: 2D output must bit-match sequential 1D calls
# ===========================================================================


@pytest.mark.parametrize("as_bgr", [False, True])
def test_precision_matches_1d_reference(as_bgr):
    """2D output is bit-identical to F sequential 1D DecodeN12ToRGB calls.

    The 2D worker internally loops the same sync 1D path F times, so the
    pixel data must match exactly (uint8, atol=0, rtol=0).
    """
    files = _select_sample_videos()
    V = len(files)
    F = 4
    frame_ids_2d = [[0, 7, 14, 21]] * V

    # Ground truth via sequential 1D + clone
    ref = _reference_decode_via_1d(files, frame_ids_2d, as_bgr)

    # Under test
    r2d = _make_reader(V=V, F=F)
    r2d.Decode(files, frame_ids_2d, as_bgr=as_bgr)
    out_2d = r2d.GetBuffer(files, frame_ids_2d, as_bgr=as_bgr)

    for v in range(V):
        for f in range(F):
            actual = torch.as_tensor(out_2d[v][f], device="cuda")
            torch.testing.assert_close(
                actual,
                ref[v][f],
                atol=0,
                rtol=0,
                msg=lambda m, vv=v, ff=f: (f"pixel mismatch at v={vv}, f={ff}, as_bgr={as_bgr}: {m}"),
            )


def test_precision_matches_1d_with_different_frame_sets_per_video():
    """Each video can request a different set of frame ids; result still matches 1D."""
    files = _select_sample_videos()
    V = len(files)
    F = 3

    # Stagger frame ids per video
    frame_ids_2d = [[v * 2 + i * 5 for i in range(F)] for v in range(V)]

    ref = _reference_decode_via_1d(files, frame_ids_2d, as_bgr=False)

    r2d = _make_reader(V=V, F=F)
    r2d.Decode(files, frame_ids_2d, as_bgr=False)
    out_2d = r2d.GetBuffer(files, frame_ids_2d, as_bgr=False)

    for v in range(V):
        for f in range(F):
            actual = torch.as_tensor(out_2d[v][f], device="cuda")
            torch.testing.assert_close(actual, ref[v][f], atol=0, rtol=0)


# ===========================================================================
# Section F — async behavior: in-flight slot, request validation, error paths
# ===========================================================================


def test_get_buffer_without_decode_raises():
    """GetBuffer with no pending task raises RuntimeError."""
    files = _select_sample_videos()
    r = _make_reader(V=len(files))
    with pytest.raises(RuntimeError):
        r.GetBuffer(files, [[0]] * len(files), False)


def test_get_buffer_request_mismatch_files_raises():
    """GetBuffer with different filepaths than Decode raises."""
    files = _select_sample_videos()
    V = len(files)
    if V < 2:
        pytest.skip("need at least 2 sample videos")
    frame_ids_2d = [[0]] * V

    r = _make_reader(V=V)
    r.Decode(files, frame_ids_2d, False)

    swapped = files[:]
    swapped[0], swapped[1] = swapped[1], swapped[0]
    with pytest.raises(RuntimeError):
        r.GetBuffer(swapped, frame_ids_2d, False)


def test_get_buffer_request_mismatch_frames_raises():
    """GetBuffer with different frame_ids than Decode raises."""
    files = _select_sample_videos()
    V = len(files)
    frame_ids_2d = [[0]] * V

    r = _make_reader(V=V)
    r.Decode(files, frame_ids_2d, False)

    different = [[7]] * V
    with pytest.raises(RuntimeError):
        r.GetBuffer(files, different, False)


def test_get_buffer_request_mismatch_as_bgr_raises():
    """GetBuffer with different as_bgr flag than Decode raises."""
    files = _select_sample_videos()
    V = len(files)
    frame_ids_2d = [[0]] * V

    r = _make_reader(V=V)
    r.Decode(files, frame_ids_2d, as_bgr=False)
    with pytest.raises(RuntimeError):
        r.GetBuffer(files, frame_ids_2d, as_bgr=True)


def test_resubmit_keeps_only_latest_result():
    """Two Decode() calls back-to-back: only the second result is retrievable.

    GetBuffer() pops then validates — on a mismatch the result is consumed and
    cannot be retrieved a second time (same semantics as the 1D async API).
    So the strongest check we can make is: after re-Decoding, only the *new*
    parameters retrieve a result; the *old* parameters error.
    """
    files = _select_sample_videos()
    V = len(files)
    F = 2
    frame_ids_a = [[0, 7]] * V
    frame_ids_b = [[14, 21]] * V

    # Variant 1: re-Decode then Get with new params — should succeed.
    r = _make_reader(V=V, F=F)
    r.Decode(files, frame_ids_a, False)
    r.Decode(files, frame_ids_b, False)
    out_b = r.GetBuffer(files, frame_ids_b, False)
    assert len(out_b) == V and len(out_b[0]) == F

    # Variant 2: re-Decode then Get with old params — should mismatch.
    r2 = _make_reader(V=V, F=F)
    r2.Decode(files, frame_ids_a, False)
    r2.Decode(files, frame_ids_b, False)
    with pytest.raises(RuntimeError, match=r"do not match buffered result"):
        r2.GetBuffer(files, frame_ids_a, False)


def test_invalid_file_propagates_exception():
    """A decode error in the worker (e.g. nonexistent file) is rethrown at Get."""
    files = _select_sample_videos()
    V = len(files)
    bad_files = list(files)
    bad_files[0] = "/__definitely_not_a_real_file__.mp4"
    frame_ids_2d = [[0]] * V

    r = _make_reader(V=V)
    r.Decode(bad_files, frame_ids_2d, False)
    with pytest.raises(RuntimeError):
        r.GetBuffer(bad_files, frame_ids_2d, False)
