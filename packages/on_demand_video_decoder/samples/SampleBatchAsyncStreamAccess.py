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
``accvlab.on_demand_video_decoder`` - 2D Batch Async Stream Access Sample

Demonstrates :class:`~accvlab.on_demand_video_decoder.PyNvBatchAsyncStreamReader`:
an async-only, 2D-indexed video decoder that returns multiple frames per video
per submission, designed for StreamPETR-like workloads where each iteration
consumes a *batch of sweeps* (V cameras × F frames).

Key differences from ``SampleStreamAsyncAccess.py`` (the 1D async sample):
- ``Decode`` accepts ``List[List[int]]`` frame_ids instead of ``List[int]``
- ``GetBuffer`` returns ``List[List[RGBFrame]]`` indexed ``[v][f]``
- A single in-flight task batches V*F decodes (vs V in the 1D version)
- The internal aggregator pool is sized at construction via
  ``max_frames_per_decode_call`` so it can hold all V*F frames simultaneously
"""

import os
import torch
import accvlab.on_demand_video_decoder as nvc


def SampleBatchAsyncStreamAccess():
    """
    Show the canonical prefetch pattern with the 2D async stream reader.

    Per iteration:
        iter 0:  Decode(batch_0) -> GetBuffer(batch_0) -> clone -> process
                 + Decode(batch_1) to prefetch the next iteration
        iter i:  GetBuffer(batch_i) -> clone -> process
                 + Decode(batch_{i+1}) to prefetch

    Each "batch" is V videos × F frames (a 2D request).
    """

    # ── Configuration ─────────────────────────────────────────────────────
    max_num_files_to_use = 6
    max_frames_per_decode_call = 4

    print("Initializing NVIDIA GPU video decoder for 2D async batch stream access...")
    reader = nvc.CreateBatchAsyncStreamReader(
        num_of_set=1,
        num_of_file=max_num_files_to_use,
        max_frames_per_decode_call=max_frames_per_decode_call,
        iGpu=0,
    )
    print(
        f"Decoder initialized on GPU 0 — V <= {max_num_files_to_use} videos, "
        f"F <= {max_frames_per_decode_call} frames per video per call"
    )

    # ── Resolve sample clip paths ────────────────────────────────────────
    #
    # In a real pipeline (e.g. StreamPETR-like training) the V file paths
    # would come from a dataset's per-sample multi-camera record, where each
    # entry maps a camera position (CAM_FRONT, CAM_BACK_LEFT, ...) to its
    # video file. See ``samples/SampleStreamAsyncAccess.py`` for the 1D
    # variant of this pattern.
    #
    # The hard-coded "moving shape" clips below ship with this package so
    # the sample runs out-of-the-box without external data.
    base_dir = os.path.dirname(__file__)
    sample_clip_dir = os.path.join(base_dir, "..", "data", "sample_clip")
    file_path_list = [
        os.path.join(sample_clip_dir, "moving_shape_circle_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_ellipse_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_hexagon_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_rect_h265.mp4"),
        os.path.join(sample_clip_dir, "moving_shape_triangle_h265.mp4"),
    ]
    V = len(file_path_list)
    F = max_frames_per_decode_call
    print(f"Processing {V} videos × {F} frames per iteration")

    # ── Build the 2D frame_id schedule ───────────────────────────────────
    # In production this schedule would come from your dataset sampler:
    # for each training step, it yields a (V × F) array of frame ids that
    # together form one batch (e.g. F sweeps × V cameras). The fixed stride
    # below is just for demonstration — replace with whatever your sampler
    # produces.
    num_iterations = 4
    step = 7  # frame stride within a batch
    batch_offset = F * step  # frame_ids[i+1] starts where frame_ids[i] ends

    def make_batch(iter_idx):
        # frame_ids_2d[v][f] = iter_idx * batch_offset + f * step
        # All V videos request the same F frames in this sample; real workloads
        # can stagger per-video frame ids (the API supports it — see tests).
        return [[iter_idx * batch_offset + f * step for f in range(F)] for _ in range(V)]

    print(f"\nStarting {num_iterations} prefetched 2D-batch iterations")
    print("Pattern: GetBuffer(N) -> process(N) -> Decode(N+1) overlaps with processing")

    # ── Main loop ────────────────────────────────────────────────────────
    for idx in range(num_iterations):
        frame_ids_2d = make_batch(idx)
        print(f"\n--- Iteration {idx + 1}/{num_iterations} ---")
        print(f"Frame ids (V×F = {V}×{F}): {frame_ids_2d[0]}")

        try:
            if idx == 0:
                # First iteration: synchronous-feeling start (Decode + immediate Get).
                print("[Async] Submitting initial decode for batch 0")
                reader.Decode(file_path_list, frame_ids_2d, as_bgr=False)
                print("[Async] Retrieving batch 0 from buffer")
                decoded = reader.GetBuffer(file_path_list, frame_ids_2d, as_bgr=False)
            else:
                # Subsequent iterations: result was prefetched at end of previous
                # iteration; this GetBuffer call may already have data ready.
                print(f"[Async] Retrieving prefetched batch {idx} from buffer")
                decoded = reader.GetBuffer(file_path_list, frame_ids_2d, as_bgr=False)

            assert len(decoded) == V, f"unexpected V: {len(decoded)}"
            for v in range(V):
                assert len(decoded[v]) == F, f"unexpected F at v={v}: {len(decoded[v])}"

            # ── CRITICAL: deep-copy before the next Decode() submission ─
            #
            # ``decoded[v][f]`` is a zero-copy RGBFrame referencing the reader's
            # internal aggregator pool. The pool is overwritten on the next
            # Decode() call, so we MUST clone every frame we want to keep
            # *before* that next Decode happens.
            print("Cloning V×F frames to PyTorch tensors (safe to keep across iterations)")
            tensor_grid = [
                [torch.as_tensor(decoded[v][f], device="cuda").clone() for f in range(F)] for v in range(V)
            ]

            # Now stack however your model wants. Example: [V, F, H, W, 3].
            batch = torch.stack([torch.stack(row, dim=0) for row in tensor_grid], dim=0)

            print(f"Batch shape: {tuple(batch.shape)}, dtype: {batch.dtype}, " f"device: {batch.device}")
            print(f"Value range: [{batch.min().item()}, {batch.max().item()}]")

            # ── Prefetch the next batch ─────────────────────────────────
            # This Decode() returns immediately. The worker decodes batch N+1
            # in the background while we "process" batch N below.
            if idx < num_iterations - 1:
                next_frame_ids_2d = make_batch(idx + 1)
                print(f"[Async] Prefetching batch {idx + 1} ({next_frame_ids_2d[0]})")
                reader.Decode(file_path_list, next_frame_ids_2d, as_bgr=False)

            # ── Simulated "process" stage (model forward, etc.) ─────────
            # This is where you'd run inference. The prefetched decode is
            # happening on the worker thread concurrently.
            print("[Processing] (simulated) — prefetched decode is running in parallel")

        except Exception as e:
            print(f"Decode failed in iteration {idx + 1}: {type(e).__name__}: {e}")
            print("Common causes:")
            print("  - frame id past last keyframe / beyond video length")
            print("  - filepaths/frame_ids mismatch between Decode and GetBuffer")
            print("  - non-existent video file")
            print("  - insufficient GPU memory for V*F*H*W*3 aggregator pool")
            continue

    print("\n" + "=" * 60)
    print("2D async batch stream decoding completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    print("NVIDIA accvlab.on_demand_video_decoder — 2D Batch Async Stream Sample")
    print("=" * 70)
    print()
    SampleBatchAsyncStreamAccess()
