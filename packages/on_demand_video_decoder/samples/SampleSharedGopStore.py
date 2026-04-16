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
``accvlab.on_demand_video_decoder`` - SharedGopStore Sample

This sample demonstrates how to use ``SharedGopStore`` for cross-process
shared GOP caching with zero-copy reads.  It simulates the typical
DataLoader pattern:

1. **Main process** creates the store before spawning workers.
2. **Worker processes** attach to the store, look up / put GOP data,
   and pass lightweight ``GopRef`` references through the IPC queue.
3. **Main process** calls ``get_batch()`` to read references as
   zero-copy numpy views and clean up orphaned shared-memory blocks.

Key Features Demonstrated:
- Cross-process shared memory cache (POSIX shm + flock)
- Zero-copy data access from main process
- LRU eviction when capacity is exceeded
- Orphan cleanup via ``get_batch()``
- GopRef pickling for DataLoader IPC

Requirements:
- Linux (POSIX shared memory)
- No GPU required (pure CPU / shared-memory demo)
"""

import multiprocessing
import os
import pickle
import time

import numpy as np

import accvlab.on_demand_video_decoder as nvc
from accvlab.on_demand_video_decoder import GopRef, SharedGopStore


# ---------------------------------------------------------------------------
# Simulated worker function (runs in a spawned child process)
# ---------------------------------------------------------------------------

def worker_fn(store_id, capacity, tasks, result_queue):
    """
    Simulate a DataLoader worker: attach to store, lookup/put, return refs.

    Args:
        store_id: SharedGopStore identifier to attach to.
        capacity: Store capacity (must match the main process).
        tasks: List of (video_path, frame_id, gop_first_frame, gop_len) to process.
        result_queue: multiprocessing.Queue to send GopRef results back.
    """
    pid = os.getpid()
    store = SharedGopStore.attach(capacity=capacity, store_id=store_id)
    print(f"  [Worker {pid}] Attached to store (id={store_id}, capacity={capacity})")

    refs = []
    for video_path, frame_id, gop_first_frame, gop_len in tasks:
        # Step 1: Try lock-free lookup
        ref = store.lookup(video_path, frame_id)
        if ref is not None:
            print(f"  [Worker {pid}] HIT  {video_path} frame={frame_id}")
        else:
            # Step 2: Cache miss -> "load from disk" (simulated with random bytes)
            fake_gop_data = np.random.randint(
                0, 256, size=4096, dtype=np.uint8)
            ref = store.put(video_path, gop_first_frame, gop_len, fake_gop_data)
            print(f"  [Worker {pid}] MISS {video_path} frame={frame_id} -> put as {ref.shm_name}")
        refs.append(ref)

    stats = store.get_stats()
    print(f"  [Worker {pid}] Stats: hits={stats['hits']}, misses={stats['misses']}, "
          f"hit_rate={stats['hit_rate']:.0%}")

    # Send lightweight GopRef list through the "IPC queue"
    result_queue.put(refs)
    store.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def SampleSharedGopStore():
    """
    Demonstrate SharedGopStore with simulated multi-process DataLoader.

    Flow:
        Main process          Worker A            Worker B
        ─────────────         ────────            ────────
        create(cap=12)
        spawn workers ──────> attach()            attach()
                              lookup/put ──┐      lookup/put ──┐
                              send refs    │      send refs    │
                                           │                   │
        receive refs <─────────────────────┘───────────────────┘
        get_batch(refs)   ← atomic open + orphan cleanup
        read data       ← zero-copy numpy views
        cleanup()
    """
    STORE_ID = 0
    CAPACITY = 12  # small for demo; production uses bs * cameras * 10
    NUM_WORKERS = 2

    print("=" * 60)
    print("SharedGopStore Demo")
    print("=" * 60)

    # ── 1. Main process creates the store ────────────────────────
    print(f"\n[Main] Creating SharedGopStore (capacity={CAPACITY}, store_id={STORE_ID})")
    store = SharedGopStore.create(capacity=CAPACITY, store_id=STORE_ID)

    # ── 2. Prepare tasks (simulating a batch of 6 cameras × 2 samples) ──
    # Each task: (video_path, target_frame_id, gop_first_frame, gop_len)
    tasks_a = [
        ("/data/video/cam0.mp4", 15, 0, 30),
        ("/data/video/cam1.mp4", 15, 0, 30),
        ("/data/video/cam2.mp4", 15, 0, 30),
        ("/data/video/cam3.mp4", 15, 0, 30),
        ("/data/video/cam4.mp4", 15, 0, 30),
        ("/data/video/cam5.mp4", 15, 0, 30),
    ]
    # Worker B requests overlapping videos -> should get cache hits
    tasks_b = [
        ("/data/video/cam0.mp4", 20, 0, 30),   # same GOP as worker A
        ("/data/video/cam1.mp4", 25, 0, 30),   # same GOP as worker A
        ("/data/video/cam2.mp4", 10, 0, 30),   # same GOP as worker A
        ("/data/video/cam6.mp4", 15, 0, 30),   # new video
        ("/data/video/cam7.mp4", 15, 0, 30),   # new video
        ("/data/video/cam8.mp4", 15, 0, 30),   # new video
    ]

    # ── 3. Spawn workers ─────────────────────────────────────────
    print(f"\n[Main] Spawning {NUM_WORKERS} workers...")
    ctx = multiprocessing.get_context("spawn")
    queue_a = ctx.Queue()
    queue_b = ctx.Queue()

    worker_a = ctx.Process(target=worker_fn,
                           args=(STORE_ID, CAPACITY, tasks_a, queue_a))
    worker_b = ctx.Process(target=worker_fn,
                           args=(STORE_ID, CAPACITY, tasks_b, queue_b))

    # Start A first, wait for it, then start B (so B sees A's data)
    worker_a.start()
    worker_a.join(timeout=30)
    assert worker_a.exitcode == 0, f"Worker A failed with exit code {worker_a.exitcode}"

    worker_b.start()
    worker_b.join(timeout=30)
    assert worker_b.exitcode == 0, f"Worker B failed with exit code {worker_b.exitcode}"

    # ── 4. Collect GopRef results ────────────────────────────────
    refs_a = queue_a.get(timeout=5)
    refs_b = queue_b.get(timeout=5)
    all_refs = refs_a + refs_b

    print(f"\n[Main] Received {len(all_refs)} GopRef references from workers")
    print(f"[Main] GopRef size: {len(pickle.dumps(all_refs[0]))} bytes "
          f"(vs ~4096 bytes of actual GOP data)")

    # ── 5. get_batch: read shm blocks + orphan cleanup ────────────
    print(f"\n[Main] Resolving {len(all_refs)} references (atomic open + orphan cleanup)...")
    arrays = store.get_batch(all_refs)

    print(f"[Main] Got {len(arrays)} zero-copy numpy views:")
    for i, arr in enumerate(arrays):
        print(f"  [{i}] shape={arr.shape}, dtype={arr.dtype}, nbytes={arr.nbytes}")

    # ── 6. Verify data integrity ─────────────────────────────────
    print(f"\n[Main] Store stats: {store.get_stats()}")

    # ── 7. Cleanup ───────────────────────────────────────────────
    store.cleanup()
    print(f"\n[Main] Cleanup complete. All shared memory released.")

    # Verify no shm files remain
    import glob
    remaining = glob.glob(f"/dev/shm/gs_{STORE_ID}_*") + \
                glob.glob(f"/dev/shm/gs_meta_{STORE_ID}") + \
                glob.glob(f"/dev/shm/gs_tick_{STORE_ID}")
    assert not remaining, f"Leaked shm files: {remaining}"
    print("[Main] Verified: no shared memory files leaked.")

    print("\n" + "=" * 60)
    print("Demo complete.")
    print("=" * 60)


if __name__ == "__main__":
    print("NVIDIA accvlab.on_demand_video_decoder - SharedGopStore Sample")
    print("=" * 60)
    print()
    SampleSharedGopStore()
