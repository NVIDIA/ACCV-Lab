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
Tests for SharedGopStore: cross-process shared GOP store.

These tests exercise the pure-Python shared memory layer.
No GPU, CUDA, or video files required -- runs on any Linux machine.
"""

import glob
import multiprocessing
import os
import pickle
import sys
import warnings

import numpy as np
import pytest

from accvlab.on_demand_video_decoder import GopRef, SharedGopStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_gop_data(size: int = 1024, seed: int = 42) -> np.ndarray:
    """Create deterministic fake GOP packet data."""
    rng = np.random.RandomState(seed)
    return rng.randint(0, 256, size=size, dtype=np.uint8)


def _shm_files_for_store(store_id: int):
    """List /dev/shm files belonging to a store."""
    return glob.glob(f"/dev/shm/gs_{store_id}_*") + \
           glob.glob(f"/dev/shm/gs_meta_{store_id}") + \
           glob.glob(f"/dev/shm/gs_tick_{store_id}") + \
           glob.glob(f"/dev/shm/gs_lock_{store_id}")


# Use a unique store_id range (9000+) to avoid collisions with real training.
_BASE_STORE_ID = 9000


@pytest.fixture
def store_id(request):
    """Provide a unique store_id per test and clean up afterwards."""
    # Use test index as offset to avoid collisions between parallel tests.
    sid = _BASE_STORE_ID + hash(request.node.name) % 500
    yield sid
    # Best-effort cleanup: remove any leftover shm files.
    for path in _shm_files_for_store(sid):
        try:
            os.unlink(path)
        except OSError:
            pass


@pytest.fixture
def store(store_id):
    """Create a SharedGopStore, yield it, then cleanup."""
    s = SharedGopStore.create(capacity=8, store_id=store_id)
    yield s
    s.cleanup()


# ---------------------------------------------------------------------------
# GopRef serialization
# ---------------------------------------------------------------------------

class TestGopRef:
    """Tests for GopRef NamedTuple."""

    def test_pickle_roundtrip(self):
        """GopRef must survive pickle roundtrip (DataLoader IPC)."""
        ref = GopRef(shm_name="gs_0_12345_0", data_size=1024,
                     first_frame_id=0, gop_len=30)
        restored = pickle.loads(pickle.dumps(ref))
        assert restored == ref
        assert type(restored) is GopRef

    def test_fields(self):
        """GopRef fields are accessible by name."""
        ref = GopRef("name", 100, 10, 30)
        assert ref.shm_name == "name"
        assert ref.data_size == 100
        assert ref.first_frame_id == 10
        assert ref.gop_len == 30

    def test_immutable(self):
        """GopRef is immutable (NamedTuple)."""
        ref = GopRef("name", 100, 10, 30)
        with pytest.raises(AttributeError):
            ref.shm_name = "other"


# ---------------------------------------------------------------------------
# Basic lifecycle
# ---------------------------------------------------------------------------

class TestLifecycle:
    """Create -> put -> lookup -> read -> cleanup."""

    def test_put_and_lookup_hit(self, store):
        """Put data, then lookup by a frame_id in range."""
        data = _make_gop_data(2048, seed=1)
        ref = store.put("/video/cam0.mp4", first_frame_id=0, gop_len=30, data=data)

        assert isinstance(ref, GopRef)
        assert ref.data_size == 2048
        assert ref.first_frame_id == 0
        assert ref.gop_len == 30

        # Lookup with frame_id inside [0, 30)
        hit = store.lookup("/video/cam0.mp4", frame_id=15)
        assert hit is not None
        assert hit.shm_name == ref.shm_name

    def test_lookup_miss(self, store):
        """Lookup with no data returns None."""
        assert store.lookup("/nonexistent.mp4", 0) is None

    def test_read_returns_correct_data(self, store):
        """read() returns zero-copy view matching the original data."""
        data = _make_gop_data(4096, seed=7)
        ref = store.put("/video/cam1.mp4", 60, 30, data)
        view = store.read(ref)
        np.testing.assert_array_equal(view, data)

    def test_cleanup_removes_all_shm(self, store_id):
        """After cleanup(), no shm files remain for this store."""
        s = SharedGopStore.create(capacity=4, store_id=store_id)
        s.put("/v.mp4", 0, 30, _make_gop_data(512))
        s.put("/v.mp4", 30, 30, _make_gop_data(512, seed=2))
        s.cleanup()

        remaining = _shm_files_for_store(store_id)
        assert remaining == [], f"Leaked shm files: {remaining}"


# ---------------------------------------------------------------------------
# Lookup boundary conditions
# ---------------------------------------------------------------------------

class TestLookupBoundary:
    """Frame-range boundary hit/miss."""

    def test_hit_at_first_frame(self, store):
        """frame_id == first_frame_id is a hit."""
        store.put("/v.mp4", first_frame_id=10, gop_len=20, data=_make_gop_data(256))
        assert store.lookup("/v.mp4", 10) is not None

    def test_hit_at_last_frame(self, store):
        """frame_id == first_frame_id + gop_len - 1 is a hit."""
        store.put("/v.mp4", first_frame_id=10, gop_len=20, data=_make_gop_data(256))
        assert store.lookup("/v.mp4", 29) is not None

    def test_miss_just_past_end(self, store):
        """frame_id == first_frame_id + gop_len is a miss."""
        store.put("/v.mp4", first_frame_id=10, gop_len=20, data=_make_gop_data(256))
        assert store.lookup("/v.mp4", 30) is None

    def test_miss_before_start(self, store):
        """frame_id < first_frame_id is a miss."""
        store.put("/v.mp4", first_frame_id=10, gop_len=20, data=_make_gop_data(256))
        assert store.lookup("/v.mp4", 9) is None

    def test_different_video_same_range(self, store):
        """Different video_path is a miss even if frame_id is in range."""
        store.put("/v1.mp4", first_frame_id=0, gop_len=30, data=_make_gop_data(256))
        assert store.lookup("/v2.mp4", 15) is None


# ---------------------------------------------------------------------------
# Put double-check
# ---------------------------------------------------------------------------

class TestPutDoubleCheck:
    """put() with same GOP twice should not create duplicate shm."""

    def test_double_put_returns_same_ref(self, store):
        data = _make_gop_data(1024)
        ref1 = store.put("/v.mp4", 0, 30, data)
        ref2 = store.put("/v.mp4", 0, 30, data)
        assert ref1.shm_name == ref2.shm_name

    def test_double_put_no_extra_slots(self, store):
        data = _make_gop_data(1024)
        store.put("/v.mp4", 0, 30, data)
        store.put("/v.mp4", 0, 30, data)
        stats = store.get_stats()
        assert stats['puts'] == 1, "Second put should be a no-op"


# ---------------------------------------------------------------------------
# LRU eviction
# ---------------------------------------------------------------------------

class TestLRUEviction:
    """Capacity full -> oldest entry evicted."""

    def test_eviction_when_full(self, store_id):
        """Fill capacity=4, then insert a 5th -> oldest evicted."""
        store = SharedGopStore.create(capacity=4, store_id=store_id)
        try:
            # Fill 4 slots
            for i in range(4):
                store.put(f"/v{i}.mp4", 0, 30, _make_gop_data(256, seed=i))

            # All 4 should be findable
            for i in range(4):
                assert store.lookup(f"/v{i}.mp4", 15) is not None

            # Insert 5th -> v0 (oldest, lowest tick) gets evicted
            store.put("/v4.mp4", 0, 30, _make_gop_data(256, seed=4))

            assert store.lookup("/v0.mp4", 15) is None, "v0 should be evicted"
            assert store.lookup("/v4.mp4", 15) is not None, "v4 should be present"

            stats = store.get_stats()
            assert stats['evictions'] == 1
        finally:
            store.cleanup()

    def test_access_refreshes_lru(self, store_id):
        """Accessing an entry prevents it from being evicted."""
        store = SharedGopStore.create(capacity=4, store_id=store_id)
        try:
            for i in range(4):
                store.put(f"/v{i}.mp4", 0, 30, _make_gop_data(256, seed=i))

            # Touch v0 so it's no longer the oldest
            store.lookup("/v0.mp4", 15)

            # Insert 5th -> v1 (now the oldest) gets evicted, not v0
            store.put("/v4.mp4", 0, 30, _make_gop_data(256, seed=4))

            assert store.lookup("/v0.mp4", 15) is not None, "v0 was touched, should survive"
            assert store.lookup("/v1.mp4", 15) is None, "v1 should be evicted"
        finally:
            store.cleanup()


# ---------------------------------------------------------------------------
# get_batch + orphan cleanup
# ---------------------------------------------------------------------------

class TestGetBatch:
    """get_batch() reads shm blocks and cleans orphans."""

    def test_get_batch_returns_correct_data(self, store):
        data0 = _make_gop_data(512, seed=10)
        data1 = _make_gop_data(768, seed=20)
        ref0 = store.put("/v0.mp4", 0, 30, data0)
        ref1 = store.put("/v1.mp4", 0, 30, data1)

        arrays = store.get_batch([ref0, ref1])
        assert len(arrays) == 2
        np.testing.assert_array_equal(arrays[0], data0)
        np.testing.assert_array_equal(arrays[1], data1)

    def test_get_batch_cleans_orphans(self, store_id):
        """After eviction + get_batch, orphaned shm blocks are unlinked."""
        store = SharedGopStore.create(capacity=2, store_id=store_id)
        try:
            ref0 = store.put("/v0.mp4", 0, 30, _make_gop_data(256, seed=0))
            store.put("/v1.mp4", 0, 30, _make_gop_data(256, seed=1))

            # Evict v0 by inserting v2
            ref2 = store.put("/v2.mp4", 0, 30, _make_gop_data(256, seed=2))

            # v0's shm block still exists on disk (not unlinked yet)
            orphan_path = f"/dev/shm/{ref0.shm_name}"
            assert os.path.exists(orphan_path), "Orphan should still exist before get_batch"

            # get_batch cleans up the orphan
            store.get_batch([ref2])

            assert not os.path.exists(orphan_path), "Orphan should be unlinked after get_batch"
        finally:
            store.cleanup()

    def test_get_batch_empty_list(self, store):
        """get_batch([]) should not crash."""
        result = store.get_batch([])
        assert result == []


# ---------------------------------------------------------------------------
# Capacity-too-small fallback (evicted before read)
# ---------------------------------------------------------------------------

class TestCapacityTooSmall:
    """read() returns zeros + RuntimeWarning when shm block was evicted."""

    def test_read_evicted_returns_zeros_with_warning(self, store_id):
        """When a GOP is evicted before main reads it, read() returns zeros."""
        store = SharedGopStore.create(capacity=2, store_id=store_id)
        try:
            data0 = _make_gop_data(512, seed=0)
            ref0 = store.put("/v0.mp4", 0, 30, data0)
            store.put("/v1.mp4", 0, 30, _make_gop_data(512, seed=1))

            # Evict v0 by inserting v2
            store.put("/v2.mp4", 0, 30, _make_gop_data(512, seed=2))

            # Manually unlink the orphaned shm block (simulates _unlink_orphans)
            orphan_path = f"/dev/shm/{ref0.shm_name}"
            if os.path.exists(orphan_path):
                from multiprocessing import shared_memory
                shm = shared_memory.SharedMemory(name=ref0.shm_name, create=False)
                shm.close()
                shm.unlink()

            assert not os.path.exists(orphan_path), "Orphan must be gone before test"

            # read() should return zeros and emit RuntimeWarning
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                result = store.read(ref0)

                assert len(w) == 1
                assert issubclass(w[0].category, RuntimeWarning)
                assert "Increase store capacity" in str(w[0].message)
                assert f"current: {store.capacity}" in str(w[0].message)
                assert ref0.shm_name in str(w[0].message)

            # Returned array should be zeros with correct size
            assert result.shape == (ref0.data_size,)
            assert result.dtype == np.uint8
            np.testing.assert_array_equal(result, np.zeros(ref0.data_size, dtype=np.uint8))
        finally:
            store.cleanup()

    def test_get_batch_with_evicted_ref(self, store_id):
        """get_batch() handles a mix of valid and evicted refs gracefully."""
        store = SharedGopStore.create(capacity=2, store_id=store_id)
        try:
            data0 = _make_gop_data(256, seed=0)
            data1 = _make_gop_data(256, seed=1)
            ref0 = store.put("/v0.mp4", 0, 30, data0)
            ref1 = store.put("/v1.mp4", 0, 30, data1)

            # Evict v0
            data2 = _make_gop_data(256, seed=2)
            ref2 = store.put("/v2.mp4", 0, 30, data2)

            # Unlink orphan
            orphan_path = f"/dev/shm/{ref0.shm_name}"
            if os.path.exists(orphan_path):
                from multiprocessing import shared_memory
                shm = shared_memory.SharedMemory(name=ref0.shm_name, create=False)
                shm.close()
                shm.unlink()

            # get_batch with [evicted_ref, valid_ref]
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                arrays = store.get_batch([ref0, ref2])

                assert len(w) == 1
                assert issubclass(w[0].category, RuntimeWarning)

            assert len(arrays) == 2
            # First is zeros (evicted)
            np.testing.assert_array_equal(arrays[0], np.zeros(ref0.data_size, dtype=np.uint8))
            # Second is valid data
            np.testing.assert_array_equal(arrays[1], data2)
        finally:
            store.cleanup()


# ---------------------------------------------------------------------------
# attach errors
# ---------------------------------------------------------------------------

class TestAttachErrors:

    def test_attach_nonexistent_raises(self):
        """attach() raises FileNotFoundError with helpful message."""
        with pytest.raises(FileNotFoundError, match="create\\(\\)"):
            SharedGopStore.attach(capacity=4, store_id=8888)


# ---------------------------------------------------------------------------
# Multiple stores isolation
# ---------------------------------------------------------------------------

class TestStoreIsolation:
    """Different store_ids do not interfere."""

    def test_two_stores_independent(self):
        sid_a = _BASE_STORE_ID + 900
        sid_b = _BASE_STORE_ID + 901
        store_a = SharedGopStore.create(capacity=4, store_id=sid_a)
        store_b = SharedGopStore.create(capacity=4, store_id=sid_b)
        try:
            store_a.put("/v.mp4", 0, 30, _make_gop_data(256, seed=1))
            store_b.put("/v.mp4", 0, 30, _make_gop_data(256, seed=2))

            ref_a = store_a.lookup("/v.mp4", 15)
            ref_b = store_b.lookup("/v.mp4", 15)

            assert ref_a is not None
            assert ref_b is not None
            assert ref_a.shm_name != ref_b.shm_name

            # Cleanup A should not affect B
            store_a.cleanup()
            assert store_b.lookup("/v.mp4", 15) is not None
        finally:
            store_b.cleanup()
            # Ensure both are cleaned
            for path in _shm_files_for_store(sid_a) + _shm_files_for_store(sid_b):
                try:
                    os.unlink(path)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Cross-process test (spawn mode)
# ---------------------------------------------------------------------------

def _worker_fn(store_id, capacity, video_path, first_frame_id, gop_len, data_bytes, result_queue):
    """Worker process: attach to store, put data, lookup, send result back."""
    store = SharedGopStore.attach(capacity=capacity, store_id=store_id)
    data = np.frombuffer(data_bytes, dtype=np.uint8)
    ref = store.put(video_path, first_frame_id, gop_len, data)
    hit = store.lookup(video_path, first_frame_id)
    result_queue.put({
        'ref': ref,
        'hit': hit,
    })
    store.close()


class TestCrossProcess:
    """Verify SharedGopStore works across spawn'd processes."""

    def test_worker_put_main_lookup(self):
        sid = _BASE_STORE_ID + 950
        capacity = 8
        store = SharedGopStore.create(capacity=capacity, store_id=sid)
        try:
            data = _make_gop_data(2048, seed=99)
            ctx = multiprocessing.get_context("spawn")
            q = ctx.Queue()

            p = ctx.Process(
                target=_worker_fn,
                args=(sid, capacity, "/cross/v.mp4", 0, 30, data.tobytes(), q),
            )
            p.start()
            p.join(timeout=30)
            assert p.exitcode == 0, f"Worker exited with code {p.exitcode}"

            result = q.get(timeout=5)
            worker_ref = result['ref']
            worker_hit = result['hit']

            # Worker's put and lookup should succeed
            assert isinstance(worker_ref, GopRef)
            assert worker_hit is not None
            assert worker_hit.shm_name == worker_ref.shm_name

            # Main process can see what worker put
            main_hit = store.lookup("/cross/v.mp4", 15)
            assert main_hit is not None
            assert main_hit.shm_name == worker_ref.shm_name

            # Main process can read the data
            view = store.read(main_hit)
            np.testing.assert_array_equal(view, data)
        finally:
            store.cleanup()


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestStats:
    """get_stats / reset_stats correctness."""

    def test_stats_after_operations(self, store):
        store.put("/v.mp4", 0, 30, _make_gop_data(256))

        store.lookup("/v.mp4", 15)   # hit
        store.lookup("/v.mp4", 15)   # hit
        store.lookup("/other.mp4", 0)  # miss

        stats = store.get_stats()
        # put() does a double-check lookup internally, so hits includes that.
        # External: 2 hits + 1 miss.  put's internal lookup was a miss (first put).
        # Total hits: 2 (external), total misses: 1 (external) + 1 (put's internal) = 2
        # But put also increments puts counter.
        assert stats['puts'] == 1
        assert stats['hits'] >= 2
        assert stats['misses'] >= 1
        assert 0.0 <= stats['hit_rate'] <= 1.0
        assert stats['pool_usage'] == "1/8"

    def test_reset_stats(self, store):
        store.put("/v.mp4", 0, 30, _make_gop_data(256))
        store.lookup("/v.mp4", 15)
        store.reset_stats()

        stats = store.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['puts'] == 0
        assert stats['evictions'] == 0
        # pool_usage is from shared metadata, not reset
        assert stats['pool_usage'] == "1/8"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
