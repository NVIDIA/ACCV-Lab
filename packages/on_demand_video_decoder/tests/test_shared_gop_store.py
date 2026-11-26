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
    return (
        glob.glob(f"/dev/shm/gs_{store_id}_*")
        + glob.glob(f"/dev/shm/gs_meta_{store_id}")
        + glob.glob(f"/dev/shm/gs_tick_{store_id}")
        + glob.glob(f"/dev/shm/gs_lock_{store_id}")
    )


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
        ref = GopRef(shm_name="gs_0_12345_0", data_size=1024, first_frame_id=0, gop_len=30)
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

    @pytest.mark.parametrize("close_first", [False, True], ids=["open", "closed"])
    def test_cleanup_removes_all_shm(self, store_id, close_first):
        """After cleanup(), no shm files remain for this store."""
        s = SharedGopStore.create(capacity=4, store_id=store_id)
        s.put("/v.mp4", 0, 30, _make_gop_data(512))
        s.put("/v.mp4", 30, 30, _make_gop_data(512, seed=2))
        if close_first:
            s.close()
        s.cleanup()

        remaining = _shm_files_for_store(store_id)
        assert remaining == [], f"Leaked shm files: {remaining}"

    def test_repeated_worker_close(self, store):
        worker = SharedGopStore.attach(store.capacity, store.store_id)
        data = _make_gop_data(32)
        ref = worker.put("/v.mp4", 0, 30, data)
        worker.close()
        worker.close()
        np.testing.assert_array_equal(store.read(ref), data)

    def test_repeated_cleanup_preserves_reused_store_id(self, store_id):
        original = SharedGopStore.create(1, store_id)
        original.cleanup()
        replacement = SharedGopStore.create(1, store_id)
        try:
            replacement.put("/v.mp4", 0, 30, _make_gop_data(32))
            files = set(_shm_files_for_store(store_id))
            original.cleanup()
            assert set(_shm_files_for_store(store_id)) == files
        finally:
            replacement.cleanup()


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

    def test_overlapping_ranges_stored_independently(self, store):
        """Two puts with distinct ``first_frame_id`` produce distinct entries
        even when their ``[first_frame_id, first_frame_id + gop_len)`` ranges
        overlap. De-duplication keys on the exact
        ``(video, first_frame_id, gop_len)`` tuple, not on range coverage.
        """
        gop_a = _make_gop_data(1024, seed=1)
        gop_b = _make_gop_data(1024, seed=2)
        ref_a = store.put("/v.mp4", first_frame_id=400, gop_len=73, data=gop_a)
        ref_b = store.put("/v.mp4", first_frame_id=470, gop_len=75, data=gop_b)
        assert ref_a.shm_name != ref_b.shm_name
        assert ref_b.first_frame_id == 470 and ref_b.gop_len == 75
        # shm names embed first_frame_id; the two entries live in distinct blocks.
        assert ref_a.shm_name.endswith("_400")
        assert ref_b.shm_name.endswith("_470")
        np.testing.assert_array_equal(store.read(ref_a), gop_a)
        np.testing.assert_array_equal(store.read(ref_b), gop_b)

    def test_exact_match_dedup_on_first_frame_id_and_gop_len(self, store):
        """Queued GOPs with the same start and different lengths retain their own bytes."""
        gop_old = _make_gop_data(1024, seed=10)
        gop_new = _make_gop_data(2048, seed=11)
        short = store.put("/v.mp4", first_frame_id=0, gop_len=20, data=gop_old)
        long = store.put("/v.mp4", first_frame_id=0, gop_len=30, data=gop_new)
        assert short.shm_name != long.shm_name
        short_view, long_view = store.get_batch([short, long])
        np.testing.assert_array_equal(short_view, gop_old)
        np.testing.assert_array_equal(long_view, gop_new)


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
            ref0 = store.put("/v0.mp4", 0, 30, _make_gop_data(256, seed=0), pin=False)
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
# Reads after eviction
# ---------------------------------------------------------------------------


class TestCapacityTooSmall:
    """A stale reference must never be accepted as a serialized GOP."""

    def test_read_evicted_raises(self, store_id):
        store = SharedGopStore.create(capacity=2, store_id=store_id)
        try:
            data0 = _make_gop_data(512, seed=0)
            ref0 = store.put("/v0.mp4", 0, 30, data0, pin=False)
            store.put("/v1.mp4", 0, 30, _make_gop_data(512, seed=1))

            # An unrelated batch evicts v0 and cleans up its block.
            ref2 = store.put("/v2.mp4", 0, 30, _make_gop_data(512, seed=2))
            store.get_batch([ref2])

            with pytest.raises(FileNotFoundError, match=ref0.shm_name):
                store.read(ref0)
        finally:
            store.cleanup()

    def test_get_batch_with_evicted_ref(self, store_id):
        """A missing block fails the batch and identifies the stale reference."""
        store = SharedGopStore.create(capacity=2, store_id=store_id)
        try:
            data0 = _make_gop_data(256, seed=0)
            data1 = _make_gop_data(256, seed=1)
            ref0 = store.put("/v0.mp4", 0, 30, data0, pin=False)
            store.put("/v1.mp4", 0, 30, data1)

            # Evict v0
            data2 = _make_gop_data(256, seed=2)
            ref2 = store.put("/v2.mp4", 0, 30, data2)

            store.get_batch([ref2])
            with pytest.raises(FileNotFoundError, match=ref0.shm_name):
                store.get_batch([ref2, ref0])
            np.testing.assert_array_equal(store.get_batch([ref2])[0], data2)
        finally:
            store.cleanup()


class TestPinnedBatch:
    def test_get_batch_releases_independent_refs_and_preserves_views(self, store_id):
        store = SharedGopStore.create(1, store_id)
        try:
            data = _make_gop_data(32)
            first = store.put("/v.mp4", 0, 30, data)
            second = store.lookup("/v.mp4", 10)
            # DataLoader can reconstruct a NamedTuple from its original fields.
            first_copy = GopRef(*first)
            a, b = store.get_batch([first, first_copy])
            with pytest.raises(BufferError, match="all slots"):
                store.put("/other.mp4", 0, 30, data)
            store.get_batch([second])
            replacement = store.put("/other.mp4", 0, 30, data[::-1])
            store.get_batch([replacement])
            np.testing.assert_array_equal(a, data)
            np.testing.assert_array_equal(b, data)
        finally:
            store.cleanup()

    def test_discarded_and_forwarded_queue_refs(self, store_id):
        store = SharedGopStore.create(1, store_id)
        worker = SharedGopStore.attach(1, store_id)
        try:
            data = _make_gop_data(32)
            source = worker.put("/v.mp4", 0, 30, data)
            first = pickle.loads(pickle.dumps(source))
            second = pickle.loads(pickle.dumps(source))
            # Reading the sender's copy or discarding the first delivery cannot release the second.
            store.read(source)
            del first
            with pytest.raises(BufferError, match="all slots"):
                worker.put("/other.mp4", 0, 30, data)
            (view,) = store.get_batch([second])
            worker.put("/other.mp4", 0, 30, data)
            np.testing.assert_array_equal(view, data)
        finally:
            worker.close()
            store.cleanup()

    def test_queued_duplicate_pins_survive_other_batch(self, store_id):
        owner = SharedGopStore.create(2, store_id)
        worker = SharedGopStore.attach(2, store_id)
        try:
            data = _make_gop_data(128, seed=140)
            first = worker.put("/wrist.mp4", 120, 30, data)
            hit = worker.lookup("/wrist.mp4", 140)
            duplicate = worker.put("/wrist.mp4", 120, 30, data)
            assert first == hit == duplicate
            other_data = _make_gop_data(128, seed=0)
            for frame in (0, 30, 60):
                other = worker.put("/other.mp4", frame, 30, other_data)
                owner.get_batch([other])

            view, other_view, hit_view, repeated_view = owner.get_batch([first, other, hit, first])
            np.testing.assert_array_equal(other_view, other_data)
            assert np.shares_memory(view, hit_view)
            assert np.shares_memory(view, repeated_view)
            busy = worker.put("/other.mp4", 90, 30, other_data)
            # Rereading a consumed ref or an unpinned probe must not release the third sample's pin.
            probe = worker.lookup("/wrist.mp4", 140, pin=False)
            owner.get_batch([first, first, probe])
            with pytest.raises(BufferError, match="all slots"):
                worker.put("/full.mp4", 0, 1, other_data)
            owner.read(duplicate)

            replacement = worker.put("/full.mp4", 0, 1, other_data)
            owner.get_batch([replacement])
            assert not os.path.exists(f"/dev/shm/{first.shm_name}")
            np.testing.assert_array_equal(view, data)
            owner.get_batch([busy])
        finally:
            worker.close()
            owner.cleanup()

    def test_failed_batch_preserves_all_pins(self, store, monkeypatch):
        data = _make_gop_data(128)
        first = store.put("/v.mp4", 0, 30, data)
        second = store.put("/v.mp4", 30, 30, data[::-1])
        refs = [first, second]
        # Keep the other slots occupied so an early release would allow eviction.
        held = [store.put("/other.mp4", i, 1, data) for i in range(store.capacity - 2)]

        read = store._open_view

        def fail_second_open(ref):
            if ref == second:
                raise FileNotFoundError(ref.shm_name)
            return read(ref)

        with monkeypatch.context() as patch:
            patch.setattr(store, "_open_view", fail_second_open)
            with pytest.raises(FileNotFoundError, match=second.shm_name):
                store.get_batch(refs)

        with pytest.raises(BufferError, match="all slots"):
            store.put("/full.mp4", 0, 1, data)
        second_view, first_view = store.get_batch([second, first])
        np.testing.assert_array_equal(first_view, data)
        np.testing.assert_array_equal(second_view, data[::-1])
        store.get_batch(held)

    def test_reinserted_gop_uses_a_fresh_handle(self, store_id):
        store = SharedGopStore.create(1, store_id)
        try:
            data = _make_gop_data(128)
            old = store.put("/v.mp4", 0, 30, data)
            (old_view,) = store.get_batch([old])
            store.put("/other.mp4", 0, 30, data)
            new = store.put("/v.mp4", 0, 30, data[::-1])
            (new_view,) = store.get_batch([new])
            np.testing.assert_array_equal(new_view, data[::-1])
            np.testing.assert_array_equal(old_view, data)
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
    result_queue.put(
        {
            'ref': ref,
            'hit': hit,
        }
    )
    store.close()


def _queue_wrist_gop(store_id, data, queue, issue_via):
    worker = SharedGopStore.attach(capacity=2, store_id=store_id)
    try:
        if issue_via == "lookup":
            ref = worker.lookup("/wrist.mp4", 140)
        else:
            ref = worker.put("/wrist.mp4", 120, 30, data)
        queue.put(ref)
    finally:
        worker.close()


def _use_forked_ref(store_id, refs, action, queue):
    if action == "read":
        worker = SharedGopStore.attach(1, store_id)
        try:
            queue.put(worker.read(refs[0]).tobytes())
        finally:
            worker.close()
    elif action == "forward":
        queue.put(refs[0])
    refs.clear()  # Drop the child's copy while the parent still holds its reference.


class TestCrossProcess:
    """Verify SharedGopStore works across processes."""

    @pytest.mark.parametrize("action", ["discard", "read", "forward"])
    @pytest.mark.parametrize("received", [False, True], ids=["issued", "received"])
    def test_forked_ref_preserves_parent_pin(self, store_id, received, action):
        store = SharedGopStore.create(1, store_id)
        ctx = multiprocessing.get_context("fork")
        queue = ctx.SimpleQueue()
        data = _make_gop_data(32)
        refs = [store.put("/v.mp4", 0, 30, data)]
        if received:
            queue.put(refs.pop())
            refs.append(queue.get())
        child = ctx.Process(target=_use_forked_ref, args=(store_id, refs, action, queue))
        try:
            child.start()
            child.join(timeout=10)
            assert child.exitcode == 0
            if action == "read":
                assert queue.get() == data.tobytes()

            with pytest.raises(BufferError, match="all slots"):
                store.put("/other.mp4", 0, 30, data)
            np.testing.assert_array_equal(store.read(refs[0]), data)
            if action == "forward":
                # The queued delivery still owns a pin after the parent opens its own ref.
                with pytest.raises(BufferError, match="all slots"):
                    store.put("/other.mp4", 0, 30, data)
                np.testing.assert_array_equal(store.read(queue.get()), data)
            store.put("/other.mp4", 0, 30, data)
        finally:
            if child.is_alive():
                child.terminate()
                child.join()
            queue.close()
            store.cleanup()

    @pytest.mark.parametrize("start_method", ["fork", "spawn"])
    @pytest.mark.parametrize("issue_via", ["lookup", "put"])
    def test_delayed_queue_consumer(self, store_id, start_method, issue_via):
        """A queued GOP must retain the worker's bytes while other batches are consumed."""
        store = SharedGopStore.create(2, store_id)
        ctx = multiprocessing.get_context(start_method)
        q = ctx.Queue()
        data = np.array([1, 2, 3, 4], dtype=np.uint8)
        worker = ctx.Process(target=_queue_wrist_gop, args=(store_id, data, q, issue_via))
        try:
            if issue_via == "lookup":
                store.get_batch([store.put("/wrist.mp4", 120, 30, data)])
            worker.start()
            worker.join(timeout=30)
            assert worker.exitcode == 0

            # Leave the wrist reference in the queue until two other GOPs cycle through the store.
            for first_frame_id in (0, 30):
                other = store.put("/other.mp4", first_frame_id, 30, data[::-1])
                store.get_batch([other])
            ref = q.get(timeout=5)
            assert ref is not None
            (actual,) = store.get_batch([ref])
            np.testing.assert_array_equal(actual, data)
            # Once the queue reference is consumed, the wrist slot is reusable.
            extra = store.put("/other.mp4", 60, 30, data)
            store.get_batch([extra])
            assert store.lookup("/wrist.mp4", 140) is None
        finally:
            if worker.is_alive():
                worker.terminate()
                worker.join()
            q.close()
            store.cleanup()

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

        store.lookup("/v.mp4", 15)  # hit
        store.lookup("/v.mp4", 15)  # hit
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


# ---------------------------------------------------------------------------
# Factory enforcement (direct construction must be blocked)
# ---------------------------------------------------------------------------


class TestFactoryEnforcement:
    """Direct __init__ calls must raise; only create()/attach() are valid."""

    def test_direct_construction_raises(self, store_id):
        """SharedGopStore(...) without the private key must raise RuntimeError."""
        with pytest.raises(RuntimeError, match="cannot be instantiated directly"):
            SharedGopStore(capacity=4, store_id=store_id, _create=True)

    def test_direct_construction_error_mentions_factories(self, store_id):
        """Error message must point users to create() and attach()."""
        with pytest.raises(RuntimeError, match=r"create\(\).*attach\(\)"):
            SharedGopStore(capacity=4, store_id=store_id, _create=True)

    def test_direct_construction_does_not_allocate_shm(self, store_id):
        """A blocked construction must reject before touching shm."""
        with pytest.raises(RuntimeError):
            SharedGopStore(capacity=4, store_id=store_id, _create=True)
        assert _shm_files_for_store(store_id) == [], "shm leaked despite blocked construction"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
