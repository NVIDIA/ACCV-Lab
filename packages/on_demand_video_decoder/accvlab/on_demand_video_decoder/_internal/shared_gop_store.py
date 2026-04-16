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
SharedGopStore: Cross-process shared GOP store backed by POSIX SharedMemory.

Workers ``put()`` GOP packet data and pass lightweight :class:`GopRef`
references through the DataLoader IPC queue.  The main process
``get_batch()`` to read references as zero-copy numpy views.

Synchronization uses file-based locking (``flock``) which works across
``spawn``'d DataLoader workers (unlike ``multiprocessing.Lock``).

Usage::

    # Main process -- create before spawning DataLoader workers
    store = SharedGopStore.create(capacity=120, store_id=0)

    # Worker -- attach to existing store
    store = SharedGopStore.attach(capacity=120, store_id=0)
    ref = store.lookup(video_path, frame_id)
    if ref is None:
        data = load_from_disk(...)
        ref = store.put(video_path, first_frame_id, gop_len, data)
    # Pass ``ref`` (lightweight) through DataLoader IPC queue

    # Main process -- get batch of GOP data, then clean orphans
    arrays = store.get_batch(refs)

    # Main process -- shutdown
    store.cleanup()
"""

import fcntl
import glob as glob_mod
import hashlib
import os
from multiprocessing import shared_memory
from typing import List, Optional

import numpy as np

from .types import GopRef

# ---------------------------------------------------------------------------
# Metadata table layout
# ---------------------------------------------------------------------------

# 96 bytes per entry -- fits in L1 cache for capacity < 500 (< 48 KB total).
ENTRY_DTYPE = np.dtype([
    ('video_path_hash', np.uint64),  # deterministic MD5 hash of video_path
    ('first_frame_id', np.int32),    # first frame ID of this GOP
    ('gop_len', np.int32),           # number of frames in this GOP
    ('data_size', np.int32),         # actual GOP data size in bytes
    ('state', np.uint8),             # 0=FREE, 1=USED
    ('_pad', np.uint8, (3,)),        # alignment padding
    ('access_tick', np.int64),       # monotonic counter for LRU ordering
    ('shm_name', 'S64'),            # SharedMemory name for the data block
])

_STATE_FREE = np.uint8(0)
_STATE_USED = np.uint8(1)

# SharedMemory name prefix used for GOP data blocks.
_SHM_PREFIX = "gs"


def _hash_video_path(video_path: str) -> np.uint64:
    """Deterministic uint64 hash for a video path string.

    Uses MD5 (truncated to 8 bytes) instead of Python's built-in ``hash()``,
    because ``hash()`` uses a per-process random seed when
    ``PYTHONHASHSEED != 0``, which breaks cross-process consistency with
    ``spawn``'d DataLoader workers.
    """
    digest = hashlib.md5(video_path.encode()).digest()[:8]
    return np.uint64(int.from_bytes(digest, 'little'))


class SharedGopStore:
    """Cross-process shared GOP store backed by POSIX SharedMemory.

    Stores GOP packet data in per-GOP SharedMemory blocks.  A small
    SharedMemory block holds the metadata table (index).  File-based
    locking (``flock``) provides cross-process safety under ``spawn`` mode.

    **Capacity sizing:**  ``capacity`` must exceed the maximum number of
    GOPs that can be "in flight" (queued in the DataLoader + being
    consumed by the training loop)::

        min_capacity > (prefetch_factor * num_workers + 1) * batch_size * num_cameras

    A recommended formula is ``batch_size * num_cameras * 10``.

    Args:
        capacity: Maximum number of GOPs to cache.
        store_id: Unique identifier (typically ``LOCAL_RANK``).
        _create: Internal flag -- use :meth:`create` / :meth:`attach`.
    """

    def __init__(self, capacity: int, store_id: int, _create: bool):
        self.capacity = capacity
        self.store_id = store_id
        self._is_creator = _create

        # --- Metadata SharedMemory ---
        meta_name = f"gs_meta_{store_id}"
        meta_size = capacity * ENTRY_DTYPE.itemsize
        if _create:
            _cleanup_stale_shm(meta_name)
            self._meta_shm = shared_memory.SharedMemory(
                name=meta_name, create=True, size=meta_size)
            # Zero-initialize all entries (state=FREE)
            np.frombuffer(self._meta_shm.buf, dtype=np.uint8)[:] = 0
        else:
            try:
                self._meta_shm = shared_memory.SharedMemory(
                    name=meta_name, create=False)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"SharedGopStore with store_id={store_id} not found. "
                    f"Call SharedGopStore.create() in the main process first."
                )

        self._entries = np.ndarray(
            capacity, dtype=ENTRY_DTYPE, buffer=self._meta_shm.buf)

        # --- File-based lock (works across spawn'd processes) ---
        self._lock_path = f"/dev/shm/gs_lock_{store_id}"
        if _create:
            with open(self._lock_path, 'w'):
                pass
        self._lock_fd = os.open(self._lock_path, os.O_RDWR)

        # --- Monotonic tick counter in a tiny SharedMemory ---
        tick_name = f"gs_tick_{store_id}"
        if _create:
            _cleanup_stale_shm(tick_name)
            self._tick_shm = shared_memory.SharedMemory(
                name=tick_name, create=True, size=8)
            np.frombuffer(self._tick_shm.buf, dtype=np.int64)[:] = 0
        else:
            self._tick_shm = shared_memory.SharedMemory(
                name=tick_name, create=False)
        self._tick_arr = np.ndarray(1, dtype=np.int64, buffer=self._tick_shm.buf)

        # --- Per-process handle cache (not shared) ---
        self._local_shm_handles = {}

        # --- Stats (per-process, not shared) ---
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0

    # ------------------------------------------------------------------ #
    #  Factory class methods
    # ------------------------------------------------------------------ #

    @classmethod
    def create(cls, capacity: int, store_id: int = 0) -> 'SharedGopStore':
        """Allocate a new store.  Call from main process before spawning workers.

        Args:
            capacity: Max number of GOPs to cache.
            store_id: Unique identifier (typically ``LOCAL_RANK``).
        """
        return cls(capacity=capacity, store_id=store_id, _create=True)

    @classmethod
    def attach(cls, capacity: int, store_id: int = 0) -> 'SharedGopStore':
        """Attach to an existing store.  Call from worker processes.

        Raises:
            FileNotFoundError: If the store has not been created yet.
        """
        return cls(capacity=capacity, store_id=store_id, _create=False)

    # ------------------------------------------------------------------ #
    #  Locking helpers (flock)
    # ------------------------------------------------------------------ #

    def _lock(self):
        fcntl.flock(self._lock_fd, fcntl.LOCK_EX)

    def _unlock(self):
        fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _next_tick(self) -> int:
        """Increment and return the monotonic tick counter.

        Not truly atomic across processes without the lock, but the tick
        is only used for LRU ordering -- a rare duplicate is harmless.
        """
        self._tick_arr[0] += 1
        return int(self._tick_arr[0])

    # ------------------------------------------------------------------ #
    #  Worker API
    # ------------------------------------------------------------------ #

    def lookup(self, video_path: str, frame_id: int) -> Optional[GopRef]:
        """Lock-free lookup for a cached GOP containing *frame_id*.

        Returns a :class:`GopRef` on hit, ``None`` on miss.  Lock-free
        design means the worst case is a stale miss (one extra disk read),
        never a correctness issue.
        """
        vp_hash = _hash_video_path(video_path)
        for i in range(self.capacity):
            e = self._entries[i]
            if (e['state'] == _STATE_USED
                    and e['video_path_hash'] == vp_hash
                    and e['first_frame_id'] <= frame_id
                    < e['first_frame_id'] + e['gop_len']):
                e['access_tick'] = self._next_tick()
                self._hits += 1
                return GopRef(
                    shm_name=e['shm_name'].decode(),
                    data_size=int(e['data_size']),
                    first_frame_id=int(e['first_frame_id']),
                    gop_len=int(e['gop_len']),
                )
        self._misses += 1
        return None

    def put(self, video_path: str, first_frame_id: int,
            gop_len: int, data: np.ndarray) -> GopRef:
        """Store GOP packet data and return a :class:`GopRef`.

        Holds ``flock`` during eviction + insertion to guarantee atomicity.
        Performs a double-check after acquiring the lock (another worker
        may have inserted while we waited).
        """
        self._lock()
        try:
            # Double-check: another worker may have inserted while we waited
            existing = self.lookup(video_path, first_frame_id)
            if existing is not None:
                return existing

            slot_idx = self._find_free_or_evict()

            # Content-addressed naming -- same GOP always gets the same name.
            vp_hash = _hash_video_path(video_path)
            shm_name = f"{_SHM_PREFIX}_{self.store_id}_{vp_hash}_{first_frame_id}"
            data_size = int(data.nbytes)

            # Clean up if same GOP was previously cached then evicted
            _cleanup_stale_shm(shm_name)

            shm = shared_memory.SharedMemory(
                name=shm_name, create=True, size=data_size)
            shm.buf[:data_size] = data.tobytes()
            shm.close()  # close handle; shm persists until unlink

            # Write metadata entry
            self._entries[slot_idx]['video_path_hash'] = vp_hash
            self._entries[slot_idx]['first_frame_id'] = first_frame_id
            self._entries[slot_idx]['gop_len'] = gop_len
            self._entries[slot_idx]['data_size'] = data_size
            self._entries[slot_idx]['access_tick'] = self._next_tick()
            self._entries[slot_idx]['shm_name'] = shm_name.encode()
            # Set state last (acts as publish barrier for lock-free readers)
            self._entries[slot_idx]['state'] = _STATE_USED

            self._puts += 1
            return GopRef(
                shm_name=shm_name,
                data_size=data_size,
                first_frame_id=first_frame_id,
                gop_len=gop_len,
            )
        finally:
            self._unlock()

    # ------------------------------------------------------------------ #
    #  Main Process API
    # ------------------------------------------------------------------ #

    def read(self, ref: GopRef) -> np.ndarray:
        """Zero-copy uint8 numpy view of GOP data in shared memory.

        Caches ``SharedMemory`` handles per-process to avoid repeated
        ``shm_open()`` system calls.
        """
        shm_name = ref.shm_name
        if shm_name not in self._local_shm_handles:
            try:
                shm = shared_memory.SharedMemory(name=shm_name, create=False)
            except FileNotFoundError:
                import warnings
                warnings.warn(
                    f"SharedGopStore: shm block '{shm_name}' not found. "
                    f"This means a GOP was evicted before the main process "
                    f"could read it. Increase store capacity "
                    f"(current: {self.capacity}) — it must exceed "
                    f"(prefetch_factor * num_workers + 1) * batch_size * num_cameras. "
                    f"Returning zeros as fallback.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                return np.zeros(ref.data_size, dtype=np.uint8)
            self._local_shm_handles[shm_name] = shm
        else:
            shm = self._local_shm_handles[shm_name]

        return np.frombuffer(shm.buf[:ref.data_size], dtype=np.uint8)

    def get_batch(self, refs: List[GopRef]) -> List[np.ndarray]:
        """Read a batch of GOPs from shared memory (zero-copy).

        Call once per training iteration from the main process.
        Holds ``flock`` during the entire operation so that no worker can
        evict a block while handles are being opened.  After opening,
        orphaned shm blocks (evicted but not yet unlinked) are cleaned up.

        Args:
            refs: Flat list of :class:`GopRef` from DataLoader workers.

        Returns:
            List of zero-copy uint8 numpy views, same order as *refs*.
        """
        self._lock()
        try:
            result = [self.read(ref) for ref in refs]
            self._unlink_orphans()
            return result
        finally:
            self._unlock()

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Per-process cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        used_slots = int(np.count_nonzero(
            self._entries['state'] == _STATE_USED))
        return {
            'hits': self._hits,
            'misses': self._misses,
            'hit_rate': hit_rate,
            'puts': self._puts,
            'evictions': self._evictions,
            'pool_usage': f"{used_slots}/{self.capacity}",
        }

    def reset_stats(self) -> None:
        """Reset per-process statistics counters."""
        self._hits = 0
        self._misses = 0
        self._puts = 0
        self._evictions = 0

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _find_free_or_evict(self) -> int:
        """Find a FREE slot, or evict the LRU entry.  Must hold flock."""
        min_tick = np.iinfo(np.int64).max
        min_idx = 0
        for i in range(self.capacity):
            if self._entries[i]['state'] == _STATE_FREE:
                return i
            if self._entries[i]['access_tick'] < min_tick:
                min_tick = int(self._entries[i]['access_tick'])
                min_idx = i

        # Evict the LRU slot.  Do NOT unlink the shm block here -- workers
        # don't know which blocks the main process still needs.  The main
        # process cleans up orphans in get_batch() -> _unlink_orphans().
        self._entries[min_idx]['state'] = _STATE_FREE
        self._evictions += 1
        return min_idx

    def _unlink_orphans(self):
        """Unlink shm blocks that were evicted but not yet cleaned up.

        Called under flock by :meth:`get_batch`.  Scans ``/dev/shm`` for
        blocks belonging to this store that are no longer in the metadata
        table.
        """
        # Active shm names from metadata table
        active_names = set()
        for i in range(self.capacity):
            if self._entries[i]['state'] == _STATE_USED:
                name = self._entries[i]['shm_name'].decode()
                if name:
                    active_names.add(name)

        # Scan /dev/shm and unlink orphans
        prefix = f"{_SHM_PREFIX}_{self.store_id}_"
        for path in glob_mod.glob(f"/dev/shm/{prefix}*"):
            name = os.path.basename(path)
            if name not in active_names:
                # Close our cached handle if any
                if name in self._local_shm_handles:
                    _force_close_shm(self._local_shm_handles[name])
                    del self._local_shm_handles[name]
                # Unlink the orphan
                try:
                    shm = shared_memory.SharedMemory(name=name, create=False)
                    shm.close()
                    shm.unlink()
                except FileNotFoundError:
                    pass

    # ------------------------------------------------------------------ #
    #  Lifecycle
    # ------------------------------------------------------------------ #

    def cleanup(self) -> None:
        """Unlink **all** SharedMemory blocks and the lock file.

        Call from the main process on shutdown.
        """
        # Close local handles (may have live numpy views -> _force_close_shm)
        for shm in self._local_shm_handles.values():
            _force_close_shm(shm)
        self._local_shm_handles.clear()

        # Glob-clean ALL shm blocks for this store (catches orphans)
        prefix = f"{_SHM_PREFIX}_{self.store_id}_"
        for path in glob_mod.glob(f"/dev/shm/{prefix}*"):
            name = os.path.basename(path)
            try:
                shm = shared_memory.SharedMemory(name=name, create=False)
                shm.close()
                shm.unlink()
            except FileNotFoundError:
                pass

        # Unlink meta and tick SharedMemory
        _force_close_shm(self._meta_shm)
        _force_close_shm(self._tick_shm)
        if self._is_creator:
            try:
                self._meta_shm.unlink()
            except FileNotFoundError:
                pass
            try:
                self._tick_shm.unlink()
            except FileNotFoundError:
                pass
            try:
                os.close(self._lock_fd)
                os.unlink(self._lock_path)
            except OSError:
                pass

    def close(self) -> None:
        """Close SharedMemory handles **without** unlinking.

        Call from worker processes before exit.
        """
        for shm in self._local_shm_handles.values():
            _force_close_shm(shm)
        self._local_shm_handles.clear()

        _force_close_shm(self._meta_shm)
        _force_close_shm(self._tick_shm)
        try:
            os.close(self._lock_fd)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Module-private helpers
# ---------------------------------------------------------------------------

def _force_close_shm(shm) -> None:
    """Close a SharedMemory handle, suppressing BufferError from live numpy views.

    When ``np.frombuffer(shm.buf)`` creates a zero-copy view, the numpy
    array holds a reference to the underlying mmap.  ``shm.close()`` then
    raises ``BufferError: cannot close exported pointers exist``.

    We catch this, then clear the internal ``_mmap`` and ``_buf`` attributes
    so that ``SharedMemory.__del__`` (called later by GC) does not re-raise
    the same error.  The actual backing memory is freed by the OS when the
    last numpy view is garbage-collected.
    """
    try:
        shm.close()
    except BufferError:
        # Prevent __del__ from retrying and printing the same traceback.
        shm._buf = None
        shm._mmap = None
    except Exception:
        pass


def _cleanup_stale_shm(name: str) -> None:
    """Remove a SharedMemory block left over from a previous run."""
    try:
        stale = shared_memory.SharedMemory(name=name, create=False)
        stale.close()
        stale.unlink()
    except FileNotFoundError:
        pass
