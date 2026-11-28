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

Workers pin serialized GOP bundles and pass lightweight :class:`GopRef`
references through the DataLoader IPC queue. The main process calls
:meth:`SharedGopStore.get_batch` to open zero-copy numpy views and
release the corresponding pins.

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

    # Main process -- read the batch in input order
    arrays = store.get_batch(refs)

    # Main process -- shutdown
    store.cleanup()
"""

import fcntl
import glob as glob_mod
import hashlib
import os
import threading
import time
from collections import Counter
from contextlib import contextmanager
from multiprocessing import shared_memory
from multiprocessing.util import ForkAwareThreadLock
from typing import List, Optional, Sequence

import numpy as np

from .types import GopRef

# ---------------------------------------------------------------------------
# Metadata table layout
# ---------------------------------------------------------------------------

# 96 bytes per entry -- fits in L1 cache for capacity < 500 (< 48 KB total).
ENTRY_DTYPE = np.dtype(
    [
        ('video_path_hash', np.uint64),  # deterministic MD5 hash of video_path
        ('first_frame_id', np.int32),  # first frame ID of this GOP
        ('gop_len', np.int32),  # number of frames in this GOP
        ('data_size', np.int32),  # actual GOP data size in bytes
        ('state', np.uint8),  # 0=FREE, 1=USED
        ('_pad', np.uint8),
        ('pins', np.uint16),  # outstanding queued references, protected by flock
        ('access_tick', np.int64),  # monotonic counter for LRU ordering
        ('shm_name', 'S64'),  # SharedMemory name for the data block
    ]
)

_STATE_FREE = np.uint8(0)
_STATE_USED = np.uint8(1)
_MAX_PINS = np.iinfo(ENTRY_DTYPE['pins']).max

# SharedMemory name prefix used for GOP data blocks.
_SHM_PREFIX = "gs"

# Private key to prevent direct instantiation of SharedGopStore.
# Mirrors the pattern used by ``CachedGopDecoder`` in ``_internal/decoder.py``.
_CREATION_KEY = object()


def _reset_process_state():
    # A raw fork can inherit a pin lock held by a thread that only exists in the parent.
    global _PROCESS_PID, _PIN_STATES, _HELD_LOCKS
    _PROCESS_PID = os.getpid()
    _PIN_STATES = threading.Lock()
    _HELD_LOCKS = threading.local()


_reset_process_state()
os.register_at_fork(after_in_child=_reset_process_state)


def _hash_video_path(video_path: str) -> np.uint64:
    """Deterministic uint64 hash for a video path string.

    Uses MD5 (truncated to 8 bytes) instead of Python's built-in ``hash()``,
    because ``hash()`` uses a per-process random seed when
    ``PYTHONHASHSEED != 0``, which breaks cross-process consistency with
    ``spawn``'d DataLoader workers.
    """
    digest = hashlib.md5(video_path.encode()).digest()[:8]
    return np.uint64(int.from_bytes(digest, 'little'))


class _PinnedName(str):
    """One hand-off pin. DataLoader may rebuild GopRef from its four fields."""

    _state = "pending"
    _issuer = None

    def __new__(cls, value):
        name = super().__new__(cls, value)
        name._owner_pid = _PROCESS_PID  # Set on issuance and again on unpickle.
        return name

    def _owns_pending_pin(self) -> bool:
        return self._owner_pid == _PROCESS_PID and self._state == "pending"

    def __deepcopy__(self, memo):
        # Copies of a GopRef in one batch share its single hand-off.
        return self

    def __reduce__(self):
        # Only the owner can transfer its pin. Each additional or fork-inherited
        # delivery needs a new pin of its own.
        with _PIN_STATES:
            if self._state == "opened":
                return str, (str(self),)
            transfer_pin = self._owns_pending_pin()
            if transfer_pin:
                self._state = "sent"
        if not transfer_pin:
            self._adjust_pin(1)
        return type(self), (str(self),), {"_location": self._location}

    def __del__(self):
        # Cache warm-ups and references discarded during batch deduplication.
        if hasattr(self, "_location") and self._owns_pending_pin():
            self._adjust_pin(-1)

    def _adjust_pin(self, change: int) -> None:
        capacity, store_id, slot = self._location
        # A fork-inherited file descriptor shares the parent's flock.
        store = self._issuer if self._owner_pid == _PROCESS_PID else None
        if store is not None and store._lock_fd is not None:
            store._adjust_pin(self, slot, change)
            return
        try:
            if os.stat(f"/dev/shm/gs_meta_{store_id}").st_size != capacity * ENTRY_DTYPE.itemsize:
                raise FileNotFoundError(str(self))
            store = SharedGopStore.attach(capacity, store_id)
        except FileNotFoundError:
            if change > 0:
                raise
            return
        try:
            store._adjust_pin(self, slot, change)
        finally:
            store.close()


class SharedGopStore:
    """Cross-process shared GOP store backed by POSIX SharedMemory.

    Stores serialized GOP bundles in per-GOP SharedMemory blocks.  A small
    SharedMemory block holds the metadata table (index).  File-based
    locking (``flock``) provides cross-process safety under ``spawn`` mode.

    **Capacity sizing:** Allow one slot per distinct GOP with an outstanding
    pin, including references held by workers and queued batches. A new GOP
    raises ``BufferError`` when all slots are pinned. After the main process
    opens the batch, its views remain readable for the duration of decoding.

    Note:
        Do not instantiate this class directly.
        Use :meth:`create` (main process, before spawning workers) or
        :meth:`attach` (worker processes) instead — these factories
        manage shared-memory creation and tear-down correctly.

    Args:
        capacity: Maximum number of GOPs to cache.
        store_id: Unique identifier (typically ``LOCAL_RANK``).
        _create: Internal flag -- use :meth:`create` / :meth:`attach`.

    Raises:
        RuntimeError: If called directly instead of via :meth:`create` /
            :meth:`attach`.
    """

    def __init__(self, capacity: int, store_id: int, _create: bool, *, _key=None):
        if _key is not _CREATION_KEY:
            raise RuntimeError(
                "SharedGopStore cannot be instantiated directly. "
                "Use SharedGopStore.create() (main process) or "
                "SharedGopStore.attach() (workers) instead."
            )
        self.capacity = capacity
        self.store_id = store_id
        self._is_creator = _create
        self._cleaned_up = False

        # --- Metadata SharedMemory ---
        meta_name = f"gs_meta_{store_id}"
        meta_size = capacity * ENTRY_DTYPE.itemsize
        if _create:
            _cleanup_stale_shm(meta_name)
            self._meta_shm = shared_memory.SharedMemory(name=meta_name, create=True, size=meta_size)
            # Zero-initialize all entries (state=FREE)
            np.frombuffer(self._meta_shm.buf, dtype=np.uint8)[:] = 0
        else:
            try:
                self._meta_shm = shared_memory.SharedMemory(name=meta_name, create=False)
            except FileNotFoundError:
                raise FileNotFoundError(
                    f"SharedGopStore with store_id={store_id} not found. "
                    f"Call SharedGopStore.create() in the main process first."
                )

        self._entries = np.ndarray(capacity, dtype=ENTRY_DTYPE, buffer=self._meta_shm.buf)

        # --- File-based lock (works across spawn'd processes) ---
        self._lock_path = f"/dev/shm/gs_lock_{store_id}"
        if _create:
            with open(self._lock_path, 'w'):
                pass
        self._lock_fd = os.open(self._lock_path, os.O_RDWR)
        self._local_lock = ForkAwareThreadLock()

        # --- Monotonic tick counter in a tiny SharedMemory ---
        tick_name = f"gs_tick_{store_id}"
        if _create:
            _cleanup_stale_shm(tick_name)
            self._tick_shm = shared_memory.SharedMemory(name=tick_name, create=True, size=8)
            # Keep delayed references distinct across successive stores with the same ID.
            np.frombuffer(self._tick_shm.buf, dtype=np.int64)[:] = time.monotonic_ns()
        else:
            self._tick_shm = shared_memory.SharedMemory(name=tick_name, create=False)
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
        return cls(capacity=capacity, store_id=store_id, _create=True, _key=_CREATION_KEY)

    @classmethod
    def attach(cls, capacity: int, store_id: int = 0) -> 'SharedGopStore':
        """Attach to an existing store.  Call from worker processes.

        Raises:
            FileNotFoundError: If the store has not been created yet.
        """
        return cls(capacity=capacity, store_id=store_id, _create=False, _key=_CREATION_KEY)

    # ------------------------------------------------------------------ #
    #  Locking helpers (flock)
    # ------------------------------------------------------------------ #

    @contextmanager
    def _locked(self):
        # The queue feeder can run on a second thread. A discarded reference can
        # also release a pin while this thread already holds the store's flock.
        held = getattr(_HELD_LOCKS, "store_ids", ())
        if self.store_id in held:
            yield
            return
        with self._local_lock:
            store_ids = (*held, self.store_id)
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX)
            _HELD_LOCKS.store_ids = store_ids
            try:
                yield
            finally:
                _HELD_LOCKS.store_ids = held
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)

    def _next_tick(self) -> int:
        """Increment the shared tick counter while holding the store lock."""
        self._tick_arr[0] += 1
        return int(self._tick_arr[0])

    # ------------------------------------------------------------------ #
    #  Worker API
    # ------------------------------------------------------------------ #

    def lookup(self, video_path: str, frame_id: int, *, pin: bool = True) -> Optional[GopRef]:
        """Look up the GOP containing *frame_id*. Return ``None`` on a miss.

        The returned reference pins its GOP until :meth:`get_batch` or :meth:`read`
        opens it. Pickling transfers the pin to the serialized reference.
        Use ``pin=False`` for a cache-only probe.
        """
        vp_hash = _hash_video_path(video_path)
        with self._locked():
            rows = self._entries
            hits = np.flatnonzero(
                (rows['state'] == _STATE_USED)
                & (rows['video_path_hash'] == vp_hash)
                & (rows['first_frame_id'] <= frame_id)
                & (frame_id < np.add(rows['first_frame_id'], rows['gop_len'], dtype=np.int64))
            )
            if hits.size:
                rows['access_tick'][hits[0]] = self._next_tick()
                ref = self._acquire_ref(int(hits[0]), pin)
                self._hits += 1
                return ref
            self._misses += 1
            return None

    def _acquire_ref(self, slot: int, pin: bool) -> GopRef:
        """Acquire a reference while holding the store lock."""
        entry = self._entries[slot]
        name = entry['shm_name'].decode()
        if pin:
            pins = int(entry['pins'])
            if pins == _MAX_PINS:
                raise OverflowError("SharedGopStore: too many outstanding references to one GOP")
            name = _PinnedName(name)
            name._issuer = self
            entry['pins'] = pins + 1
            name._location = (self.capacity, self.store_id, slot)
        return GopRef(name, int(entry['data_size']), int(entry['first_frame_id']), int(entry['gop_len']))

    def put(
        self, video_path: str, first_frame_id: int, gop_len: int, data: np.ndarray, *, pin: bool = True
    ) -> GopRef:
        """Store a serialized GOP bundle and return a :class:`GopRef`.

        Both a new insertion and a cache hit pin the returned reference by default.
        Raises ``BufferError`` if all slots have outstanding references.
        """
        with self._locked():
            # Skip allocation if an identical GOP — same video, same
            # first_frame_id, same gop_len — is already in the table.
            vp_hash = _hash_video_path(video_path)
            for i in range(self.capacity):
                e = self._entries[i]
                if (
                    e['state'] == _STATE_USED
                    and e['video_path_hash'] == vp_hash
                    and e['first_frame_id'] == first_frame_id
                    and e['gop_len'] == gop_len
                ):
                    e['access_tick'] = self._next_tick()
                    return self._acquire_ref(i, pin)

            slot_idx = self._find_free_or_evict()

            # A fresh name keeps older references and cached handles bound to their own bytes,
            # even when a GOP with the same starting frame is inserted again.
            tick = self._next_tick()
            shm_name = f"{_SHM_PREFIX}_{self.store_id}_{tick}_{first_frame_id}"
            data_size = int(data.nbytes)

            # A previous run with the same store ID may have left this name behind.
            _cleanup_stale_shm(shm_name)

            shm = shared_memory.SharedMemory(name=shm_name, create=True, size=data_size)
            shm.buf[:data_size] = data.tobytes()
            shm.close()  # close handle; shm persists until unlink

            # Write metadata entry
            self._entries[slot_idx]['video_path_hash'] = vp_hash
            self._entries[slot_idx]['first_frame_id'] = first_frame_id
            self._entries[slot_idx]['gop_len'] = gop_len
            self._entries[slot_idx]['data_size'] = data_size
            self._entries[slot_idx]['access_tick'] = tick
            self._entries[slot_idx]['shm_name'] = shm_name.encode()
            self._entries[slot_idx]['pins'] = 0
            self._entries[slot_idx]['state'] = _STATE_USED

            self._puts += 1
            return self._acquire_ref(slot_idx, pin)

    # ------------------------------------------------------------------ #
    #  Main Process API
    # ------------------------------------------------------------------ #

    def read(self, ref: GopRef) -> np.ndarray:
        """Open a single zero-copy uint8 view and release its hand-off pin."""
        return self.get_batch([ref])[0]

    def _open_view(self, ref: GopRef) -> np.ndarray:
        shm_name = str(ref.shm_name)
        if shm_name not in self._local_shm_handles:
            self._local_shm_handles[shm_name] = shared_memory.SharedMemory(name=shm_name, create=False)
        shm = self._local_shm_handles[shm_name]
        return np.frombuffer(shm.buf[: ref.data_size], dtype=np.uint8)

    def get_batch(self, refs: Sequence[GopRef]) -> List[np.ndarray]:
        """Return one zero-copy view per input reference, in order.

        Pass the batch as received. Repeated GOPs share the same mapping.
        If any open fails, no hand-off pins are released and the batch can be retried.
        """
        refs = tuple(refs)
        with self._locked():
            arrays = [self._open_view(ref) for ref in refs]
            self._unlink_orphans()
            with _PIN_STATES:
                # Reusing one reference for several views consumes one pin. Separate
                # worker lookups for the same GOP have equal fields but own one pin each.
                names = (ref.shm_name for ref in refs)
                pending = {
                    id(name): name
                    for name in names
                    if isinstance(name, _PinnedName) and name._owns_pending_pin()
                }
                counts = Counter((name._location[2], str(name)) for name in pending.values())
                pins = self._entries['pins']
                for (slot, name), count in counts.items():
                    if (
                        slot >= self.capacity
                        or self._entries['shm_name'][slot] != name.encode()
                        or pins[slot] < count
                    ):
                        raise ValueError(f"Unbalanced GOP acknowledgement: {name}")
                for (slot, _), count in counts.items():
                    pins[slot] -= count
                for name in pending.values():
                    name._state = "opened"
                    name._issuer = None
            return arrays

    def _adjust_pin(self, name: str, slot: int, change: int) -> None:
        """Account for a discarded reference or another delivery of an issued one."""
        with self._locked():
            entry = self._entries[slot]
            if entry['state'] != _STATE_USED or entry['shm_name'] != name.encode():
                if change > 0:
                    raise FileNotFoundError(name)
                return
            pins = int(entry['pins']) + change
            if not 0 <= pins <= _MAX_PINS:
                raise OverflowError(f"Invalid SharedGopStore pin count for {name}: {pins}")
            entry['pins'] = pins

    # ------------------------------------------------------------------ #
    #  Stats
    # ------------------------------------------------------------------ #

    def get_stats(self) -> dict:
        """Per-process cache statistics."""
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        used_slots = int(np.count_nonzero(self._entries['state'] == _STATE_USED))
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
        """Find a free slot or evict the least recently used unpinned entry under flock."""
        min_tick = np.iinfo(np.int64).max
        min_idx = None
        for i in range(self.capacity):
            if self._entries[i]['state'] == _STATE_FREE:
                return i
            if not self._entries[i]['pins'] and self._entries[i]['access_tick'] < min_tick:
                min_tick = int(self._entries[i]['access_tick'])
                min_idx = i

        if min_idx is None:
            raise BufferError(f"SharedGopStore: all slots are pinned (capacity={self.capacity})")
        # Leave the block linked until the main process cleans up orphans.
        self._entries[min_idx]['state'] = _STATE_FREE
        self._evictions += 1
        return min_idx

    def _unlink_orphans(self):
        """Unlink shm blocks that were evicted but not yet cleaned up.

        Called under flock by get_batch. Scans ``/dev/shm`` for
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

        Call from the main process on shutdown, including after :meth:`close`.
        """
        if self._cleaned_up:
            return  # A new store may now own the same names.
        self.close()

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
                os.unlink(self._lock_path)
            except FileNotFoundError:
                pass
        self._cleaned_up = True

    def close(self) -> None:
        """Close SharedMemory handles **without** unlinking.

        Call from worker processes before exit.
        """
        lock_fd = self._lock_fd
        if lock_fd is None:
            return
        self._lock_fd = None
        for shm in self._local_shm_handles.values():
            _force_close_shm(shm)
        self._local_shm_handles.clear()

        _force_close_shm(self._meta_shm)
        _force_close_shm(self._tick_shm)
        try:
            os.close(lock_fd)
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
