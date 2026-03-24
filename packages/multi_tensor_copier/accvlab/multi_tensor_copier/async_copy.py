# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import numpy as np

from . import _ext


@dataclass
class AsyncCopyHandle:
    """Handle to an in-progress asynchronous copy started by :func:`start_copy`.

    Use :meth:`ready` to poll for completion without blocking, or :meth:`get` to block until
    the result is available. The handle must be kept alive until the copy is consumed or goes
    out of scope; dropping it early triggers a synchronous wait in the destructor to ensure
    staging buffers are not freed while transfers are in flight.
    """

    _h: Any

    def ready(self) -> bool:
        """Return whether the copy has completed.

        Returns:
            ``True`` if the copy has completed, ``False`` otherwise.
        """
        return bool(self._h.ready())

    def get(self) -> list[Any] | tuple[Any, ...] | dict[Any, Any] | torch.Tensor:
        """Block until the copy is done and return the result.

        The returned structure mirrors the input to :func:`start_copy`, with all tensors
        copied to the target device.

        Returns:
            The structure with the contained tensors copied to the target device (and numpy arrays replaced by
            PyTorch tensors).

        Raises:
            RuntimeError: If the copy fails

        """
        return self._h.get()


def start_copy(
    data: list[Any] | tuple[Any, ...] | dict[Any, Any] | torch.Tensor | np.ndarray,
    device: str | torch.device,
    *,
    use_pinned_staging: bool = True,
    pack_cpu_tensors: bool = True,
    min_packed_alignment_bytes: int = 16,
    max_packed_chunk_bytes: int = 32 * 1024 * 1024,
    use_background_thread: bool = True,
) -> AsyncCopyHandle:
    """Asynchronously copy tensors in a nested structure to ``device``.

    Traverses an arbitrarily nested combination of :class:`list`, :class:`tuple`, and :class:`dict`
    containers, copies every :class:`torch.Tensor` and :class:`numpy.ndarray` (automatically converted to
    PyTorch tensors) leaf to ``device``, and returns an :class:`AsyncCopyHandle` whose
    :meth:`~AsyncCopyHandle.get` method yields the copied structure. The output preserves container types and
    passes through non-tensor, non-container leaves (e.g. strings) unchanged.

    The primary optimization target is **CPU → GPU** transfers of many small tensors in non-pinned
    memory. Other copy directions (GPU → CPU, GPU → GPU, CPU → CPU) are supported and benefit from
    some optimizations (e.g. background-thread scheduling for all directions, parallel pinned staging for
    D2H), but are not the main focus.

    .. note::

       The input tensors do not need to all be on the same device, copying tensors from different devices is
       supported. If some tensors are already on the target device, they will be re-used as is.

    .. important::

        Packing of small tensors (see ``pack_cpu_tensors`` parameter below) is a major contribution to the
        overall performance optimization vs. using standard PyTorch ``.to()`` calls on the individual tensors.
        For this optimization to be applied, the input CPU tensors must be contiguous.

    .. warning::

        The caller must not **free** or **modify in-place** any input tensors until the copy has completed
        (i.e. until :meth:`~AsyncCopyHandle.get` returns or :meth:`~AsyncCopyHandle.ready` returns ``True``).
        Because copies are submitted asynchronously — potentially on a background thread — input tensor memory
        may still be read by the GPU after this function returns. Violating this contract leads to undefined
        behavior (silent data corruption, stale reads, or CUDA errors).

    Important:
        Only :class:`list`, :class:`tuple`, and :class:`dict` are recognized as container types.
        Other container-like objects (e.g. custom classes, named tuples) are treated as opaque
        leaves and returned unchanged; any tensors nested inside them will **not** be copied.

    Args:
        data: The structure to copy. May be a single :class:`torch.Tensor` or
            :class:`numpy.ndarray`, or a nested :class:`list`/:class:`tuple`/:class:`dict` objects containing
            tensor/numpy arrays.
        device: Target PyTorch device (e.g. ``"cuda:0"``, ``"cpu"``).
        use_pinned_staging: When ``True``, allocate pinned (page-locked) host buffers as
            intermediate staging for CPU → CUDA and CUDA → CPU transfers. For H2D this enables
            ``non_blocking`` copies; for D2H the pinned buffer **is** the returned output tensor.
            Has no effect on CPU → CPU or GPU → GPU copies.
        pack_cpu_tensors: When ``True``, pack multiple small contiguous CPU tensors (≤ 256 KB
            each, mixed dtypes supported) into one or more staging buffers (each at most
            ``max_packed_chunk_bytes``) and issue one H2D transfer per chunk instead of per tensor.
            Only applies to CPU → CUDA copies.
        min_packed_alignment_bytes: Minimum byte-alignment of each tensor's start offset within
            the packed buffer. The effective alignment for each tensor which participates in the packing is
            ``max(min_packed_alignment_bytes, tensor.element_size())``.
        max_packed_chunk_bytes: Maximum payload size in bytes of each packed staging chunk (tensor
            data plus inter-tensor alignment padding; the actual allocation may be slightly larger to
            satisfy buffer-start alignment).  When the total packed data exceeds this limit, multiple
            packed chunks are allocated and transferred.  Defaults to 32 MB.
        use_background_thread: When ``True``, the copy orchestration (buffer allocation, staging,
            and CUDA copy submission) runs on a C++ background thread (from a shared pool) so that this
            function returns before the copies complete. Note that CPU staging is done parallelly regardless
            of this setting. Benefits all copy directions.

    Returns:
        Handle to the in-progress copy. Call :meth:`~AsyncCopyHandle.get` to block until completion
        and retrieve the result, or :meth:`~AsyncCopyHandle.ready` to poll without blocking.

    Raises:
        RuntimeError: If the copy fails (propagated on :meth:`~AsyncCopyHandle.get`).

    Examples:
        Copy a nested structure of tensors to the GPU::

            data = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6])]
            handle = start_copy(data, "cuda:0")
            # ... do other work ...
            result = handle.get()  # [tensor([1,2,3], device='cuda:0'), ...]

        Convert numpy arrays to CPU tensors (makes use of the fact that numpy arrays can be used as inputs and
        benefits from background-thread scheduling)::

            data = [np.array([1, 2, 3]), np.array([4, 5, 6])]
            handle = start_copy(data, "cpu")
            result = handle.get()  # [tensor([1, 2, 3]), tensor([4, 5, 6])]
    """
    dev = torch.device(device)
    h = _ext.start_copy(
        data,
        str(dev),
        bool(use_pinned_staging),
        bool(use_background_thread),
        bool(pack_cpu_tensors),
        int(min_packed_alignment_bytes),
        int(max_packed_chunk_bytes),
    )
    handle = AsyncCopyHandle(h)
    return handle
