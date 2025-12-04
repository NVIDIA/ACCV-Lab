Introduction
============

The ``multi_tensor_copier`` package provides functionality for efficient copying of tensors contained in 
nested structures (lists, tuples, dicts) between devices.
Its primary goal is to optimize **CPU to GPU transfers**, especially for many small tensors in non-pinned 
memory. Other copy directions (GPU to CPU, GPU to GPU, CPU to CPU) are also supported and benefit from 
some of the optimizations, but are not the main focus.

Motivation
----------

Standard PyTorch copy operations (e.g. :meth:`torch.Tensor.to`) have two properties that make them 
inefficient for the scenario of transferring many small tensors to the GPU (e.g. for transferring per-sample
meta-data to the GPU):

1. **Non-pinned memory cannot be copied asynchronously.** PyTorch's ``non_blocking=True`` only yields 
   truly asynchronous host-to-device (H2D) transfers when the source tensor resides in pinned (page-locked) 
   memory. In some workloads -- e.g. when tensors originate from a :class:`torch.utils.data.DataLoader` 
   with ``pin_memory=False`` or when they are obtained e.g. by reading pickled numpy arrays -- this 
   precondition is not met, and every transfer blocks the calling thread.

2. **Per-tensor overhead dominates for small tensors.** Each call to ``.to()`` incurs overhead. For small 
   tensors (e.g. variable-length annotations in object detection), this overhead can exceed the actual 
   transfer time, so that if many small tensors are present, this can lead to a considerable overhead and 
   dominate the actual transfer time.

.. note::

    Apart from improving copying efficiency, the package also makes copying multiple tensors more convenient 
    by automatically traversing the input structure and copying all contained tensors to the target device.

Features
--------

The package addresses the efficiency issues through the following optimizations, all of which are 
configurable (i.e. can be enabled or disabled):

**Automatic packing of small tensors** (``pack_cpu_tensors``, default: enabled)
  Multiple small contiguous CPU tensors (up to 256 KB each, mixed dtypes supported) are **automatically** 
  packed into one or more fixed-size byte buffers and transferred with one H2D copy per buffer. On the
  GPU side, per-tensor views into the packed allocations are created with configurable alignment
  (``min_packed_alignment_bytes``) enforced for the individual outputs. This optimization is **only
  applicable to CPU to GPU transfers**.

  .. important::

    This feature is a major contribution to the overall performance optimization vs. using standard 
    PyTorch ``.to()`` calls on the individual tensors. For this optimization to be applied, the input CPU 
    tensors must be contiguous.

**Parallel pinned memory staging** (``use_pinned_staging``, default: enabled)
  For CPU to GPU transfers, input tensors are first copied into pinned host buffers (in parallel) so that the 
  subsequent H2D transfer can use ``non_blocking=True``. For GPU to CPU transfers, 
  output is written directly into a pinned host buffer via an asynchronous D2H copy on a CUDA stream, and the 
  pinned tensor is returned as the result.

**Background-thread scheduling** (``use_background_thread``, default: enabled)
  The copy orchestration (buffer allocation, staging, and CUDA copy submission) runs on a C++ background 
  thread rather than the calling Python thread. 
  :func:`~accvlab.multi_tensor_copier.start_copy` returns a handle before the copies complete; 
  the caller can do other work and retrieve results via 
  :meth:`~accvlab.multi_tensor_copier.AsyncCopyHandle.get`. Note that parallel CPU staging is used regardless 
  of this setting. The background-thread scheduling benefits all copy directions, including CPU to CPU.

**Nested structure traversal**
  Input may be an arbitrarily nested combination of :class:`list`, :class:`tuple`, and :class:`dict` 
  containers with :class:`torch.Tensor` or :class:`numpy.ndarray` leaves. The output preserves the original 
  structure. Non-tensor, non-container leaves (e.g. strings) are passed through unchanged. Numpy 
  arrays are converted to PyTorch tensors during traversal. The automatic handling of nested structures 
  **greatly simplifies copying of nested structures of tensors** while also allowing for **automatic packing 
  of small tensors** (see above) without the need for manual bookkeeping.

Integration
-----------

The copy can be started wherever the data is needed -- not e.g. only directly after it is obtained from 
a PyTorch DataLoader. For example, if the GPU-resident data is only required for loss computation, 
:func:`~accvlab.multi_tensor_copier.start_copy` can be called at the beginning of the loss 
computation step, ideally with some work performed in the meantime to overlap with the asynchronous 
copy. This means the package can be integrated into existing training loops with only local 
modifications, and can also be used with data originating from other sources than a DataLoader.

.. note::

   At the time of the :func:`~accvlab.multi_tensor_copier.start_copy` call, the active PyTorch
   streams on all involved CUDA devices are captured.  All copy work is then enqueued on, or
   synchronized with, these captured streams so that transfers are correctly ordered with respect
   to preceding GPU operations — no manual synchronization is required.  Non-default stream
   contexts (e.g. :func:`torch.cuda.stream`) are respected.

Supported Copy Directions
-------------------------

The table below summarizes which optimizations apply to each copy direction:

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 15 15

   * - Optimization
     - CPU → GPU
     - GPU → CPU
     - GPU → GPU
     - CPU → CPU
   * - Pinned staging
     - ✓
     - ✓
     -
     -
   * - Packing
     - ✓
     -
     -
     -
   * - Background thread
     - ✓
     - ✓
     - ✓
     - ✓

.. seealso::

   Refer to the :doc:`api` for the full parameter reference and the :doc:`example` for a usage example.
