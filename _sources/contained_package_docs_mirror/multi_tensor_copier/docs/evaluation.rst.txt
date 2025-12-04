Evaluation
==========

This page summarizes the performance of ``multi-tensor-copier`` compared to standard PyTorch
``.to()`` calls when copying training meta-data from CPU to GPU.

Setup
-----

The benchmark uses the same data structure as the :doc:`example <example>`: per-sample meta-data from
a multi-camera 3D object detection pipeline, containing variable-size bounding boxes, class IDs,
active flags, depths, and projection matrices for 6 cameras, plus ground truth 3D bounding boxes
with associated attributes. See the :doc:`example <example>` for the full data structure description.

.. list-table:: Benchmark Configuration
   :header-rows: 1

   * - Parameter
     - Value
   * - Batch size
     - 16 samples
   * - Total tensors per batch
     - 528
   * - Total transfer size per batch
     - ~150 KB
   * - Runs
     - 10
   * - Warmup iterations (per run)
     - 100
   * - Measured iterations (per run)
     - 1000

Two baselines are compared against ``multi-tensor-copier``:

- **``.to()`` hardcoded** -- per-tensor ``.to(device)`` calls with the data structure known at
  development time (representative of a manual implementation in a training pipeline).
- **``.to()`` generic** -- a recursive traversal that copies all tensors in an arbitrary nested
  structure using ``.to(device)``, with ``isinstance`` checks and dictionary key iteration at each
  level.

.. note::
    
    The evaluation measures only the copy time itself, without any concurrent work. In practice,
    ``multi_tensor_copier`` allows overlapping the copy with other computation (see the 
    :doc:`example <example>`), which can hide some of the latency. The speedups 
    reported here therefore reflect the improvement in raw copy throughput, not necessarily the full 
    potential benefit in an end-to-end training loop.

Hardware
~~~~~~~~

.. list-table:: System Configuration
   :header-rows: 1

   * - GPU
     - CPU
   * - NVIDIA RTX 5000 Ada Generation
     - AMD Ryzen 9 7950X 16-Core Processor


Results
-------

.. list-table:: Runtime and Speedup (mean +/- std over 10 runs)
   :header-rows: 1

   * - Method
     - Runtime [ms]
     - Speedup
   * - ``.to()`` hardcoded
     - 3.035 +/- 0.006
     - (baseline)
   * - ``.to()`` generic
     - 3.172 +/- 0.006
     - (baseline)
   * - ``multi_tensor_copier``
     - 0.375 +/- 0.008
     - **8.10x** +/- 0.16 vs hardcoded, **8.47x** +/- 0.16 vs generic

The ``multi-tensor-copier`` package achieves a speedup of approximately **8x** over both baselines.
The generic traversal baseline is slightly slower than the hardcoded baseline due to Python overhead from
``isinstance`` checks and dictionary key iteration, but the difference is small compared to the
overall runtime.

.. note::

    In this example the absolute copy time of the baseline (~3 ms with ``.to()``) is moderate. As the
    complexity of the meta-data grows (e.g. with additional variable-length annotations such as lane
    geometry with multiple lanes per sample that cannot be combined into single tensors), the number of
    tensors and thus the overall transfer overhead increases, leading to larger optimization potential.
    Similarly, larger batch sizes multiply the number of tensors proportionally.

.. seealso::

   The evaluation script can be found at ``packages/multi_tensor_copier/example/evaluation.py``.
