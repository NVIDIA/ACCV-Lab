Runtime Performance Evaluation and NVTX Profiling
=================================================

This page explains how to use the ``optim_test_tools`` to measure runtime and add NVTX ranges for profiling.
We focus on the most important measurements for the training iteration and loss/optimizer computation.

We rely on two singletons from ``accvlab.optim_test_tools``:

- :class:`~accvlab.optim_test_tools.Stopwatch`: lightweight runtime measurements with optional CPU usage 
  sampling
- :class:`~accvlab.optim_test_tools.NVTXRangeWrapper`: NVTX push/pop helpers with additional functionality to 
  annotate critical sections for Nsight Systems (& Nsight Compute).

Enable & Configure in the Training Script
-----------------------------------------

The configuration is centralized in the main training script (``demos/streampetr/tools/train.py``). The 
runtime measurement and NVTX range wrapper are enabled based on the ``runtime_meas_config`` section in the 
config. The use of the configuration file enables us to use different configurations for different types of 
runs, e.g.:

  - Training (actual usage): All disabled
  - Evaluation: Only Stopwatch enabled
  - Profiling: All enabled, nvtx range mismatches not checked (to avoid overhead)
  - (Debugging: All enabled, nvtx range mismatches checked (to help debugging range mismatches))

Note that the ``Stopwatch`` and ``NVTXRangeWrapper`` are enabled & configured centrally here, and can be used 
throughout the codebase relying on this configuration.

.. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
   :language: python
   :caption: demos/streampetr/tools/train.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Runtime Measurement & Profiling Configuration =====
   :end-at: # ===== End of Runtime Measurement & Profiling Configuration =====

Measuring the Training Iteration
--------------------------------

The core iteration timings are implemented in the custom runner
(``IterBasedRunnerWithTime``) to capture the outer training loop and the inner training step,
including optional CPU usage measurement. Note that this means that this runner needs to be used instead
of the original implementation (from the ``mmcv`` package) to enable the runtime measurements. The selection
can be done by setting ``runner.type`` in the config file to ``IterBasedRunnerWithTime``.

Outer loop timing with CPU usage sampling in ``run()``:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/adjusted_impl/iter_based_runner_with_time.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/adjusted_impl/iter_based_runner_with_time.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Training loop & timing =====
   :end-at: # ===== End of Training loop & timing =====

Training step timing in ``train()``:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/adjusted_impl/iter_based_runner_with_time.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/adjusted_impl/iter_based_runner_with_time.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Training function & timing =====
   :end-at: # ===== End of Training function & timing =====

Measuring Optimizer/Loss Step and NVTX Ranges
---------------------------------------------

The optimizer step is instrumented with both ``Stopwatch`` and ``NVTXRangeWrapper`` to
annotate and time the critical loss backward/step section. Note that similar to the runner, this 
implementation needs to be used instead of the original implementation by setting up the ``optimizer_config`` 
parameter in the config file with ``type='Fp16OptimizerHookWithTime'``.

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/adjusted_impl/optimizer_with_time.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/adjusted_impl/optimizer_with_time.py
   :linenos:
   :lineno-match:
   :start-at: # ===== Optimizer Step Timing and NVTX Ranges =====
   :end-at: # ===== End of Optimizer Step Timing and NVTX Ranges =====

NVTX ranges also identify operations in the :doc:`matching implementation <../batching_helpers/batching_matching>`
and the loss computations described in the :doc:`Batching Helpers demo <../batching_helpers/index>`.

NVTX Range Mismatch Checking
----------------------------

To enable NVTX range mismatch checking, set both ``enable`` and ``keep_track_of_range_order`` to ``True``
in ``runtime_meas_config.nvtx_range_wrapper``. When checking is enabled, calls to
``range_pop(expected_name)`` verify that the range being closed has the expected name. Disable
``keep_track_of_range_order`` during performance profiling to avoid the checking overhead.
For example, ``HungarianAssigner2DBatched`` checks its outermost range when the assignment finishes:

.. note-literalinclude:: ../../../../demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :language: python
   :caption: demos/streampetr/projects/mmdet3d_plugin/core/bbox/assigners/hungarian_assigner_2d_batched.py
   :linenos:
   :lineno-match:
   :start-at: #@NOTE: Types of range pops
   :end-at: nvtx_wrp.range_pop("HungarianAssigner2DBatched.assign")

Viewing Results
---------------

- ``Stopwatch`` results are printed from the main script at the end of training:

  .. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
     :language: python
     :caption: demos/streampetr/tools/train.py
     :linenos:
     :lineno-match:
     :start-at: # ===== Stopwatch: Print eval times =====
     :end-at: # ===== End of Stopwatch: Print eval times =====

- For NVTX, launch your run under Nsight Systems/Compute. The training script additionally uses 
  ``torch.cuda.cudart().cudaProfilerStart()`` and ``torch.cuda.cudart().cudaProfilerStop()`` to start and stop
  the profiling session. Note that this is done only if the use of NVTX is enabled in the config (and 
  therefore, the NVTX range wrapper is enabled).

  .. note-literalinclude:: ../../../../demos/streampetr/tools/train.py
     :language: python
     :caption: demos/streampetr/tools/train.py
     :linenos:
     :lineno-match:
     :start-at: # ===== Start profiling (if running with nsys) =====
     :end-at: # ===== End of Stop profiling (if running with nsys) =====
