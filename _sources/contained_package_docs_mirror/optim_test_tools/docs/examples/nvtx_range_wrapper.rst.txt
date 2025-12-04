NVTX Range Wrapper Example
==========================

The NVTX range wrapper helps annotate code with NVTX ranges to profile code with Nsight Systems.
It follows the same singleton pattern as the stopwatch (see :doc:`stopwatch`): enable once, then push/pop 
ranges from any part of your code.

Compared to using nvtx ranges directly, it offers the following advantages:
  - It is possible to centrally configure whether CUDA synchronization is performed when pushing/popping a 
    range. While the ranges themselves have already minimal overhead if there is no profiling, the
    synchronization adds an additional overhead, and otherwise needs to be handled manually.
  - It is possible to check for range push/pop mismatches (only use for debugging purposes and leave disabled 
    otherwise, as it has an overhead). It can be very handy to check for unexpected range pops for some key
    ranges, as manual search for mismatches can be tedious for large codebases.

.. seealso::
    
    The code of this example can be found in the repository under 
    ``packages/optim_test_tools/examples/nvtx_range_wrapper_example.py``.

Overview
--------

- Singleton: ``NVTXRangeWrapper()`` returns the global instance.
- Call ``enable(...)`` once to activate; otherwise calls are no-ops with minimal overhead.
- Push/pop named ranges; optionally verify that the popped name matches expectations.

Example
-------

Please see the notes in the code for more details.

.. note-literalinclude:: ../../examples/nvtx_range_wrapper_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/nvtx_range_wrapper_example.py
   :linenos:
   :lineno-match:

Related examples
----------------

- See the :doc:`stopwatch` example for similar usage with timings and warm-up handling.
- The tensor dumper examples (:doc:`tensor_dumper_comparison` and :doc:`tensor_dumper_dumping`) demonstrate
  another singleton tool for capturing tensors and gradients.


