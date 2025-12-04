NVTX Ranges in Numba Code Example
=================================

This example shows how to annotate Numba JIT-compiled code with NVTX ranges using the 
``accvlab.optim_test_tools.numba_nvtx`` module.
Numba-compiled functions run as native code; this module allows you to push and pop NVTX
ranges inside such functions.

.. seealso::
    
    The code of this example can be found in the repository under 
    ``packages/optim_test_tools/examples/numba_nvtx_example.py``.

.. important::
  
  The functionality described here is specifically aimed at adding NVTX ranges to Numba-compiled code,
  where alternative solutions, e.g. the Python ``nvtx`` module or ``torch.cuda.nvtx``, cannot be used.
  While ``accvlab.optim_test_tools.numba_nvtx`` also works for plain Python code, it is not intended for 
  that use-case and does not provide benefits over using available alternatives in this case.
  
  For profiling Python code, please also see the :doc:`nvtx_range_wrapper` example.
  

Overview
--------

- Import ``accvlab.optim_test_tools.numba_nvtx`` (e.g. as ``nvtx``).
- Register range names with ``register_string("...")`` to obtain integer handles; this must be done outside
  the JIT function, before compilation.
- Inside an ``@numba.njit`` function, call ``nvtx.range_push(handle)`` and ``nvtx.range_pop()`` around
  the region to profile.

Example
-------

Please see the notes in the code for more details.

.. note-literalinclude:: ../../examples/numba_nvtx_example.py
   :language: python
   :caption: packages/optim_test_tools/examples/numba_nvtx_example.py
   :linenos:
   :lineno-match:
