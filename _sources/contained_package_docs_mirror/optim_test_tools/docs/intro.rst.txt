Introduction
============

Design
------

This package contains helpers which can be used for evaluating runtime optimizations as well as for 
debugging. 

Most of the helper functionality are singleton classes which are designed to

  - have minimal overhead if no evaluation is performed (i.e. the helper is not enabled)
  - have a convenient way to be enabled & configured if used (e.g. from main training script)
  - being able to be used in different parts of the implementation without the need for coordination between 
    the parts. This simplifies their use as e.g.,
    
      - it allows to configure the helpers from the main training script, without the need to propagate the 
        configuration to the parts of the code which use the helpers. This includes
  
        - Detail configurations such as e.g. whether to perform CUDA synchronization on starting/stopping a 
          measurement
        - Enabling/disabling the helpers from the main training script
  
      - it allows to e.g. start and end measurements in different parts of the code without the need to 
        coordinate the data exchange (such as time stamps) between the parts. All measurements can be combined 
        centrally and a summary can be obtained easily.

Functionalities
---------------

This package contains the following helper classes (singletons):
  - :class:`~accvlab.optim_test_tools.Stopwatch`: Can be used to conveniently measure runtime of different 
    parts of the code (including defining a warm-up phase, keeping track of the 
    iterations, averaging measurements etc.). Additionally, it supports measuring average CPU usage.
  - :class:`~accvlab.optim_test_tools.NVTXRangeWrapper`: Wrapper for NVTX ranges which (optionally) 
    automatically performs CUDA synchronizations on entering/exiting a range & has additional functionality to 
    debug range push/pop mismatches.
  - :class:`~accvlab.optim_test_tools.TensorDumper`: Can be used to dump tensors as well as gradients for 
    debugging purposes. It supports different dump formats (including images, binary files, JSON) and 
    additional functionality to dump different types of data, potentially processing the data first to obtain 
    a more easy-to-interpret representation. Please see the API documentation and the samples usage for more 
    details. Additionally, it supports comparing data from previous dumps to detect mismatches. This can be 
    used to detect e.g. differences due to bugs introduced in the code while working on optimization (which 
    should not change the results).

Additionally, this package provides the :mod:`~accvlab.optim_test_tools.numba_nvtx` module for adding NVTX 
ranges to Numba JIT-compiled (``@njit``) code. Numba-compiled functions run as native code, so Python-level 
NVTX tools (such as the ``nvtx`` package or ``torch.cuda.nvtx``) cannot be called from within them. The 
:mod:`~accvlab.optim_test_tools.numba_nvtx` module solves this by providing 
:func:`~accvlab.optim_test_tools.numba_nvtx.range_push` and 
:func:`~accvlab.optim_test_tools.numba_nvtx.range_pop` functions that can be used 
directly inside ``@njit`` functions. It works by registering C-level NVTX symbols using Numba's 
:func:`~numba.core.extending.overload` mechanism and emitting the corresponding calls in the compiled code.

.. seealso::

    Please see :doc:`api` for more detailed information about the classes and :doc:`examples` on how to use 
    them.


.. important::

   This package implements the helper classes as singletons to ensure that they can be enabled from any part 
   of the code, and also to e.g. ensure correct runtime measurement (consistent counting of iterations, allow 
   to start & end runtime measurement in different parts of the code etc.). In order for the singleton to work 
   correctly, it is important to use it as a package and not to only copy the code into a project and e.g. use 
   relative imports. The latter may lead to multiple instances of the helper classes to be created (as the 
   import manager may handle the imports as independent packages), leading to errors/misconfigurations.


