Examples
========

This section contains runnable examples for the tools in ``accvlab.optim_test_tools``. The examples are
kept concise and highlight common usage patterns. The corresponding classes in this package are singletons, 
allowing them to be used in different parts of the code without the need for coordination between the parts.
The stopwatch and NVTX range wrapper examples show this pattern explicitly, while the tensor dumper examples
focus mainly on the actual usage of the tools. Please see the individual examples for more details. These
are listed below in the recommended reading order.

.. toctree::
   :maxdepth: 1

   examples/stopwatch
   examples/nvtx_range_wrapper
   examples/tensor_dumper_comparison
   examples/tensor_dumper_dumping

.. seealso::
    
    The code of the examples can be found in the repository under ``packages/optim_test_tools/examples/``.
