Introduction
============

This is the documentation for the **example_package** package, which demonstrates how to create ACCV-Lab 
namespace packages with C++ and CUDA extensions.

.. note::

    This documentation is also used as an example for how to create documentation for a namespace package.
    Please refer to the :doc:`../../../guides/DOCUMENTATION_SETUP_GUIDE` for more details.
    details.

Package Overview
----------------

The example package provides:

* **C++ Extensions**: Vector and matrix operations implemented in C++
* **CUDA Extensions**: GPU-accelerated vector operations
* **Python Wrappers**: Easy-to-use Python functions that wrap the extensions
* **Build System**: Complete setup for building C++/CUDA extensions

Key Features
------------

* Vector sum and matrix transpose using C++ extensions
* Element-wise vector multiplication and reduction using CUDA
* Automatic build information display
* Simple hello function for testing

Basic Usage
-----------

Here's a quick example of how to use the package:

.. code-block:: python

    import torch
    import accvlab.example_package as example_pkg
    
    # C++ extension example
    vector = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    vector_sum = example_pkg.cpp_vector_sum(vector)
    print(f"Vector sum: {vector_sum}")
    
    # CUDA extension example (if CUDA is available)
    if torch.cuda.is_available():
        a = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
        b = torch.tensor([2.0, 3.0, 4.0, 5.0], device='cuda')
        multiplied = example_pkg.cuda_vector_multiply(a, b)
        print(f"Element-wise multiplication: {multiplied}")

Examples
--------

For examples, see :doc:`examples`. The example makes use of ``note-literalinclude`` to include the 
example code in the documentation and highlight notes in the code (comment blocks starting with ``# @NOTE``).

Generated Documentation Assets
------------------------------

This package also demonstrates package-local documentation asset generation. The docs build reads committed
data from ``evaluation_results/simple_plot.csv`` and writes the generated plot to
``docs/_generated/simple_plot.png``. The source documentation remains static and references the generated
image using a normal relative path.

.. figure:: _generated/simple_plot.png
   :alt: Simple generated plot from committed CSV data
   :align: center
   :width: 70%

   Example plot generated from committed CSV data during documentation generation.

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples