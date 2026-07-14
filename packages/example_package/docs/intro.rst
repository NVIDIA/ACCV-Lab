Introduction
============

This is the documentation for the **example_package** package, which demonstrates how to create ACCV-Lab
namespace packages with PyTorch C++ and CUDA extensions built through ``CppExtension`` and ``CUDAExtension``.

.. note::

    CMake-based native implementations are also supported. For the CMake/``scikit-build`` package pattern,
    see the :doc:`../../example_skbuild_package/docs/index` package.

.. note::

    This documentation is also used as an example for how to create documentation for a namespace package.
    Please refer to the :doc:`../../../guides/DOCUMENTATION_SETUP_GUIDE` for more details.

Package Overview
----------------

The example package provides:

* **PyTorch C++ Extensions**: Vector and matrix operations implemented with ``CppExtension``
* **PyTorch CUDA Extensions**: GPU-accelerated vector operations implemented with ``CUDAExtension``
* **Python Wrappers**: Easy-to-use Python functions that wrap the extensions
* **Build System**: Complete setuptools setup for building PyTorch C++/CUDA extensions

Key Features
------------

* Vector sum and matrix transpose using C++ extensions
* Element-wise vector multiplication and reduction using CUDA
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

Further Examples
----------------

For examples, see :doc:`examples`. That page uses ``note-literalinclude`` to include the example code,
highlights notes in the code (comment blocks starting with ``# @NOTE``), and explains how
``docu_referenced_dirs.txt`` mirrors the package-level ``examples/`` directory so the combined documentation
site can be built while keeping relative references valid in the rendered documentation.

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
