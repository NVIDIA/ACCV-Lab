Introduction
============

This is the documentation for the **example_skbuild_package** package, which demonstrates the supported
CMake-based ACCV-Lab package pattern using ``scikit-build``.

.. note::

    For PyTorch extensions built directly with ``CppExtension`` and ``CUDAExtension`` in ``setup.py``, see the
    :doc:`../../example_package/docs/index` package.

Package Overview
----------------

The example SKBuild package provides:

* **CMake integration**: Native C++/CUDA sources are configured through ``ext_impl/CMakeLists.txt``.
* **Python packaging**: ``setup.py`` uses ``skbuild.setup`` with ``cmake_source_dir`` and
  ``cmake_install_dir`` so the built extension is installed into the package.
* **Python wrappers**: The package imports the built extension module and exposes Python-facing functions.
* **Native tests**: Optional C++/CUDA tests are defined as CMake/CTest targets under ``ext_impl/utest``.

Basic Usage
-----------

The package exposes Python wrappers around the CMake-built extension:

.. code-block:: python

    import torch
    import accvlab.example_skbuild_package as example_pkg

    if torch.cuda.is_available():
        a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
        result = example_pkg.vector_add(a, b)

Native Test Example
-------------------

The native-test example uses the GoogleTest submodule under ``ext_impl/external/googletest``. After
initializing the repository submodules, run it from the repository root with:

.. code-block:: bash

    git submodule update --init --recursive
    ./scripts/run_cpp_unit_tests.sh \
        --config packages/example_skbuild_package/cpp_unit_tests.yaml \
        example_skbuild_package

For details about configuring native-test packages, including selecting the CUDA architecture strategy, see
:ref:`native-test-cuda-architecture-strategy`.

Use this package as the reference when creating a CMake-based namespace package. For documentation layout
details that apply to all namespace packages, see :doc:`../../../guides/DOCUMENTATION_SETUP_GUIDE`.
