# Development Guide

This guide covers the development aspects of ACCV-Lab: how the project is structured, how to add new namespace 
packages, and how to work with the build system.

> **ℹ️ Note**: For installation instructions, see the [Installation Guide](INSTALLATION_GUIDE.md).

## How It Works

The project uses a shared configuration system where namespace packages are explicitly defined in 
`namespace_packages_config.py` (please also note the comments in the code snippet for more details):

```python
# List of all ACCV-Lab namespace packages
# Each namespace package should:
# - Be a directory under the packages/ subdirectory
# - Have a pyproject.toml and setup.py file for building
# - Be added to this list to be included in builds and documentation
# Please note that:
# - Packages that are not listed here will be ignored when installing all packages, building the 
#   documentation, running the tests, etc.
# - The order in which the packages are listed here is the order in which they will be installed, and in which
#   they will appear in the documentation.
NAMESPACE_PACKAGES = [
    # The commented out packages below this line are examples (see the development guide):
    # 'accvlab.example_package',
    # 'accvlab.example_skbuild_package',
    'accvlab.on_demand_video_decoder',
    'accvlab.batching_helpers',
    'accvlab.multi_tensor_copier',
    'accvlab.dali_pipeline_framework',
    'accvlab.lane_helpers',
    'accvlab.draw_heatmap',
    'accvlab.optim_test_tools',
    # Add new namespace packages in the same way as above
]
```

Each namespace package is self-contained with its own `setup.py` and `pyproject.toml` files and can be built 
and installed independently. The build configuration is handled directly in each package's `setup.py` using 
shared build utilities from the `accvlab_build_config` package (located in the `build_config/` directory
and installed as part of the ACCV-Lab installation) as well as the `pyproject.toml` file for the package; 
see the [Installation Guide](INSTALLATION_GUIDE.md) for more details on how to build and the 
[Build Configuration](#the-build-configuration-system) section for more details on how to use the shared build 
utilities inside the package's `setup.py` file.

## Package Structure Overview & Adding new Packages

There are two example projects which showcase supported package patterns:
- `packages/example_package`: Shows a setuptools package with PyTorch C++/CUDA extensions built using
  `CppExtension` and `CUDAExtension`. It also includes a package-local documentation asset hook that
  generates a simple plot from committed CSV data under `evaluation_results/` during the docs build.
- `packages/example_skbuild_package`: Shows the supported CMake-based package pattern using `scikit-build`.
  See [CMake-Based Packages with SKBuild](#cmake-based-packages-with-skbuild) for details.

First, we will focus on `example_package` and explain how to set up a setuptools/PyTorch-extension package
from scratch. For CMake-based packages, use the SKBuild pattern described later in this guide.

### Overview

To add a new namespace package (e.g., `example_package`), you need to create:

| Component | Directory | Purpose |
|-----------|-----------|---------|
| **Implementation** | `packages/example_package/accvlab/example_package/` | Python package with your actual code |
| **Native extension sources** | `packages/example_package/accvlab/example_package/csrc/` and `packages/example_package/accvlab/example_package/include/` | C++/CUDA sources and headers used by PyTorch `CppExtension`/`CUDAExtension` packages |
| **Documentation** | `packages/example_package/docs/` | API reference and user guides (template will be auto-generated) |
| **Tests** | `packages/example_package/tests/` | Python-based unit tests (will be automatically picked up by the test runner) |
| **Setup** | `packages/example_package/setup.py` | Package build configuration |
| **Project Config** | `packages/example_package/pyproject.toml` | Modern Python project configuration and authoritative dependency definition |
| **Documentation include list (optional)** | `packages/example_package/docu_referenced_dirs.txt` | List additional directories referenced by the docs (besides `docs/`). See [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md) for more details.|
| **Documentation asset hook (optional)** | `packages/example_package/docs/_on_doc_generation.py` | Generate package-owned docs assets such as plots from committed evaluation data. See [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md#package-local-generated-assets). |
| **Evaluation results (optional)** | `packages/example_package/evaluation_results/` | Package-owned committed inputs for generating docs assets, such as data to plot. |

> **ℹ️ Note**: Apart from the above, further folders/files can be included (and made use of manually or added to the 
> documentation) if needed. A typical use case is to include e.g. an `examples` directory which is:
> - Referenced by the documentation (added to `docu_referenced_dirs.txt`) and from which code snippets are 
>   included in the documentation.
> - Can be used directly by navigating to the `examples/` directory and running the contained code.

### Structure

The following diagram shows the relevant project structure containing the folders which correspond to the 
`example_package` namespace package:

```
accvlab/
├── packages/                         # Namespace packages directory
│   ├── optim_test_tools/...
│   ├── batching_helpers/...
│   └── example_package/              # ← New namespace package
│       ├── accvlab/                  # ← Namespace root
│       │   └── example_package/      # ← Implementation for "example_package" package
│       │       ├── __init__.py
│       │       ├── csrc/             # ← C++/CUDA sources
│       │       └── include/          # ← Headers
│       ├── tests/                    # ← Tests for "example_package" package
│       ├── examples/                 # ← Optional runnable examples referenced by the docs
│       ├── evaluation_results/       # ← Optional committed inputs for generated docs assets
│       ├── docs/                     # ← Documentation for "example_package" package
│       │   ├── _on_doc_generation.py # ← Optional docs asset hook
│       │   └── ...
│       ├── setup.py                  # ← Package build configuration
│       ├── pyproject.toml            # ← Project configuration (including dependencies)
│       └── docu_referenced_dirs.txt  # ← Optional: list additional directories referenced by the docs (besides `docs/`)
├── build_config/                     # Shared build utilities
├── docs/                             # Main documentation
└── namespace_packages_config.py      # ← Namespace package needs to be listed here
```

Note that inside the package, there is the directory structure `accvlab/example_package`. This is where the 
Python implementation of the namespace package is located, and it is named according to the package name (in 
this case `accvlab.example_package`). Other packages are structured in the same way, with `example_package` 
replaced by the name of the respective package. 

### Adding a new Package: Step-by-Step Process

To add a new namespace package (e.g., `example_package`):

#### 1. Create the Directory Structure

```bash
# Create the namespace package directory
mkdir -p packages/example_package
mkdir -p packages/example_package/accvlab/example_package           # For the implementation
mkdir -p packages/example_package/accvlab/example_package/csrc      # For C++/CUDA sources
mkdir -p packages/example_package/accvlab/example_package/include   # For headers

# Documentation include list file (optional - see Documentation Setup Guide)
touch packages/example_package/docu_referenced_dirs.txt

# Examples directory (optional - see Documentation Setup Guide)
mkdir -p packages/example_package/examples                   

# Tests directory
mkdir -p packages/example_package/tests

# Documentation directory created automatically by docs system as:
#   packages/example_package/docs/
```

#### 2. Create the Implementation

Add your implementation in the `packages/example_package/accvlab/example_package` folder (including the 
package's Python code and native extensions using the `CppExtension` and `CUDAExtension` approach if 
applicable).

For CMake-based native implementations, use an SKBuild package instead. The supported pattern is documented in
[CMake-Based Packages with SKBuild](#cmake-based-packages-with-skbuild) and demonstrated by
`packages/example_skbuild_package`.

#### 3. Create `setup.py`

Create the setup configuration file that defines how to compile PyTorch C++/CUDA extensions. See
`packages/example_package/setup.py` for a complete working example. Do not hardcode `version=` in `setup()`;
ACCV-Lab packages derive the package version from SCM metadata, with `pyproject.toml` configuring
`setuptools-scm` (see below).

The setuptools/PyTorch-extension pattern imports the shared build helpers, loads the build configuration, and
passes the generated compile flags into `CppExtension` and `CUDAExtension` definitions.

#### 4. Create `pyproject.toml`

Create the Python project configuration. This file also defines the package's runtime and optional dependencies.
See `packages/example_package/pyproject.toml` for a complete working example.

Add your package metadata and dependencies to `pyproject.toml`. For example, the
`packages/example_package/pyproject.toml` file contains:

```toml
[build-system]
requires = [
    "setuptools>=64",
    "wheel",
    "torch>=2.0.0",
    "pybind11>=2.10.0",
    "setuptools-scm>=8",
    "accvlab-build-config @ file:../../build_config",
]
build-backend = "setuptools.build_meta"

[project]
name = "accvlab.example_package"
dynamic = ["version"]
description = "ACCV-Lab Example Package"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.22.2",
]

[project.optional-dependencies]
optional = [
    "matplotlib",
    "pytest",
]

[tool.setuptools.packages.find]
where = ["."]
include = ["accvlab.example_package*"]

[tool.setuptools_scm]
version_scheme = "no-guess-dev"
fallback_version = "0.0.0"
root = "../.."
```

Use this pattern for your own namespace package, adapting the dependency names as needed.

Use `[project.optional-dependencies].optional` for dependencies needed by tests, examples, or package-local
documentation asset hooks, but not by the core package at runtime. For example, if a docs hook generates plots
from committed data, put the plotting library in the package's optional dependencies rather than in the base
`[project].dependencies`.

> **ℹ️ Note**: The `accvlab-build-config @ file:../../build_config` build dependency is intentionally a
> local path reference. From a package under `packages/<package_name>/`, it resolves to the repository's `build_config/` package 
> so isolated pip builds use the local helper package. See
> [Installing with Build Isolation](INSTALLATION_GUIDE.md#installing-with-build-isolation) for the related precautions.

> **ℹ️ Note**: ACCV-Lab packages use `setuptools-scm` for versioning. The package version is derived from SCM
> metadata, while `[project]` marks it as dynamic and `[tool.setuptools_scm]` configures how it is resolved,
> instead of hardcoding it in `setup.py` or `[project].version`. The top-level package `__init__.py` should also
> expose `__version__` using distribution metadata rather than a hardcoded string. For example:
> ```python
> from importlib.metadata import PackageNotFoundError, version
>
> try:
>     __version__ = version("accvlab.example_package")
> except PackageNotFoundError:
>     __version__ = "0.0.0"
> ```
> This keeps `import accvlab.<package_name>; print(accvlab.<package_name>.__version__)` working for installed
> packages. If the module defines `__all__`, include `"__version__"` there as well.

#### 5. Add to NAMESPACE_PACKAGES List

This ensures that your package is included in various places, e.g.
- in the installation by the `package_manager.sh` script
- in the documentation
- in the code formatting by the `format.sh` script

```python
# In namespace_packages_config.py
NAMESPACE_PACKAGES = [
    'accvlab.on_demand_video_decoder',
    'accvlab.batching_helpers',
    'accvlab.multi_tensor_copier',
    'accvlab.dali_pipeline_framework',
    'accvlab.lane_helpers',
    'accvlab.draw_heatmap',
    'accvlab.optim_test_tools',
    'accvlab.example_package',  # Add your new namespace package here
]
```

#### 6. Create Tests

Create test files for your namespace package. See `packages/example_package/tests/` for examples.

> **ℹ️ Note**: This covers Python-based tests. For native C++/CUDA tests in SKBuild-based packages, see
> [Native Tests](#native-tests).

#### 7. Set Up Documentation

The documentation system will automatically create the structure when you add the namespace package and build
the documentation (see the [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md) for more details).

Alternatively, you can generate the documentation structure manually using the following commands:

```bash
# Generate documentation structure for the new namespace package
cd docs # ← Assuming you are in the main docs directory (not the package's docs directory)
python3 generate_new_namespace_package_docs.py
python3 update_docs_index.py
```

This creates:
- `packages/example_package/docs/index.rst` - Table of contents (needs to be present; can be edited if needed)
- `packages/example_package/docs/intro.rst` - Manual introduction (customize this!)
- `packages/example_package/docs/api.rst` - Auto-generated API reference (can be edited if needed)

**Customize the introduction:**
Edit `packages/example_package/docs/intro.rst` to add
- Package overview and purpose
- Basic usage examples
- Performance characteristics
- Integration notes

Note that you are mostly free to modify, add, or remove the files & their contents of the documentation.
However, `index.rst` serves as the "entry point" for the documentation and needs to be present.

Most of the contained packages extend this basic structure considerably to provide more detailed
documentation. Please see the [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md) for more details on
the documentation system and how to set it up.

If your package needs generated docs assets, add `packages/<package_name>/docs/_on_doc_generation.py`. The
documentation build creates `packages/<package_name>/docs/_generated/`, keeps it untracked, and passes that
directory to the hook. Keep user-facing `.rst`/`.md` files static and reference generated assets with relative
paths such as `_generated/<asset_name>.png`. The hook should generate those assets from committed inputs and
fail clearly if required inputs are missing. Store committed plot or evaluation inputs outside the package
`docs/` folder, for example under `packages/<package_name>/evaluation_results/`, so Sphinx does not discover
data tables as standalone documentation pages.

> **⚠️ Important**: Documentation asset hooks must not run evaluations, benchmarks, or other measurement
> workflows. They should only regenerate documentation assets, such as plots, from data that is already
> available in the repository.

#### 8. Test Your Package

```bash
# 1. Install the package with your new namespace package
cd packages/example_package
# IMPORTANT: Do not use  editable installation (`-e`) for SKBuild-based packages (e.g.
# `on_demand_video_decoder` or `dali_pipeline_framework`), as it would lead to missing binaries and import
# errors.
# It is ok to use `-e` for other packages. However, keep in mind that for any changes in C++ code to take
# effect, the package needs to be re-installed regardless of whether it is installed in editable mode or not.
pip install . --no-build-isolation

# 2. Run tests
pytest tests/ -v
```

#### 9. Build the Documentation

> **⚠️ Important**: Ensure all configured namespace packages are installed before building the docs.
> For detailed instructions (commands and development targets), see the
> [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md#building-documentation-locally).

### Summary Checklist

When adding a new namespace package, ensure you have:

- [ ] **Implementation**: `packages/<package_name>/accvlab/<package_name>/` with `__init__.py` and source
  code
- [ ] **Setup**: `packages/<package_name>/setup.py` with build configuration
- [ ] **Project Config**: `packages/<package_name>/pyproject.toml` with project metadata
- [ ] **Configuration**: Added to `namespace_packages_config.py`
- [ ] **Tests**: `packages/<package_name>/tests/` with test files
- [ ] **Documentation**: Generated with docs scripts and customized intro
- [ ] **Documentation include list (optional)**: `docu_referenced_dirs.txt` created and populated if extra
  folders (e.g. `examples/`) are referenced and are needed to build the documentation
- [ ] **Documentation asset hook (optional)**: `_on_doc_generation.py` added if the package needs generated
  documentation assets
- [ ] **Evaluation results (optional)**: `packages/<package_name>/evaluation_results/` contains committed
  inputs for generated docs assets if needed
- [ ] **Examples (optional)**: `packages/<package_name>/examples/` created and referenced from docs if used
- [ ] **Dependencies**: Declared runtime and optional dependencies in `pyproject.toml`
- [ ] **Verification**: Package installs, tests pass, docs build

## CMake-Based Packages with SKBuild

Use **SKBuild** (`scikit-build`) for namespace packages whose native implementation is built with CMake. The
canonical example is `packages/example_skbuild_package/`.

SKBuild packages cannot combine CMake targets with PyTorch's `CppExtension` or `CUDAExtension` declarations in
`setup.py`. If a CMake package needs PyTorch integration, define those targets in `CMakeLists.txt` instead.

SKBuild packages:
- Use CMake for building C++/CUDA extensions
- Install built extension modules through CMake install targets
- In `setup.py`
  - Use `from skbuild import setup` instead of `from setuptools import setup`
  - Include `cmake_source_dir` and `cmake_install_dir` parameters

### SKBuild Package Structure

An SKBuild-based package follows this structure:

```
packages/example_skbuild_package/
├── ...
├── cpp_unit_tests.yaml             # Optional: repository native-test runner configuration
├── setup.py
├── pyproject.toml
└── ext_impl/
    ├── CMakeLists.txt
    ├── external/                   # Optional third-party dependencies, often added as git submodules
    │   └── googletest/             # Native-test dependency used by this example
    ├── src/
    │   ├── external_cuda_ops.cpp
    │   └── external_cuda_ops.cu
    ├── include/
    │   └── external_cuda_ops.h
    └── utest/                      # Optional: native C++/CUDA tests
        └── ...
```

> **ℹ️ Note**: The `utest/` directory is optional and only needed if native C++/CUDA tests are implemented. See
> [Native Tests](#native-tests) for more details.

### SKBuild Setup Configuration

#### 1. Setting up `setup.py` with SKBuild

See `packages/example_skbuild_package/setup.py` for a complete working example.

The SKBuild setup:
- Uses `cmake_source_dir` and `cmake_install_dir` parameters in `setup.py`. Please see the 
  [important note for cmake install configuration](#important-note-for-cmake-install-configuration) 
  section below for details on how these parameters need to be configured.
- Cannot be used in combination with PyTorch's `CppExtension` and `CUDAExtension` in `setup.py`. The
  extensions must be set up as targets in the `CMakeLists.txt`.

> **ℹ️ Note**: The shared ACCV-Lab build configuration can be forwarded to SKBuild-based CMake builds. In this case, `setup.py`
> typically calls `build_cmake_args(cuda_arch_strategy=...)` and passes the result to `setup()` via the
> `cmake_args` parameter. See the
> [Build Configuration System](#the-build-configuration-system) section below for details.

SKBuild-based packages can also define opt-in native C++/CUDA test targets in their CMake project. These tests can e.g. be used
for implementation details which are not exposed by the Python API and therefore cannot be tested from Python. See
[Native Tests](#native-tests) for the repository runner and the example in `packages/example_skbuild_package/ext_impl/utest`.

#### 2. CMakeLists.txt Configuration

See `packages/example_skbuild_package/ext_impl/CMakeLists.txt` for a complete working example.

The CMake project needs an install target that places the built extension under the package directory. See the
[important note for CMake install configuration](#important-note-for-cmake-install-configuration).

#### 3. Setting up `pyproject.toml` with SKBuild

See `packages/example_skbuild_package/pyproject.toml` for a complete working example.

SKBuild packages include `scikit-build` and `pybind11` as additional build requirements.

#### 4. Python Package Structure

See `packages/example_skbuild_package/accvlab/example_skbuild_package/__init__.py` and 
`packages/example_skbuild_package/accvlab/example_skbuild_package/functions/functions.py` for complete working 
examples.

The built extension module is installed into the final Python package, where it can be imported by Python
wrappers and expose functionality defined in the C++ implementation using `pybind11`.

### SKBuild Best Practices

The CMake snippets in this section are adapted from
`packages/example_skbuild_package/ext_impl/CMakeLists.txt`, which is the complete reference implementation for
the example SKBuild package.

#### 1. Package Discovery Configuration & CMake Install Configuration

#### Package Discovery Configuration

SKBuild requires careful configuration to ensure extensions are placed correctly in the final package:

```python
# In setup.py
setup(
    # ... other configuration ...
    cmake_install_dir="accvlab/example_skbuild_package",  # Must match package structure
)
```

##### CMake Install Configuration

The CMake install target must match the `cmake_install_dir`. Note that this directory is the working directory as used
when setting up the `install` target in the `CMakeLists.txt`, so that the path needs to be set to `.`:

```cmake
# Install target - destination must be relative to cmake_install_dir
install(TARGETS accvlab_example_skbuild_package_ext
    LIBRARY DESTINATION .  # Installs to cmake_install_dir
    RUNTIME DESTINATION .
)
```

##### Important Note for CMake Install Configuration

Note that the `cmake_install_dir` (set to `accvlab/example_skbuild_package`) in `setup.py` and the CMake 
installation directory (set to `.`) in `CMakeLists.txt` must point to the same directory in order for the 
binary to be included in the wheel in the correct way (i.e. not as `data`). If this is not the case, it cannot 
be imported from the installed package. 

However, the paths in `setup.py` and `CMakeLists.txt` both use relative paths, but relative to different base 
locations. While the `cmake_install_dir` in `setup.py` relative to that `setup.py`, the CMake install 
configuration is relative to the package root. The package root is located at 
`accvlab/example_skbuild_package` (relative to the `setup.py`) in this case (and in general at 
`accvlab/<package_name>` for the setup used in this repo). Therefore, the general guideline is:
- In `setup.py`, set `cmake_install_dir` to the root of the package (i.e. `accvlab/<package_name>`)
- In the `CMakeLists.txt`, set the installation directory to `.`.

#### 2. Extension Module Naming

Ensure the extension module name matches Python import expectations:

```cmake
# Set output name to match Python import
set_target_properties(accvlab_example_skbuild_package_ext PROPERTIES
    OUTPUT_NAME "_ext"  # Results in _ext.so
    PREFIX ""           # No lib prefix
)
```

#### 3. PyTorch Integration

For CMake-built PyTorch extensions, the CMake project needs three pieces of PyTorch-specific configuration.

First, point CMake at the PyTorch CMake files installed inside the active Python environment, then call
`find_package(Torch REQUIRED)`. PyTorch wheels usually install their CMake package under
`torch/share/cmake`, which is not always on `CMAKE_PREFIX_PATH` by default:

```cmake
# Find PyTorch CMake files
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os; import torch; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)
```

Second, apply the compile flags exported by Torch to every target that includes or links against Torch. Parse
the string with `separate_arguments()` so CMake passes each option as a separate compiler argument:

```cmake
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(accvlab_example_skbuild_package_ext PRIVATE ${TORCH_CXX_FLAGS_LIST})
```

This includes settings such as PyTorch's C++ ABI selection. Omitting these flags can produce build or runtime
incompatibilities even when `target_link_libraries(... ${TORCH_LIBRARIES})` succeeds. Apply the same pattern to
native test and benchmark executables that consume Torch. Dependencies compiled as part of the same CMake
project must also use the matching ABI when they exchange C++ standard-library types with those targets.

Third, define the extension module name expected by the C++ binding code:

```cmake
# The CMake target name can be descriptive; OUTPUT_NAME controls the Python module filename.
set_target_properties(accvlab_example_skbuild_package_ext PROPERTIES
    OUTPUT_NAME "_ext"
    PREFIX ""
)

# Add PyTorch extension definitions
target_compile_definitions(accvlab_example_skbuild_package_ext PRIVATE
    TORCH_EXTENSION_NAME=_ext
    TORCH_API_INCLUDE_EXTENSION_H
)
```

`TORCH_EXTENSION_NAME=_ext` must match the installed module name because the binding source uses
`PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)`. In the example package this expands to `PYBIND11_MODULE(_ext, m)`,
matching the `_ext` module imported by the Python package. `TORCH_API_INCLUDE_EXTENSION_H` enables the Torch
extension API declarations expected by PyTorch/pybind11 extension builds.


## The Build Configuration System

ACCV-Lab uses a centralized build configuration provided by the `accvlab_build_config` package to ensure 
consistency across packages and simplify adding new ones.

### Installation

The `build_config` package is a build dependency of the contained namespace packages. It is installed as part 
of the ACCV-Lab installation using the unified installer script. However, it can also be built and installed 
manually as follows:
```bash
cd build_config
# Inside the build_config directory, call:
pip install . --no-build-isolation
```

### Purpose and Benefits

The `build_config/` package serves several key purposes:

1. **Shared Build Utilities**: Provides common functions for C++/CUDA extension building, dependency 
   management, and configuration handling
2. **Consistency**: Ensures all namespace packages use the same build logic, compiler flags, and configuration 
   patterns
3. **Maintainability**: Centralizes build logic so bug fixes and improvements benefit all packages
4. **Simplicity**: Reduces amount of boilerplate code needed in the `setup.py` files of individual packages


### Shared Build & Configuration Utilities

The `accvlab_build_config` package provides the following shared build & configuration utilities:
- `load_config()` - Loads build configuration from environment variables (shared across all build types). Please see the 
  [Available Build Variables](INSTALLATION_GUIDE.md#available-build-variables) section of the 
  [Installation Guide](INSTALLATION_GUIDE.md) for the list of the supported variables.
- `detect_cuda_info()` - Detects CUDA availability and GPU architectures. Missing PyTorch or CPU-only PyTorch raises a
  build configuration error.
- `get_compile_flags()` - Generates compiler flags for PyTorch extensions; based on the variable values obtained from 
  `load_config()`. The generated flags can then be passed to the PyTorch extensions (see example below).
- `build_cmake_args(cuda_arch_strategy)` - Produces the full CMake `-D` argument list for CMake-based builds.
  Each CMake-based package must pass `CUDA_ARCH_STRATEGY_CMAKE` or
  `CUDA_ARCH_STRATEGY_TORCH` depending on whether its `CMakeLists.txt` uses native CMake CUDA targets or
  `find_package(Torch)`. It contains two parts:
  - **Environment-derived build settings**: Converts ACCV-Lab build variables into CMake cache entries:
    - `DEBUG_BUILD` → `CMAKE_BUILD_TYPE`
    - `CPP_STANDARD` → `CMAKE_CXX_STANDARD`, `CMAKE_CUDA_STANDARD`
    - `CUSTOM_CUDA_ARCHS` → `CMAKE_CUDA_ARCHITECTURES` when using `CUDA_ARCH_STRATEGY_CMAKE`, or
      `-DACCVLAB_TORCH_CUDA_ARCH_LIST=...` when using `CUDA_ARCH_STRATEGY_TORCH`. Torch CMake projects must
      set `TORCH_CUDA_ARCH_LIST` from that cache variable before `find_package(Torch)`.
    - `VERBOSE_BUILD` → `CMAKE_VERBOSE_MAKEFILE`
    - `OPTIMIZE_LEVEL`, `USE_FAST_MATH`, `ENABLE_PROFILING` → appended to `CMAKE_CXX_FLAGS`, `CMAKE_CUDA_FLAGS`
    - Always sets `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
  - **Repository-aligned version info**: Adds 
    `-DACCVLAB_PACKAGE_CMAKE_VERSION=<major>.<minor>.<patch>` (digits only, e.g. `0.1.0`), as derived from the version obtained 
    using `setuptools-scm`. CMake projects may read this variable if they need a repo-aligned numeric version; others can ignore 
    it.
> **ℹ️ Note**: The authoritative list of supported build variables, defaults, and CUDA 
> architecture handling is in the 
> [Available Build Variables](INSTALLATION_GUIDE.md#available-build-variables) section of the 
> [Installation Guide](INSTALLATION_GUIDE.md).

### Usage in Namespace Packages

Each namespace package's `setup.py` imports and uses these shared utilities, for example:

```python
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension

# The real setup.py files wrap this import in a guarded block with a prominent
# missing-dependency message. The message body is omitted here for brevity.
from accvlab_build_config import (
    load_config,
    detect_cuda_info,
    get_abs_setup_dir,
    get_compile_flags,
)

# Load config and generate flags
config = load_config()
cuda_info = detect_cuda_info()
compile_flags = get_compile_flags(config, cuda_info)
source_dir = "accvlab/<pkg>"
include_dirs = [str(get_abs_setup_dir(__file__) / source_dir / "include")]

# Example: wire flags into extensions
ext_modules = [
    CppExtension(
        name="accvlab.<pkg>._cpp",
        sources=["accvlab/<pkg>/csrc/cpp_functions.cpp"],
        include_dirs=include_dirs,
        extra_compile_args=compile_flags['cxx'],
        language="c++",
        verbose=config["VERBOSE_BUILD"],
    ),
    CUDAExtension(
        name="accvlab.<pkg>._cuda",
        sources=["accvlab/<pkg>/csrc/cuda_functions.cu"],
        include_dirs=include_dirs,
        extra_compile_args={"cxx": compile_flags["cxx"], "nvcc": compile_flags["nvcc"]},
        language="c++",
        verbose=config["VERBOSE_BUILD"],
    ),
]
```

This approach ensures that:
- All packages use the same build logic
- Configuration is consistent across packages
- Build improvements benefit all packages automatically
- Individual package setup.py files remain simple and focused

Please see the `setup.py` files of the example packages (e.g. `packages/example_package/setup.py` and 
`packages/example_skbuild_package/setup.py`) for complete working examples for different package types.

### How Build Variables Are Picked Up

> **ℹ️ Note**: The authoritative list of supported build variables, defaults, and CUDA 
> architecture handling is in the 
> [Available Build Variables](INSTALLATION_GUIDE.md#available-build-variables) section of the 
> [Installation Guide](INSTALLATION_GUIDE.md). The 
> [Shared Build & Configuration Utilities](#shared-build--configuration-utilities) section explains how the 
> shared helper utilities consume those variables.

Depending on the package type, build variables are consumed as follows:

- Setuptools (PyTorch extensions):
  - In `setup.py`, call `config = load_config()` and `cuda_info = detect_cuda_info()`, then pass 
    `compile_flags = get_compile_flags(config, cuda_info)` to `CppExtension`/`CUDAExtension`:
  ```python
  config = load_config()
  cuda_info = detect_cuda_info()
  compile_flags = get_compile_flags(config, cuda_info)
  ext = CUDAExtension(
      name='accvlab.<pkg>._ext',
      sources=[...],
      extra_compile_args={'cxx': compile_flags['cxx'], 'nvcc': compile_flags['nvcc']},
      language='c++',
      verbose=config['VERBOSE_BUILD'],
  )
  ```
  - This forwards `DEBUG_BUILD`, `OPTIMIZE_LEVEL`, `CPP_STANDARD`, `VERBOSE_BUILD`, `CUSTOM_CUDA_ARCHS`, 
    `USE_FAST_MATH`, and `ENABLE_PROFILING` to host and device compilers.

- Scikit-build packages:
  - In `setup.py`, pass CMake arguments from the helper. Use `CUDA_ARCH_STRATEGY_CMAKE` for native CMake CUDA
    targets, or `CUDA_ARCH_STRATEGY_TORCH` when `ext_impl/CMakeLists.txt` calls `find_package(Torch)`:
  ```python
  # The real setup.py files wrap this import in a guarded block with a prominent
  # missing-dependency message. The message body is omitted here for brevity.
  from accvlab_build_config import build_cmake_args, CUDA_ARCH_STRATEGY_CMAKE

  _cmake_args = build_cmake_args(cuda_arch_strategy=CUDA_ARCH_STRATEGY_CMAKE)
  setup(
      ...,
      cmake_source_dir="ext_impl",
      cmake_install_dir="accvlab/<pkg>",
      cmake_args=_cmake_args,
  )
  ```
  - For Torch CMake projects, replace `CUDA_ARCH_STRATEGY_CMAKE` in the import and call with
    `CUDA_ARCH_STRATEGY_TORCH`.
  - In `ext_impl/CMakeLists.txt`, guard defaults:
  ```cmake
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  ```
  - If the CMake project calls `find_package(Torch)`, set the Torch architecture list before that call:
  ```cmake
  if(DEFINED ACCVLAB_TORCH_CUDA_ARCH_LIST)
    set(TORCH_CUDA_ARCH_LIST "${ACCVLAB_TORCH_CUDA_ARCH_LIST}")
  endif()
  find_package(Torch REQUIRED)
  ```


### Configuration Options

For the list of supported environment variables, their defaults, and descriptions, see the
[Available Build Variables](INSTALLATION_GUIDE.md#available-build-variables) in the installation guide.
These variables can be set per-package during installation or globally for all packages.


## Development Workflow

### Testing During Development

#### Running Tests

Tests of a package can be performed with:
```bash
cd packages/<package_name>/tests
pytest .
```

> **⚠️ Important**: Note that the tests do not necessarily need to be run from within the `tests` directory.
> However, it is advised to not run tests from the root directory of a namespace package (i.e.
> not from `packages/<package_name>`), as in this way, importing the package is ambiguous.
> For example, the import `import accvlab.<package_name>` could mean either refer to the installed package, 
> or to the original source files inside the current directory.

> **ℹ️ Note**: See the section [Repository test runner](#repository-test-runner-scriptsrun_testssh) for how to run 
> tests across all packages in the repository.

#### Native Tests

Native C++/CUDA tests are CMake/CTest targets for implementation details that are not naturally covered by the
Python test suite. They are opt-in and are not part of a normal `pip install` package build. They can e.g. be used to test
implementation details which are not exposed by the Python API.

> **ℹ️ Note**: The `packages/example_skbuild_package` package provides a complete example of how to implement
> native tests. Useful reference files are:
> - `packages/example_skbuild_package/cpp_unit_tests.yaml`: repository runner configuration.
> - `packages/example_skbuild_package/ext_impl/CMakeLists.txt`: package-level CMake option and `utest/`
>   subdirectory wiring.
> - `packages/example_skbuild_package/ext_impl/utest/CMakeLists.txt`: GoogleTest setup, CTest registration,
>   and aggregate test target.
> - `packages/example_skbuild_package/ext_impl/utest/dummy_native_test.cpp`: minimal native test executable.
>
> The production `lane_helpers` package demonstrates a component-level layout under
> `packages/lane_helpers/ext_impl/polyline/utest/`.

The following conventions are recommended when adding native tests:

- Put native tests under an opt-in CMake subdirectory. Common locations are `ext_impl/utest/` for a
  package-level test suite or `ext_impl/<component>/utest/` for a component-level test suite.
- Gate the test subdirectory with a package-specific CMake option, such as
  `ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_BUILD_CPP_TESTS`.
- Call `enable_testing()` in the root `CMakeLists.txt`, for example in
  `packages/<package_name>/ext_impl/CMakeLists.txt`, when the native test option is enabled. This creates the top-level
  CTest metadata used by the runner.
- Define test executables with `EXCLUDE_FROM_ALL` so they are not built with the package target.
- Register each executable with CTest using `add_test(...)`.
- Add an explicit aggregate target, such as `accvlab_example_skbuild_package_run_cpp_tests`, that depends on
  the test executables and runs `ctest --output-on-failure`.

To run native tests with the repository runner:

- A `cpp_unit_tests.yaml` config must be present for the package. See below for the expected fields.
- A single aggregate CMake target must build the native test executables and run CTest; this is the target
  referenced by the YAML config.
- The configured test target must return a nonzero exit code when tests fail.

> **ℹ️ Note**: Native tests typically use GoogleTest, added as a git submodule under the package's
> `ext_impl/external/` directory and included from CMake with `add_subdirectory(...)`. The example package uses
> this pattern at `packages/example_skbuild_package/ext_impl/external/googletest`. GoogleTest is not strictly
> required: any native executable can be registered with CTest as long as it returns a nonzero exit code on
> failure.

When the package calls `find_package(Torch)`, compile both the native test executables and GoogleTest with
`TORCH_CXX_FLAGS_LIST`. PyTorch binaries can use either libstdc++ ABI, depending on the release and wheel
variant. Applying Torch's exported flags to every target on the GoogleTest boundary keeps `std::string` and
other standard-library types ABI-compatible:

```cmake
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

add_subdirectory(
    "${ACCVLAB_GTEST_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/googletest"
    EXCLUDE_FROM_ALL
)

foreach(gtest_target IN ITEMS gtest gtest_main gmock gmock_main)
    if(TARGET ${gtest_target})
        target_compile_options(${gtest_target} PRIVATE ${TORCH_CXX_FLAGS_LIST})
    endif()
endforeach()

target_compile_options(my_native_test PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_link_libraries(my_native_test PRIVATE GTest::gtest_main ${TORCH_LIBRARIES})
```

The runner reads `namespace_packages_config.py` and checks only those configured packages for package-local
native test configuration files named `cpp_unit_tests.yaml` in the package root directory. Packages not listed there must
be supplied explicitly with `--config packages/<package_name>/cpp_unit_tests.yaml`. For example, the SKBuild
example package is not configured by default, and its config lives at
`packages/example_skbuild_package/cpp_unit_tests.yaml`. The config looks like this:

```yaml
cmake_source_dir: ext_impl
cuda_arch_strategy: torch
test_option: ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_BUILD_CPP_TESTS
test_target: accvlab_example_skbuild_package_run_cpp_tests
```

The fields have the following meaning:

- `cmake_source_dir`: CMake project to configure, relative to the package root. For SKBuild packages, this is
  usually `ext_impl`.
- `cuda_arch_strategy`: Required CUDA architecture strategy for the CMake project. Use `torch` when the
  project calls `find_package(Torch)`, or `cmake` for native CMake CUDA targets.
- `test_option`: Package-specific CMake option that enables the otherwise opt-in native test subdirectory. The
  runner automatically forwards it as `-D<test_option>=ON`.
- `test_target`: Explicit aggregate target that builds the native test executables and runs CTest.
- `cmake_args`: Optional additional CMake arguments for package-specific needs.

(native-test-cuda-architecture-strategy)=
##### CUDA Architecture Strategy

The runner passes `cuda_arch_strategy` to `build_cmake_args(...)`, so the value controls how the shared
`CUSTOM_CUDA_ARCHS` setting reaches the package's CMake project:

- Use `torch` when the package CMake project calls `find_package(Torch)`. The shared helper then provides
  `ACCVLAB_TORCH_CUDA_ARCH_LIST`, which the project must assign to `TORCH_CUDA_ARCH_LIST` before finding Torch.
- Use `cmake` for CMake projects that compile native CUDA targets without Torch architecture handling. The
  shared helper then sets `CMAKE_CUDA_ARCHITECTURES`.

Select the strategy for the package's root CMake project, even when an individual test target contains only
C++. The value in `cpp_unit_tests.yaml` must match the strategy passed by the package's `setup.py`.

Choose a `test_option` name that includes the package name to avoid collisions when several package CMake
projects are configured by the same repository tooling. A good pattern is
`ACCVLAB_<PACKAGE_NAME>_BUILD_CPP_TESTS`, for example
`ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_BUILD_CPP_TESTS`.

The package name and build directory are derived by the runner from the config location. For the example above,
the package name becomes `example_skbuild_package` and the build directory becomes
`build/cpp_unit_tests/example_skbuild_package`.

The example package is not enabled in `namespace_packages_config.py` by default, so it is not picked up by
`all`. To run the example directly, pass its config explicitly:

```bash
./scripts/run_cpp_unit_tests.sh --config packages/example_skbuild_package/cpp_unit_tests.yaml example_skbuild_package
```

Or enable the package in `namespace_packages_config.py` and run it as:

```bash
./scripts/run_cpp_unit_tests.sh example_skbuild_package
```

The runner also forwards the shared ACCV-Lab CMake build arguments before package-specific arguments.
That means the native test targets use the same build variables as package builds, including build type,
compiler flags, CUDA architecture selection, verbosity, and the configured C++/CUDA standard. See
[Shared Build & Configuration Utilities](#shared-build--configuration-utilities) for how the arguments are
created and [Available Build Variables](INSTALLATION_GUIDE.md#available-build-variables) for the supported
environment variables. The test option from the YAML config is added automatically after those shared
arguments. Test targets should avoid overriding these settings unless they have a specific reason to do so.

The runner requires the `accvlab-build-config` package in the active environment. It is installed by the
repository package manager, or can be installed directly with `pip install build_config/`.

Use the repository-level runner for native tests, e.g.:

```bash
# List native test packages discovered from namespace_packages_config.py.
./scripts/run_cpp_unit_tests.sh --list

# Run native tests for one configured package.
./scripts/run_cpp_unit_tests.sh lane_helpers

# Run native tests for all configured packages with native test configs.
./scripts/run_cpp_unit_tests.sh all

# Configure one package without building or running its tests.
./scripts/run_cpp_unit_tests.sh --configure-only lane_helpers
```

When adding native tests to another configured SKBuild package, follow the same structure and add a
package-local `cpp_unit_tests.yaml`.


#### Install in Development Mode

This can be done with
```bash
pip install -e . --no-build-isolation
```

Note that
- The editable mode refers to the Python code. For any changes in C++ code to take effect, the package
  needs to be re-installed regardless of whether it is installed in editable mode or not.
- SKBuild does not support editable installs. SKBuild-based packages need to always be installed without the `-e` flag.

### Documentation Development

> **⚠️ Important**: Ensure all configured namespace packages are installed before building the docs.
> For detailed instructions (commands and development targets), see the 
> [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md#building-documentation-locally).

## Code Formatting

The ACCV-Lab project uses automated code formatting to maintain consistent code style across common code and 
all namespace packages. Please see the [Formatting Guide](FORMATTING_GUIDE.md) for details.

### Script Features

As all scripts, the formatting script automatically discovers configured namespace packages from 
`namespace_packages_config.py`.

Formatting for the whole repo can be run as:
```bash
bash scripts/format.sh
```

It is also possible to run the formatting for individual namespace packages. To list available packages and 
format one of them:
```bash
# List available namespace packages
python3 -c "from namespace_packages_config import get_package_names; print('\n'.join(get_package_names()))"

# Format a specific package (example)
./scripts/format.sh --package on_demand_video_decoder
```

You can also format only a specific language:
```bash
# Format Python code only (common + all packages)
./scripts/format.sh --python

# Format C++/CUDA code only (all packages)
./scripts/format.sh --cpp

# Combine with a package filter
./scripts/format.sh --python --package example_package
./scripts/format.sh --cpp --package batching_helpers
```

### Typical Workflows

#### During Development
```bash
# Format the namespace package you're working on
./scripts/format.sh --package example_package
```

#### Before Committing
```bash
# Format everything to ensure consistency
./scripts/format.sh
```

## Repository Test Runner (`scripts/run_tests.sh`)

The repository provides a convenience script to run pytest for all configured namespace packages:
```bash
./scripts/run_tests.sh              # run all tests
./scripts/run_tests.sh -- -k smoke  # pass arguments after -- directly to pytest (`-k smoke` used as an example)
```

> **ℹ️ Note**: This script only runs Python-based tests under `packages/<package_name>/tests`. For native
> C++/CUDA tests, use the native test runner described in [Native Tests](#native-tests).

How it discovers and runs tests:
- Discovers package names from `namespace_packages_config.py` (function `get_package_names()`).
- For each package `<name>`, looks for `packages/<name>/tests`. If missing, it warns and skips.
- Executes `pytest` from inside each `packages/<name>/tests` directory to avoid importing local sources.
- Exits non-zero if any package test run fails.

To ensure your tests are picked up:
- Place tests under `packages/<package_name>/tests`.
- Name files following pytest conventions (e.g., `test_*.py` or `*_test.py`).
- If you need custom pytest flags, pass them after `--`, e.g. 
  `./scripts/run_tests.sh -- -q -k "gpu and not slow"`.

Additional notes:
- Inside your test scripts, import the installed package (e.g., `import accvlab.<package_name>`), not local 
  source paths.
- Ensure the package is installed in the current environment before running the script (editable or standard 
  install). This can be done with `pip install -e .[optional] --no-build-isolation` or 
  `pip install .[optional] --no-build-isolation` (see the Installation Guide for more details). Note that for
  SKBuild-based packages, the editable install is not supported and will result in missing binaries & import 
  errors.

> **⚠️ Important**: The tests often rely on optional dependencies. Therefore, it is recommended to install 
> the package with optional dependencies, as described in the [Installation Guide](INSTALLATION_GUIDE.md).

## Namespace Package Structure and Configuration

### Understanding the Package Structure

ACCV-Lab uses implicit namespace packages where each package directory maps to a namespace under `accvlab`. 
For example:
- `packages/example_package/accvlab/example_package/` → `accvlab.example_package`
- `packages/batching_helpers/accvlab/batching_helpers/` → `accvlab.batching_helpers`

Note that e.g. the directory `packages/example_package/` is the root directory of the package, containing
the `pyproject.toml` & `setup.py` files, native sources, documentation, and related package files. The
remainder of the path (i.e. `[...]/accvlab/example_package/`) reflects the package
name as installed (`accvlab.example_package` in this case) and contains the actual Python package 
implementation as will be included in the installation.

### Package Discovery Configuration

The `pyproject.toml` configuration is crucial for proper package discovery:

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["accvlab.example_package*"]
```

This configuration:
- `where = ["."]` - Searches in the current directory
- `include = ["accvlab.example_package*"]` - Includes all packages starting with `accvlab.example_package`

### Handling Subpackages

If your namespace package contains subpackages (like `accvlab.example_package.functions`), the 
`include = ["accvlab.example_package*"]` pattern will automatically discover and include all subpackages.

### Notes

1. **Namespace package structure**: The directory structure must match the namespace structure:
   ```
   packages/example_package/
   ├── accvlab/                    # ← Namespace root
   │   └── example_package/       # Maps to accvlab.example_package
   │       ├── __init__.py
   │       ├── functions/         # Maps to accvlab.example_package.functions
   │       │   └── __init__.py
   │       └── ...
   ├── setup.py
   ├── pyproject.toml
   └── ...
   ```

2. As usual for modules, `__init__.py` files need to be present and otherwise, e.g. the automatic package 
discovery may fail.

3. **Build artifacts**: Always clean build artifacts (`build/`, `dist/`, `*.egg-info/`) when testing package 
installation to ensure that the installation can be performed from a clean state.
