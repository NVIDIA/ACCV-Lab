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
    #'accvlab.example_package',
    #'accvlab.example_skbuild_package',
    'accvlab.on_demand_video_decoder',
    'accvlab.batching_helpers',
    'accvlab.dali_pipeline_framework',
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

There are two example projects which showcase how a namespace package is structured. These are
- `packages/example_package`: Showcases a package containing PyTorch extensions built using
  `CppExtension` and `CUDAExtension` provided by PyTorch as well as an external implementation (see
  [External Implementations](#external-implementations) section for more details on external implementations)
  as described below.
- `packages/example_skbuild_package`: Showcases a package using `scikit-build` for C++/CUDA implementation 
  (see the [Alternative: SKBuild-Based Packages](#alternative-skbuild-based-packages) section for more 
  details on this approach).

First, we will focus on the `example_package`, and explain how to set it up from scratch.
For `scikit-build` based packages, the setup is similar. The SKBuild-based approach is described in the 
[Alternative: SKBuild-Based Packages](#alternative-skbuild-based-packages) section.

### Overview

To add a new namespace package (e.g., `example_package`), you need to create:

| Component | Directory | Purpose |
|-----------|-----------|---------|
| **Implementation** | `packages/example_package/accvlab/example_package/` | Python package with your actual code |
| **External Implementation** | `packages/example_package/ext_impl/` | Parts of the package which are built externally (e.g. using CMake) and then copied to the main package |
| **Documentation** | `packages/example_package/docs/` | API reference and user guides (template will be auto-generated) |
| **Tests** | `packages/example_package/tests/` | Unit tests (will be automatically picked up by the test runner) |
| **Setup** | `packages/example_package/setup.py` | Package build configuration |
| **Project Config** | `packages/example_package/pyproject.toml` | Modern Python project configuration and authoritative dependency definition |
| **Documentation include list (optional)** | `packages/example_package/docu_referenced_dirs.txt` | List additional directories referenced by the docs (besides `docs/`). See [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md) for more details.|

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
├── packages/                        # Namespace packages directory
│   ├── optim_test_tools/...
│   ├── batching_helpers/...
│   └── example_package/             # ← New namespace package
│       ├── accvlab/                 # ← Namespace root
│       │   └── example_package/     # ← Implementation for "example_package" package
│       │       ├── __init__.py
│       │       ├── csrc/            # ← C++/CUDA sources
│       │       └── include/         # ← Headers
│       ├── ext_impl/                # ← Optional: external implementation
│       │   ├── build_and_copy.sh
│       │   └── ...
│       ├── tests/                   # ← Tests for "example_package" package
│       ├── docs/                    # ← Documentation for "example_package" package
│       ├── setup.py                 # ← Package build configuration
│       ├── pyproject.toml           # ← Project configuration (including dependencies)
│       └── docu_referenced_dirs.txt # ← Optional: list additional directories referenced by the docs (besides `docs/`)
├── build_config/                    # Shared build utilities
├── docs/                            # Main documentation
└── namespace_packages_config.py     # ← Namespace package needs to be listed here
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

# External implementation directory (optional - see External Implementations section)
mkdir -p packages/example_package/ext_impl

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

If more complex C++/CUDA implementations are used, an external implementation can be added in 
`packages/example_package/ext_impl`.

> **ℹ️ Note**: For complex C++/CUDA implementations that require custom build procedures, third-party libraries, 
> or CMake-based builds, there are two alternatives:

1. **External Implementations** - Uses custom build scripts and CMake (used in this example)
2. **SKBuild-based packages** - Uses `scikit-build` for CMake integration

Both approaches are described in detail later in this guide. The `example_package` uses the External 
Implementations approach. For more details, see:
- [External Implementations](#external-implementations) section for the approach used in this example
- [Alternative: SKBuild-Based Packages](#alternative-skbuild-based-packages) section for the SKBuild approach
- [Comparison: External Implementations vs. SKBuild](#comparison-external-implementations-vs-skbuild) section 
  for a detailed comparison of both approaches

#### 3. Create `setup.py`

Create the setup configuration file that defines how to compile C++/CUDA extensions and configure any external
builds. See `packages/example_package/setup.py` for a complete working example.

> **⚠️ Important**: If your package uses external builds (see [External Implementations](#external-implementations) 
> section), you need to:
> - Call `run_external_build()` before calling `setup()` to ensure that the external binaries are included in 
>   the resulting package.
> - Include the parameters as shown below to the call to `setup()` (with `example_package` replaced by your 
>   package's name).

```python
...
run_external_build(str(Path(__file__).parent))
setup(
    ...
    include_package_data=True,
    package_data={
        'accvlab.example_package': ['*.so'],
    },
    zip_safe=False,
    ....
)
```
The first 2 arguments ensure that the external binaries are included, while the last argument ensures that the 
package is not stored as a zip-file, which would prevent the external binaries from being imported correctly.

#### 4. Create `pyproject.toml`

Create the Python project configuration. This file also defines the package's runtime and optional dependencies.
See `packages/example_package/pyproject.toml` for a complete working example.

#### 5. Define Dependencies in `pyproject.toml`

Add your package dependencies to the `[project]` section of `pyproject.toml`. For example, the
`packages/example_package/pyproject.toml` file contains:

```toml
[project]
name = "accvlab.example_package"
version = "0.1.0"
description = "ACCV-Lab Example Package"
requires-python = ">=3.8"
dependencies = [
    "torch>=2.0.0",
    "numpy>=1.22.2",
    "accvlab-build-config>=0.1.0",
]

[project.optional-dependencies]
optional = [
    "pytest",
]
```

Use this pattern for your own namespace package, adapting the dependency names and versions as needed.

#### 6. Add to NAMESPACE_PACKAGES List

This ensures that your package is included in various places, e.g.
- in the installation by the `package_manager.sh` script
- in the documentation
- in the code formatting by the `format.sh` script

```python
# In namespace_packages_config.py
NAMESPACE_PACKAGES = [
    'accvlab.optim_test_tools',
    'accvlab.batching_helpers',
    'accvlab.dali_pipeline_framework',
    'accvlab.on_demand_video_decoder',
    'accvlab.example_package',  # Add your new namespace package here
]
```

#### 7. Create Tests

Create test files for your namespace package. See `packages/example_package/tests/` for examples.

Note that this includes only tests for the Python functionality. If you use C++ tests, please ensure
that they are performed during the build and in case of external implementation, that 
the `build_and_copy.sh` script returns an error code (e.g. a value != 0) if tests fail.

#### 8. Set Up Documentation

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

#### 9. Test Your Package

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

#### 10. Build the Documentation

> **⚠️ Important**: Ensure all configured namespace packages are installed before building the docs.
> For detailed instructions (commands and development targets), see the 
> [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md#building-documentation-locally).

### Summary Checklist

When adding a new namespace package, ensure you have:

- [ ] **Implementation**: `packages/example_package/accvlab/example_package/` with `__init__.py` and source 
  code
- [ ] **Setup**: `packages/example_package/setup.py` with build configuration
- [ ] **Project Config**: `packages/example_package/pyproject.toml` with project metadata
- [ ] **Configuration**: Added to `namespace_packages_config.py`
- [ ] **Tests**: `packages/example_package/tests/` with test files
- [ ] **Documentation**: Generated with docs scripts and customized intro
- [ ] **Documentation include list (optional)**: `docu_referenced_dirs.txt` created and populated if extra 
  folders (e.g. `examples/`) are referenced
- [ ] **Examples (optional)**: `packages/<package_name>/examples/` created and referenced from docs if used
- [ ] **Dependencies**: Declared runtime and optional dependencies in `pyproject.toml`
- [ ] **External implementation**: (Optional) `packages/example_package/ext_impl/` for external builds
- [ ] **External build binaries**: (If using external builds) `include_package_data=True` and `package_data` 
  configuration in `setup.py`
- [ ] **Verification**: Package installs, tests pass, docs build

## External Implementations

External implementations provide a way to build complex components outside of the standard Python/pybind11 
workflow and then integrate them into your namespace package. This is particularly useful for:

- **Complex CMake projects** with multiple dependencies
- **Third-party libraries** that need specific build configurations
- **Other cases with the need for flexible manual setups**

For relatively small and self-contained C++/CUDA implementations, consider using PyTorch's `CppExtension` and 
`CUDAExtension` instead.

> **ℹ️ Note**: This section describes the External Implementations approach used in the `example_package`. For an 
> alternative approach using `scikit-build`, see the 
> [Alternative: SKBuild-Based Packages](#alternative-skbuild-based-packages) section. A detailed comparison of 
> both approaches is available in the 
> [Comparison: External Implementations vs. SKBuild](#comparison-external-implementations-vs-skbuild) section.

### External Implementation Structure

External implementations are located in the `packages/example_package/ext_impl/` directory. An example is 
shown below.

```
accvlab/
├── packages/
│   └── example_package/
│       ├── ...
│       ├── setup.py
│       ├── pyproject.toml
│       └── ext_impl/
│           ├── build_and_copy.sh
│           ├── CMakeLists.txt
│           ├── src/
│           │   ├── external_algo.cpp
│           │   └── external_algo.cu
│           ├── include/
│           │   └── external_algo.h
│           ├── third_party/
│           └── build/
```

### Required Components

#### 1. Build and Copy Script (`build_and_copy.sh`)

Every external implementation must have this script,
which handles the external build process and copies results to the main package. See 
`packages/example_package/ext_impl/build_and_copy.sh` for a complete working example.

#### 2. Build Configuration (`CMakeLists.txt`)

Typically, the external build is performed using `CMake` (although this is not strictly necessary).
See `packages/example_package/ext_impl/CMakeLists.txt` for a complete working example.

### Integration with Build System

External implementations are automatically handled by the main `setup.py` in a generic manner (by calling 
`accvlab_build_config.run_external_build()`). Please ensure that you set up the `setup.py` correctly (as 
described in the [setup.py configuration](#3-create-setuppy) section above).


> **ℹ️ Note**: External implementations can be used alongside standard C++/CUDA extensions in the same namespace 
> package. You can combine both approaches - use external implementations for complex components while still 
> having standard extensions for simpler functionality.

The setup process works as follows:

1. **Selective Detection**: `setup.py` detects whether an external implementation is present and calls the 
   `build_and_copy.sh` script (which is responsible for building & inserting the binaries into the package).
2. **Standard Processing**: After the external build completes for a namespace package, the obtained binaries 
   are inside the package, and standard installation (including extension processing) can be used. Note
   that the build binaries are copied to the installed package if `setup()` is called with the correct
   parameters (see the [setup.py configuration](#3-create-setuppy) section above).

This means your `setup.py` remains simple, and only the changes described in the 
[setup.py configuration](#3-create-setuppy) section above are needed to use external implementations.

**Key Points:**
- External implementations are only built for namespace packages listed in `namespace_packages_config.py` (as 
  installation is only triggered for those).
- External builds are executed **before** each namespace package's call to `setup()`, i.e. before potential 
  extension processing.
- The external build script copies its outputs to the main package directory.
- **External implementations can coexist with standard extensions** - you can have both in the same namespace 
  package.
- No special external build logic is needed in `setup.py`, only ensure to call `run_external_build()` and 
  ensure the resulting files are included in the package (see [setup.py configuration](#3-create-setuppy)).

> **ℹ️ Note**: The fact that external builds are performed before calls to `setup()` and are copied into the 
> package means that it is possible to use the external builds inside standard PyTorch extensions built with 
> `CppExtension` and `CUDAExtension`.

### External Implementation Best Practices

#### Error Handling
- Use `set -e` in build scripts to fail fast on errors
- Provide clear error messages for build failures
- Check for required tools and dependencies before building

#### Path Management
- Make scripts work from any working directory
- Use variables for commonly referenced paths

#### Build Reproducibility
- Pin dependency versions where possible
- Use consistent compiler flags and options
- Clean build directories before building

#### Including External Build Binaries

When using external builds, you must configure `setup.py` to include the built binaries in the installed 
package:

```python
setup(
    # ... other configuration ...
    
    # `include_package_data` and `package_data` are needed for the binaries created by the external build
    # to be included in the package.
    include_package_data=True,
    package_data={
        'accvlab.example_package': ['*.so'],  # Include all .so files in the package
    },
    zip_safe=False,

    # ... rest of configuration ...
)
```

> **⚠️ Important**: Without this configuration, the binaries created by external builds will not be included in the 
> installed package, causing import errors when users try to use your package.

**Pattern matching**: The `package_data` configuration uses glob patterns to match files. Common patterns:
- `['*.so']` - Include all shared object files
- `['lib/*.so']` - Include .so files in a lib subdirectory

## Alternative: SKBuild-Based Packages

You can also use **SKBuild** (scikit-build) instead of the external implementation approach discussed above. 
See `packages/example_skbuild_package/` for a complete example. Note that SKBuild-based packages cannot be 
used in combination with PyTorch's `CppExtension` and `CUDAExtension` in `setup.py`. All PyTorch extensions 
must be set up as targets in the `CMakeLists.txt` instead.

SKBuild packages:
- Use `from skbuild import setup` instead of `from setuptools import setup`
- Include `cmake_source_dir` and `cmake_install_dir` parameters in `setup.py` (see below for details)
- Use CMake for building C++/CUDA extensions
- Don't require additional `build_and_copy.sh` scripts (and calls to `run_external_build()`), simplifying 
  the setup

### SKBuild Package Structure

An SKBuild-based package follows this structure:

```
packages/example_skbuild_package/
├── ...
├── setup.py
├── pyproject.toml
└── ext_impl/
    ├── CMakeLists.txt
    ├── src/
    │   ├── external_cuda_ops.cpp
    │   └── external_cuda_ops.cu
    └── include/
        └── external_cuda_ops.h
```

### SKBuild Setup Configuration

#### 1. Setting up `setup.py` with SKBuild

See `packages/example_skbuild_package/setup.py` for a complete working example.

**Key Differences from the external implementation-based approach**:
- No `run_external_build()` calls
- No `package_data` configuration (SKBuild handles this automatically)
- Uses `cmake_source_dir` and `cmake_install_dir` parameters in `setup.py`. Please see the 
  [important note for cmake install configuration](#important-note-for-cmake-install-configuration) 
  section below for details on how these parameters need to be configured.
- Cannot be used in combination with PyTorch's `CppExtension` and `CUDAExtension` in `setup.py`. The
  extensions must be set up as targets in the `CMakeLists.txt`.

#### 2. CMakeLists.txt Configuration

See `packages/example_skbuild_package/ext_impl/CMakeLists.txt` for a complete working example.

**Key Differences from External Implementations**: 
- The overall setup is simpler - no need for `build_and_copy.sh`, i.e. no manual path calculations or copying 
  of files
- Install target needs to be set up (see the 
  [important note for cmake install configuration](#important-note-for-cmake-install-configuration)).

#### 3. Setting up `pyproject.toml` with SKBuild

See `packages/example_skbuild_package/pyproject.toml` for a complete working example.

**Key Difference from External Implementations**: Includes `scikit-build` and `pybind11` as build 
requirements.

#### 4. Python Package Structure

See `packages/example_skbuild_package/accvlab/example_skbuild_package/__init__.py` and 
`packages/example_skbuild_package/accvlab/example_skbuild_package/functions/functions.py` for complete working 
examples.

This is similar to the external implementation-based case in that a binary is present in the final package, 
which can be imported in Python and exposes relevant functionality as defined in the C++ implementation using 
`pybind11`.

### SKBuild Best Practices

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

The CMake install target must match the `cmake_install_dir`. Note that this directory is the working as used
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

For PyTorch extensions, ensure proper CMake configuration:

```cmake
# Find PyTorch CMake files
execute_process(
    COMMAND "python3" -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

# Add PyTorch extension definitions
target_compile_definitions(accvlab_example_skbuild_package_ext PRIVATE 
    TORCH_EXTENSION_NAME=_ext
    TORCH_API_INCLUDE_EXTENSION_H
)
```

## Comparison: External Implementations vs. SKBuild

Both External Implementations and SKBuild use CMake for building C++/CUDA extensions, but they differ 
significantly in how they integrate with Python packaging:

### SKBuild vs. External Implementations (CMake-Based)

| Aspect | External Implementations | SKBuild |
|--------|-------------------------|---------|
| **Build Integration** | Automated via `build_and_copy.sh` called by setup.py | Integrated with pip/setuptools |
| **Package Discovery** | Automated file copying via build script | Automatic package inclusion |
| **Development Workflow** | Single `pip install` command (build script runs automatically) | Single `pip install` command |
| **Complexity** | Medium to High (automated build script + setup.py integration) | Medium (CMake + Python integration) |
| **Use Case** | Complex external projects with custom build procedures | CMake-based Python extensions |
| **Build Scripts** | Required custom `build_and_copy.sh` | No custom scripts needed |
| **File Management** | Copying of `.so` files via build script | Automatic placement in package |

### SKBuild vs. External Implementations: Workflow Comparison

#### External Implementations Workflow:
```bash
# Single step - setup.py automatically calls build_and_copy.sh
cd packages/example_package
pip install {-e} . --no-build-isolation
```

#### SKBuild Workflow:
```bash
# Single step - SKBuild handles everything
cd packages/example_skbuild_package
pip install . --no-build-isolation
```

> **ℹ️ Note**: `{-e}` means that this argument is optional. It is not present in the SKBuild-based example as 
> SKBuild does not support editable installs.

> **ℹ️ Note**: Both approaches use a single `pip install` command. The difference is that external implementations 
> automatically execute the `build_and_copy.sh` script during the setup process, while SKBuild handles the CMake 
> build process internally.

### When to Use SKBuild vs. External Implementations

**Use SKBuild when:**
- You have CMake-based C++/CUDA code that you want to integrate as Python extensions
- You want to avoid writing custom build scripts and rely on SKBuild's built-in CMake integration
- You prefer a more standardized approach to CMake-based Python packaging
- You want to leverage CMake's features while maintaining Python packaging standards

**Use External Implementations when:**
- You want to use CMakeLists for parts of the implementation while defining PyTorch extensions directly in 
  `setup.py`
- You have very complex external projects with their own build systems
- You need to integrate with third-party libraries that require specific build procedures
- You want maximum flexibility in build configuration and file placement
- You have existing build scripts that you want to preserve and integrate
- You need to build components that exist outside the Python package structure
- You require custom build steps that go beyond standard CMake workflows


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
- `load_config()` - Loads build configuration from environment variables (shared across all build types).
  Please see the [Available Build Variables](INSTALLATION_GUIDE.md#available-build-variables) section of the 
  [Installation Guide](INSTALLATION_GUIDE.md) for the list of the supported variables.
- `detect_cuda_info()` - Detects CUDA availability and version
- `get_compile_flags()` - Generates compiler flags for PyTorch extensions; based on the variable values
   obtained from `load_config()`. The generated flags can then be passed to the PyTorch extensions (see 
   example below).
- `build_cmake_args_from_env()` - Translates the variable values obtained from `load_config()` into CMake `-D` 
  arguments for scikit-build and external CMake builds. The mappings are as follows:
  - `DEBUG_BUILD` → `CMAKE_BUILD_TYPE`
  - `CPP_STANDARD` → `CMAKE_CXX_STANDARD`, `CMAKE_CUDA_STANDARD`
  - `CUSTOM_CUDA_ARCHS` → `CMAKE_CUDA_ARCHITECTURES` (auto-detect if unset)
  - `VERBOSE_BUILD` → `CMAKE_VERBOSE_MAKEFILE`
  - `OPTIMIZE_LEVEL`, `USE_FAST_MATH`, `ENABLE_PROFILING` → appended to `CMAKE_CXX_FLAGS`, `CMAKE_CUDA_FLAGS`
  - Always sets `-DCMAKE_EXPORT_COMPILE_COMMANDS=ON`
- `run_external_build()` - Executes `build_and_copy.sh` build script (used in external implementations, see
   [External Implementations](#external-implementations) section for more details).

### Usage in Namespace Packages

Each namespace package's `setup.py` imports and uses these shared utilities, for example:

```python
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
from accvlab_build_config import (
    load_config,
    detect_cuda_info,
    get_compile_flags,
)

# Load config and generate flags
config = load_config()
cuda_info = detect_cuda_info()
compile_flags = get_compile_flags(config, cuda_info)

# Example: wire flags into extensions
ext_modules = [
    CppExtension(
        name="accvlab.<pkg>._cpp",
        sources=["accvlab/<pkg>/csrc/cpp_functions.cpp"],
        extra_compile_args=compile_flags['cxx'],
        language="c++",
        verbose=config["VERBOSE_BUILD"],
    ),
    CUDAExtension(
        name="accvlab.<pkg>._cuda",
        sources=[
            "accvlab/<pkg>/csrc/cuda_functions.cpp",
            "accvlab/<pkg>/csrc/cuda_functions.cu",
        ],
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

- External implementation (manual CMake):
  - In `ext_impl/build_and_copy.sh`, forward variables using the helper to produce `cmake -D` args:
  ```bash
  readarray -t CMAKE_ARGS < <(python -c "from accvlab_build_config.helpers.cmake_args import build_cmake_args_from_env; print('\n'.join(build_cmake_args_from_env()))")
  cmake .. "${CMAKE_ARGS[@]}" ...
  cmake --build . --parallel
  ```
  - In `CMakeLists.txt`, avoid hardcoding and guard defaults so passed values win:
  ```cmake
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  ```

- Scikit-build packages:
  - In `setup.py`, pass CMake arguments from the helper:
  ```python
  from accvlab_build_config.helpers.cmake_args import build_cmake_args_from_env
  _cmake_args = build_cmake_args_from_env()
  setup(
      ...,
      cmake_source_dir="ext_impl",
      cmake_install_dir="accvlab/<pkg>",
      cmake_args=_cmake_args,
  )
  ```
  - In `ext_impl/CMakeLists.txt`, guard defaults:
  ```cmake
  if(NOT DEFINED CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
  endif()
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
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
> not from `packages/<package_name>`, as in this way, importing the package is ambiguous.
> For example, the import `import accvlab.<package_name>` could mean either refer to the installed package, 
> or to the original source files inside the current directory).

> **ℹ️ Note**: See the section [Repository test runner](#repository-test-runner-scriptsrun_testssh) for how to run 
> tests across all packages in the repository.


#### Install in Development Mode

This can be done with
```bash
pip install -e . --no-build-isolation
```

Note that
- The editable mode refers to the Python code. For any changes in C++ code to take effect, the package
  needs to be re-installed regardless whether it is installed in editable mode or not.
- SKBuild does not support editable installs, and instead, the SKBuild-based packages need to be
  always be installed without the `-e` flag.

### Documentation Development

> **⚠️ Important**: Ensure all configured namespace packages are installed before building the docs.
> For detailed instructions (commands and development targets), see the 
> [Documentation Setup Guide](DOCUMENTATION_SETUP_GUIDE.md#building-documentation-locally).

## Code Formatting

The ACCV-Lab project uses automated code formatting to maintain consistent code style across all namespace 
packages. Please see the [Formatting Guide](FORMATTING_GUIDE.md) for details.

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
the `pyproject.toml` & `setup.py` files, external C++ implementations or CMake projects as used by SKBuild, 
documentation, etc. The remainder of the path (i.e. `[...]/accvlab/example_package/`) reflects the package 
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
