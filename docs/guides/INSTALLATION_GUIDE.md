# Installation Guide

This guide covers how to install and use ACCV-Lab, a CUDA-accelerated library with multiple namespace packages.

## Prerequisites

### Environment Setup

Before installing ACCV-Lab, please ensure that the environment is set up correctly. The recommended way is to
use the Docker image as described in the [Docker Guide](DOCKER_GUIDE.md).

You can also have a look at the [Dockerfile](../../docker/Dockerfile) for more details on the base
environment.

### Submodules

ACCV-Lab uses submodules in some of the packages (currently only the `on_demand_video_decoder` package).
To clone the repository with the submodules, you can use the following command:
```bash
git clone --recurse-submodules https://github.com/NVIDIA/ACCV-Lab.git
```
If you have already cloned the repository without the submodules, you can add them later with the following 
command:
```bash
git submodule update --init --recursive
```

## Installation Methods

### 1. Installation Using the Package Manager Script

#### Overview

> **⚠️ Important**: The editable installation (`-e`) is not supported for scikit-build based packages
> (e.g. `on_demand_video_decoder` or `dali_pipeline_framework`). This can lead to missing binaries
> and import errors.

The standard way to install ACCV-Lab is using the unified installer script that handles all namespace packages 
automatically. By default it installs packages with their **basic** dependencies only; to also install optional
dependencies (needed for some tests and examples), pass the `--optional` flag explicitly:

```bash
# Install all namespace packages
./scripts/package_manager.sh install

# Install in development mode (editable installation)
./scripts/package_manager.sh install -e

# Install with optional dependencies
./scripts/package_manager.sh install --optional

# Install in development mode with optional dependencies
./scripts/package_manager.sh install -e --optional
```

> **⚠️ Important**: Installing with optional dependencies is required for workflows that rely on packages
> outside the core library, including contained tests, contained examples, and documentation generation.
> Documentation generation may run package-local asset hooks, for example to regenerate plots from committed
> data, and those hooks can require plotting or data-processing packages. Tests commonly require tools such as
> `pytest` and may require further packages.

The package manager script:
- Automatically installs the required `accvlab_build_config` helper package (see the `build_config` directory
  in the repository root)
- Installs all configured namespace packages from `namespace_packages_config.py` (see the 
  [development guide](DEVELOPMENT_GUIDE.md) for more details)
- Installs the individual namespace packages with `pip install` and the `--no-build-isolation` flag by 
  default. This is the recommended way to install the packages. However, you can pass `--with-build-isolation` to the script if
  you want pip to use build isolation; see [Installing with Build Isolation](#installing-with-build-isolation) for the
  required precautions.
- Tests imports after installation
- Provides detailed progress feedback


### 2. Installation Using the Convenience Wrapper Script

**Alternative**: You can also use the convenience wrapper `install_local.sh` which calls the package manager 
automatically and **performs a single default install with optional dependencies enabled**.

```bash
# Install all packages with optional dependencies (default local setup)
./scripts/install_local.sh
```

> **ℹ️ Note**: `install_local.sh` does not accept any parameters. If you need fine-grained control (e.g.,
> installing **without** optional dependencies, using editable installs, or building wheels), use
> `scripts/package_manager.sh` directly instead of `install_local.sh`.


### 3. Building and Installing Wheels

You can also build the wheels:

```bash
# Build wheels for all namespace packages
./scripts/package_manager.sh wheel

# Build wheels with optional dependencies
./scripts/package_manager.sh wheel --optional

# Build wheels in a specific directory
./scripts/package_manager.sh wheel -o /path/to/wheels
```

The wheel building script:
- Creates wheels for all namespace packages
- Saves wheels to `./wheels/` directory by default
- Supports various build configurations for different deployment scenarios
- Passes `--no-build-isolation` to `pip wheel` by default. This means that the resulting wheel will be built in the current
  environment, which is the preferred approach. You can pass `--with-build-isolation` to the script if you want pip to use
  build isolation; see [Installing with Build Isolation](#installing-with-build-isolation) for the required precautions.
- Uses `--no-deps` by default. This means the wheelhouse will contain wheels for the ACCV-Lab packages only, not wheels for
  their dependencies. You can pass `--with-deps` to also include dependency wheels using the default `pip wheel` behavior.

> **ℹ️ Note**: Wheel versions are derived from git metadata via `setuptools-scm`. To get meaningful version
> numbers, build from a repository checkout where `.git`, tags, and sufficient history are available. If git
> metadata is missing (for example in a source export or shallow CI checkout without tags), the package version
> may fall back to `0.0.0`.

> **ℹ️ Note**: Even in `wheel` mode, the script installs or updates the `accvlab-build-config` helper package
> in the active Python environment before building the wheels, because that helper is used by the other
> ACCV-Lab packages during the build. No wheel is generated for `accvlab-build-config`, because it is only
> needed while building the ACCV-Lab package wheels (or installing from source), not when installing from already generated 
> wheels.

#### Installing from Built Wheels

After building wheels, you can install them:

```bash
# Install all wheels from the wheels directory
pip install wheels/*.whl

# Install specific wheels
pip install wheels/accvlab_optim_test_tools-*.whl
pip install wheels/accvlab_batching_helpers-*.whl
pip install wheels/accvlab_dali_pipeline_framework-*.whl
pip install wheels/accvlab_on_demand_video_decoder-*.whl
```

### 4. Installing Individual Packages with `pip`

For development or when you only need specific packages, you can install them individually directly with 
`pip`.

The examples below use `--no-build-isolation`, matching the package manager's default and preferred behavior. For isolated
builds, omit `--no-build-isolation` when calling `pip` directly; see
[Installing with Build Isolation](#installing-with-build-isolation) for the required precautions when building in this way.

> **ℹ️ Note**: `{-e}` means that the `-e` (editable) option is optional.

> **ℹ️ Note**: The `-e` option is not supported for scikit-build based packages (e.g. 
> `dali_pipeline_framework`, `on_demand_video_decoder`).

```bash
# Install individual packages
cd packages/optim_test_tools && pip install {-e} . --no-build-isolation
cd packages/batching_helpers && pip install {-e} . --no-build-isolation
cd packages/dali_pipeline_framework && pip install . --no-build-isolation
cd packages/on_demand_video_decoder && pip install . --no-build-isolation
```

#### Installing with Optional Dependencies

For individual package installation with optional dependencies:

```bash
# Install individual packages with optional dependencies
cd packages/optim_test_tools && pip install {-e} .[optional] --no-build-isolation
cd packages/batching_helpers && pip install {-e} .[optional] --no-build-isolation
cd packages/dali_pipeline_framework && pip install .[optional] --no-build-isolation
cd packages/on_demand_video_decoder && pip install .[optional] --no-build-isolation
```

## Installing with Build Isolation

By default, the ACCV-Lab package manager disables pip build isolation. This is the recommended approach: It reuses the active
Python environment when building, ensuring compatibility by using the exact installed packages, especially packages with
version-, CUDA-, or ABI-sensitive build behavior such as PyTorch and Nvidia DALI.

> **⚠️ Important**: Installing ACCV-Lab without build isolation is recommended. It uses the same (i.e. the current) environment 
> for building and running ACCV-Lab, keeping build-time and runtime dependencies consistent without the need for the extra 
> precautions described in this section.

If you enable build isolation, `pip` creates a temporary build environment and installs each package's
`[build-system].requires` dependencies into that environment. This is more self-contained, but it also means `pip` resolves
build dependencies independently of the active Python environment. For version-sensitive dependencies such as PyTorch (`torch`)
and Nvidia DALI (`nvidia-dali-cuda120`), this can select versions that are ABI-incompatible or built for a different CUDA
setup than the versions you plan to use ACCV-Lab with. For PyTorch specifically, the selected wheel may be CPU-only or it may
be CUDA-enabled with the default CUDA runtime for that package index and PyTorch release.

For PyTorch, avoiding this mismatch involves:
- Manually adjusting the build dependencies in `pyproject.toml` files to ensure that the correct `torch` version is used.
- Setting the `pip` package indices so that the pulled `torch` package corresponds to the correct CUDA version.

These two steps control different parts of the selected PyTorch package. Pinning `torch==2.6.0` in
`[build-system].requires` constrains the PyTorch version used in the isolated build environment, but it does not specify the
CUDA wheel variant. Without explicit index configuration, `pip` may select a CPU-only wheel or a CUDA-enabled wheel with a
predefined CUDA runtime for that package index and PyTorch release. Configuring `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL`
influences which wheel variant `pip` selects, for example a CUDA 12.4 wheel instead of the default wheel contained in the current 
index.

The PyTorch-specific steps and other version-sensitive build dependencies are described in the remainder of this section.

> **ℹ️ Note**: Installation with build isolation is currently not supported when using `uv`.

### Ensuring the Correct PyTorch Version

> **ℹ️ Note**: The steps described here are only needed when building with build isolation.

By default `pip` will automatically resolve the `torch` version, typically choosing the newest version compatible with the
declared `[build-system].requires` dependency. However, this may not be the version you use in your environment, and its wheel
variant may be CPU-only or may use a different default CUDA runtime. Building ACCV-Lab against a mismatched `torch` version or
CUDA runtime may break the PyTorch custom extensions used in the individual ACCV-Lab packages.

This build dependency is separate from `[project].dependencies`, which controls the package's runtime dependencies (and which
will not trigger a `torch` installation if your version already satisfies the specified version range). To avoid
build-time mismatches, set the version of the `torch` build dependency manually in the individual `pyproject.toml` files of
the contained packages (sub-directories of the `packages` directory in the repo root). Make this adjustment for every package
that declares `torch` in `[build-system].requires`.

For example, the following adjustments need to be made if the required torch version is `2.6.0`:

```toml
# Example from: packages/example_package/pyproject.toml
[build-system]
requires = [
    "setuptools>=64",
    "wheel",
    "torch==2.6.0",  # For build-isolated installs only. Original: "torch>=2.0.0"
    "pybind11>=2.10.0",
    "setuptools-scm>=8",
    "accvlab-build-config @ file:../../build_config",
]
# ... rest of the file remains unchanged
```

Also pin the `torch` entry in `build_config/pyproject.toml` under `[project].dependencies`, because the
`accvlab-build-config` helper package is used at build time by the other packages.

```toml
# build_config/pyproject.toml
[project]
dependencies = [
    "torch==2.6.0",  # Original: "torch>=2.0.0"
    "setuptools-scm>=8",
]
# ... rest of the file remains unchanged
```

> **ℹ️ Note**: The `torch` version specifier alone does not select a CUDA-enabled wheel or a specific CUDA runtime. Please see
> [Setting up `pip` Indices for CUDA-enabled PyTorch](#setting-up-pip-indices-for-cuda-enabled-pytorch)
> below for more details on how to handle this.

### Setting up `pip` Indices for CUDA-enabled PyTorch

> **ℹ️ Note**: The steps described here are only needed when building with build isolation.

Configure `pip`'s standard index environment variables so the isolated build environment installs
the desired CUDA-enabled wheel variant. For example, for CUDA 12.4 PyTorch wheels:

```bash
PIP_INDEX_URL=https://download.pytorch.org/whl/cu124 \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
./scripts/package_manager.sh install --with-build-isolation
```

The same pattern works when building wheels:

```bash
PIP_INDEX_URL=https://download.pytorch.org/whl/cu124 \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
./scripts/package_manager.sh wheel --with-build-isolation
```

It also works with direct `pip` commands:

```bash
cd packages/batching_helpers
PIP_INDEX_URL=https://download.pytorch.org/whl/cu124 \
PIP_EXTRA_INDEX_URL=https://pypi.org/simple \
pip install .
```

`PIP_INDEX_URL` selects the main package index. `PIP_EXTRA_INDEX_URL` keeps PyPI available for non-PyTorch build
dependencies that are not hosted on the PyTorch wheel index.

> **⚠️ Important**: `pip` considers candidates from both `PIP_INDEX_URL` and `PIP_EXTRA_INDEX_URL`, so it is
> not guaranteed to select `torch` from the PyTorch index. If the selected PyTorch wheel is CPU-only or uses an unexpected CUDA
> runtime, first check whether the required `torch` version is available in the selected PyTorch index. If the issue persists, use 
> the default installation flow without build isolation instead.

### Other Version-sensitive Build Dependencies

> **ℹ️ Note**: The steps described here are only needed when building with build isolation.

Other build dependencies can have similar version-mismatch risks. For example, `packages/dali_pipeline_framework` uses
NVIDIA DALI when building a custom DALI operator (`nvidia-dali-cuda120` in `[build-system].requires`) and when running the
package (`nvidia-dali-cuda120` in `[project].dependencies`). The custom operator is linked against the DALI version present
during the build, and DALI's ABI may change between versions.

If the build-time and runtime DALI versions differ, the resulting wheel may fail to import or run correctly. If you build this
package with build isolation, pin the DALI dependency consistently in both locations. For example, if the required DALI version
is `1.51.2`:

```toml
# packages/dali_pipeline_framework/pyproject.toml
[build-system]
requires = [
    # ... other entries remain unchanged
    "nvidia-dali-cuda120==1.51.2",  # Original: "nvidia-dali-cuda120>=1.51.2"
]

[project]
dependencies = [
    # ... other entries remain unchanged
    "nvidia-dali-cuda120==1.51.2",  # Original: "nvidia-dali-cuda120>=1.51.2"
]
```

## Verifying Installation

### Basic Verification

Test that the packages installed correctly:

```bash
# Test basic import
python -c "import accvlab; print('ACCV-Lab loaded successfully')"

# Test specific namespace packages
python -c "import accvlab.optim_test_tools; print('Optim test tools loaded successfully')"
python -c "import accvlab.batching_helpers; print('Batching helpers loaded successfully')"
python -c "import accvlab.dali_pipeline_framework; print('DALI pipeline framework loaded successfully')"
python -c "import accvlab.on_demand_video_decoder; print('On-demand video decoder loaded successfully')"
```

### Check Installed Versions

Each top-level ACCV-Lab package exposes `__version__`, and you can also query the installed distribution
metadata directly, e.g.:

```bash
# Check package-level __version__
python -c "import accvlab.on_demand_video_decoder as pkg; print(pkg.__version__)"
```

> **ℹ️ Note**: `accvlab` itself is an implicit namespace package and therefore does not provide
> `accvlab.__version__`. Query a concrete package such as `accvlab.on_demand_video_decoder.__version__`,
> or use `importlib.metadata.version(...)` with a distribution name.

### Check Available Namespace Packages

```bash
# List all configured namespace packages
python -c "from namespace_packages_config import get_namespace_packages; print('\n'.join(get_namespace_packages()))"
```

### Running all Unit Tests

The repository provides a convenience script to run pytest for all configured namespace packages:

```bash
./scripts/run_tests.sh
```

> **⚠️ Important**: If you want to run the tests, please make sure to install the packages with optional 
> dependencies, as they may be required for the tests.

> **⚠️ Important**: If you want to run the tests inside a docker container, you need to 
> install and use the Nvidia container runtime to ensure that the `on_demand_video_decoder` package can be 
> used. 
> Please see the [Docker Guide](DOCKER_GUIDE.md) for more details on how to set up and run with the Nvidia 
> container runtime.
>
> Alternatively, you can also remove the `on_demand_video_decoder` package from the installation (by removing 
> it from the list of namespace packages in the `namespace_packages_config.py` file, also see the 
> [Development Guide](DEVELOPMENT_GUIDE.md)).

## Build Configuration

You can customize the build process using environment variables. Note that this works across the package manager (including the
convenience wrapper script), wheel builds, and direct `pip` installs.

```bash
# Debug build with verbose output
DEBUG_BUILD=1 VERBOSE_BUILD=1 ./scripts/package_manager.sh install

# Optimized build for production
OPTIMIZE_LEVEL=3 USE_FAST_MATH=1 ./scripts/package_manager.sh install

# Custom CUDA architectures (if you need to override auto-detection)
CUSTOM_CUDA_ARCHS="70,75,80" ./scripts/package_manager.sh install

# Enable profiling support
ENABLE_PROFILING=1 ./scripts/package_manager.sh install
```

> **ℹ️ Note**: These build variables are honored across all build types in ACCV-Lab: setuptools (PyTorch 
> extensions), external CMake builds (via the provided helper script), and scikit-build packages.

### Available Build Variables

| Variable | Type/Values | Default | Description |
|----------|-------------|---------|-------------|
| `DEBUG_BUILD` | bool: `0`/`1`, `true`/`false`, `yes`/`no`, `on`/`off` | `0` | Enable debug symbols and assertions |
| `OPTIMIZE_LEVEL` | int: `0`–`3` | `3` | Compiler optimization level |
| `CPP_STANDARD` | string: `c++17` | `c++17` | C++ standard to use |
| `VERBOSE_BUILD` | bool: `0`/`1`, `true`/`false`, `yes`/`no`, `on`/`off` | `0` | Show detailed build output |
| `CUSTOM_CUDA_ARCHS` | list: e.g. `"70,75,80"` or `"75;80;86"` | PyTorch auto-detect, then package default | Explicit CUDA architecture override |
| `USE_FAST_MATH` | bool: `0`/`1`, `true`/`false`, `yes`/`no`, `on`/`off` | `1` | Enable fast math optimizations |
| `ENABLE_PROFILING` | bool: `0`/`1`, `true`/`false`, `yes`/`no`, `on`/`off` | `0` | Enable profiling support |

> **⚠️ Important**: Currently only C++17 is supported across all packages and toolchains. Set 
> `CPP_STANDARD=c++17`. Using newer standards (e.g., C++20) may not be supported for CUDA builds for some 
> of the packages.

> **⚠️ Important**: If `CUSTOM_CUDA_ARCHS` is not set, ACCV-Lab tries to auto-detect
> GPU architectures via CUDA-enabled PyTorch. Missing PyTorch or CPU-only PyTorch is treated as a build
> configuration error.
>
> Auto-detected architectures are emitted as real/cubin targets only when the installed `nvcc` exactly supports
> them. If a detected architecture is unsupported, ACCV-Lab emits a supported PTX target below the detected
> architecture, preferring base architectures whose number is divisible by 10 (for example, `100` for a
> detected `103` architecture).
>
> `CUSTOM_CUDA_ARCHS` is an explicit override. When it is set, ACCV-Lab passes those architectures through
> unchanged instead of applying the auto-detection fallback logic.
>
> If PyTorch is CUDA-enabled but no architecture can be detected
> (for example because no CUDA device is visible), ACCV-Lab does not pass `CMAKE_CUDA_ARCHITECTURES`;
> package-specific CMake defaults then apply.

## Additional Information

For information about extending ACCV-Lab or adding new namespace packages, see the 
[development guide](DEVELOPMENT_GUIDE.md).
