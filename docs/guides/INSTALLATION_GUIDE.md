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

> **⚠️ Important**: Installing with optional dependencies is required if you plan to run the contained
> tests, as they rely on optional dependencies such as `pytest` (and possibly other dependencies). It may be 
> also required for the contained examples, as they may use additional packages which are otherwise 
> not used in the core library.

The package manager script:
- Automatically installs the required `accvlab_build_config` helper package (see the `build_config` directory
  in the repository root)
- Installs all configured namespace packages from `namespace_packages_config.py` (see the 
  [development guide](DEVELOPMENT_GUIDE.md) for more details).
- Installs the individual namespace packages with `pip install` and the `--no-build-isolation` flag by 
  default. You can pass `--with-build-isolation` to the script if you want pip to use build isolation.
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
- Includes the `build_config` helper package wheel
- Supports various build configurations for different deployment scenarios
- Uses `--no-build-isolation` by default. This means that the resulting wheel will be built in the current 
  environment. You can pass `--with-build-isolation` to the script if you want pip to use build isolation.
- Uses `--no-deps` by default. This means that only a wheel for the package itself will be built, and no 
  wheels for dependencies will be prepared. You can pass `--with-deps` to the script if you want to use the 
  default `pip wheel` behavior instead (i.e. prepare wheels for all dependencies as well).

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

You can customize the build process using environment variables. Note that this works for any of the 
installation methods described above.

```bash
# Debug build with verbose output
DEBUG_BUILD=1 VERBOSE_BUILD=1 ./scripts/package_manager.sh install

# Optimized build for production
OPTIMIZE_LEVEL=3 USE_FAST_MATH=1 ./scripts/package_manager.sh install

# Custom CUDA architectures (if you know your GPU architecture)
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
| `CUSTOM_CUDA_ARCHS` | list: e.g. `"70,75,80"` or `"75;80;86"` | Auto-detect | Target CUDA architectures (overrides auto-detect) |
| `USE_FAST_MATH` | bool: `0`/`1`, `true`/`false`, `yes`/`no`, `on`/`off` | `1` | Enable fast math optimizations |
| `ENABLE_PROFILING` | bool: `0`/`1`, `true`/`false`, `yes`/`no`, `on`/`off` | `0` | Enable profiling support |

> **⚠️ Important**: Currently only C++17 is supported across all packages and toolchains. Set 
> `CPP_STANDARD=c++17`. Using newer standards (e.g., C++20) may not be supported for CUDA builds for some 
> of the packages.

> **⚠️ Important**: If `CUSTOM_CUDA_ARCHS` is not set, auto-detection via PyTorch is attempted. If this fails, 
> the build will use CMake's native detection (`CMAKE_CUDA_ARCHITECTURES=native`) to select the appropriate 
> GPU architectures on the current system.

## Additional Information

For information about extending ACCV-Lab or adding new namespace packages, see the 
[development guide](DEVELOPMENT_GUIDE.md).
