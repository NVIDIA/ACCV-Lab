# ACCV-Lab Overview

Accelerated Computer Vision Lab (ACCV-Lab) is a systematic collection of packages with the common goal to 
facilitate end-to-end efficient training in the ADAS domain, each package offering tools & best practices for 
a specific aspect/task in this domain.

## Resources

The following resources provide an overview of the project and its capabilities:

| Type | Title | Date | Link |
|------|-------|------|------|
| 📝 Blog | Accelerate Assisted Driving Training with NVIDIA ACCV-Lab Open-Source Toolkit 【CN】 | 2025-12-24 | [NVIDIA Tech Blog](https://developer.nvidia.cn/blog/accelerate-assisted-driving-training-with-nvidia-accv-lab-open-source-toolkit/) |
| 🎬 Video | ADAS & Robotics - Overview of ACCV-Lab for Data Loading and Model Training Acceleration 【EN】 | 2025-11-19 | [Bilibili](https://www.bilibili.com/video/BV14yyEB3E5B/) |
| 🎬 Video | Video Training for ADAS 【CN】 and DALI-Based Image Preprocessing Pipeline 【EN】 | 2025-06-12 | [Bilibili](https://www.bilibili.com/video/BV1LMMwzNEnF/) |
| 🎬 Video | Batching Helpers: Optimizing Loss Computation 【EN】 | 2025-06-12 | [Bilibili](https://www.bilibili.com/video/BV1dKMwzvEQ5/) |

## Contained Packages

This section provides a brief overview of the contained packages & evaluations performed on them. Please also 
see the `CONTAINED PACKAGES` section in the documentation for a more detailed description of the packages, 
their functionality and usage, as well as the API reference for each package and examples demonstrating the 
usage of the packages. Evaluation results are also provided for some of the packages (On-Demand Video Decoder, 
Batching Helpers, DALI Pipeline Framework).

> **ℹ️ Note**: We are planning to add demos for packages contained in ACCV-Lab in the future. Apart from 
> acting as tutorials show-casing real-world examples, they will include the implementation of the experiments 
> which were used to obtain the evaluation results as shown in the documentation of the contained packages.

The contained packages are:
- **On-Demand Video Decoder**: Designed for efficiently obtaining video frames from video files. Focused on 
  the ADAS training domain, implementing access patterns typical for training workloads to ensure high 
  throughput and low latency. Enables training directly from videos with performance similar to image-based 
  training.
- **Batching Helpers**: Facilitates easy-to-implement batching for non‑uniform sample sizes, a common issue in 
  loss computation in the ADAS domain.
- **DALI Pipeline Framework**: Framework on top of 
  [NVIDIA DALI](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/) that simplifies creation of 
  pipelines for typical ADAS use‑cases and enables integration into existing training implementations with 
  minimal changes.
- **Draw Heatmap**: Utility for drawing Gaussian heatmaps (e.g. for object detection training) on the GPU.
  Optionally, this package can process data in the batched format defined in the Batching Helpers package.
- **Optim Test Tools**: Utilities for testing and profiling the optimizations. This functionality can be used
  during development and evaluation of optimizations.

For more details on the packages (including performance evaluation results), please see the documentation for 
each package in the `CONTAINED PACKAGES` section of the documentation.

> **⚠️ Important**: If you are reading this in the `README.md` file (and not in the HTML documentation), 
> please note that there is a full documentation available, containing this page, the guides, and additional 
> content (e.g. API reference, examples with detailed explanations, ...). You can view the documentation 
> online at [NVIDIA.github.io/ACCV-Lab](https://NVIDIA.github.io/ACCV-Lab), or build a local copy as outlined 
> in the `Quick Start` section below.

(project_overview-quick-start)=
## Quick Start

A docker file setting up the environment to install and use the ACCV-Lab package is available in the 
`docker/` directory. It allows building an image and using it as root (build without any parameters),
but also allows to configure a specific user (see the Dockerfile for the parameters).

The image can be built as follows (run from inside the `docker/` directory):

```bash
# Build container to be used as root
docker build -t image_name:tag .

# Build container for the current user
docker build \
  --build-arg USER_ID=$(id -u) \
  --build-arg USER_NAME=$(id -un) \
  --build-arg GROUP_ID=$(id -g) \
  --build-arg GROUP_NAME=$(id -gn) \
  -t image_name:tag .
```

> **⚠️ Important**: If you want to run the On-Demand Video Decoder package inside the docker container, you 
> need to install and use the Nvidia container runtime.
> Please see the [Docker Guide](../guides/DOCKER_GUIDE.md) for more details.

Before installing the packages, you need to make sure that git submodules are initialized. This can e.g. be 
done by running the following command in your cloned version of the repository:
```bash
git submodule update --init --recursive
```

There are scripts available for the common tasks such as installing the contained packages or building the 
documentation. The basic commands are:

```bash
# Install all namespace packages
./scripts/install_local.sh

# Install in development mode (see "Important" note below)
./scripts/install_local.sh -e

# Build documentation. This needs to be done after installing the packages, as the generation of the API 
# documentation relies on the installed packages.
./scripts/build_docs.sh
```

> **⚠️ Important**: Note that for some of the contained namespace packages, an editable build (i.e. 
> development mode) may not be possible. This is the case if the package uses `scikit-build`. In this case, 
> the package will be installed, but the binaries will be missing, leading to import errors when using the 
> package. It is recommended to only use the editable installation while working on specific packages.

> **⚠️ Important**: Before generating the documentation, the package needs to be installed, as the generation 
> of the API documentation relies on the package being available (i.e. it should be possible to import it 
> during the generation). Note that installing the package also ensures that binaries of the used C++ 
> extensions are available, so that any docstrings defined there can also be extracted. 

> **ℹ️ Note**: Pre-built documentation is available online at 
> [NVIDIA.github.io/ACCV-Lab](https://NVIDIA.github.io/ACCV-Lab).

## Project Structure

- `packages/` - Individual namespace packages (each with their own `setup.py`)
- `build_config/` - Shared build utilities and configuration
- `docs/` - Documentation generation implementation & common (i.e. not package-specific) part of documentation
- `scripts/` - Development and build scripts
- `docs/guides/` - Guides on how to work with this repository (also included in the documentation)
- `docker/` - Dockerfile for setting up a container able to run `ACCV-Lab`

## Available Guides

> **ℹ️ Note**: The following guides are also available in the documentation (in the `Guides` section).
> If you are reading this in the `README.md` file, we recommend building the documentation first (see the 
> `Quick Start` section above) for better formatting and navigation, as well as to have access to the full
> documentation.

The following guides provide comprehensive information for different aspects of the project:

### 🚀 [Installation Guide](../guides/INSTALLATION_GUIDE.md)
Step-by-step installation instructions for users and developers.

### 🐳 [Docker Guide](../guides/DOCKER_GUIDE.md)
Guide for setting up a container able to run `accvlab` from the provided Dockerfile and how to use it.

### 📚 [Development Guide](../guides/DEVELOPMENT_GUIDE.md)
Complete guide for developers on project structure, adding new namespace packages, and working with the build 
system.

### 🎨 [Formatting Guide](../guides/FORMATTING_GUIDE.md)
Guide to code formatting standards and tools. Explains how to use the unified formatting script and maintain 
consistent code style across all namespace packages.

### 📖 [Documentation Setup Guide](../guides/DOCUMENTATION_SETUP_GUIDE.md)
Comprehensive guide for setting up and maintaining project documentation. Covers Sphinx configuration, 
automatic documentation generation, and best practices for documenting namespace packages.

### 🤝 [Contribution Guide](../guides/CONTRIBUTION_GUIDE.md)
Guide for contributors on how to contribute to the project.


## Namespace Packages

The project includes several namespace packages, each providing specific functionality.
Please see the `CONTAINED PACKAGES` section in the documentation for details on the packages.

Each namespace package contains its own documentation, C++ extensions, etc. and can be installed independently
(see the [Installation Guide](../guides/INSTALLATION_GUIDE.md) for details).

## Development

For development information, see the [Development Guide](../guides/DEVELOPMENT_GUIDE.md) which covers:
- Project architecture and namespace package structure
- Adding new namespace packages
- Build system configuration
- Testing and documentation workflows

## Contributing

1. Follow the [Development Guide](../guides/DEVELOPMENT_GUIDE.md) for development environment setup
2. Use the [Formatting Guide](../guides/FORMATTING_GUIDE.md) to maintain code style
3. Ensure documentation is updated following the 
   [Documentation Setup Guide](../guides/DOCUMENTATION_SETUP_GUIDE.md)

Please also see the [Contribution Guide](../guides/CONTRIBUTION_GUIDE.md) for more a more detailed description

## Useful Links

- [Online Documentation](https://NVIDIA.github.io/ACCV-Lab)
- [WeChat Discussion Group](https://github.com/NVIDIA/ACCV-Lab/issues/2): A real-time channel for ACCV-Lab Q&A and news.

## License

This project is licensed under the Apache License, Version 2.0. See the `LICENSE` file for details.
