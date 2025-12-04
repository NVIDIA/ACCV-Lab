# Installation

## Overview

This package is distributed as part of the ACCV-Lab package, and can be installed as described in the 
[Installation Guide](../../../guides/INSTALLATION_GUIDE.md).

> **ℹ️ Note**: In most cases, it is recommended to follow the the general ACCV-Lab installation guide (see 
> above) to install the package.

## Independent Installation

If needed, the package can be installed independently of the ACCV-Lab package and the default environment
using the steps outlined in this section.

### Prerequisites
- **CUDA**: >= 11.0
- **Operating System**: Linux
- **Python**: 3.8+
- **cmake**: >= 3.21
- **scikit-build**

### Step 1: Pull Base Image
Choose an appropriate NVIDIA PyTorch container from 
[NVIDIA Deep Learning Framework Documentation](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html). 
Images released after 22.11 are required for successful compilation.

**Example using PyTorch 23.02:**
```bash
docker pull nvcr.io/nvidia/pytorch:23.02-py3
```

### Step 2: Install Dependencies

#### Install ``scikit-build``
```bash
pip install scikit-build
```

#### Install ``accvlab_build_config``
```bash
cd build_config  # ← Assuming you are in the root directory of the repository
pip install . --no-build-isolation
```

#### Build and Install ``FFmpeg``
[FFMpeg](https://ffmpeg.org/) version==4.4 has been tested, and we use this version here.
```bash
apt install yasm nasm -y
wget https://ffmpeg.org/releases/ffmpeg-4.4.6.tar.xz
tar -xf ffmpeg-4.4.6.tar.xz

cd ffmpeg-4.4.6
# FFmpeg Libraries used in this package can be built using the following configure options. The configuration 
# ensures that the FFmpeg libraries are built with the minimum required functionality to support 
# the On-demand Video Decoder Package, while also ensuring that only the components of FFmpeg that are 
# licensed under the LGPL license is used.
LDSOFLAGS=-Wl,-rpath,\''$$$$ORIGIN'\'  ./configure --enable-shared --disable-encoders --disable-decoders --enable-decoder=vp9 --arch=x86_64 --prefix=/PATH/TO/YOUR/FFMPEG

make -j
make install
# The FFMPEG_DIR path is used by the On-demand Video Decoder Package to find the FFmpeg libraries during 
# build time. 
export FFMPEG_DIR=/PATH/TO/YOUR/FFMPEG
```

### Step 3: Build and Install
```bash
cd packages/on_demand_video_decoder
pip install . --no-build-isolation
```

## Dataset Preparation
Please refer to the [Dataset Preparation Guide](dataset_preparation.md) for instructions on how to prepare 
the NuScenes (mini) dataset for testing and profiling the video decoder, as well as for potential use 
of the video dataset for training purposes.

## Development

### Debugging
For development and debugging purposes, you can trigger a debug build via the environment variables described 
in the main [Installation Guide](../../../guides/INSTALLATION_GUIDE.md). 
The `pip install` command will internally run the CMake/scikit-build build, so you do not need to call 
`cmake` or `make` manually:

```bash
cd packages/on_demand_video_decoder
# Debug build with verbose output
DEBUG_BUILD=1 VERBOSE_BUILD=1 pip install . --no-build-isolation
```

### Performance Profiling
Use NVIDIA Nsight Systems for comprehensive performance analysis:

```bash
nsys profile \
    --trace-fork-before-exec true \
    -w true \
    -f true \
    -t cuda,nvtx,osrt,cudnn,cublas,nvvideo \
    --gpu-video-device all \
    -x true \
    -o report \
    python samples/ProfileNuscenesGopDecoder.py
```

**Profile Options Explained:**
- `--trace-fork-before-exec true`: Enable fork tracing
- `-w true`: Overwrite existing output files
- `-f true`: Force profiling
- `-t`: Specify trace categories
- `--gpu-video-device all`: Profile all GPU video devices
- `-x true`: Export results
- `-o`: Output file prefix
