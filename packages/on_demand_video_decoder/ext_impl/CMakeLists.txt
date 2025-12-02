# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.20)

# ============================================================================
# CUDA Architecture Configuration
# ============================================================================
# WARNING: When building via `pip wheel` or `python setup.py`, the 
# `accvlab_build_config.build_cmake_args_from_env()` function will AUTO-DETECT
# the current machine's GPU and pass -DCMAKE_CUDA_ARCHITECTURES=<detected_arch>
# to CMake. This OVERRIDES the default values below!
#
# To build for multiple GPU architectures (recommended for distribution):
#   export CUSTOM_CUDA_ARCHS="60,70,75,80,86,89"
#   pip wheel . -w ./
#
# The defaults below are ONLY used when:
#   1. Building directly with CMake (not via pip/setup.py), AND
#   2. CMAKE_CUDA_ARCHITECTURES is not set via command line or environment
# ============================================================================
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES
        60  # Pascal (P100, GTX 10xx)
        70  # Volta (V100)
        75  # Turing (T4, RTX 20xx)
        80  # Ampere (A100)
        86  # Ampere (A40, A10, RTX 30xx)
        89  # Ada (L40, RTX 40xx)
    )
endif()

project(accvlab.on_demand_video_decoder)
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()

if(DEFINED ENV{FFMPEG_DIR})
    set(FFMPEG_DIR "$ENV{FFMPEG_DIR}" CACHE PATH "path for ffmpeg installed")
else()
    set(FFMPEG_DIR "/usr/local" CACHE PATH "path for ffmpeg installed")
endif()

# Validate that FFmpeg is actually installed at FFMPEG_DIR
message(STATUS "Checking for FFmpeg at: ${FFMPEG_DIR}")

# Check for FFmpeg header files
if(NOT EXISTS "${FFMPEG_DIR}/include/libavcodec/avcodec.h")
    message(FATAL_ERROR 
        "FFmpeg not found at ${FFMPEG_DIR}\n"
        "  Expected header file: ${FFMPEG_DIR}/include/libavcodec/avcodec.h\n"
        "  Please either:\n"
        "    1. Install FFmpeg to ${FFMPEG_DIR}, or\n"
        "    2. Set FFMPEG_DIR environment variable to your FFmpeg installation path:\n"
        "       export FFMPEG_DIR=/path/to/ffmpeg\n"
        "    3. You can install FFMPEG followed by `packages/on_demand_video_decoder/docs/installation.md`"
        "  After that, you can `rm -rf _skbuild/` and rebuild the project."
    )
endif()

# Check for FFmpeg library directory
if(NOT EXISTS "${FFMPEG_DIR}/lib")
    message(FATAL_ERROR 
        "FFmpeg library directory not found at ${FFMPEG_DIR}/lib\n"
        "  Please ensure FFmpeg is properly installed with libraries.\n"
    )
endif()

message(STATUS "FFmpeg found at: ${FFMPEG_DIR}")

set(NV_FFMPEG_LIBRARIES  "${CMAKE_CURRENT_SOURCE_DIR}/src/VideoCodecSDKUtils/Interface" CACHE PATH "path for VideoCodecSDKUtils ffmpeg path")
set(NVTX_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/NVTX/c/include" CACHE PATH "path for nvtx head file") 
set(DLPACK_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/dlpack" CACHE PATH "path for dlpack lib") 
set(PYBIND11_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/external/pybind11" CACHE PATH "path for pybind11 lib") 

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Set CMake variable
    set(IS_DEBUG_BUILD "true")
    # Set pre-processor macro
    add_definitions(-DIS_DEBUG_BUILD)
endif()

add_subdirectory(src)

#If we need to run with santizer, the way is ASAN_OPTIONS=protect_shadow_gap=0:replace_intrin=0:detect_leaks=1 ./cpp_samples/run_decoding
#set (CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")
#set (CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

message("${CMAKE_BUILD_TYPE}")
