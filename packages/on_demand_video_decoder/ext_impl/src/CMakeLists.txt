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

# NOTE: CMAKE_CUDA_ARCHITECTURES is inherited from parent ext_impl/CMakeLists.txt
# To customize, set CUSTOM_CUDA_ARCHS environment variable before building.

project(accvlab.on_demand_video_decoder)

#Undef MIN & MAX & C runtime warnings for Windows
if(WIN32)
    add_definitions(-DNOMINMAX)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif(WIN32)

option(PROCESS_SYNC "Run the demuxer and decoder in synchornous" OFF)
set(PROCESS_SYNC $ENV{PROCESS_SYNC})
if (PROCESS_SYNC)
  add_compile_definitions(PROCESS_SYNC)
endif()

message("PROCESS_SYNC environment variable: $ENV{PROCESS_SYNC}")

option(USE_NVTX "enable nvtx support" FALSE)
set(USE_NVTX $ENV{USE_NVTX})

find_package(CUDAToolkit 11.2 REQUIRED)

add_subdirectory(VideoCodecSDKUtils)
add_subdirectory(PyNvOnDemandDecoder)
