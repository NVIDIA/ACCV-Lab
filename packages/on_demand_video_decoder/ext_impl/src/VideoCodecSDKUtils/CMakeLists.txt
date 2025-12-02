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

project(VideoCodecSDKUtils LANGUAGES CXX CUDA)

set(USE_NVTX $ENV{USE_NVTX})

set(CODEC_SOURCES
 helper_classes/NvCodec/NvDecoder/NvDecoder.cpp
 helper_classes/NvCodec/NvEncoder/NvEncoder.cpp
)
set(CODEC_HDRS
 helper_classes/NvCodec/NvDecoder/NvDecoder.h
 helper_classes/NvCodec/NvEncoder/NvEncoder.h
 helper_classes/Utils/NvCodecUtils.h
 helper_classes/Utils/FFmpegDemuxer.h
 helper_classes/Utils/ColorSpace.h
 helper_classes/Utils/FFmpegStreamer.h
 helper_classes/Utils/Logger.h
 helper_classes/Utils/cuvidFunctions.h
 helper_classes/Utils/cuvid_function_pointer.h
 helper_classes/Utils/cuvid_dlopen.h
 helper_classes/Utils/cuvid_dlopen_unix.cpp
 helper_classes/Utils/cuvid_dlopen_windows.cpp
 helper_classes/Utils/
 Interface/cuviddec.h
 Interface/nvcuvid.h
 Interface/nvEncodeAPI.h
)

set(CODEC_CUDA_UTILS
 helper_classes/Utils/ColorSpace.cu
)

if(WIN32)
    set(NV_FFMPEG_HDRS ${FFMPEG_DIR}/include)
endif()

find_package(CUDAToolkit 11.2 REQUIRED)
include(GenerateExportHeader)

set(CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_50,code=\"sm_50,compute_50\")
# CUDA 11.7 and later is compiled with /MT option (staticaly linked with C runtime), 
# since we are statically linking cudart_static.lib with the app, 
# we need to make sure that all .obj files linked are compiled with /MT to avoid linker warning LNK4098
if (WIN32 AND CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 7)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-Xcompiler /MT)
endif()
if ( CMAKE_COMPILER_IS_GNUCC )
    if(NOT "${CUDA_NVCC_FLAGS}" MATCHES "-std=c\\+\\+11" )
        list(APPEND CUDA_NVCC_FLAGS -std=c++11)
    endif()
endif()

if(MSVC)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

if (WIN32 AND CUDA_VERSION_MAJOR GREATER_EQUAL 11 AND CUDA_VERSION_MINOR GREATER_EQUAL 7)
    target_compile_options(${PROJECT_NAME} PRIVATE "/MT")
endif()

add_library(
    VideoCodecSDKUtils
    ${CODEC_SOURCES}
    ${CODEC_HDRS}
    ${NV_FFMPEG_HDRS}
    ${CODEC_CUDA_UTILS}
)

set(TC_VERSION_MAJOR 1)
set(TC_VERSION_MINOR 0)

generate_export_header(VideoCodecSDKUtils)

target_link_libraries(
    VideoCodecSDKUtils
    PUBLIC
        CUDA::cuda_driver
)


target_compile_features(VideoCodecSDKUtils PRIVATE cxx_std_17)
set_property(
    TARGET VideoCodecSDKUtils
    PROPERTY
        # required for shared Python modules in case library is build statically on Unix
        POSITION_INDEPENDENT_CODE
        ON
)

set(VIDEO_CODEC_INTERFACE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/Interface"
    CACHE PATH "Path to Video Codec SDK interface headers"
)
target_include_directories(
    VideoCodecSDKUtils
    PUBLIC
        helper_classes/NvCodec
        helper_classes/Utils
        ${VIDEO_CODEC_INTERFACE_DIR} # TODO: check if it can be made private!
        ${CMAKE_CURRENT_BINARY_DIR}
)

if(NOT EXISTS ${VIDEO_CODEC_INTERFACE_DIR}/nvEncodeAPI.h)
    message(
        FATAL_ERROR
        "Could not find nvEncodeAPI.h! "
        "Please set VIDEO_CODEC_INTERFACE_DIR=\"${VIDEO_CODEC_INTERFACE_DIR}\" to a location that contains nvEncodeAPI.h!"
    )
endif()

if(UNIX)
    target_link_libraries(VideoCodecSDKUtils PUBLIC pthread)
endif(UNIX)

include(../cmake/ffmpeg.cmake)
if(NV_FFMPEG_HAS_BSF)
    target_compile_definitions(VideoCodecSDKUtils PUBLIC -DHAS_BSF=1)
endif()
target_include_directories(VideoCodecSDKUtils PUBLIC ${FFMPEG_DIR}/include)
target_link_libraries(VideoCodecSDKUtils PUBLIC ${NV_FFMPEG_LIBRARIES})
message("LINKTEST  ${NV_FFMPEG_LIBRARIES}")
option(USE_NVTX "Use NVTX for profiling" FALSE)

if(USE_NVTX)
   target_link_libraries(VideoCodecSDKUtils PRIVATE nvtx3-cpp)
   target_compile_definitions(VideoCodecSDKUtils PRIVATE -DUSE_NVTX=1)
endif()

message(STATUS "HAS_BSF=${HAS_BSF}")
