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

project(accvlab.on_demand_video_decoder VERSION 0.1.0 LANGUAGES CXX CUDA)

set(PYNVONDEMANDDEC_VERSION_MAJOR 0)
set(PYNVONDEMANDDEC_VERSION_MINOR 1)
set(PYNVONDEMANDDEC_VERSION_PATCH 0)

if(POLICY CMP0135)
  #  https://cmake.org/cmake/help/latest/policy/CMP0135.html
  # From docs:
  # CMake 3.24 and above prefers to set the timestamps of all extracted contents to the time of the extraction.
  # This ensures that anything that depends on the extracted contents will be rebuilt whenever the URL changes.
  cmake_policy(SET CMP0135 NEW)
endif()

configure_file("inc/Version.hpp.in" "pynod_version.h")

find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter Development)


add_subdirectory(${PYBIND11_SOURCE_DIR}
                 ${CMAKE_BINARY_DIR}/external/pybind11)

message(STATUS "FFmpeg installed path:\"${FFMPEG_DIR}/")

set(PY_SOURCES
      src/PyNvOnDemandDecoder.cpp
      src/PyCAIMemoryView.cpp
      src/PyDecodedFrameExt.cpp
      src/PyNvGopDecoder_common.cpp
      src/PyNvGopDecoder_constructors.cpp
      src/PyNvGopDecoder_random_decoder.cpp
      src/PyNvGopDecoder_separate_decoder.cpp
      src/PyNvVideoReader.cpp
      src/PyNvSampleReader.cpp
      src/PyNvGopDemuxer.cpp
      src/PyRGBFrame.cpp
      src/GPUMemoryPool.cpp
      src/ColorConvertKernels.cu
      src/DLPackUtils.cpp
      src/ExternalBuffer.cpp
  )
set(PY_HDRS
      inc
      ${FFMPEG_DIR}/include/
)

if (DEFINED IS_DEBUG_BUILD)
    pybind11_add_module(_PyNvOnDemandDecoder SHARED 
        ${PY_SOURCES}
    )
else ()
    pybind11_add_module(_PyNvOnDemandDecoder MODULE 
        ${PY_SOURCES}
    ) 
endif()

if(DEFINED CMAKE_CXX_STANDARD)
  set_property(TARGET _PyNvOnDemandDecoder PROPERTY CXX_STANDARD ${CMAKE_CXX_STANDARD})
endif()
target_include_directories(_PyNvOnDemandDecoder 
    PUBLIC 
    ${PY_HDRS}
)

target_include_directories(_PyNvOnDemandDecoder PUBLIC ${NVTX_INCLUDE_DIR})

target_include_directories(_PyNvOnDemandDecoder PRIVATE "${DLPACK_SOURCE_DIR}/include")

target_link_libraries(_PyNvOnDemandDecoder PUBLIC VideoCodecSDKUtils)
set_target_properties(_PyNvOnDemandDecoder PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(_PyNvOnDemandDecoder PROPERTIES BUILD_RPATH "$ORIGIN")
# Prefer DT_RPATH over DT_RUNPATH so embedded rpath takes precedence
if (CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_link_options(_PyNvOnDemandDecoder PRIVATE -Wl,--disable-new-dtags)
endif()
# Do not augment INSTALL_RPATH with link-time paths
set_target_properties(_PyNvOnDemandDecoder PROPERTIES INSTALL_RPATH_USE_LINK_PATH FALSE)

include(GNUInstallDirs)
# Install runtime dependencies (i.e. FFMPEG, nppi DLLS) on Windows but not Linux



if(WIN32)
  message(STATUS "FFMPEG_DIR/bin/=${FFMPEG_DIR}/bin/")
  install(TARGETS _PyNvOnDemandDecoder
      RUNTIME_DEPENDENCIES DIRECTORIES "${FFMPEG_DIR}/lib/"
               PRE_EXCLUDE_REGEXES "api-ms-" "ext-ms-" "python" "nvcuda"
               POST_EXCLUDE_REGEXES ".*system32/.*\\.dll"
      RUNTIME DESTINATION .
      LIBRARY DESTINATION .
  )
else()
  install(TARGETS _PyNvOnDemandDecoder
      RUNTIME DESTINATION .
      LIBRARY DESTINATION .
  )
  message("======Not WIN32=====")  
  install(
  	DIRECTORY 
  	"${FFMPEG_DIR}/lib/" 
  	DESTINATION .)

endif()

