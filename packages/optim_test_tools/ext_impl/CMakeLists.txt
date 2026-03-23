# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18)
project(accvlab_optim_test_tools_numba_ext LANGUAGES CXX)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Get pybind11 CMake directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

find_package(pybind11 REQUIRED)

# NVTX headers are provided via a git submodule (mirrors on_demand_video_decoder, but separate instance).
set(ACCVLAB_NVTX_INCLUDE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/external/NVTX/c/include"
)

if(NOT EXISTS "${ACCVLAB_NVTX_INCLUDE_DIR}/nvtx3/nvToolsExt.h")
  message(FATAL_ERROR "NVTX submodule not found. Run: git submodule update --init --recursive")
endif()

pybind11_add_module(accvlab_optim_test_tools_numba_ext MODULE
    src/nvtx_numba.cpp
)

set_target_properties(accvlab_optim_test_tools_numba_ext PROPERTIES
    CXX_STANDARD 17
    OUTPUT_NAME "_nvtx_numba_ext"
    PREFIX ""
)

target_include_directories(accvlab_optim_test_tools_numba_ext PRIVATE
    "${ACCVLAB_NVTX_INCLUDE_DIR}"
)

install(TARGETS accvlab_optim_test_tools_numba_ext
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)


