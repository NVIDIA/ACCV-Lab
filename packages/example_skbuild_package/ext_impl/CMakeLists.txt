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

cmake_minimum_required(VERSION 3.18)
project(accvlab_example_skbuild_package_ext LANGUAGES CXX CUDA)

if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch CMake path
execute_process(
    COMMAND "python3" -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Add PyTorch CMake path to CMAKE_PREFIX_PATH
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")

# Find CUDA
find_package(CUDA REQUIRED)

# PyTorch CMake ignores CMAKE_CUDA_ARCHITECTURES and uses TORCH_CUDA_ARCH_LIST.
# setup.py passes cuda_arch_strategy=TORCH via build_cmake_args().
if(DEFINED ACCVLAB_TORCH_CUDA_ARCH_LIST)
  set(TORCH_CUDA_ARCH_LIST "${ACCVLAB_TORCH_CUDA_ARCH_LIST}")
endif()

# Find PyTorch
find_package(Torch REQUIRED)

# Find Python
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Get pybind11 CMake directory
execute_process(
    COMMAND "${Python_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Find pybind11
find_package(pybind11 REQUIRED)

# Set CUDA architectures (allow override via -DCMAKE_CUDA_ARCHITECTURES)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
    set(CMAKE_CUDA_ARCHITECTURES native)
  else()
    # Fallback list for older CMake versions without 'native'
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86")
  endif()
endif()


# Create the PyTorch extension using pybind11_add_module
pybind11_add_module(accvlab_example_skbuild_package_ext MODULE
    src/external_cuda_ops.cu
    src/external_cuda_ops.cpp
)

# Set CUDA properties and PyTorch compatibility
set_target_properties(accvlab_example_skbuild_package_ext PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CXX_STANDARD 17
    CUDA_STANDARD 17
    OUTPUT_NAME "_ext"
    PREFIX ""
)

# Add PyTorch extension definitions
target_compile_definitions(accvlab_example_skbuild_package_ext PRIVATE 
    TORCH_EXTENSION_NAME=_ext
    TORCH_API_INCLUDE_EXTENSION_H
)

# Add PyTorch C++ compilation flags
target_compile_definitions(accvlab_example_skbuild_package_ext PRIVATE ${TORCH_CXX_FLAGS})

# Find torch_python library path
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS ${TORCH_INSTALL_PREFIX}/lib)

# Link libraries
target_link_libraries(accvlab_example_skbuild_package_ext PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

# Include directories
target_include_directories(accvlab_example_skbuild_package_ext PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    include/
)

# Install target
install(TARGETS accvlab_example_skbuild_package_ext
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
) 