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

project(draw_heatmap_benchmark_cpp)

cmake_minimum_required(VERSION 3.14)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
# Find required packages
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
            "import os; import torch; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake'))"
    OUTPUT_VARIABLE TORCH_CMAKE_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PATH}")
find_package(Torch REQUIRED)
separate_arguments(TORCH_CXX_FLAGS_LIST NATIVE_COMMAND "${TORCH_CXX_FLAGS}")

include_directories(${Python3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../accvlab/draw_heatmap/include)

add_executable(bench_flattened 
    bench_flattened.cpp
    ../accvlab/draw_heatmap/csrc/draw_heatmap_cuda.cu
)
add_executable(bench_batched 
    bench_batched.cpp
    ../accvlab/draw_heatmap/csrc/draw_heatmap_cuda.cu
)
add_executable(bench_batched_classwise 
    bench_batched_classwise.cpp
    ../accvlab/draw_heatmap/csrc/draw_heatmap_cuda.cu
)
target_compile_options(bench_flattened PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_compile_options(bench_batched PRIVATE ${TORCH_CXX_FLAGS_LIST})
target_compile_options(bench_batched_classwise PRIVATE ${TORCH_CXX_FLAGS_LIST})

target_link_libraries(bench_flattened ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
target_link_libraries(bench_batched ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})
target_link_libraries(bench_batched_classwise ${TORCH_LIBRARIES} ${CUDA_LIBRARIES} ${CUDA_CUDART_LIBRARY})



