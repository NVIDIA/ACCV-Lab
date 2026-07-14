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

set(ACCVLAB_LANE_HELPERS_GTEST_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../../external/googletest"
)

if(NOT EXISTS "${ACCVLAB_LANE_HELPERS_GTEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GoogleTest submodule not found. Run: git submodule update --init --recursive")
endif()

add_subdirectory(
    "${ACCVLAB_LANE_HELPERS_GTEST_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/googletest"
    EXCLUDE_FROM_ALL
)

set(ACCVLAB_LANE_HELPERS_POLYLINE_TEST_TARGETS)

add_library(accvlab_lane_helpers_polyline_native_test_impl STATIC EXCLUDE_FROM_ALL
    ../src/polyline.cu
    ../src/polyline_cpu.cpp
)

target_compile_options(accvlab_lane_helpers_polyline_native_test_impl PRIVATE ${TORCH_CXX_FLAGS_LIST})

target_include_directories(accvlab_lane_helpers_polyline_native_test_impl PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    "${CMAKE_CURRENT_LIST_DIR}/../include"
)

target_link_libraries(accvlab_lane_helpers_polyline_native_test_impl PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

function(accvlab_lane_helpers_add_polyline_test target_name source_file)
    add_executable(${target_name} EXCLUDE_FROM_ALL ${source_file})

    target_compile_options(${target_name} PRIVATE ${TORCH_CXX_FLAGS_LIST})

    target_include_directories(${target_name} PRIVATE
        ${TORCH_INCLUDE_DIRS}
        ${CUDA_INCLUDE_DIRS}
        "${CMAKE_CURRENT_LIST_DIR}/../include"
    )

    target_link_libraries(${target_name} PRIVATE
        GTest::gtest_main
        accvlab_lane_helpers_polyline_native_test_impl
        ${TORCH_LIBRARIES}
        ${CUDA_LIBRARIES}
        ${TORCH_PYTHON_LIBRARY}
    )

    add_test(NAME ${target_name} COMMAND ${target_name})
    set_tests_properties(${target_name} PROPERTIES LABELS "lane_helpers_polyline_native")

    list(APPEND ACCVLAB_LANE_HELPERS_POLYLINE_TEST_TARGETS ${target_name})
    set(ACCVLAB_LANE_HELPERS_POLYLINE_TEST_TARGETS
        ${ACCVLAB_LANE_HELPERS_POLYLINE_TEST_TARGETS}
        PARENT_SCOPE
    )
endfunction()

accvlab_lane_helpers_add_polyline_test(polyline_host_api_test polyline_host_api_test.cu)
accvlab_lane_helpers_add_polyline_test(single_kernel_binary_search_test single_kernel_binary_search_test.cu)
accvlab_lane_helpers_add_polyline_test(single_kernel_compute_distances_test single_kernel_compute_distances_test.cu)
accvlab_lane_helpers_add_polyline_test(single_kernel_kernel_test single_kernel_kernel_test.cu)
accvlab_lane_helpers_add_polyline_test(single_kernel_kernel_test_batch2 single_kernel_kernel_test_batch2.cu)
accvlab_lane_helpers_add_polyline_test(single_kernel_prefix_scan_test single_kernel_prefix_scan_test.cu)
accvlab_lane_helpers_add_polyline_test(single_kernel_sample_distances_test single_kernel_sample_distances_test.cu)

set(ACCVLAB_LANE_HELPERS_POLYLINE_CTEST_CONFIG_ARGS)
if(CMAKE_CONFIGURATION_TYPES)
    set(ACCVLAB_LANE_HELPERS_POLYLINE_CTEST_CONFIG_ARGS -C $<CONFIG>)
endif()

add_custom_target(accvlab_lane_helpers_polyline_run_cpp_tests
    COMMAND ${CMAKE_CTEST_COMMAND}
            --output-on-failure
            ${ACCVLAB_LANE_HELPERS_POLYLINE_CTEST_CONFIG_ARGS}
            -L lane_helpers_polyline_native
    DEPENDS ${ACCVLAB_LANE_HELPERS_POLYLINE_TEST_TARGETS}
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running lane_helpers polyline native tests"
    COMMAND_EXPAND_LISTS
)
