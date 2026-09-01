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

set(ACCVLAB_ON_DEMAND_VIDEO_DECODER_GTEST_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../external/googletest"
)

if(NOT EXISTS "${ACCVLAB_ON_DEMAND_VIDEO_DECODER_GTEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GoogleTest submodule not found. Run: git submodule update --init --recursive")
endif()

add_subdirectory(
    "${ACCVLAB_ON_DEMAND_VIDEO_DECODER_GTEST_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/googletest"
    EXCLUDE_FROM_ALL
)

find_package(Threads REQUIRED)

add_executable(on_demand_video_decoder_thread_pool_test EXCLUDE_FROM_ALL
    thread_pool_test.cpp
)

target_include_directories(on_demand_video_decoder_thread_pool_test PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/../src/PyNvOnDemandDecoder/inc"
)

target_link_libraries(on_demand_video_decoder_thread_pool_test PRIVATE
    GTest::gtest_main
    Threads::Threads
)

add_test(
    NAME on_demand_video_decoder_thread_pool_test
    COMMAND on_demand_video_decoder_thread_pool_test
)
set_tests_properties(on_demand_video_decoder_thread_pool_test PROPERTIES
    LABELS "on_demand_video_decoder_native"
)

set(ACCVLAB_ON_DEMAND_VIDEO_DECODER_CTEST_CONFIG_ARGS)
if(CMAKE_CONFIGURATION_TYPES)
    set(ACCVLAB_ON_DEMAND_VIDEO_DECODER_CTEST_CONFIG_ARGS -C $<CONFIG>)
endif()

add_custom_target(accvlab_on_demand_video_decoder_run_cpp_tests
    COMMAND ${CMAKE_CTEST_COMMAND}
            --output-on-failure
            ${ACCVLAB_ON_DEMAND_VIDEO_DECODER_CTEST_CONFIG_ARGS}
            -L on_demand_video_decoder_native
    DEPENDS on_demand_video_decoder_thread_pool_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running on_demand_video_decoder native tests"
    COMMAND_EXPAND_LISTS
)
