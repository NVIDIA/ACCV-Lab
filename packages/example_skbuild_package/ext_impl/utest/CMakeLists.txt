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

set(ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_GTEST_SOURCE_DIR
    "${CMAKE_CURRENT_LIST_DIR}/../external/googletest"
)

if(NOT EXISTS "${ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_GTEST_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "GoogleTest submodule not found. Run: git submodule update --init --recursive")
endif()

add_subdirectory(
    "${ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_GTEST_SOURCE_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/googletest"
    EXCLUDE_FROM_ALL
)

add_executable(example_skbuild_package_dummy_native_test EXCLUDE_FROM_ALL
    dummy_native_test.cpp
)

target_link_libraries(example_skbuild_package_dummy_native_test PRIVATE
    GTest::gtest_main
)

add_test(
    NAME example_skbuild_package_dummy_native_test
    COMMAND example_skbuild_package_dummy_native_test
)
set_tests_properties(example_skbuild_package_dummy_native_test PROPERTIES
    LABELS "example_skbuild_package_native"
)

set(ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_CTEST_CONFIG_ARGS)
if(CMAKE_CONFIGURATION_TYPES)
    set(ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_CTEST_CONFIG_ARGS -C $<CONFIG>)
endif()

add_custom_target(accvlab_example_skbuild_package_run_cpp_tests
    COMMAND ${CMAKE_CTEST_COMMAND}
            --output-on-failure
            ${ACCVLAB_EXAMPLE_SKBUILD_PACKAGE_CTEST_CONFIG_ARGS}
            -L example_skbuild_package_native
    DEPENDS example_skbuild_package_dummy_native_test
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running example_skbuild_package native tests"
    COMMAND_EXPAND_LISTS
)
