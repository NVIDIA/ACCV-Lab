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

pybind11_add_module(accvlab_lane_helpers_polyline_sampling MODULE
    src/polyline.cu
    src/polyline_cpu.cpp
    src/polyline.cpp
)

set_target_properties(accvlab_lane_helpers_polyline_sampling PROPERTIES
    CXX_STANDARD 17
    CUDA_STANDARD 17
    OUTPUT_NAME "_polyline_sampling"
    PREFIX ""
)

target_compile_definitions(accvlab_lane_helpers_polyline_sampling PRIVATE
    TORCH_EXTENSION_NAME=_polyline_sampling
    TORCH_API_INCLUDE_EXTENSION_H
)

target_compile_options(accvlab_lane_helpers_polyline_sampling PRIVATE ${TORCH_CXX_FLAGS_LIST})

target_link_libraries(accvlab_lane_helpers_polyline_sampling PRIVATE
    ${TORCH_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

target_include_directories(accvlab_lane_helpers_polyline_sampling PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    include/
)

install(TARGETS accvlab_lane_helpers_polyline_sampling
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
)
