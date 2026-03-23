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

import accvlab.optim_test_tools.numba_nvtx as nvtx
from numba import njit

# @NOTE
# Register range names outside the JIT function to obtain integer handles.
# Handles must be created before compilation because Numba compiles the function
# and the handle value is baked into the compiled code.
h_example_range = nvtx.register_string("example_range")
h_example_range_inner = nvtx.register_string("example_range_inner")


@njit
def compute(x):
    # @NOTE: Push the range using the handle created above.
    nvtx.range_push(h_example_range)
    y = x - 1
    # @NOTE: Push the inner range.
    nvtx.range_push(h_example_range_inner)
    y = x + 2
    # @NOTE: Pop both ranges.
    nvtx.range_pop()
    nvtx.range_pop()
    return y


if __name__ == "__main__":
    result = compute(41)
