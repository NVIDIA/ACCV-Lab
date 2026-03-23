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

import pytest


def test_import_numba_nvtx_module():
    import accvlab.optim_test_tools.numba_nvtx as nvtx

    assert hasattr(nvtx, "register_string")
    assert hasattr(nvtx, "range_push")
    assert hasattr(nvtx, "range_pop")


def test_numba_njit_calls_do_not_crash():
    numba = pytest.importorskip("numba")

    import accvlab.optim_test_tools.numba_nvtx as nvtx

    h = nvtx.register_string("test_range")
    assert isinstance(h, int)

    @numba.njit
    def f(x):
        nvtx.range_push(h)
        y = x + 1
        nvtx.range_pop()
        return y

    assert f(41) == 42


if __name__ == "__main__":
    pytest.main([__file__])
