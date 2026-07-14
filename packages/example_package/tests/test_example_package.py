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

import pytest
import torch
import accvlab.example_package


def test_examples_import():
    assert accvlab.example_package is not None


def test_cpp_extension_available():
    import accvlab.example_package._cpp

    assert hasattr(accvlab.example_package._cpp, "vector_sum")


def test_cuda_extension_available():
    import accvlab.example_package._cuda

    assert hasattr(accvlab.example_package._cuda, "vector_multiply")


def test_cpp_vector_sum():
    x = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    result = accvlab.example_package.cpp_vector_sum(x)
    assert result == 6.0


def test_cuda_vector_multiply():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).cuda()
    b = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32).cuda()
    result = accvlab.example_package.cuda_vector_multiply(a, b)
    expected = torch.tensor([2.0, 6.0, 12.0], dtype=torch.float32).cuda()
    assert torch.allclose(result, expected)


if __name__ == "__main__":
    pytest.main()
