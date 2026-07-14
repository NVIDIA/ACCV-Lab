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
import accvlab.example_skbuild_package


def test_examples_import():
    assert accvlab.example_skbuild_package is not None


def test_external_module_available():
    try:
        import accvlab.example_skbuild_package._ext

        assert accvlab.example_skbuild_package._ext is not None
    except ImportError:
        pytest.fail("External module _ext not found.")


def test_cuda_vector_add():
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).cuda()
    b = torch.tensor([2.0, 3.0, 4.0], dtype=torch.float32).cuda()
    # Use the public wrapper function; it calls `_ext.external_vector_add_cuda` internally.
    result = accvlab.example_skbuild_package.vector_add(a, b)
    expected = torch.tensor([3.0, 5.0, 7.0], dtype=torch.float32).cuda()
    assert torch.allclose(result, expected)


def test_cuda_vector_scale():
    a = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).cuda()
    scale = 2.5
    # Use the public wrapper function; it calls `_ext.external_vector_scale_cuda` internally.
    result = accvlab.example_skbuild_package.vector_scale(a, scale)
    expected = torch.tensor([2.5, 5.0, 7.5], dtype=torch.float32).cuda()
    assert torch.allclose(result, expected)
