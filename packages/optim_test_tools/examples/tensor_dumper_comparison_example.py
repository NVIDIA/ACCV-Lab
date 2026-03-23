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

import os
import torch
from accvlab.optim_test_tools import TensorDumper

# @NOTE
# In this example, we do not divide the code into different parts which correspond to e.g. different source
# files in the actual use case, to make the example more concise. However, as the `TensorDumper` is a
# singleton, this can be easily done in practice. Please see the stopwatch example ("stopwatch_example.py")
# or the nvtx range wrapper example ("nvtx_range_wrapper_example.py") for examples of how to do this. The
# same approach can be used with the tensor dumper.

# @NOTE: Get instance and enable the dumper.
_current_dir = os.path.dirname(os.path.abspath(__file__))
dumper = TensorDumper()
dumper.enable(os.path.join(_current_dir, "test_dump"))

# @NOTE: Test configuration. You can play around with this configuration and see how it affects the comparison.

# Set to `True` to make the tensor data for `i==1` inconsistent with the reference data (`i==0`)
make_tensor_data_inconsistent = True
# Set to `True` to make the stored structure inconsistent with the reference data
make_structure_inconsistent = True
# Set to `True` to allow missing data in the current data (one of the two inconsistencies introduced if
# `make_structure_inconsistent` is `True`)
allow_missing_data_in_current = False

# @NOTE
# Here, we dump tensor in iteration 0 and compare to the dumped data in iteration 1.
# In a typical use-case, one would dump the data in one run, and then compare to the dumped data in subsequent
# runs, while e.g. working on optimizations, to ensure that the optimizations do not introduce errors (which
# would lead to a mismatch in the dumped data).
for i in range(2):
    # @NOTE: Generate data to dump/compare to
    if not make_tensor_data_inconsistent:
        torch.manual_seed(42)
    tensor1 = torch.randn(10, 10)
    tensor2 = torch.randn(10, 10)
    tensor3 = torch.randn(10, 10)

    # @NOTE: Add a single tensor to be dumped/compared
    dumper.add_tensor_data("tensor1", tensor1, TensorDumper.Type.JSON, None)
    # To introduce mismatches, set `make_structure_inconsistent` to `True`
    if i == 1 and make_structure_inconsistent:
        # @NOTE
        # Add a dictionary of tensors to be dumped/compared. Note that the BINARY format is used here.
        # See the API documentation of the `TensorDumper.Type` enum for more details.
        dumper.add_tensor_data(
            "other_tensors", {"tensor2": tensor2, "tensor4": tensor3}, TensorDumper.Type.BINARY, None
        )
    else:
        # @NOTE: Add a dictionary of tensors to be dumped/compared
        dumper.add_tensor_data(
            "other_tensors", {"tensor2": tensor2, "tensor3": tensor3}, TensorDumper.Type.BINARY, None
        )

    # @NOTE
    # Add gradients of tensor to be dumped/compared. Note that we need to ensure that the gradients are
    # actually required by the tensor. If this is not the case, no gradients will be computed and the dump
    # will contain the note that no gradients are needed (instead of the actual gradients).
    tensor1.requires_grad = True
    dumper.add_grad_data("tensor1", tensor1, TensorDumper.Type.JSON, None)
    sin_tensor1 = torch.sin(tensor1)
    sum_sin_tensor1 = torch.sum(sin_tensor1)

    # @NOTE
    # Overwrite the dump type for all tensors and gradients to JSON to ensure that the reference data is
    # stored in the JSON format (which is the only format supported for comparison).
    # This is not needed if the dump type for all tensors and gradients is set to JSON when generating the
    # reference data. However, as we set some of the data to be in BINARY format, we need to override this to
    # use the comparison functionality.
    #
    # In practice, this may be useful to switch between dumping the data for manual inspection (originally
    # selected formats) or the automatic comparison functionality used in this example (all data in JSON
    # format).
    dumper.set_dump_type_for_all(TensorDumper.Type.JSON)

    # @NOTE
    # To set the gradients (for the tensors for which gradients are dumped), we need to call the
    # `set_gradients()` method and provide a sequence of scalars (typically loss values) from which to compute
    # the gradients.
    dumper.set_gradients([sum_sin_tensor1])

    # @NOTE: Dump or compare the data (depending on the iteration).
    if i == 0:
        dumper.dump()
    else:
        # @NOTE
        # There are some parameters which can be used to configure the comparison. Please see the API
        # documentation for more details.
        dumper.compare_to_dumped_data(
            num_errors_per_tensor_to_show=3,
            as_warning=True,
            allow_missing_data_in_current=allow_missing_data_in_current,
        )

    # @NOTE
    # Note that the dumper allows for dumping/comparison of dumped data from multiple iterations. This can be
    # useful e.g. when the dumping/comparison spans multiple training iterations. Each time `dump()` is
    # called, the iteration count is incremented, and data is stored in separate sub-directories.
    #
    # However, here we have a special case: we want to pretend that the two iterations of the loop are two
    # different runs (the original run storing the data, and a following run where data is compared).
    # Therefore, reset the dump count to start from the first dump again (i.e. the next iteration `i==1` will
    # be compared to the first dump `i==0`).
    #
    # Note that in the typical use case (comparison of two runs), this resetting is not needed.
    # However, it may be useful for debugging e.g. to rerun the same code multiple times to check for
    # determinism, while always comparing to the same dumped data.
    dumper.reset_dump_count()
