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
import numpy as np

from accvlab.batching_helpers import RaggedBatch

from accvlab.optim_test_tools import TensorDumper

# @NOTE
# In this example, we do not divide the code into different parts which correspond to e.g. different source
# files in the actual use case, to make the example more concise. However, as the `TensorDumper` is a
# singleton, this can be easily done in practice. Please see the stopwatch example ("stopwatch_example.py")
# or the nvtx range wrapper example ("nvtx_range_wrapper_example.py") for examples of how to do this. The
# same approach can be used with the tensor dumper.


# ------------------------- Helper: Create synthetic inputs -------------------------

# @NOTE
# Here, helper functions are defined to create dummy data which will be dumped. Additionally, a wrapper class
# for demonstrating the custom converter functionality is defined.


def create_simple_gradient_image(
    height: int = 256, width: int = 256, blue_channel_value: float = 0.5
) -> torch.Tensor:
    """Create a tensor representing an image with smooth gradients on different channels."""

    # Create coordinate grids as a single tensor
    coords = torch.stack(
        torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing='ij'), dim=-1
    )

    # Create smooth gradients on different channels
    image = torch.zeros(height, width, 3)
    image[:, :, 0] = coords[:, :, 1]  # Red channel: horizontal gradient
    image[:, :, 1] = coords[:, :, 0]  # Green channel: vertical gradient
    image[:, :, 2] = blue_channel_value  # Blue channel: diagonal gradient

    return image


def create_bboxes(num_bboxes: int, image_shape: tuple[int, int]) -> torch.Tensor:
    """Create a tensor representing bounding boxes."""
    bboxes = []
    for _ in range(num_bboxes):
        x1 = torch.randint(0, image_shape[1], (1,))
        y1 = torch.randint(0, image_shape[0], (1,))
        x2 = torch.randint(x1.item(), image_shape[1], (1,))
        y2 = torch.randint(y1.item(), image_shape[0], (1,))
        bboxes.append(torch.tensor([x1, y1, x2, y2]))
    bboxes = torch.stack(bboxes, dim=0)
    return bboxes


# @NOTE: Define a simple wrapper class for demonstrating the custom converter functionality.
class TensorWrapper:
    def __init__(self, tensor: torch.Tensor):
        self.tensor = tensor
        self.some_addition_text = "hello!"


# ------------------- Initialize and configure the dumper -------------------
# @NOTE: Get instance and enable the dumper.
_current_dir = os.path.dirname(os.path.abspath(__file__))
dumper = TensorDumper()
dumper.enable(os.path.join(_current_dir, "test_dump"))
# @NOTE: Configure early‑exit after 3 dumps. This can be useful to only dump a few iterations and exit,
# without any code-changes for the exit functionality. Other functions can also be set up to be called after a
# fixed number of dumps.
dumper.perform_after_dump_count(3, exit)

# ------------------------- Register custom converters -------------------------
# @NOTE
# Register a custom converter for `TensorWrapper`. Any tensors returned by the converter are treated
# the same as tensors added directly via `add_tensor_data`. Converters are applied recursively, i.e. if a
# converter returns types for which other custom converters are registered, these are called in turn.
dumper.register_custom_converter(
    TensorWrapper, lambda x: {"tensor": x.tensor, "some_addition_text": x.some_addition_text}
)

# ------------------------------- Main loop -------------------------------
# @NOTE: While the loop has 10 iterations, the program will exit after 3 dumps due to the configuration above.
for i in range(10):
    # --------------------------- Create the test data ---------------------------
    # @NOTE: Generate simple images, small tensors, and random bounding boxes.
    i_scaled = i * 0.1
    test_image_1 = create_simple_gradient_image(256, 256, 0.0 + i_scaled)
    test_image_2 = create_simple_gradient_image(256, 256, 0.5 + i_scaled)
    test_image_3 = create_simple_gradient_image(256, 256, 1.0 + i_scaled)
    test_image_4 = create_simple_gradient_image(256, 256, 0.25 + i_scaled)
    test_image_5 = create_simple_gradient_image(256, 256, 0.75 + i_scaled)
    test_simple_tensor1 = create_simple_gradient_image(2, 2, 0.0 + i_scaled)
    test_simple_tensor2 = create_simple_gradient_image(2, 2, 0.5 + i_scaled)
    test_bboxes1 = create_bboxes(10, (256, 256))
    test_bboxes2 = create_bboxes(10, (256, 256))
    test_bboxes3 = create_bboxes(10, (256, 256))

    # @NOTE: This is a non‑tensor object that needs custom handling (converter registered above).
    wrapped_tensor1 = TensorWrapper(test_image_1)

    # @NOTE: Mark selected tensors as requiring gradients to demonstrate gradient dumping.
    test_image_3.requires_grad = True
    test_image_5.requires_grad = True
    test_image_1.requires_grad = True

    # @NOTE: Create a combined image
    test_image_combined = torch.zeros(2, 1, 2, 256, 256, 3)
    test_image_combined[0, 0, 0, :, :] = test_image_1
    test_image_combined[0, 0, 1, :, :] = test_image_2
    test_image_combined[1, 0, 0, :, :] = test_image_3
    test_image_combined[1, 0, 1, :, :] = test_image_4
    # @NOTE
    # Add a permutation (to demonstrate the permute axes override, see beow).
    # The inverse permutation to be applied is: (0, 1, 5, 2, 4, 3)
    test_image_combined = torch.permute(test_image_combined, (0, 1, 3, 5, 4, 2))

    # ------------------------------- Add tensors -------------------------------
    # @NOTE: Add a single image at a new path
    dumper.add_tensor_data("images.image_1", test_image_1, TensorDumper.Type.IMAGE_RGB)
    # @NOTE: Add a dictionary of images; override dump type for `bboxes` to JSON while keeping images as RGB.
    dumper.add_tensor_data(
        "images.other_images",
        {"image_2": test_image_2, "image_3": test_image_3, "bboxes": test_bboxes1},
        TensorDumper.Type.IMAGE_RGB,
        dump_type_override={"bboxes": TensorDumper.Type.JSON},
    )
    # @NOTE
    # Add a dictionary of images. This time, set the to be dumped as binary files.
    # Note that
    # 1. `test_image_combined` is inside the structure, but is handled differently (i.e. a permutation is
    #    applied and it is dumped as an image). This is useful to e.g. dump structures which contain many
    #    tensors, but only one or a few of them need to be handled differently. In this way, this can be
    #    achieved within a single call to `add_tensor_data`.
    # 2. `wrapped_tensor1` is a non-tensor object, which needs custom handling.
    #    The custom handling is done by adding a custom extension to the dumper, which is then used to dump
    #    the object (the custom converter is registered above).
    # 3. `unneeded_data` is excluded from the dump.
    #    This is useful to e.g. exclude data which is part of the structure, but either not needed in the 
    #    dump, or which will be added to the dump later via custom processing logic (see below for bounding 
    #    box images).
    dumper.add_tensor_data(
        "images.other_images",
        {
            "binary_image_1": test_image_4,
            "binary_image_2": test_image_5,
            "image_combined": test_image_combined,
            "wrapped_tensor": wrapped_tensor1,
            "unneeded_data": torch.randn(10, 10),
        },
        TensorDumper.Type.BINARY,
        dump_type_override={"image_combined": TensorDumper.Type.IMAGE_RGB},
        permute_axes_override={"image_combined": (0, 1, 5, 2, 4, 3)},
        exclude=["unneeded_data"],
    )
    # @NOTE: Add small tensors dumped directly into main JSON.
    dumper.add_tensor_data(
        "images.other_images",
        {"tensor_1": test_simple_tensor1, "tensor_2": test_simple_tensor2},
        TensorDumper.Type.JSON,
    )

    # ------------------------------- Add gradients ------------------------------
    # @NOTE
    # Add gradients for single tensors and for a dictionary of tensors. Note that the gradients are computed
    # later automatically (loss value(s) need to be provided to the `set_gradients` method, see below).
    dumper.add_grad_data("images.image_1", test_image_1, TensorDumper.Type.IMAGE_RGB, None)
    dumper.add_grad_data(
        "images.other_images",
        {"image_2": test_image_2, "image_3": test_image_3},
        TensorDumper.Type.IMAGE_RGB,
    )
    # @NOTE: Add gradients dumped as binary files.
    dumper.add_grad_data(
        "images.other_images",
        {"binary_image_1": test_image_4, "binary_image_2": test_image_5},
        TensorDumper.Type.BINARY,
    )

    # --------------------- Custom processing prior to dumping --------------------
    # @NOTE
    # Convert bounding boxes to images through custom logic that only runs if the dumper is enabled.
    # This pattern allows flexible transformations or aggregations before dumping, while still incurring no
    # overhead if the dumper is not enabled.
    #
    # Note that the custom processing does not have parameters and instead, uses the enclosing function's
    # scope. This enables writing the custom processing logic as if it runs in-line with the rest of the code,
    # and also avoids the need to pass parameters to the custom processing function.
    def process_and_add_bboxes():
        def draw_bboxes(bboxes: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
            image = torch.zeros(image_shape)
            for bbox in bboxes:
                x1, y1, x2, y2 = bbox
                # Draw the outline of the bbox
                image[y1, x1:x2] = 1  # Top edge
                image[y2 - 1, x1:x2] = 1  # Bottom edge
                image[y1:y2, x1] = 1  # Left edge
                image[y1:y2, x2 - 1] = 1  # Right edge

                # Fill the interior: set to 0.1 if current value is 0.0 and do not change otherwise
                interior_region = image[y1 + 1 : y2 - 1, x1 + 1 : x2 - 1]
                interior_region += 0.1
                interior_region[interior_region > 1.0] = 1.0
            return image

        test_bboxes1_processed = draw_bboxes(test_bboxes1, (256, 256))
        test_bboxes2_processed = draw_bboxes(test_bboxes2, (256, 256))
        test_bboxes3_processed = draw_bboxes(test_bboxes3, (256, 256))
        dumper.add_tensor_data(
            "images.other_images",
            {
                "bboxes1": test_bboxes1_processed,
                "additional_boxes": {
                    "bboxes2": test_bboxes2_processed,
                    "bboxes3": test_bboxes3_processed,
                },
            },
            TensorDumper.Type.IMAGE_I,
        )

    # @NOTE
    # Run the custom processing logic if the dumper is enabled. Note that the function also hanfles calls
    # to `add_tensor_data` internally.
    dumper.run_if_enabled(process_and_add_bboxes)

    # ---------------------------- RaggedBatch dumping ----------------------------
    # @NOTE: Dump RaggedBatch structures both as per‑sample and as full RaggedBatch objects.
    ragged_batch_1 = RaggedBatch(torch.randn(3, 5), sample_sizes=torch.tensor([3, 5, 1]))
    ragged_batch_2 = RaggedBatch(torch.randn(3, 5), sample_sizes=torch.tensor([3, 5, 1]))
    # @NOTE
    # Toggling `as_per_sample` and dumping the data in the desired format (per-sample or as a RaggedBatch
    # structure).
    dumper.enable_ragged_batch_dumping(as_per_sample=True)
    dumper.add_tensor_data("ragged_batches.batch_1", ragged_batch_1, TensorDumper.Type.JSON)
    dumper.enable_ragged_batch_dumping(as_per_sample=False)
    dumper.add_tensor_data("ragged_batches.batch_2", ragged_batch_2, TensorDumper.Type.JSON)

    # --------------------------------- Inner loop --------------------------------
    # @NOTE
    # Here, the same data is added to the dump in multiple iterations. Care needs to be taked to disambiguate
    # the names of the individual data entries. This can be conveniently done using ranges. Ranges will become
    # part of the dump path, and the names of the data entries will be appended to the range name.
    for inner_iteration in range(10):
        # @NOTE
        # The string formatting is performed inside a lambda function instead of passing the formatted string
        # directly. `push_range` supports both strings and callables. If formatting is needed, a callable
        # should be passed as it is called only if the dumper is enabled, avoiding unnecessary overhead for
        # formatting if the dumper is not enabled.
        dumper.push_range(lambda: f"inner_loop_{inner_iteration}")
        dummy_tensor = torch.tensor(inner_iteration)
        # @NOTE: While the path "dummy.tensor" remains the same, the range provides context and ensures
        # that naming collisions are avoided. The range (or ranges) is added at the beginning of the path.
        # For example, the json structure for `inner_iteration == 0` would be:
        #   "inner_loop_0": {
        #     "dummy": {
        #       "tensor": 0
        #   }
        dumper.add_tensor_data("dummy.tensor", dummy_tensor, TensorDumper.Type.JSON)
        dumper.pop_range()

    # ------------------- Placeholder for e.g. loss computation -------------------
    # @NOTE
    # Dummy loss computation to demonstrate auto-computing & dumping of gradients.
    # In a real use case, a final loss value would be used instead of `summed_3` and `summer_5`.
    # Although the final loss is one value, we use `summed_3` and `summed_5` to demonstrate that the
    # gradients can also be computed for more than one output tensor.
    image_sin_3 = torch.sin(test_image_3 * 2.0 * np.pi * 3.0)
    image_sin_5 = torch.sin(test_image_5 * 2.0 * np.pi * 3.0)
    summed_3 = torch.sum(image_sin_3)
    summed_5 = torch.sum(image_sin_5)

    # ----------------------------- Set the gradients -----------------------------
    # @NOTE
    # Provide (scalar) loss tensors from which gradients are computed for all tensors that require them.
    dumper.set_gradients([summed_3, summed_5])

    # ---------------------------------- Dump ----------------------------------
    # @NOTE: Trigger a dump for this iteration.
    dumper.dump()
