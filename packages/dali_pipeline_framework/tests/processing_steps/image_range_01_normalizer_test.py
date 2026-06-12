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

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import numpy as np
import torch

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import ImageRange01Normalizer


def normalize_uint8_to_float32(uint8_array):
    """Helper function to normalize uint8 values to float32 range [0.0, 1.0]."""
    return torch.tensor(uint8_array.astype(np.float32) / 255.0, dtype=torch.float32)


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Create test images with different values to test normalization
        # Main image: 3x3 RGB image with values 0-255
        res["image"] = np.array(
            [
                [[0, 128, 255], [64, 192, 32], [16, 240, 96]],
                [[48, 176, 144], [80, 112, 208], [224, 160, 88]],
                [[128, 64, 192], [32, 96, 160], [240, 16, 128]],
            ],
            dtype=np.uint8,
        )

        # Second image: 2x2 RGB image with different values
        res["camera"]["image"] = np.array(
            [
                [[255, 0, 128], [64, 192, 255]],
                [[128, 64, 0], [192, 128, 64]],
            ],
            dtype=np.uint8,
        )

        # Third image: 1x3 RGB image
        res["camera"]["annotation"]["image"] = np.array(
            [[[0, 255, 128], [128, 0, 255], [255, 128, 0]]], dtype=np.uint8
        )

        # Add some non-image data to ensure it's not affected
        res["metadata"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res["camera"]["metadata"] = np.array([10.0, 20.0, 30.0])

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()

        # Main image field
        res.add_data_field("image", DALIDataType.UINT8)

        # Camera group with image
        camera = SampleDataGroup()
        camera.add_data_field("image", DALIDataType.UINT8)
        camera.add_data_field("metadata", DALIDataType.FLOAT)

        # Camera annotation with image
        camera_annotation = SampleDataGroup()
        camera_annotation.add_data_field("image", DALIDataType.UINT8)
        camera.add_data_group_field("annotation", camera_annotation)

        res.add_data_group_field("camera", camera)
        res.add_data_field("metadata", DALIDataType.FLOAT)

        return res


def test_image_range_01_normalizer():
    """Comprehensive test for ImageRange01Normalizer processing step."""
    # Set up pipeline once
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = ImageRange01Normalizer(image_name="image")

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["metadata"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=1,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Define original values for validation
    original_main = np.array(
        [
            [[0, 128, 255], [64, 192, 32], [16, 240, 96]],
            [[48, 176, 144], [80, 112, 208], [224, 160, 88]],
            [[128, 64, 192], [32, 96, 160], [240, 16, 128]],
        ],
        dtype=np.uint8,
    )

    original_camera = np.array(
        [
            [[255, 0, 128], [64, 192, 255]],
            [[128, 64, 0], [192, 128, 64]],
        ],
        dtype=np.uint8,
    )

    original_annotation = np.array([[[0, 255, 128], [128, 0, 255], [255, 128, 0]]], dtype=np.uint8)

    # Calculate expected normalized values
    expected_main = normalize_uint8_to_float32(original_main)
    expected_camera = normalize_uint8_to_float32(original_camera)
    expected_annotation = normalize_uint8_to_float32(original_annotation)

    # Test 1: Verify all three images were normalized correctly
    main_image = res["image"]
    camera_image = res["camera"]["image"]
    annotation_image = res["camera"]["annotation"]["image"]

    # Check data types
    assert main_image.dtype == torch.float32
    assert camera_image.dtype == torch.float32
    assert annotation_image.dtype == torch.float32

    # Check that values are in range [0.0, 1.0]
    assert torch.all(main_image >= 0.0) and torch.all(main_image <= 1.0)
    assert torch.all(camera_image >= 0.0) and torch.all(camera_image <= 1.0)
    assert torch.all(annotation_image >= 0.0) and torch.all(annotation_image <= 1.0)

    # Check specific normalized values
    assert torch.allclose(main_image, expected_main, atol=1e-3)
    assert torch.allclose(camera_image, expected_camera, atol=1e-3)
    assert torch.allclose(annotation_image, expected_annotation, atol=1e-3)

    # Test 2: Verify non-image data is unchanged (test for exact equality, accounting for batch dimension)
    assert torch.equal(res["metadata"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(res["camera"]["metadata"][0], torch.tensor([10.0, 20.0, 30.0]))

    # Check that the structure is preserved (no fields were added or removed)
    expected_fields = {"image", "camera", "metadata"}
    assert set(res.keys()) == expected_fields

    expected_camera_fields = {"image", "annotation", "metadata"}
    assert set(res["camera"].keys()) == expected_camera_fields

    expected_annotation_fields = {"image"}
    assert set(res["camera"]["annotation"].keys()) == expected_annotation_fields


def test_image_range_01_normalizer_no_images_found():
    """Test that an error is raised when no images with the specified name are found."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Try to normalize images with a name that doesn't exist
    step = ImageRange01Normalizer(image_name="nonexistent_image")

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["metadata"],
    )

    # This should raise an error during pipeline building
    with pytest.raises(KeyError, match="No occurrences of images found"):
        pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=1,
            prefetch_queue_depth=1,
            num_threads=1,
            py_start_method="spawn",
        )


if __name__ == "__main__":
    pytest.main([__file__])
