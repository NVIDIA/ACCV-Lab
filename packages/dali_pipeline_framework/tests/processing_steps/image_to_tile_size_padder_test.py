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
from accvlab.dali_pipeline_framework.processing_steps import ImageToTileSizePadder


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Create test images with different sizes
        # Main image: 64x64 RGB image
        res["image"] = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)

        # Second image: 32x48 RGB image
        res["camera"]["image"] = np.random.randint(0, 256, (32, 48, 3), dtype=np.uint8)

        # Third image: 16x32 RGB image
        res["camera"]["annotation"]["image"] = np.random.randint(0, 256, (16, 32, 3), dtype=np.uint8)

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


@pytest.mark.parametrize(
    "tile_size,expected_sizes",
    [
        (8, [[64, 64], [32, 48], [16, 32]]),  # All images already aligned
        (16, [[64, 64], [32, 48], [16, 32]]),  # All images already aligned
        (32, [[64, 64], [32, 64], [32, 32]]),  # Some padding needed
        (64, [[64, 64], [64, 64], [64, 64]]),  # All images need padding
        ([32, 16], [[64, 64], [32, 64], [16, 32]]),  # Different tile sizes for H/W
        (1, [[64, 64], [32, 48], [16, 32]]),  # Tile size 1
    ],
    ids=["tile_8", "tile_16", "tile_32", "tile_64", "tile_16x32", "tile_1"],
)
def test_image_to_tile_size_padder(tile_size, expected_sizes):
    """Test ImageToTileSizePadder with different tile sizes."""
    # Set up pipeline
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = ImageToTileSizePadder(
        image_name="image",
        tile_size_to_pad_to=tile_size,
    )

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

    # Test 1: Verify image dimensions match the expected padded sizes
    main_image = res["image"]
    camera_image = res["camera"]["image"]
    annotation_image = res["camera"]["annotation"]["image"]

    # Check that images have the expected dimensions (accounting for batch dimension)
    assert main_image.shape[1:] == (expected_sizes[0][0], expected_sizes[0][1], 3)
    assert camera_image.shape[1:] == (expected_sizes[1][0], expected_sizes[1][1], 3)
    assert annotation_image.shape[1:] == (expected_sizes[2][0], expected_sizes[2][1], 3)

    # Test 2: Verify non-image data is unchanged
    assert torch.equal(res["metadata"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(res["camera"]["metadata"][0], torch.tensor([10.0, 20.0, 30.0]))

    # Test 3: Verify structure is preserved
    expected_fields = {"image", "camera", "metadata"}
    assert set(res.keys()) == expected_fields

    expected_camera_fields = {"image", "annotation", "metadata"}
    assert set(res["camera"].keys()) == expected_camera_fields

    expected_annotation_fields = {"image"}
    assert set(res["camera"]["annotation"].keys()) == expected_annotation_fields


def test_image_to_tile_size_padder_no_images_found():
    """Test that an error is raised when no images with the specified name are found."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Use a non-existent image name
    step = ImageToTileSizePadder(
        image_name="nonexistent_image",
        tile_size_to_pad_to=32,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["metadata"],
    )

    # Should raise KeyError during pipeline construction when trying to find images
    with pytest.raises(KeyError, match="No occurrences of images found"):
        pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=1,
            prefetch_queue_depth=1,
            num_threads=1,
            py_start_method="spawn",
        )


def test_image_to_tile_size_padder_already_padded():
    """Test that images that are already properly padded remain unchanged."""
    # Use tile size that matches the original image dimensions
    tile_size = 64  # Main image is 64x64, so no padding needed

    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = ImageToTileSizePadder(
        image_name="image",
        tile_size_to_pad_to=tile_size,
    )

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

    # Verify image dimensions match the expected padded sizes
    # Main image should remain 64x64 (no padding needed)
    assert res["image"].shape[1:] == (64, 64, 3)

    # Camera image should be padded to 64x64
    assert res["camera"]["image"].shape[1:] == (64, 64, 3)

    # Annotation image should be padded to 64x64
    assert res["camera"]["annotation"]["image"].shape[1:] == (64, 64, 3)


if __name__ == "__main__":
    pytest.main([__file__])
