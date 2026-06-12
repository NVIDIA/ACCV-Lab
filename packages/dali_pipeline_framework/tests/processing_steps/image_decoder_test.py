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
import cv2

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import ImageDecoder


def create_test_image(height, width, channels=3):
    """Helper function to create a test image with known content in RGB format."""
    # Create an image with complex patterns in RGB format
    image = np.zeros((height, width, channels), dtype=np.uint8)

    # Create a more complex pattern for better testing
    for i in range(height):
        for j in range(width):
            # Red channel: horizontal gradient (RGB format)
            image[i, j, 0] = int((j / width) * 255)
            # Green channel: vertical gradient
            image[i, j, 1] = int((i / height) * 255)
            # Blue channel: diagonal pattern
            image[i, j, 2] = int(((i + j) / (height + width)) * 255)

    return image


def encode_image_to_jpeg(image):
    """Helper function to encode image to JPEG format with maximum quality.

    Args:
        image: RGB image as numpy array (H, W, 3)

    Returns:
        JPEG encoded image as numpy array of bytes
    """
    # OpenCV expects BGR format, so convert RGB to BGR for encoding
    image_bgr = image[:, :, ::-1]  # RGB to BGR

    # Encode image to JPEG format with maximum quality (100 out of 100)
    # in order to minimize compression artifacts
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 100]
    success, encoded_image = cv2.imencode(".jpg", image_bgr, encode_params)
    if not success:
        raise ValueError("Failed to encode image to JPEG")
    # Convert to numpy array (to be used as input to the DALI pipeline)
    return np.frombuffer(encoded_image.tobytes(), dtype=np.uint8)


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Create test images with different sizes
        # Main image: 64x64 RGB image
        main_image = create_test_image(64, 64, 3)
        res["image"] = encode_image_to_jpeg(main_image)

        # Second image: 32x48 RGB image
        camera_image = create_test_image(32, 48, 3)
        res["camera"]["image"] = encode_image_to_jpeg(camera_image)

        # Third image: 16x32 RGB image
        annotation_image = create_test_image(16, 32, 3)
        res["camera"]["annotation"]["image"] = encode_image_to_jpeg(annotation_image)

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
    "use_device_mixed,as_bgr",
    [(False, False), (True, False), (False, True)],
    ids=["rgb_cpu", "rgb_mixed_device", "bgr_cpu"],
)
def test_image_decoder(use_device_mixed, as_bgr):
    """Basic test for ImageDecoder processing step."""
    # Set up pipeline
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = ImageDecoder(
        image_name="image",
        use_device_mixed=use_device_mixed,
        as_bgr=as_bgr,
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

    # Get first batch
    batch = next(iterator_iter)

    # Check that images are decoded (should be tensors, not bytes)
    assert isinstance(batch["image"], torch.Tensor)
    assert batch["image"].shape[1:] == (64, 64, 3)  # Batch dimension + HWC
    assert batch["image"].dtype == torch.uint8

    # Check device placement based on configuration
    if use_device_mixed:
        # With mixed device, images should be on GPU
        assert batch["image"].device.type == "cuda"
        assert batch["camera"]["image"].device.type == "cuda"
        assert batch["camera"]["annotation"]["image"].device.type == "cuda"
    else:
        # Without mixed device, images should be on CPU
        assert batch["image"].device.type == "cpu"
        assert batch["camera"]["image"].device.type == "cpu"
        assert batch["camera"]["annotation"]["image"].device.type == "cpu"

    # Check nested images
    assert batch["camera"]["image"].shape[1:] == (32, 48, 3)
    assert batch["camera"]["annotation"]["image"].shape[1:] == (16, 32, 3)

    # Check image content with tolerance for JPEG compression
    decoded_main = batch["image"][0].cpu().numpy()  # Move to CPU if on GPU

    # Get the original image in RGB format
    original_rgb = create_test_image(64, 64, 3)  # Created in RGB format

    if as_bgr:
        # For BGR test, the step should return BGR format
        # We need to convert the RGB reference to BGR for comparison
        original_bgr = original_rgb[:, :, ::-1]  # RGB to BGR
        reference_image = original_bgr
    else:
        # For RGB test, the step should return RGB format
        reference_image = original_rgb

    # Use numpy.allclose with rtol=0 and atol=5.0 to focus on maximum absolute error
    # rtol=0 means we only care about absolute tolerance, not relative
    assert np.allclose(
        decoded_main.astype(np.float32), reference_image.astype(np.float32), rtol=0, atol=5.0
    ), f"Image content differs too much from reference (max allowed abs error: 5.0)"

    # Check that non-image data is preserved
    assert torch.all(batch["metadata"] == torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.all(batch["camera"]["metadata"] == torch.tensor([10.0, 20.0, 30.0]))


def test_image_decoder_no_images_found():
    """Test ImageDecoder when no images are found."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Use a non-existent image name
    step = ImageDecoder(
        image_name="nonexistent_image",
        use_device_mixed=False,
        as_bgr=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["metadata"],
    )

    # This should raise an error during pipeline creation
    with pytest.raises(KeyError, match="No occurrences of images found"):
        pipeline_def.get_dali_pipeline(
            enable_conditionals=True,
            batch_size=1,
            prefetch_queue_depth=1,
            num_threads=1,
            py_start_method="spawn",
        )


def test_image_decoder_multiple_batches():
    """Test ImageDecoder processes multiple batches correctly."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=2,  # Use batch size of 2
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = ImageDecoder(
        image_name="image",
        use_device_mixed=False,
        as_bgr=False,
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["metadata"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=2,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)

    # Process multiple batches
    for i in range(3):
        batch = next(iterator_iter)

        # Check batch size
        assert batch["image"].shape[0] == 2
        assert batch["camera"]["image"].shape[0] == 2
        assert batch["camera"]["annotation"]["image"].shape[0] == 2

        # Check dimensions for each sample in batch
        assert batch["image"].shape[1:] == (64, 64, 3)
        assert batch["camera"]["image"].shape[1:] == (32, 48, 3)
        assert batch["camera"]["annotation"]["image"].shape[1:] == (16, 32, 3)


if __name__ == "__main__":
    pytest.main([__file__])
