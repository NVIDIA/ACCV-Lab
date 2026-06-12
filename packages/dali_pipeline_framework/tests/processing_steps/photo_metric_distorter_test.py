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
import numpy as np
import torch
import cv2

try:
    from typing import override
except ImportError:
    from typing_extensions import override

import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType

from _dali_fake_random_generator import DaliFakeRandomGenerator

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import PhotoMetricDistorter

# @TODO:
#  - Decide whether to use accurate transformations instead of the used approximations
#  - Decide whether to change to applied transformations and their order
#  - Adjust the tests accordingly & check the results

USING_APPROXIMATIONS = True


def set_dali_uniform_generator_and_get_orig_and_replacement(sequences):
    """Set up fake random generator and return original generator for restoration."""
    generator = DaliFakeRandomGenerator(sequences)
    original_generator = fn.random.uniform
    fn.random.uniform = generator.get_generator()
    return original_generator, generator


def restore_generator(generator):
    """Restore the original random generator."""
    fn.random.uniform = generator


def to_float01(img: np.ndarray) -> np.ndarray:
    """Convert image to float32 in [0, 1] if input is uint8; otherwise return as-is."""
    return img.astype(np.float32) / 255.0 if img.dtype == np.uint8 else img


def build_basic_sequences_for_dtype(use_uint8: bool):
    """Build the deterministic fake-random sequences used in the basic tests.

    Brightness delta replacement is scaled to the input dtype domain to match
    how DALI expects the parameter for uint8 vs float inputs.
    """
    return [
        DaliFakeRandomGenerator.RangeReplacement([0.0, 1.0], [0.3, 0.2, 0.1, 0.4, 0.6]),
        DaliFakeRandomGenerator.RangeReplacement([0, 2], [0]),
        DaliFakeRandomGenerator.RangeReplacement([-0.1, 0.1], [0.05 * (255.0 if use_uint8 else 1.0)]),
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [1.1]),
        DaliFakeRandomGenerator.RangeReplacement([-10, 10], [10]),
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [1.15]),
        DaliFakeRandomGenerator.RangeReplacement([0, 6], [2]),
    ]


def apply_brightness_reference(image, delta):
    """Reference implementation for brightness adjustment."""
    result = image + delta
    return np.clip(result, 0.0, 1.0)


def apply_contrast_reference(image, alpha):
    """Reference implementation for contrast adjustment."""
    result = image * alpha
    return np.clip(result, 0.0, 1.0)


def apply_hue_reference(image, hue_delta, is_bgr=False):
    """Reference implementation for hue adjustment."""
    # Convert BGR/RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV if is_bgr else cv2.COLOR_RGB2HSV)
    # Adjust hue (H channel is the first channel)
    # hue_delta is in degrees, need to convert to the appropriate range
    if hsv.dtype == np.uint8:
        # For uint8, hue is in [0, 180] range (i.e. scale factor of 2 to obtain degrees)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_delta // 2) % 180
    else:
        # For float images, hue is in [0, 1] range, so convert degrees to [0,1] range
        # 360 degrees = 1.0 in float representation
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue_delta, 360.0)
    # Convert back
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR if is_bgr else cv2.COLOR_HSV2RGB)


def apply_saturation_reference(image, saturation_factor, is_bgr=False):
    """Reference implementation for saturation adjustment."""
    # Convert BGR/RGB to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV if is_bgr else cv2.COLOR_RGB2HSV)
    # Adjust saturation (S channel is the second channel)
    hsv[:, :, 1] = hsv[:, :, 1] * saturation_factor
    # Convert back
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR if is_bgr else cv2.COLOR_HSV2RGB)


def apply_channel_swap_reference(image, permutation):
    """Reference implementation for channel swapping."""
    return image[:, :, permutation]


def apply_photometric_distortion_reference(image, augmentation_params, is_bgr=False):
    """Reference implementation that applies all photometric distortions in the correct order."""
    result = image.copy()

    # Apply brightness
    if augmentation_params["aug_brightness"]:
        result = apply_brightness_reference(result, augmentation_params["delta"])

    # Apply contrast (mode 1: before HSV operations)
    if augmentation_params["contrast_mode"] == 1 and augmentation_params["aug_contrast"]:
        result = apply_contrast_reference(result, augmentation_params["alpha"])

    # Apply saturation
    if augmentation_params["aug_saturation"]:
        result = apply_saturation_reference(result, augmentation_params["saturation"], is_bgr)

    # Apply hue
    if augmentation_params["aug_hue"]:
        result = apply_hue_reference(result, augmentation_params["hue"], is_bgr)

    # Apply contrast (mode 0: after HSV operations)
    if augmentation_params["contrast_mode"] == 0 and augmentation_params["aug_contrast"]:
        result = apply_contrast_reference(result, augmentation_params["alpha"])

    # Apply channel swapping
    if augmentation_params["aug_swap_channels"]:
        permutation = augmentation_params["channel_permutation"]
        result = apply_channel_swap_reference(result, permutation)

    return result


class TestProvider(DataProvider):
    """Test data provider for PhotoMetricDistorter tests."""

    def __init__(self, use_uint8: bool = False):
        self._use_uint8 = use_uint8

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Base float images in [0, 1]
        main_image = np.array(
            [
                [[0.0, 0.5, 1.0], [0.25, 0.75, 0.125], [0.0625, 0.9375, 0.375]],
                [[0.1875, 0.6875, 0.5625], [0.3125, 0.4375, 0.8125], [0.875, 0.625, 0.34375]],
                [[0.5, 0.25, 0.75], [0.125, 0.375, 0.625], [0.9375, 0.0625, 0.5]],
            ],
            dtype=np.float32,
        )
        camera_image = np.array(
            [
                [[1.0, 0.0, 0.5], [0.25, 0.75, 1.0]],
                [[0.5, 0.25, 0.0], [0.75, 0.5, 0.25]],
            ],
            dtype=np.float32,
        )
        annotation_image = np.array([[[0.0, 1.0, 0.5], [0.5, 0.0, 1.0], [1.0, 0.5, 0.0]]], dtype=np.float32)
        red_image = np.concatenate(
            [
                np.ones((3, 3, 1), dtype=np.float32),
                np.zeros((3, 3, 1), dtype=np.float32),
                np.zeros((3, 3, 1), dtype=np.float32),
            ],
            axis=2,
        )

        if self._use_uint8:
            to_u8 = lambda x: np.clip(np.round(x * 255.0), 0, 255).astype(np.uint8)
            res["image"] = to_u8(main_image)
            res["camera"]["image"] = to_u8(camera_image)
            res["camera"]["annotation"]["image"] = to_u8(annotation_image)
            res["red"]["image"] = to_u8(red_image)
        else:
            res["image"] = main_image
            res["camera"]["image"] = camera_image
            res["camera"]["annotation"]["image"] = annotation_image
            res["red"]["image"] = red_image

        # Non-image metadata
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

        img_type = DALIDataType.UINT8 if self._use_uint8 else DALIDataType.FLOAT

        # Main image field
        res.add_data_field("image", img_type)

        # Camera group with image
        camera = SampleDataGroup()
        camera.add_data_field("image", img_type)
        camera.add_data_field("metadata", DALIDataType.FLOAT)

        # Camera annotation with image
        camera_annotation = SampleDataGroup()
        camera_annotation.add_data_field("image", img_type)
        camera.add_data_group_field("annotation", camera_annotation)

        # red image
        red_image = SampleDataGroup()
        red_image.add_data_field("image", img_type)
        res.add_data_group_field("red", red_image)

        res.add_data_group_field("camera", camera)
        res.add_data_field("metadata", DALIDataType.FLOAT)

        return res


@pytest.mark.parametrize(
    "transformation_type,min_max_brightness,min_max_hue,min_max_saturation,prob_brightness_aug,prob_hue_aug,prob_saturation_aug,expected_augmentation",
    [
        (
            "brightness",
            [0.05, 0.05],  # Fixed brightness delta of 0.05
            [0.0, 0.0],  # No hue change
            [0.8, 1.2],  # Not used
            1.0,  # Always apply brightness
            0.0,  # Never apply hue
            0.0,  # Never apply saturation
            {
                "contrast_mode": 0,
                "aug_brightness": True,
                "aug_contrast": False,
                "aug_saturation": False,
                "aug_hue": False,
                "aug_swap_channels": False,
                "delta": 0.05,
                "alpha": 0.0,
                "hue": 0.0,
                "saturation": 0.0,
                "channel_permutation": [0, 1, 2],
            },
        ),
        (
            "hue",
            [-0.1, 0.1],  # Not used
            [7.0, 7.0],  # Fixed hue delta of 7.0 degrees
            [0.8, 1.2],  # Not used
            0.0,  # Never apply brightness
            1.0,  # Always apply hue
            0.0,  # Never apply saturation
            {
                "contrast_mode": 0,
                "aug_brightness": False,
                "aug_contrast": False,
                "aug_saturation": False,
                "aug_hue": True,
                "aug_swap_channels": False,
                "delta": 0.0,
                "alpha": 0.0,
                "hue": 7.0,
                "saturation": 0.0,
                "channel_permutation": [0, 1, 2],
            },
        ),
        (
            "saturation",
            [-0.1, 0.1],  # Not used
            [0.0, 0.0],  # No hue change
            [0.9, 0.9],  # Fixed saturation factor of 0.9
            0.0,  # Never apply brightness
            0.0,  # Never apply hue
            1.0,  # Always apply saturation
            {
                "contrast_mode": 0,
                "aug_brightness": False,
                "aug_contrast": False,
                "aug_saturation": True,
                "aug_hue": False,
                "aug_swap_channels": False,
                "delta": 0.0,
                "alpha": 0.0,
                "hue": 0.0,
                "saturation": 0.9,
                "channel_permutation": [0, 1, 2],
            },
        ),
    ],
)
def test_photometric_distorter_single_transformation(
    transformation_type,
    min_max_brightness,
    min_max_hue,
    min_max_saturation,
    prob_brightness_aug,
    prob_hue_aug,
    prob_saturation_aug,
    expected_augmentation,
):
    """Test photometric distortion with only one transformation at a time."""
    # Set up pipeline with deterministic transformation
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = PhotoMetricDistorter(
        image_name="image",
        min_max_brightness=min_max_brightness,
        min_max_hue=min_max_hue,
        min_max_contrast=[0.8, 1.2],
        min_max_saturation=min_max_saturation,
        prob_brightness_aug=prob_brightness_aug,
        prob_hue_aug=prob_hue_aug,
        prob_contrast_aug=0.0,  # Never apply contrast
        prob_saturation_aug=prob_saturation_aug,
        prob_swap_channels=0.0,  # Never apply channel swap
        is_bgr=False,
        enforce_process_on_gpu=False,
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

    iterator = DALIStructuredOutputIterator(1, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)
    res = next(iterator_iter)

    # Get the processed images
    main_image = res["image"][0].numpy()  # Remove batch dimension
    camera_image = res["camera"]["image"][0].numpy()
    annotation_image = res["camera"]["annotation"]["image"][0].numpy()
    red_image = res["red"]["image"][0].numpy()

    # Get original images from the provider for reference
    original_data = provider.get_data(0)
    original_main = original_data["image"]
    original_camera = original_data["camera"]["image"]
    original_annotation = original_data["camera"]["annotation"]["image"]
    original_red = original_data["red"]["image"]

    # Apply reference transformations
    expected_main = apply_photometric_distortion_reference(original_main, expected_augmentation, is_bgr=False)
    expected_camera = apply_photometric_distortion_reference(
        original_camera, expected_augmentation, is_bgr=False
    )
    expected_annotation = apply_photometric_distortion_reference(
        original_annotation, expected_augmentation, is_bgr=False
    )
    expected_red = apply_photometric_distortion_reference(original_red, expected_augmentation, is_bgr=False)

    # Test that the transformations were applied correctly
    if not USING_APPROXIMATIONS or not transformation_type in ["hue", "saturation"]:
        assert np.allclose(main_image, expected_main, atol=1e-2)
        assert np.allclose(camera_image, expected_camera, atol=1e-2)
        assert np.allclose(annotation_image, expected_annotation, atol=1e-2)
        assert np.allclose(red_image, expected_red, atol=1e-2)
    else:
        assert not np.allclose(main_image, original_main, atol=0.03)
        assert not np.allclose(camera_image, original_camera, atol=0.03)
        assert not np.allclose(annotation_image, original_annotation, atol=0.03)
        assert not np.allclose(red_image, original_red, atol=0.03)

    # Test that non-image data is unchanged
    assert torch.equal(res["metadata"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    assert torch.equal(res["camera"]["metadata"][0], torch.tensor([10.0, 20.0, 30.0]))


@pytest.mark.parametrize("use_uint8", [False, True])
def test_photometric_distorter_basic(use_uint8):
    """Test basic photometric distortion functionality with multiple augmentations."""
    # Set up fake random generator with predefined values
    # The same augmentation will be applied to all images
    sequences = [
        DaliFakeRandomGenerator.RangeReplacement(
            [0.0, 1.0], [0.3, 0.2, 0.1, 0.4, 0.6]
        ),  # Multiple augmentations enabled
        DaliFakeRandomGenerator.RangeReplacement([0, 2], [0]),  # Contrast mode 0 (after HSV)
        DaliFakeRandomGenerator.RangeReplacement(
            [-0.1, 0.1], [0.05 * (255.0 if use_uint8 else 1.0)]
        ),  # Brightness delta
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [1.1]),  # Contrast alpha
        DaliFakeRandomGenerator.RangeReplacement([-10, 10], [10]),  # Hue delta
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [1.15]),  # Saturation factor
        DaliFakeRandomGenerator.RangeReplacement([0, 6], [2]),  # Channel permutation
    ]

    original_generator, generator = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)

    try:
        # Set up pipeline
        provider = TestProvider(use_uint8=use_uint8)
        input_callable = ShuffledShardedInputCallable(
            provider,
            batch_size=1,
            num_shards=1,
            shard_id=0,
            shuffle=False,
        )

        step = PhotoMetricDistorter(
            image_name="image",
            min_max_brightness=[-0.1, 0.1],
            min_max_hue=[-10.0, 10.0],
            min_max_contrast=[0.8, 1.2],
            min_max_saturation=[0.8, 1.2],
            prob_brightness_aug=0.5,
            prob_hue_aug=0.5,
            prob_contrast_aug=0.5,
            prob_saturation_aug=0.5,
            prob_swap_channels=0.5,
            is_bgr=False,
            enforce_process_on_gpu=False,
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

        iterator = DALIStructuredOutputIterator(
            1, pipeline, pipeline_def.check_and_get_output_data_structure()
        )
        iterator_iter = iter(iterator)
        res = next(iterator_iter)

        # Get the processed images
        main_image = res["image"][0].numpy()  # Remove batch dimension
        camera_image = res["camera"]["image"][0].numpy()
        annotation_image = res["camera"]["annotation"]["image"][0].numpy()
        red_image = res["red"]["image"][0].numpy()

        # Get original images from the provider for reference
        original_data = provider.get_data(0)
        original_main = original_data["image"]
        original_camera = original_data["camera"]["image"]
        original_annotation = original_data["camera"]["annotation"]["image"]
        original_red = original_data["red"]["image"]

        # Based on the fake random generator values, we can predict the augmentation parameters
        # The first random value 0.3 < 0.5, so brightness augmentation is applied
        # The second random value 0.2 < 0.5, so contrast augmentation is applied
        # The third random value 0.1 < 0.5, so saturation augmentation is applied
        # The fourth random value 0.4 < 0.5, so hue augmentation is applied
        # The fifth random value 0.6 > 0.5, so channel swap is not applied
        # Contrast mode is 0 (after HSV operations)
        # Brightness delta is 0.05
        # Contrast alpha is 1.1
        # Hue delta is 5.0
        # Saturation factor is 1.15
        # Channel permutation index is 2 (which gives [1, 0, 2]) but not applied since swap is False

        expected_augmentation = {
            "contrast_mode": 0,
            "aug_brightness": True,
            "aug_contrast": True,
            "aug_saturation": True,
            "aug_hue": True,
            "aug_swap_channels": False,
            "delta": 0.05,
            "alpha": 1.1,
            "hue": 5.0,
            "saturation": 1.15,
            "channel_permutation": [1, 0, 2],
        }

        # Apply reference transformations for operations we can reliably replicate
        # Note: We skip hue and saturation in reference comparison as DALI's fn.hue/fn.saturation
        # may have different behavior than OpenCV color space conversions
        reference_augmentation = expected_augmentation.copy()
        reference_augmentation["aug_hue"] = False
        reference_augmentation["aug_saturation"] = False

        expected_main = apply_photometric_distortion_reference(
            to_float01(original_main), reference_augmentation, is_bgr=False
        )
        expected_camera = apply_photometric_distortion_reference(
            to_float01(original_camera), reference_augmentation, is_bgr=False
        )
        expected_annotation = apply_photometric_distortion_reference(
            to_float01(original_annotation), reference_augmentation, is_bgr=False
        )
        expected_red = apply_photometric_distortion_reference(
            to_float01(original_red), reference_augmentation, is_bgr=False
        )

        # Test that the transformations were actually applied (images are different from original)
        if not USING_APPROXIMATIONS:
            assert np.allclose(to_float01(main_image), expected_main, atol=1e-2)
            assert np.allclose(to_float01(camera_image), expected_camera, atol=1e-2)
            assert np.allclose(to_float01(annotation_image), expected_annotation, atol=1e-2)
            assert np.allclose(to_float01(red_image), expected_red, atol=1e-2)
        else:
            assert not np.allclose(to_float01(main_image), to_float01(original_main), atol=0.03)
            assert not np.allclose(to_float01(camera_image), to_float01(original_camera), atol=0.03)
            assert not np.allclose(to_float01(annotation_image), to_float01(original_annotation), atol=0.03)
            assert not np.allclose(to_float01(red_image), to_float01(original_red), atol=0.03)
            red_image_01 = to_float01(red_image)
            assert (
                red_image_01[0, 0, 0] > 0.9 and red_image_01[0, 0, 1] > 0.01
            ), "Red image should be slightly shifted towards green"

        # Test that the images are still in valid range
        if use_uint8:
            assert np.all(main_image >= 0) and np.all(main_image <= 255)
            assert np.all(camera_image >= 0) and np.all(camera_image <= 255)
            assert np.all(annotation_image >= 0) and np.all(annotation_image <= 255)
        else:
            assert np.all(main_image >= 0.0) and np.all(main_image <= 1.0)
            assert np.all(camera_image >= 0.0) and np.all(camera_image <= 1.0)
            assert np.all(annotation_image >= 0.0) and np.all(annotation_image <= 1.0)

        # Test that non-image data is unchanged
        assert torch.equal(res["metadata"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
        assert torch.equal(res["camera"]["metadata"][0], torch.tensor([10.0, 20.0, 30.0]))

    finally:
        restore_generator(original_generator)


def test_photometric_distorter_float_uint8_compare():
    """Run the same augmentations on float and uint8 inputs and compare outputs.

    We ensure the same augmentation parameters are drawn by seeding via the
    fake random generator. For uint8 input, brightness deltas are scaled to
    the 0..255 domain so that the effective change matches the float case.
    """

    def run_once(use_uint8: bool):
        sequences = build_basic_sequences_for_dtype(use_uint8)
        original_generator, _ = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)
        try:
            provider = TestProvider(use_uint8=use_uint8)
            input_callable = ShuffledShardedInputCallable(
                provider,
                batch_size=1,
                num_shards=1,
                shard_id=0,
                shuffle=False,
            )

            step = PhotoMetricDistorter(
                image_name="image",
                min_max_brightness=[-0.1, 0.1],
                min_max_hue=[-10.0, 10.0],
                min_max_contrast=[0.8, 1.2],
                min_max_saturation=[0.8, 1.2],
                prob_brightness_aug=0.5,
                prob_hue_aug=0.5,
                prob_contrast_aug=0.5,
                prob_saturation_aug=0.5,
                prob_swap_channels=0.5,
                is_bgr=False,
                enforce_process_on_gpu=False,
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

            iterator = DALIStructuredOutputIterator(
                1, pipeline, pipeline_def.check_and_get_output_data_structure()
            )
            iterator_iter = iter(iterator)
            res = next(iterator_iter)

            # Copies are needed to avoiding access after free errors after the pipeline goes out of scope.
            return {
                "main": res["image"][0].numpy().copy(),
                "camera": res["camera"]["image"][0].numpy().copy(),
                "annotation": res["camera"]["annotation"]["image"][0].numpy().copy(),
                "red": res["red"]["image"][0].numpy().copy(),
                "metadata": res["metadata"][0].clone(),
                "camera_metadata": res["camera"]["metadata"][0].clone(),
            }
        finally:
            restore_generator(original_generator)

    # Run both cases
    out_float = run_once(use_uint8=False)
    out_u8 = run_once(use_uint8=True)

    # Normalize to float [0,1] for fair comparison
    float_main = to_float01(out_float["main"])
    u8_main = to_float01(out_u8["main"])
    float_cam = to_float01(out_float["camera"])
    u8_cam = to_float01(out_u8["camera"])
    float_ann = to_float01(out_float["annotation"])
    u8_ann = to_float01(out_u8["annotation"])
    float_red = to_float01(out_float["red"])
    u8_red = to_float01(out_u8["red"])

    # Compare within reasonable tolerance due to dtype and rounding differences
    atol = 0.02
    rtol = 1e-3
    assert np.allclose(float_main, u8_main, atol=atol, rtol=rtol)
    assert np.allclose(float_cam, u8_cam, atol=atol, rtol=rtol)
    assert np.allclose(float_ann, u8_ann, atol=atol, rtol=rtol)
    assert np.allclose(float_red, u8_red, atol=atol, rtol=rtol)

    # Non-image metadata must be identical
    assert torch.equal(out_float["metadata"], out_u8["metadata"])
    assert torch.equal(out_float["camera_metadata"], out_u8["camera_metadata"])


def test_photometric_distorter_bgr():
    """Test photometric distortion with BGR format."""
    # Set up fake random generator with predefined values
    sequences = [
        DaliFakeRandomGenerator.RangeReplacement(
            [0.0, 1.0], [0.1, 0.2, 0.3, 0.4, 0.5]
        ),  # All augmentations applied
        DaliFakeRandomGenerator.RangeReplacement([0, 2], [0]),  # Contrast mode 0 (after HSV)
        DaliFakeRandomGenerator.RangeReplacement([0, 6], [1]),  # Channel permutation [0, 2, 1]
        DaliFakeRandomGenerator.RangeReplacement([-0.1, 0.1], [0.08]),  # Brightness delta
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [0.9]),  # Contrast alpha
        DaliFakeRandomGenerator.RangeReplacement([-10, 10], [-7]),  # Hue delta
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [0.85]),  # Saturation factor
    ]

    original_generator, generator = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)

    try:
        # Set up pipeline with BGR format
        provider = TestProvider()
        input_callable = ShuffledShardedInputCallable(
            provider,
            batch_size=1,
            num_shards=1,
            shard_id=0,
            shuffle=False,
        )

        step = PhotoMetricDistorter(
            image_name="image",
            min_max_brightness=[-0.1, 0.1],
            min_max_hue=[-10.0, 10.0],
            min_max_contrast=[0.8, 1.2],
            min_max_saturation=[0.8, 1.2],
            prob_brightness_aug=0.5,
            prob_hue_aug=0.5,
            prob_contrast_aug=0.5,
            prob_saturation_aug=0.5,
            prob_swap_channels=0.5,
            is_bgr=True,  # Use BGR format
            enforce_process_on_gpu=False,
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

        iterator = DALIStructuredOutputIterator(
            1, pipeline, pipeline_def.check_and_get_output_data_structure()
        )
        iterator_iter = iter(iterator)
        res = next(iterator_iter)

        # Get the processed images
        main_image = res["image"][0].numpy()
        camera_image = res["camera"]["image"][0].numpy()
        annotation_image = res["camera"]["annotation"]["image"][0].numpy()
        red_image = res["red"]["image"][0].numpy()

        # Get original images from the provider for reference
        original_data = provider.get_data(0)
        original_main = original_data["image"]
        original_camera = original_data["camera"]["image"]
        original_annotation = original_data["camera"]["annotation"]["image"]
        original_red = original_data["red"]["image"]

        # Test that the transformations were actually applied (images are different from original)
        assert not np.allclose(main_image, original_main, atol=1e-6)
        assert not np.allclose(camera_image, original_camera, atol=1e-6)
        assert not np.allclose(annotation_image, original_annotation, atol=1e-6)
        assert not np.allclose(red_image, original_red, atol=1e-6)

        # Test that the images are still in valid range [0, 1] (assuming input was in [0, 1])
        assert np.all(main_image >= -0.2) and np.all(main_image <= 1.2)  # Allow some overflow for brightness

    finally:
        restore_generator(original_generator)


def test_photometric_distorter_no_augmentations():
    """Test photometric distortion when no augmentations are applied."""
    # Set up fake random generator with values that disable all augmentations
    sequences = [
        DaliFakeRandomGenerator.RangeReplacement(
            [0.0, 1.0], [0.9, 0.8, 0.7, 0.6, 0.5]
        ),  # All probabilities > 0.5
        DaliFakeRandomGenerator.RangeReplacement([0, 2], [0]),  # Contrast mode
        DaliFakeRandomGenerator.RangeReplacement([0, 6], [0]),  # Channel permutation
        DaliFakeRandomGenerator.RangeReplacement([-0.1, 0.1], [0.05]),  # Brightness delta (not used)
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [1.1]),  # Contrast alpha (not used)
        DaliFakeRandomGenerator.RangeReplacement([-10, 10], [5]),  # Hue delta (not used)
        DaliFakeRandomGenerator.RangeReplacement([0.8, 1.2], [1.15]),  # Saturation factor (not used)
    ]

    original_generator, generator = set_dali_uniform_generator_and_get_orig_and_replacement(sequences)

    try:
        # Set up pipeline
        provider = TestProvider()
        input_callable = ShuffledShardedInputCallable(
            provider,
            batch_size=1,
            num_shards=1,
            shard_id=0,
            shuffle=False,
        )

        step = PhotoMetricDistorter(
            image_name="image",
            min_max_brightness=[-0.1, 0.1],
            min_max_hue=[-10.0, 10.0],
            min_max_contrast=[0.8, 1.2],
            min_max_saturation=[0.8, 1.2],
            prob_brightness_aug=0.5,
            prob_hue_aug=0.5,
            prob_contrast_aug=0.5,
            prob_saturation_aug=0.5,
            prob_swap_channels=0.5,
            is_bgr=False,
            enforce_process_on_gpu=False,
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

        iterator = DALIStructuredOutputIterator(
            1, pipeline, pipeline_def.check_and_get_output_data_structure()
        )
        iterator_iter = iter(iterator)
        res = next(iterator_iter)

        # Get the processed images
        main_image = res["image"][0].numpy()
        camera_image = res["camera"]["image"][0].numpy()
        annotation_image = res["camera"]["annotation"]["image"][0].numpy()
        red_image = res["red"]["image"][0].numpy()

        # Get original images from the provider for reference
        original_data = provider.get_data(0)
        original_main = original_data["image"]
        original_camera = original_data["camera"]["image"]
        original_annotation = original_data["camera"]["annotation"]["image"]
        original_red = original_data["red"]["image"]

        # Since all augmentations are disabled, images should remain unchanged
        assert np.allclose(main_image, original_main, atol=1e-6)
        assert np.allclose(camera_image, original_camera, atol=1e-6)
        assert np.allclose(annotation_image, original_annotation, atol=1e-6)
        assert np.allclose(red_image, original_red, atol=1e-6)

    finally:
        restore_generator(original_generator)


def test_photometric_distorter_no_images_found():
    """Test that an error is raised when no images with the specified name are found."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Try to process images with a name that doesn't exist
    step = PhotoMetricDistorter(
        image_name="nonexistent_image",
        min_max_brightness=[-0.1, 0.1],
        min_max_hue=[-10.0, 10.0],
        min_max_contrast=[0.8, 1.2],
        min_max_saturation=[0.8, 1.2],
    )

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
