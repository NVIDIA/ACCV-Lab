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

from nvidia.dali.types import DALIDataType

from accvlab.dali_pipeline_framework.inputs import ShuffledShardedInputCallable, DataProvider
from accvlab.dali_pipeline_framework.pipeline import (
    SampleDataGroup,
    PipelineDefinition,
    DALIStructuredOutputIterator,
)
from accvlab.dali_pipeline_framework.processing_steps import PaddingToUniform


class SimpleTestProvider(DataProvider):
    """Simple test provider with minimal data structure for testing padding."""

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Create test data with different shapes based on sample_id
        # Simple case: just vary the height of images
        heights = [32, 48, 64]
        height = heights[sample_id % len(heights)]

        # Create image with varying height but same width
        res["image"] = np.random.randint(0, 256, (height, 64, 3), dtype=np.uint8)

        # Create nested metadata with varying lengths
        lengths = [3, 4, 5]
        length = lengths[sample_id % len(lengths)]

        # Main metadata field
        res["metadata"]["values"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0][:length])

        # Nested metadata field with different varying lengths
        nested_lengths = [2, 3, 4]
        nested_length = nested_lengths[sample_id % len(nested_lengths)]
        res["metadata"]["features"] = np.array([10.0, 20.0, 30.0, 40.0][:nested_length])

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 3

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()
        res.add_data_field("image", DALIDataType.UINT8)

        # Create nested metadata structure
        metadata = SampleDataGroup()
        metadata.add_data_field("values", DALIDataType.FLOAT)
        metadata.add_data_field("features", DALIDataType.FLOAT)
        res.add_data_group_field("metadata", metadata)

        return res


@pytest.mark.parametrize(
    "field_names,fill_value,expected_image_shape,expected_metadata_values_shape,expected_metadata_features_shape",
    [
        (None, 0.0, (64, 64, 3), (5,), (4,)),  # All fields padded to max size
        (None, 255, (64, 64, 3), (5,), (4,)),  # All fields padded to max size with different fill value
        ("image", 0.0, (64, 64, 3), (5,), (4,)),  # Only image field padded
        (["image", "values"], 0.0, (64, 64, 3), (5,), (4,)),  # Image and specific nested field padded
        (["image", "features"], 0.0, (64, 64, 3), (5,), (4,)),  # Image and specific nested field padded
    ],
    ids=["all_fields_0", "all_fields_255", "only_image", "image_and_values", "image_and_features"],
)
def test_padding_to_uniform_parametrized(
    field_names,
    fill_value,
    expected_image_shape,
    expected_metadata_values_shape,
    expected_metadata_features_shape,
):
    """Test PaddingToUniform with different field names and fill values."""
    provider = SimpleTestProvider()
    input_callable = ShuffledShardedInputCallable(
        data_provider=provider,
        batch_size=3,
        shard_id=0,
        num_shards=1,
        shuffle=False,
    )

    step = PaddingToUniform(field_names=field_names, fill_value=fill_value)

    passthrough_copy_kwargs = {}
    if field_names == "image":
        passthrough_copy_kwargs = {
            "copy_external_source_passthrough_outputs": True,
            "passthrough_copy_branch_paths": ["metadata"],
        }
    elif field_names == ["image", "values"]:
        passthrough_copy_kwargs = {
            "copy_external_source_passthrough_outputs": True,
            "passthrough_copy_field_names": ["features"],
            "passthrough_copy_field_names_scope_paths": ["metadata"],
        }
    elif field_names == ["image", "features"]:
        passthrough_copy_kwargs = {
            "copy_external_source_passthrough_outputs": True,
            "passthrough_copy_field_names": ["values"],
            "passthrough_copy_field_names_scope_paths": ["metadata"],
        }

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        **passthrough_copy_kwargs,
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=3,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    # Build the pipeline
    pipeline.build()

    # Get the raw DALI output (tensor lists) - this allows non-uniform shapes
    output = pipeline.run()

    # Get the field names in the order they appear in the output
    field_names_flat = pipeline_def.check_and_get_output_data_structure().field_names_flat

    # Convert field_names to list if it's a string
    if isinstance(field_names, str):
        field_names_list = [field_names]
    elif field_names is None:
        field_names_list = ["image", "values", "features"]
    else:
        field_names_list = field_names

    # Check that all fields are present
    assert "image" in field_names_flat
    assert "metadata.values" in field_names_flat
    assert "metadata.features" in field_names_flat

    # Get field indices
    image_idx = field_names_flat.index("image")
    metadata_values_idx = field_names_flat.index("metadata.values")
    metadata_features_idx = field_names_flat.index("metadata.features")

    # Check image field
    image_tensor_list = output[image_idx]
    assert len(image_tensor_list) == 3  # batch_size=3

    # Check metadata fields
    metadata_values_tensor_list = output[metadata_values_idx]
    metadata_features_tensor_list = output[metadata_features_idx]
    assert len(metadata_values_tensor_list) == 3  # batch_size=3
    assert len(metadata_features_tensor_list) == 3  # batch_size=3

    # Check that all samples in the batch have the expected shape for specified fields
    for i in range(3):
        image_tensor = image_tensor_list[i]
        metadata_values_tensor = metadata_values_tensor_list[i]
        metadata_features_tensor = metadata_features_tensor_list[i]

        # Check image shape
        if "image" in field_names_list or field_names is None:
            # Should be padded to uniform shape
            assert list(image_tensor.shape()) == list(expected_image_shape)
        else:
            # Should have original varying shapes
            original_heights = [32, 48, 64]
            expected_height = original_heights[i % len(original_heights)]
            assert list(image_tensor.shape()) == [expected_height, 64, 3]

        # Check metadata.values shape
        if "values" in field_names_list or field_names is None:
            # Should be padded to uniform shape
            assert list(metadata_values_tensor.shape()) == list(expected_metadata_values_shape)
        else:
            # Should have original varying shapes
            original_lengths = [3, 4, 5]
            expected_length = original_lengths[i % len(original_lengths)]
            assert list(metadata_values_tensor.shape()) == [expected_length]

        # Check metadata.features shape
        if "features" in field_names_list or field_names is None:
            # Should be padded to uniform shape
            assert list(metadata_features_tensor.shape()) == list(expected_metadata_features_shape)
        else:
            # Should have original varying shapes
            original_nested_lengths = [2, 3, 4]
            expected_nested_length = original_nested_lengths[i % len(original_nested_lengths)]
            assert list(metadata_features_tensor.shape()) == [expected_nested_length]


def test_padding_to_uniform_nested_structure():
    """Test PaddingToUniform with nested structure - padding the entire metadata group."""
    provider = SimpleTestProvider()
    input_callable = ShuffledShardedInputCallable(
        data_provider=provider,
        batch_size=3,
        shard_id=0,
        num_shards=1,
        shuffle=False,
    )

    step = PaddingToUniform(field_names="metadata", fill_value=0.0)

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["image"],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=3,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    # Build the pipeline
    pipeline.build()

    # Get the raw DALI output (tensor lists) - this allows non-uniform shapes
    output = pipeline.run()

    # Get the field names in the order they appear in the output
    field_names_flat = pipeline_def.check_and_get_output_data_structure().field_names_flat

    # Check that all fields are present
    assert "image" in field_names_flat
    assert "metadata.values" in field_names_flat
    assert "metadata.features" in field_names_flat

    # Get field indices
    image_idx = field_names_flat.index("image")
    metadata_values_idx = field_names_flat.index("metadata.values")
    metadata_features_idx = field_names_flat.index("metadata.features")

    # Check image field
    image_tensor_list = output[image_idx]
    assert len(image_tensor_list) == 3  # batch_size=3

    # Check metadata fields
    metadata_values_tensor_list = output[metadata_values_idx]
    metadata_features_tensor_list = output[metadata_features_idx]
    assert len(metadata_values_tensor_list) == 3  # batch_size=3
    assert len(metadata_features_tensor_list) == 3  # batch_size=3

    # Check that all samples in the batch have the expected shape
    for i in range(3):
        image_tensor = image_tensor_list[i]
        metadata_values_tensor = metadata_values_tensor_list[i]
        metadata_features_tensor = metadata_features_tensor_list[i]

        # Image should have original varying shapes (not in field_names)
        original_heights = [32, 48, 64]
        expected_height = original_heights[i % len(original_heights)]
        assert list(image_tensor.shape()) == [expected_height, 64, 3]

        # metadata.values should be padded to uniform shape (part of "metadata" group)
        assert list(metadata_values_tensor.shape()) == [5]  # max of [3, 4, 5]

        # metadata.features should be padded to uniform shape (part of "metadata" group)
        assert list(metadata_features_tensor.shape()) == [4]  # max of [2, 3, 4]


def test_padding_to_uniform_with_iterator():
    """Test PaddingToUniform with iterator to ensure uniform shapes for tensor output."""
    provider = SimpleTestProvider()
    input_callable = ShuffledShardedInputCallable(
        data_provider=provider,
        batch_size=3,
        shard_id=0,
        num_shards=1,
        shuffle=False,
    )

    step = PaddingToUniform(field_names=None, fill_value=0.0)

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
    )

    pipeline = pipeline_def.get_dali_pipeline(
        enable_conditionals=True,
        batch_size=3,
        prefetch_queue_depth=1,
        num_threads=1,
        py_start_method="spawn",
    )

    iterator = DALIStructuredOutputIterator(10, pipeline, pipeline_def.check_and_get_output_data_structure())
    iterator_iter = iter(iterator)

    # Get one batch
    batch = next(iterator_iter)

    # Check that the pipeline runs without errors
    assert batch is not None
    assert "image" in batch
    assert "metadata" in batch

    # Check that images have uniform shape (should be padded to max height 64)
    assert batch["image"].shape[1:] == (64, 64, 3)

    # Check that nested metadata fields have uniform shapes
    assert batch["metadata"]["values"].shape[1:] == (5,)  # max of [3, 4, 5]
    assert batch["metadata"]["features"].shape[1:] == (4,)  # max of [2, 3, 4]


def test_padding_to_uniform_data_format_check():
    """Test that PaddingToUniform correctly checks data format."""
    step = PaddingToUniform(field_names="image", fill_value=0.0)

    # Create a simple data structure with nested metadata
    data = SampleDataGroup()
    data.add_data_field("image", DALIDataType.UINT8)

    metadata = SampleDataGroup()
    metadata.add_data_field("values", DALIDataType.FLOAT)
    metadata.add_data_field("features", DALIDataType.FLOAT)
    data.add_data_group_field("metadata", metadata)

    # This should not raise an error
    result = step._check_and_adjust_data_format_input_to_output(data)
    assert result is data


def test_padding_to_uniform_field_not_found():
    """Test that PaddingToUniform raises error for non-existent fields."""
    step = PaddingToUniform(field_names="nonexistent_field", fill_value=0.0)

    # Create a simple data structure
    data = SampleDataGroup()
    data.add_data_field("image", DALIDataType.UINT8)

    # This should raise a ValueError
    with pytest.raises(ValueError, match="No fields to be padded with name 'nonexistent_field' were found"):
        step._check_and_adjust_data_format_input_to_output(data)


if __name__ == "__main__":
    pytest.main([__file__])
