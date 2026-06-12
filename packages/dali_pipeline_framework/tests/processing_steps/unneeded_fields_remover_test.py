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
from accvlab.dali_pipeline_framework.processing_steps import UnneededFieldRemover


class TestProvider(DataProvider):

    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure
        # Create test data with various fields to test removal
        res["annotation"]["num_lidar_points"] = np.array([0, 1, 0, 1, 5, 10])
        res["annotation"]["num_radar_points"] = np.array([9, 10, 0, 1, 0, 0])
        res["annotation"]["visibility_levels"] = np.array([1, 1, 1, 0, 2, 3])
        res["annotation"]["categories"] = np.array([0, 1, 2, 3, 4, 5])
        res["annotation"]["is_bbox_in_range"] = np.array([True, True, False, True, True, False])
        res["annotation"]["other_field"] = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
        res["annotation"]["temp_field"] = np.array([100, 200, 300, 400, 500, 600])

        # Add nested annotation inside the main annotation for testing nested structure
        res["annotation"]["annotation"]["num_lidar_points"] = np.array([0, 0, 0, 1])
        res["annotation"]["annotation"]["num_radar_points"] = np.array([0, 100, 0, 1])
        res["annotation"]["annotation"]["visibility_levels"] = np.array([1, 2, 1, -1])
        res["annotation"]["annotation"]["categories"] = np.array([0, 1, 2, 3])
        res["annotation"]["annotation"]["is_bbox_in_range"] = np.array([True, True, False, True])
        res["annotation"]["annotation"]["other_field"] = np.array([100.0, 200.0, 300.0, 400.0])
        res["annotation"]["annotation"]["temp_field"] = np.array([1000, 2000, 3000, 4000])

        # Add some top-level fields
        res["other_data"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        res["metadata"] = np.array([1000, 2000, 3000, 4000, 5000, 6000])
        res["temp_field"] = np.array([10000, 20000, 30000, 40000, 50000, 60000])

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()

        # Main annotation group
        annotation = SampleDataGroup()
        annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
        annotation.add_data_field("num_radar_points", DALIDataType.INT32)
        annotation.add_data_field("visibility_levels", DALIDataType.INT32)
        annotation.add_data_field("categories", DALIDataType.INT32)
        annotation.add_data_field("is_bbox_in_range", DALIDataType.BOOL)
        annotation.add_data_field("other_field", DALIDataType.FLOAT)
        annotation.add_data_field("temp_field", DALIDataType.INT32)

        # Nested annotation group inside the main annotation (same name for testing find_all_occurrences)
        nested_annotation = SampleDataGroup()
        nested_annotation.add_data_field("num_lidar_points", DALIDataType.INT32)
        nested_annotation.add_data_field("num_radar_points", DALIDataType.INT32)
        nested_annotation.add_data_field("visibility_levels", DALIDataType.INT32)
        nested_annotation.add_data_field("categories", DALIDataType.INT32)
        nested_annotation.add_data_field("is_bbox_in_range", DALIDataType.BOOL)
        nested_annotation.add_data_field("other_field", DALIDataType.FLOAT)
        nested_annotation.add_data_field("temp_field", DALIDataType.INT32)
        annotation.add_data_group_field("annotation", nested_annotation)  # Nested with same name

        res.add_data_group_field("annotation", annotation)
        res.add_data_field("other_data", DALIDataType.FLOAT)
        res.add_data_field("metadata", DALIDataType.INT32)
        res.add_data_field("temp_field", DALIDataType.INT32)
        return res


class TestProviderNestedFields(DataProvider):
    @override
    def get_data(self, sample_id: int) -> SampleDataGroup:
        res = self.sample_data_structure

        # Create nested structure with opposite nesting orders & nested fields with the same name
        # Opposite nesting orders
        res["a_to_delete"]["z_to_delete"] = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        res["z_to_delete"]["a_to_delete"] = np.array([4.0, 5.0, 6.0], dtype=np.float32)
        # Nested fields with the same name
        res["a_to_delete"]["a_to_delete"] = np.array([7.0, 8.0, 9.0], dtype=np.float32)

        # Add some fields that should remain
        res["keep_this_field"] = np.array([10.0, 20.0], dtype=np.float32)
        res["nested"]["keep_this_too"] = np.array([30.0, 40.0], dtype=np.float32)

        return res

    @override
    def get_number_of_samples(self) -> int:
        return 10

    @property
    @override
    def sample_data_structure(self) -> SampleDataGroup:
        res = SampleDataGroup()

        # Create nested structure for a_to_delete
        a_to_delete_1 = SampleDataGroup()
        a_to_delete_1.add_data_field("z_to_delete", DALIDataType.FLOAT)
        a_to_delete_1.add_data_field("a_to_delete", DALIDataType.FLOAT)
        res.add_data_group_field("a_to_delete", a_to_delete_1)

        # Create nested structure for z_to_delete
        z_to_delete_1 = SampleDataGroup()
        z_to_delete_1.add_data_field("a_to_delete", DALIDataType.FLOAT)
        res.add_data_group_field("z_to_delete", z_to_delete_1)

        # Add fields that should remain
        res.add_data_field("keep_this_field", DALIDataType.FLOAT)

        # Add nested structure that should remain
        nested = SampleDataGroup()
        nested.add_data_field("keep_this_too", DALIDataType.FLOAT)
        res.add_data_group_field("nested", nested)

        return res


def test_remove_single_field():
    """Test removing a single field."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = UnneededFieldRemover(["temp_field"])

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data", "metadata"],
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

    # Check that temp_field was removed from all locations
    assert "temp_field" not in res
    assert "temp_field" not in res["annotation"]
    assert "temp_field" not in res["annotation"]["annotation"]

    # Check that the structure is preserved
    assert "annotation" in res
    assert "annotation" in res["annotation"]  # Nested annotation

    # Check that other fields are preserved
    assert "num_lidar_points" in res["annotation"]
    assert "num_radar_points" in res["annotation"]
    assert "visibility_levels" in res["annotation"]
    assert "categories" in res["annotation"]
    assert "is_bbox_in_range" in res["annotation"]
    assert "other_field" in res["annotation"]
    assert "other_data" in res
    assert "metadata" in res

    # Check that remaining fields have correct data
    assert torch.equal(
        res["annotation"]["num_lidar_points"][0], torch.tensor([0, 1, 0, 1, 5, 10], dtype=torch.int32)
    )
    assert torch.equal(
        res["annotation"]["num_radar_points"][0], torch.tensor([9, 10, 0, 1, 0, 0], dtype=torch.int32)
    )
    assert torch.equal(
        res["annotation"]["visibility_levels"][0], torch.tensor([1, 1, 1, 0, 2, 3], dtype=torch.int32)
    )
    assert torch.equal(
        res["annotation"]["categories"][0], torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.int32)
    )
    assert torch.equal(
        res["annotation"]["is_bbox_in_range"][0],
        torch.tensor([True, True, False, True, True, False], dtype=torch.bool),
    )
    assert torch.equal(
        res["annotation"]["other_field"][0], torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    )
    assert torch.equal(res["other_data"][0], torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))
    assert torch.equal(
        res["metadata"][0], torch.tensor([1000, 2000, 3000, 4000, 5000, 6000], dtype=torch.int32)
    )


@pytest.mark.parametrize(
    "field_names",
    [
        ["temp_field", "metadata", "other_field"],
        ("temp_field", "metadata", "other_field"),
    ],
)
def test_remove_multiple_fields(field_names):
    """Test removing multiple fields at once."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = UnneededFieldRemover(field_names)

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data"],
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

    # Check that all specified fields were removed
    assert "temp_field" not in res
    assert "temp_field" not in res["annotation"]
    assert "temp_field" not in res["annotation"]["annotation"]
    assert "metadata" not in res
    assert "other_field" not in res["annotation"]
    assert "other_field" not in res["annotation"]["annotation"]

    # Check that other fields are preserved
    assert "num_lidar_points" in res["annotation"]
    assert "num_radar_points" in res["annotation"]
    assert "visibility_levels" in res["annotation"]
    assert "categories" in res["annotation"]
    assert "is_bbox_in_range" in res["annotation"]
    assert "other_data" in res


def test_remove_nonexistent_fields():
    """Test that removing non-existent fields doesn't cause errors."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    step = UnneededFieldRemover(["nonexistent_field_1", "nonexistent_field_2"])

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_branch_paths=["annotation", "other_data", "metadata", "temp_field"],
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

    # Check that all original fields are preserved
    assert "num_lidar_points" in res["annotation"]
    assert "num_radar_points" in res["annotation"]
    assert "visibility_levels" in res["annotation"]
    assert "categories" in res["annotation"]
    assert "is_bbox_in_range" in res["annotation"]
    assert "other_field" in res["annotation"]
    assert "temp_field" in res["annotation"]
    assert "other_data" in res
    assert "metadata" in res
    assert "temp_field" in res


def test_remove_all_fields():
    """Test removing all fields from a specific group."""
    provider = TestProvider()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Remove all fields from the nested annotation
    step = UnneededFieldRemover(
        [
            "num_lidar_points",
            "num_radar_points",
            "visibility_levels",
            "categories",
            "is_bbox_in_range",
            "other_field",
            "temp_field",
        ]
    )

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["other_data", "metadata"],
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

    # Check that all specified fields were removed from both main and nested annotation
    for field in [
        "num_lidar_points",
        "num_radar_points",
        "visibility_levels",
        "categories",
        "is_bbox_in_range",
        "other_field",
        "temp_field",
    ]:
        assert field not in res["annotation"]
        assert field not in res["annotation"]["annotation"]

    # Check that top-level fields that were not in the removal list are preserved
    assert "other_data" in res
    assert "metadata" in res
    # Note: temp_field was also removed from top-level because remove_all_occurrences removes ALL instances


def test_remove_nested_to_remove_fields():
    """Test removing nested fields to ensure no crashes."""
    provider = TestProviderNestedFields()
    input_callable = ShuffledShardedInputCallable(
        provider,
        batch_size=1,
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    # Remove both a_to_delete and z_to_delete fields (which are nested in opposite orders)
    step = UnneededFieldRemover(["a_to_delete", "z_to_delete"])

    pipeline_def = PipelineDefinition(
        data_loading_callable_iterable=input_callable,
        preprocess_functors=[step],
        copy_external_source_passthrough_outputs=True,
        passthrough_copy_field_names=["keep_this_field"],
        passthrough_copy_branch_paths=["nested"],
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

    # Check that both nested fields were successfully deleted
    assert "a_to_delete" not in res
    assert "z_to_delete" not in res

    # Check that fields that should remain are still present
    assert "keep_this_field" in res
    assert "nested" in res
    assert "keep_this_too" in res["nested"]

    # Verify the data values are correct
    assert torch.equal(res["keep_this_field"][0], torch.tensor([10.0, 20.0]))
    assert torch.equal(res["nested"]["keep_this_too"][0], torch.tensor([30.0, 40.0]))


if __name__ == "__main__":
    pytest.main([__file__])
